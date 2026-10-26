"""Обёртка над best_accuracy.pth: препроцесс + CTC-декод + уверенность."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.model import DEFAULT_CHARSET, ANPRNet, load_model


@dataclass
class HeadResult:
    text: str
    conf: float
    chars: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class RecogResult:
    text: str
    conf: float
    chars: list[tuple[str, float]]
    type_code: str = ""
    type_conf: float = 0.0


class CTCDecoder:
    """Greedy CTC + ограниченный beam-search по top-k символам."""

    def __init__(self, charset: str = DEFAULT_CHARSET):
        if not charset:
            raise ValueError("charset пустой")
        # индекс 0 = blank
        self.charset = charset
        self.itos = ["[blank]"] + list(charset)

    def __len__(self) -> int:
        return len(self.itos)

    def greedy(self, probs: np.ndarray) -> HeadResult:
        """probs: [T, num_class] softmax."""
        idx = probs.argmax(axis=1)
        pmax = probs.max(axis=1)
        chars: list[tuple[str, float]] = []
        prev = -1
        for t, k in enumerate(idx):
            if k != prev and k != 0:
                chars.append((self.itos[k], float(pmax[t])))
            prev = int(k)
        text = "".join(c for c, _ in chars)
        # уверенность: геометрическое среднее по всем шагам (как в EasyOCR, но без
        # взрыва длины — prod^(1/T)), устойчиво к длине последовательности
        conf = float(np.exp(np.log(np.clip(pmax, 1e-12, 1.0)).mean())) if len(pmax) else 0.0
        return HeadResult(text=text, conf=conf, chars=chars)

    def beam_alternatives(
        self, probs: np.ndarray, topk: int = 3, beam: int = 12, max_alts: int = 24
    ) -> list[tuple[str, float]]:
        """Варианты строки от greedy-пути с заменой символов на top-k альтернативы.

        Дешёвая замена полноценного beam search: берём greedy-выравнивание, для
        каждой выданной позиции рассматриваем topk символов и ищем комбинации с
        максимальным score. Нужно для форматно-ограниченного декода (plate_rules).
        """
        idx = probs.argmax(axis=1)
        positions: list[np.ndarray] = []
        prev = -1
        for t, k in enumerate(idx):
            if k != prev and k != 0:
                positions.append(probs[t])
            prev = int(k)
        if not positions:
            return []

        cands: list[tuple[str, float]] = [("", 0.0)]
        for step in positions:
            order = np.argsort(step)[::-1][:topk]
            nxt: list[tuple[str, float]] = []
            for text, logp in cands:
                for k in order:
                    if k == 0:
                        continue
                    nxt.append((text + self.itos[k], logp + float(np.log(max(step[k], 1e-12)))))
            nxt.sort(key=lambda x: -x[1])
            cands = nxt[:beam]
        n = len(positions)
        return [(t, float(np.exp(lp / max(n, 1)))) for t, lp in cands[:max_alts]]


def find_line_gap(crop: np.ndarray) -> int | None:
    """Строка для разреза двухстрочной пластины — по минимуму «чернил» в середине.

    Возвращает индекс строки или None, если разумного разреза нет.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return None
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    profile = binary.mean(axis=1) / 255.0
    k = max(3, h // 14) | 1
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    lo, hi = int(h * 0.32), int(h * 0.68)
    if hi - lo < 2:
        return None
    gap = int(lo + profile[lo:hi].argmin())
    if min(gap, h - gap) < 8:  # строки слишком тонкие, читать нечего
        return None
    return gap


def split_lines(crop: np.ndarray) -> list[np.ndarray]:
    """Режет кроп на две строки (сверху вниз) — порядок чтения номеров УЗ."""
    gap = find_line_gap(crop)
    if gap is None:
        return [crop]
    h = crop.shape[0]
    pad = max(1, h // 25)
    return [crop[: min(h, gap + pad)], crop[max(0, gap - pad) :]]


#: пропорции однострочной пластины УЗ — под них обучалась модель.
#: Короткую строку («2373») нельзя растягивать на все 100 px: символы становятся
#: почти в два раза шире тренировочных, и CTC начинает их дублировать (2373 -> 23773).
PLATE_AR = 4.3


def pad_to_ratio(img: np.ndarray, target_ar: float = PLATE_AR) -> np.ndarray:
    """Досыпает справа фон, пока кроп не станет пропорциями похож на пластину."""
    h, w = img.shape[:2]
    need = int(h * target_ar)
    if h <= 0 or need <= w:
        return img
    edge = img[:, -max(2, w // 15) :]
    flat = edge.reshape(-1, edge.shape[2]) if img.ndim == 3 else edge.reshape(-1, 1)
    fill = np.median(flat, axis=0)
    shape = (h, need, img.shape[2]) if img.ndim == 3 else (h, need)
    out = np.empty(shape, img.dtype)
    out[:] = fill.astype(img.dtype)
    out[:, :w] = img
    return out


def layout_variants(crop: np.ndarray, max_ar: float = 2.3) -> list[list[np.ndarray]]:
    """Варианты раскладки пластины для распознавания.

    Квадратные кропы (прицепы, мототранспорт) могут быть и одной строкой с большими
    полями, и двумя строками — по одной геометрии не отличить. Поэтому читаем оба
    варианта, а выбор оставляем правилам форматов: двухстрочный номер, прочитанный
    как одна строка, даёт мусор, который ни в один формат не укладывается.

    Первый вариант всегда однострочный — на него же откат, если ничего не валидно.
    """
    h, w = crop.shape[:2]
    if h <= 0 or w / h > max_ar:
        return [[crop]]
    lines = split_lines(crop)
    if len(lines) == 1:
        return [[crop]]
    return [[crop], [pad_to_ratio(ln) for ln in lines]]


def unfold_lines(crop: np.ndarray, max_ar: float = 2.6) -> np.ndarray:
    """Двухстрочный кроп -> одна строка: верхняя строка, справа от неё нижняя.

    Порядок строк на пластине прицепа совпадает с порядком текста номера:
    «8815» сверху, «BA 01» снизу -> «8815BA01» (проверено глазами на выборке).
    Развёртка по пропорциям похожа на однострочную пластину и читается за один
    проход, а не двумя, как строки по отдельности (184 мс против 97 на видео).

    Читать развёртку модель учится в tools.train_recog — до дообучения она её
    не знает. Обучение и чтение обязаны резать одной и той же функцией, иначе
    модель увидит в работе не то, на чём училась.

    Не квадратный кроп или кроп без разреза возвращается как есть — тем же
    объектом, по этому вызывающий отличает «развернул» от «не стал».
    """
    h, w = crop.shape[:2]
    if h <= 0 or w / h > max_ar:
        return crop
    lines = split_lines(crop)
    if len(lines) != 2:
        return crop
    lh = min(ln.shape[0] for ln in lines)  # общая высота, ширина — пропорционально
    parts = [
        cv2.resize(
            ln, (max(1, round(ln.shape[1] * lh / ln.shape[0])), lh), interpolation=cv2.INTER_AREA
        )
        for ln in lines
    ]
    return np.hstack(parts)


def moto_unfold(crop: np.ndarray, min_ar: float = 0.6, max_ar: float = 1.9) -> np.ndarray:
    """Квадратный номер мото/спецтехники (сетка 2x2) -> одна строка.

    Раскладка не как у прицепа: регион стоит в отдельном окне справа-сверху, а не
    в конце нижней строки.

        197 | 01        читается 197 -> DA -> 01  (регион последний в ТЕКСТЕ,
        DA  | флаг                                 но физически он в правом окне)

    Поэтому обычная развёртка «верх+низ» дала бы 19701DA — неверно. Режем иначе:
    левый блок (цифры над буквами) разворачиваем как две строки -> «197»+«DA»,
    затем приклеиваем регион из правого верхнего окна -> «197»+«DA»+«01».

    Не квадрат — возвращаем как есть (тем же объектом), по этому вызывающий
    отличает «развернул» от «не стал».
    """
    h, w = crop.shape[:2]
    if h <= 0 or not min_ar <= w / h <= max_ar:
        return crop
    # левый блок — основной (цифры/буквы), правый — окно региона с флагом УЗ
    split = int(w * 0.66)
    left, right = crop[:, :split], crop[:, split:]
    region = right[: int(right.shape[0] * 0.55)]  # верх правого окна — цифры региона
    main = unfold_lines(left, max_ar=max_ar)  # «197» над «DA» -> в строку
    if main is left:  # левый блок не раскрылся на две строки — не мото-раскладка
        return crop
    lh = main.shape[0]
    rw = max(1, round(region.shape[1] * lh / max(region.shape[0], 1)))
    region = cv2.resize(region, (rw, lh), interpolation=cv2.INTER_AREA)
    return np.hstack([main, region])


def unfold_variants(crop: np.ndarray, max_ar: float = 2.6) -> list[list[np.ndarray]]:
    """Как layout_variants, но двухстрочный вариант — одна развёрнутая строка.

    Однострочный вариант остаётся первым: квадратный кроп бывает и однострочным
    номером с большими полями, а выбор между ними делают правила форматов.
    """
    flat = unfold_lines(crop, max_ar)
    return [[crop]] if flat is crop else [[crop], [flat]]


def prep_gray(
    img: np.ndarray, h: int, w: int, preprocess: str = "resize", contrast_boost: bool = False
) -> np.ndarray:
    """Кроп -> массив [h, w] в диапазоне [-1, 1], как его видит модель.

    Вынесено из PlateRecognizer, чтобы обучение (tools.train_recog) готовило
    кропы тем же кодом, что и чтение.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if contrast_boost:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

    if preprocess == "pad":
        ratio = img.shape[1] / max(img.shape[0], 1)
        new_w = min(w, max(1, int(np.ceil(h * ratio))))
        resized = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((h, w), dtype=resized.dtype)
        canvas[:, :new_w] = resized
        if new_w < w:  # padding краевым столбцом, как NormalizePAD в DTRB/EasyOCR
            canvas[:, new_w:] = resized[:, -1:]
        img = canvas
    else:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    arr = img.astype(np.float32) / 255.0
    return (arr - 0.5) / 0.5


class PlateRecognizer:
    """Ленивая загрузка модели, батч-инференс по кропам номеров."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        charset: str = DEFAULT_CHARSET,
        preprocess: str = "resize",  # resize | pad
        contrast_boost: bool = False,
        num_threads: int = 0,
        text_head: str = "anpr",
        type_head: str | None = "ctype",
        two_line: bool = True,
        two_line_max_ar: float = 2.6,
        two_line_mode: str = "split",
    ):
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        self.device = device
        self.preprocess = preprocess
        self.contrast_boost = contrast_boost
        self.text_head = text_head
        self.type_head = type_head
        self.two_line = two_line
        self.two_line_max_ar = two_line_max_ar
        if two_line_mode not in ("split", "unfold"):
            raise ValueError(f"two_line_mode: split | unfold, получено {two_line_mode!r}")
        self.two_line_mode = two_line_mode
        self.model: ANPRNet
        self.model, self.info = load_model(weights, device=device)
        self.decoder = CTCDecoder(charset)
        if len(self.decoder) != self.info.num_class:
            raise ValueError(
                f"charset даёт {len(self.decoder)} классов, а в чекпоинте {self.info.num_class}. "
                "Поправьте ANPR_CHARSET в .env"
            )
        if self.text_head not in self.info.heads:
            raise ValueError(f"головы {self.info.heads}, запрошена {self.text_head!r}")
        if self.type_head and self.type_head not in self.info.heads:
            self.type_head = None

    # ------------------------------------------------------------------ препроцесс
    def _prep_one(self, img: np.ndarray) -> np.ndarray:
        return prep_gray(
            img, self.info.img_h, self.info.img_w, self.preprocess, self.contrast_boost
        )

    def to_tensor(self, crops: list[np.ndarray]) -> torch.Tensor:
        batch = np.stack([self._prep_one(c) for c in crops])[:, None, :, :]
        return torch.from_numpy(batch).to(self.device)

    # ------------------------------------------------------------------- инференс
    @torch.inference_mode()
    def raw_probs(self, crops: list[np.ndarray]) -> dict[str, np.ndarray]:
        """-> {head: [B, T, num_class]} softmax."""
        if not crops:
            return {}
        out = self.model(self.to_tensor(crops))
        return {k: torch.softmax(v, dim=2).cpu().numpy() for k, v in out.items()}

    def recognize(self, crops: list[np.ndarray]) -> list[RecogResult]:
        probs = self.raw_probs(crops)
        if not probs:
            return []
        results: list[RecogResult] = []
        text_p = probs[self.text_head]
        type_p = probs.get(self.type_head) if self.type_head else None
        for i in range(text_p.shape[0]):
            main = self.decoder.greedy(text_p[i])
            r = RecogResult(text=main.text, conf=main.conf, chars=main.chars)
            if type_p is not None:
                t = self.decoder.greedy(type_p[i])
                r.type_code, r.type_conf = t.text, t.conf
            results.append(r)
        return results

    def recognize_with_alternatives(
        self, crops: list[np.ndarray], topk: int = 3, beam: int = 12
    ) -> list[tuple[RecogResult, list[tuple[str, float]]]]:
        """То же, но с альтернативами для форматно-ограниченного выбора."""
        probs = self.raw_probs(crops)
        if not probs:
            return []
        text_p = probs[self.text_head]
        type_p = probs.get(self.type_head) if self.type_head else None
        out = []
        for i in range(text_p.shape[0]):
            main = self.decoder.greedy(text_p[i])
            r = RecogResult(text=main.text, conf=main.conf, chars=main.chars)
            if type_p is not None:
                t = self.decoder.greedy(type_p[i])
                r.type_code, r.type_conf = t.text, t.conf
            alts = self.decoder.beam_alternatives(text_p[i], topk=topk, beam=beam)
            out.append((r, alts))
        return out

    # ---------------------------------------------------------- чтение с учётом строк
    def read(
        self,
        crops: list[np.ndarray],
        topk: int = 3,
        beam: int = 12,
        validate: Callable[[str], bool] | None = None,
    ) -> list[tuple[RecogResult, list[tuple[str, float]]]]:
        """Основной вход: сам разбирается с двухстрочными пластинами.

        Для квадратных кропов читаются обе раскладки (одна строка / две), и выигрывает
        та, чей текст проходит `validate` (правила форматов). Все строки всех вариантов
        уходят в модель одним батчем, так что цена — только лишний элемент батча.
        """
        if not crops:
            return []
        if not self.two_line:
            return self.recognize_with_alternatives(crops, topk=topk, beam=beam)

        pieces: list[np.ndarray] = []
        plan: list[list[list[int]]] = []  # crop -> варианты -> индексы строк
        layouts = unfold_variants if self.two_line_mode == "unfold" else layout_variants
        for crop in crops:
            variants = layouts(crop, max_ar=self.two_line_max_ar)
            plan.append(self._register(variants, pieces))

        parts = self.recognize_with_alternatives(pieces, topk=topk, beam=beam)

        out: list[tuple[RecogResult, list[tuple[str, float]]]] = []
        for variants in plan:
            cands = [
                parts[idx[0]] if len(idx) == 1 else _merge_lines([parts[i] for i in idx], beam)
                for idx in variants
            ]
            out.append(_pick_variant(cands, validate))
        return out

    @staticmethod
    def _register(variants: list[list[np.ndarray]], pieces: list[np.ndarray]) -> list[list[int]]:
        """Складывает строки всех вариантов в общий батч, возвращает их индексы."""
        idx: list[list[int]] = []
        for lines in variants:
            idx.append(list(range(len(pieces), len(pieces) + len(lines))))
            pieces.extend(lines)
        return idx


def _pick_variant(
    cands: list[tuple[RecogResult, list[tuple[str, float]]]],
    validate: Callable[[str], bool] | None,
) -> tuple[RecogResult, list[tuple[str, float]]]:
    """Выбор между раскладками: побеждает только валидный формат.

    Если ни один вариант не укладывается в формат — возвращаем первый (однострочный).
    Так альтернативная раскладка не может испортить результат: она либо даёт валидный
    номер, либо просто игнорируется.
    """
    if len(cands) == 1 or validate is None:
        return cands[0]
    valid = [c for c in cands if validate(c[0].text) or any(validate(t) for t, _ in c[1][:4])]
    return max(valid, key=lambda c: c[0].conf) if valid else cands[0]


def _merge_lines(
    parts: list[tuple[RecogResult, list[tuple[str, float]]]], max_alts: int = 12
) -> tuple[RecogResult, list[tuple[str, float]]]:
    """Склеивает строки двухстрочного номера сверху вниз."""
    text = "".join(p[0].text for p in parts)
    # уверенность всей пластины = слабейшая строка: ошибка в любой ломает номер
    conf = min(p[0].conf for p in parts)
    chars: list[tuple[str, float]] = []
    for p in parts:
        chars.extend(p[0].chars)
    merged = RecogResult(
        text=text,
        conf=conf,
        chars=chars,
        type_code="".join(p[0].type_code for p in parts),
        type_conf=min(p[0].type_conf for p in parts),
    )

    # альтернативы — произведение вариантов по строкам, score = min по строкам
    combos: list[tuple[str, float]] = [("", 1.0)]
    for p in parts:
        alts = p[1][:4] or [(p[0].text, p[0].conf)]
        combos = [(t + at, min(c, ac)) for t, c in combos for at, ac in alts]
        combos.sort(key=lambda x: -x[1])
        combos = combos[:max_alts]
    return merged, combos

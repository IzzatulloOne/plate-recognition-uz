"""Правила номеров Узбекистана: форматы, цвет фона, исправление путаницы символов.

Форматы лежат в PLATE_FORMATS и переопределяются файлом plate_formats.json
(см. README) — их легко подправить под свои данные без правки кода.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: буква 'I' в номерах УЗ не используется (подтверждено анализом весов head_anpr)
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

#: коды регионов УЗ
VALID_REGIONS = {f"{i:02d}" for i in range(1, 96)}


@dataclass(frozen=True)
class PlateFormat:
    name: str
    regex: str
    plate_class: str  # private | legal | electric | foreign | trailer | moto | special
    region_group: int = 1  # какая группа содержит код региона (у прицепов он последний)
    pretty: str = ""  # шаблон вывода, например "{0} {1} {2} {3}"

    def compiled(self) -> re.Pattern:
        return re.compile(self.regex)


#: порядок важен — первый матч выигрывает.
#: Форматы сверены с реальными номерами из галереи platesmania.com/uz (175 шт.):
#:   DD L DDD LL  (148)  01 A 111 AK   физлица
#:   DD DDD LLL   ( 19)  01 006 DFA    юрлица
#:   DDDD LL DD   (  8)  4169 AA 70    прицепы/спецтехника — регион в КОНЦЕ, две строки
PLATE_FORMATS: list[PlateFormat] = [
    # 01 A 123 BC — физлица
    PlateFormat("uz_private", rf"^(\d{{2}})([{LETTERS}])(\d{{3}})([{LETTERS}]{{2}})$", "private"),
    # 01 123 ABC — юрлица
    PlateFormat("uz_legal", rf"^(\d{{2}})(\d{{3}})([{LETTERS}]{{3}})$", "legal"),
    # 4169 AA 70 — прицепы и спецтехника: квадратная пластина в две строки,
    # регион последний (проверено по фото platesmania)
    PlateFormat(
        "uz_trailer",
        rf"^(\d{{4}})([{LETTERS}]{{2}})(\d{{2}})$",
        "trailer",
        region_group=3,
    ),
    # Девятисимвольные: регион + одна буква серии + 6 цифр. Именно из этого набора
    # состоят цветные номера УЗ, и алфавит второй головы модели (цифры + H + M) ровно
    # ему соответствует — см. ANPR_TYPE_HEAD_COLORS в README.
    #   01 M 018461 — зелёные, электромобили (встречены в выборке platesmania)
    #   01 H 018461 — жёлтые, транспорт иностранцев (буква — по смыслу серии)
    PlateFormat("uz_electric", r"^(\d{2})(M)(\d{6})$", "electric"),
    PlateFormat("uz_foreign", r"^(\d{2})(H)(\d{6})$", "foreign"),
    PlateFormat("uz_series6", rf"^(\d{{2}})([{LETTERS}])(\d{{6}})$", "special"),
    # 01 A 1234 — мототранспорт
    PlateFormat("uz_moto", rf"^(\d{{2}})([{LETTERS}])(\d{{4}})$", "moto"),
    # 01 AB 123 — служебные/спец
    PlateFormat("uz_special", rf"^(\d{{2}})([{LETTERS}]{{2}})(\d{{3}})$", "special"),
]

_FORMATS_JSON = Path(__file__).resolve().parent.parent.parent / "plate_formats.json"
if _FORMATS_JSON.exists():
    PLATE_FORMATS = [PlateFormat(**f) for f in json.loads(_FORMATS_JSON.read_text())]

_COMPILED = [(f, f.compiled()) for f in PLATE_FORMATS]

#: OCR-путаница: чем можно заменить символ, если он стоит не на своём месте
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}


@dataclass
class PlateMatch:
    text: str
    format: str | None
    plate_class: str
    groups: tuple[str, ...] = ()
    region_ok: bool = False
    corrected: bool = False

    @property
    def valid(self) -> bool:
        return self.format is not None

    def pretty(self) -> str:
        return " ".join(self.groups) if self.groups else self.text


def normalize(text: str) -> str:
    """Верхний регистр, только [0-9A-Z]."""
    return re.sub(r"[^0-9A-Z]", "", text.upper())


def match(text: str) -> PlateMatch:
    t = normalize(text)
    for fmt, rx in _COMPILED:
        m = rx.match(t)
        if m:
            return PlateMatch(
                text=t,
                format=fmt.name,
                plate_class=fmt.plate_class,
                groups=m.groups(),
                region_ok=m.group(fmt.region_group) in VALID_REGIONS,
            )
    return PlateMatch(text=t, format=None, plate_class="unknown")


def _coerce_to(text: str, pattern: str) -> str | None:
    """Пробует привести строку к шаблону, меняя цифры<->похожие буквы по позициям.

    pattern — строка из 'D' (цифра) и 'L' (буква) той же длины.
    """
    if len(text) != len(pattern):
        return None
    out = []
    for ch, kind in zip(text, pattern, strict=True):
        if kind == "D":
            if ch in DIGITS:
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch])
            else:
                return None
        else:
            if ch in LETTERS:
                out.append(ch)
            elif ch in TO_LETTER and TO_LETTER[ch] in LETTERS:
                out.append(TO_LETTER[ch])
            else:
                return None
    return "".join(out)


#: скелеты позиций, соответствующие PLATE_FORMATS (D=цифра, L=буква)
_SKELETONS = ["DDLDDDLL", "DDDDDLLL", "DDDDLLDD", "DDLDDDDDD", "DDLDDDD", "DDLLDDD"]


def dedup_variants(text: str, max_removals: int = 2) -> list[str]:
    """Варианты строки с удалением повторяющихся подряд символов.

    CTC при широких пластинах склонен дублировать символ (`01M0021672` вместо
    `01M021672`, `23773` вместо `2373`). Просто схлопнуть все повторы нельзя —
    в номерах есть законные `007`, — поэтому перебираем варианты и принимаем
    только те, что укладываются в формат.
    """
    out: list[str] = []
    seen = {text}
    frontier = [text]
    for _ in range(max_removals):
        nxt: list[str] = []
        for t in frontier:
            for i in range(1, len(t)):
                if t[i] == t[i - 1]:
                    cand = t[:i] + t[i + 1 :]
                    if cand not in seen:
                        seen.add(cand)
                        nxt.append(cand)
                        out.append(cand)
        frontier = nxt
    return out


def repair(text: str) -> PlateMatch:
    """Матч как есть; иначе правим путаницу символов и лишние повторы от CTC."""
    m = match(text)
    if m.valid:
        return m
    t = normalize(text)

    for skel in _SKELETONS:
        cand = _coerce_to(t, skel)
        if cand:
            m2 = match(cand)
            if m2.valid:
                m2.corrected = True
                return m2

    for shrunk in dedup_variants(t):
        m2 = match(shrunk)
        if m2.valid:
            m2.corrected = True
            return m2
        for skel in _SKELETONS:
            cand = _coerce_to(shrunk, skel)
            if cand:
                m3 = match(cand)
                if m3.valid:
                    m3.corrected = True
                    return m3
    return m


def pick_best(
    alternatives: list[tuple[str, float]], min_conf: float = 0.0
) -> tuple[str, float, PlateMatch]:
    """Из вариантов CTC выбирает лучший, дающий валидный формат.

    Возвращает (text, conf, match). Если валидных нет — берёт вариант №1
    и пытается его починить.
    """
    if not alternatives:
        return "", 0.0, PlateMatch(text="", format=None, plate_class="unknown")

    for text, conf in alternatives:
        if conf < min_conf:
            continue
        m = match(text)
        if m.valid and m.region_ok:
            return m.text, conf, m
    for text, conf in alternatives:
        m = match(text)
        if m.valid:
            return m.text, conf, m

    # ни один вариант не подошёл как есть — пробуем починить каждый по порядку
    # (важно для вывода второй головы: он приходит последним кандидатом)
    for text, conf in alternatives:
        m = repair(text)
        if m.valid:
            return m.text, conf, m

    text, conf = alternatives[0]
    m = repair(text)
    return m.text, conf, m


# --------------------------------------------------------------------- цвет фона
#: (имя, H_min, H_max, S_min, V_min) в OpenCV HSV (H: 0..179)
_COLOR_BANDS = [
    ("yellow", 15, 38, 70, 90),
    ("green", 39, 89, 60, 60),
    ("blue", 90, 130, 70, 60),
    ("red", 0, 10, 90, 70),
]


def classify_color(crop: np.ndarray) -> tuple[str, float]:
    """Цвет фона номера по центральной области кропа.

    Жёлтый/зелёный/синий/красный определяем по HSV, иначе white/black по яркости.
    Возвращает (цвет, доля пикселей).
    """
    if crop.ndim != 3 or crop.size == 0:
        return "unknown", 0.0
    h, w = crop.shape[:2]
    y0, y1 = int(h * 0.15), max(int(h * 0.85), int(h * 0.15) + 1)
    x0, x1 = int(w * 0.05), max(int(w * 0.95), int(w * 0.05) + 1)
    roi = crop[y0:y1, x0:x1]
    if roi.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    total = float(H.size)

    best, best_frac = "", 0.0
    for name, hmin, hmax, smin, vmin in _COLOR_BANDS:
        mask = (H >= hmin) & (H <= hmax) & (S >= smin) & (V >= vmin)
        frac = float(mask.sum()) / total
        if frac > best_frac:
            best, best_frac = name, frac
    if best_frac >= 0.22:
        return best, best_frac

    bright = float((V >= 140).mean())
    if bright >= 0.45:
        return "white", bright
    return "black", 1.0 - bright

"""Сборка датасета узбекских номеров из galery platesmania — для дообучения.

Готовой YOLO, обученной на номерах УЗ, в открытом доступе нет (проверено: на
HuggingFace таких моделей нет, на Roboflow Universe есть проект из 40 картинок).
Зато platesmania отдаёт вместе с каждым фото точный текст номера, поэтому разметку
можно получить автоматически:

    фото + текст номера (с сайта)
        -> YOLO ищет пластину          -> бокс
        -> распознаватель читает кроп   -> текст
        -> текст совпал с сайтом?       -> да: бокс и кроп размечены ВЕРНО

Что получается на выходе:

  dataset/   YOLO-формат (images/labels/data.yaml) — дообучение детектора под УЗ:
             ракурсы, фон, пропорции пластин, грузовики и прицепы.
  recog/     кропы + gt.txt (`путь<TAB>ТЕКСТ`) — дообучение распознавателя
             в формате deep-text-recognition-benchmark / EasyOCR. Здесь метки
             настоящие (с сайта), а не предсказанные, — включая двухстрочные
             квадратные номера прицепов, которые текущий чекпоинт не читает.
  missed/    фото, где пластина не найдена вовсе — их надо разметить руками,
             автоматика тут бессильна (детектор не может научиться находить то,
             что сам не находит).

    python -m tools.build_dataset --fetch 300        # скачать выборку
    python -m tools.build_dataset --build            # разметить и разложить
    python -m tools.build_dataset --fetch 300 --build --include-unverified
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2

from app.config import settings
from app.core import plate_rules
from tools.eval_uz import fetch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "uz"
INDEX = DATA / "index.json"
EVAL_INDEX = ROOT / "data" / "eval" / "index.json"

DATA_YAML = """# Датасет номеров Узбекистана, разметка получена автоматически
# (бокс от YOLO, проверен совпадением OCR с текстом номера с platesmania).
path: {path}
train: images/train
val: images/val
names:
  0: plate
"""


#: доли по тексту номера: эталон eval-v2, валидация, остальное — обучение
TEST_SHARE, VAL_SHARE = 0.10, 0.10
BENCH = ROOT / "data" / "eval-v2"


def _split(plate: str) -> str:
    """train / val / test — детерминированно по тексту номера.

    По тексту, а не по pid фото: 56 номеров сняты на нескольких фото, и при
    разбиении по фото одна и та же машина попадала бы и в обучение, и в эталон.

    test — новый эталон data/eval-v2. Он не попадает ни в обучение, ни в
    валидацию, и в отличие от старого data/eval (80 фото, из них прицепов 4)
    в нём сотни номеров каждого массового класса.
    """
    h = int(hashlib.sha1(plate.encode()).hexdigest()[:8], 16) % 100 / 100.0
    if h < TEST_SHARE:
        return "test"
    return "val" if h < TEST_SHARE + VAL_SHARE else "train"


def write_benchmark(items: dict) -> Path:
    """Эталон eval-v2 из отложенной части выборки: формат index.json как у eval_uz.

    Пути к фото — относительно папки эталона, сами фото не копируются.
    Модель машины едет вместе с номером: на ней можно мерить и распознавание
    марки, когда оно появится.
    """
    BENCH.mkdir(parents=True, exist_ok=True)
    bench = {
        pid: {"gt": v["gt"], "file": f"../uz/{v['file']}", "photo": v.get("photo", ""),
              "model": v.get("model", "")}
        for pid, v in items.items()
        if _split(v["gt"]) == "test"
    }
    (BENCH / "index.json").write_text(json.dumps(bench, ensure_ascii=False, indent=1))
    by_class: dict[str, int] = {}
    for v in bench.values():
        c = plate_rules.match(v["gt"]).plate_class or "?"
        by_class[c] = by_class.get(c, 0) + 1
    with_model = sum(bool(v["model"]) for v in bench.values())
    print(f"эталон {BENCH}: {len(bench)} фото, {by_class}, с моделью машины {with_model}")
    return BENCH


def _eval_holdout() -> tuple[set[str], set[str]]:
    """pid и тексты эталонной выборки tools.eval_uz — в датасет им нельзя.

    data/uz набиралась с тех же страниц галереи, что и data/eval: общих фото
    нашлось 41. Модель, видевшая экзамен, дала бы на замере 72/80 красивую
    цифру, которая ничего не значит. Текст исключаем тоже: та же машина на
    другом снимке для модели почти тот же пример.
    """
    if not EVAL_INDEX.exists():
        return set(), set()
    ev = json.loads(EVAL_INDEX.read_text())
    return set(ev), {v["gt"] for v in ev.values()}


def _lev(a: str, b: str) -> int:
    """Расстояние Левенштейна — сколько правок отделяет одну строку от другой."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def _link(src: Path, dst: Path) -> None:
    """Положить фото в датасет жёсткой ссылкой, а не копией.

    Копии съели 5 ГБ из 20 свободных: сборка клала каждое фото ещё раз в
    dataset/images и в missed. Ссылка — тот же файл на диске, места не занимает.
    На другой файловой системе ссылка невозможна — тогда обычная копия.
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def dedupe() -> int:
    """Копии фото в dataset/ и missed/ -> жёсткие ссылки на оригиналы. Возвращает байты.

    Старые сборки клали копии; ссылка — тот же файл, место освобождается.
    Совпадение проверяется по размеру и содержимому, а не по имени.
    """
    import filecmp

    photos = {f.stem: f for f in (DATA / "photos").glob("*.jpg")}
    freed = 0
    for f in [*(DATA / "dataset" / "images").rglob("*.jpg"), *(DATA / "missed").glob("*.jpg")]:
        src = photos.get(f.stem.split("_")[-1])
        if src is None or f.stat().st_ino == src.stat().st_ino:
            continue
        if f.stat().st_size == src.stat().st_size and filecmp.cmp(f, src, shallow=False):
            freed += f.stat().st_size
            _link(src, f)
    return freed


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _trailer_block(crop):
    """Для прицепа сверяем верхнюю строку (4 цифры). None — кроп не двухстрочный."""
    from app.core.recognizer import pad_to_ratio, split_lines

    lines = split_lines(crop)
    return pad_to_ratio(lines[0]) if len(lines) == 2 else None


def _moto_block(crop):
    """Для мото сверяем левый верхний блок (3 цифры): регион справа, буквы снизу."""
    h, w = crop.shape[:2]
    return crop[: int(h * 0.55), : int(w * 0.66)]


#: как подтверждать квадратный номер: диапазон пропорций, чем сверять, сколько
#: первых символов GT, чем разворачивать для полного чтения дообученной моделью
SQUARE = {
    "trailer": (0.9, 2.6, _trailer_block, 4, "unfold_lines"),
    "moto": (0.6, 1.9, _moto_block, 3, "moto_unfold"),
    "special": (0.6, 1.9, _moto_block, 3, "moto_unfold"),
}

DET_BENCH = ROOT / "data" / "det-bench" / "items.json"


def _hand_boxes() -> dict[str, list[list[int]]]:
    """Размеченные вручную рамки из эталона детектора: {pid -> [[x1,y1,x2,y2], ...]}.

    Для мото это надёжнее авто-проверки (у неё было 6 из 20): человек разметил
    точно. test-доля эталона в обучение всё равно не идёт — её отсекает _split.
    """
    if not DET_BENCH.exists():
        return {}
    from tools.label_boxes import _load_boxes

    out: dict[str, list[list[int]]] = {}
    for pid, v in json.loads(DET_BENCH.read_text()).items():
        if v.get("done"):
            boxes = _load_boxes(pid, v["w"], v["h"]) or []
            if boxes:
                out[pid] = boxes
    return out


def _from_hand(pipe, img, boxes, match):
    """Из рамок человека выбрать пластину номера -> (бокс, кроп, прочие рамки).

    Пластину узнаём по опорному блоку (как в _square_crop): у мото левые 3 цифры.
    Если ни одна рамка не прочиталась — берём самую квадратную, рамки всё равно
    верные (их ставил человек), а кроп нужен лишь распознавателю.
    """
    from app.core.detector import Detection, crop_plate

    lo, hi, block_fn, nchars, _fold = SQUARE[match.plate_class]
    expect = match.text[:nchars]
    scored = []
    for box in boxes:
        det = Detection(x1=box[0], y1=box[1], x2=box[2], y2=box[3], conf=1.0)
        crop = crop_plate(img, det, settings.crop_padding)
        if not crop.size:
            continue
        ar = crop.shape[1] / max(crop.shape[0], 1)
        if not lo <= ar <= hi:  # мото-пластина квадратная; широкая рамка — не она
            continue
        block = block_fn(crop)
        dist = 9
        if block is not None and getattr(block, "size", 0):
            read = pipe.recognizer.recognize([block])[0].text
            dist = _lev("".join(c for c in plate_rules.normalize(read) if c.isdigit()), expect)
        squareness = abs(1.0 - ar)  # чем ближе к квадрату, тем вероятнее мото-пластина
        scored.append((dist, squareness, box, crop))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    _, _, box, crop = scored[0]
    others = [b for b in boxes if b is not box]
    return box, crop, others


def _square_crop(pipe, img, match, verifier=None):
    """Квадратный номер (прицеп/мото), подтверждённый чтением -> (det, кроп, прочие).

    Целиком такую пластину исходная модель не читает, а опорный блок — верхнюю
    строку прицепа или левые цифры мото — читает. Совпал с началом номера с
    точностью до правки — значит, вырезана та самая пластина. Так отсеивается то,
    что детектор берёт по ошибке: номер тягача, надпись на борту, бренд на баке
    мотоцикла (SBK вместо пластины).

    verifier — дообученная модель: читает развёртку целиком и сверяет весь номер.
    На прицепах это вернуло 91 из 606 против 36 у сверки по одному блоку.

    Три прохода детектора: обычный, крупный, крупный с низким порогом — нужную
    пластину детектор часто видит с уверенностью ниже 0.10, а ложные рамки того
    же порога отсекает чтение. Прочие уверенные пластины первого прохода идут в
    разметку YOLO, иначе детектор учился бы не видеть соседние номера.
    """
    from app.core.detector import crop_plate
    from app.core.recognizer import moto_unfold, unfold_lines

    cfg = pipe.settings
    lo, hi, block_fn, nchars, fold_name = SQUARE[match.plate_class]
    fold = {"unfold_lines": unfold_lines, "moto_unfold": moto_unfold}[fold_name]
    expect = match.text[:nchars]
    first: list = []
    passes = ((cfg.det_imgsz, None), (cfg.det_retry_imgsz, cfg.det_retry_conf),
              (cfg.det_retry_imgsz, 0.03))
    for n, (imgsz, conf) in enumerate(passes):
        dets = pipe.detector.detect(img, imgsz=imgsz, conf=conf)
        if n == 0:
            first = dets
        best = None
        for det in dets:
            crop = crop_plate(img, det, cfg.crop_padding)
            h, w = crop.shape[:2]
            if not crop.size or not lo <= w / max(h, 1) <= hi:
                continue
            block = block_fn(crop)
            if block is None or not getattr(block, "size", 0):
                continue
            read = pipe.recognizer.recognize([block])[0].text
            dist = _lev("".join(ch for ch in plate_rules.normalize(read) if ch.isdigit()),
                        expect)
            if dist > 1 and verifier is not None and fold(crop) is not crop:
                full = plate_rules.normalize(verifier.recognize([fold(crop)])[0].text)
                dist = _lev(full, match.text)
            if dist <= 1 and (best is None or dist < best[0]):
                best = (dist, det, crop)
        if best:
            box = [best[1].x1, best[1].y1, best[1].x2, best[1].y2]
            others = [[d.x1, d.y1, d.x2, d.y2] for d in first
                      if _iou([d.x1, d.y1, d.x2, d.y2], box) < 0.3]
            return best[1], best[2], others
    return None

def write_manifest(verifier_weights: Path | None = None, stats: dict | None = None) -> Path:
    """Что лежит в датасете и как собрано — считается по самим файлам, а не по памяти."""
    from tools import provenance

    by_class: dict[str, int] = {}
    gt_lines = [ln for ln in (DATA / "recog" / "gt.txt").read_text().splitlines() if ln.strip()]
    for ln in gt_lines:
        c = plate_rules.match(ln.split("\t")[1]).plate_class or "?"
        by_class[c] = by_class.get(c, 0) + 1
    yolo = {part: len(list((DATA / "dataset" / "images" / part).glob("*.jpg")))
            for part in ("train", "val")}
    bench_index = BENCH / "index.json"
    bench = json.loads(bench_index.read_text()) if bench_index.exists() else {}
    return provenance.write(DATA / "manifest.json", {
        "что": "датасет номеров УЗ (tools.build_dataset --build)",
        "индекс": {"фото": len(json.loads(INDEX.read_text())),
                   "sha256": provenance.sha256(INDEX)},
        "распознаватель": {"кропов": len(gt_lines), "по классам": by_class,
                           "gt.txt sha256": provenance.sha256(DATA / "recog" / "gt.txt")},
        "детектор YOLO": yolo,
        "эталон eval-v2": len(bench),
        "разбиение": {"ключ": "текст номера (sha1)", "test": TEST_SHARE, "val": VAL_SHARE},
        "проверяльщик квадратных": ({"файл": str(verifier_weights),
                                      "sha256": provenance.sha256(verifier_weights)}
                                     if verifier_weights else "исходная модель, опорный блок"),
        "итог сборки": stats or {},
    })


def build(include_unverified: bool = False, verifier_weights: Path | None = None) -> int:
    from app.core.detector import Detection, crop_plate
    from app.core.pipeline import ANPRPipeline

    if not INDEX.exists():
        print("нет выборки — сначала: python -m tools.build_dataset --fetch 300")
        return 2
    items = json.loads(INDEX.read_text())
    pipe = ANPRPipeline(settings)
    verifier = None
    if verifier_weights:
        from app.core.recognizer import PlateRecognizer

        verifier = PlateRecognizer(verifier_weights, device=settings.device, two_line=False)
        print(f"проверяльщик прицепов: {verifier_weights}")

    ds, recog, missed = DATA / "dataset", DATA / "recog", DATA / "missed"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    (recog / "images").mkdir(parents=True, exist_ok=True)
    missed.mkdir(parents=True, exist_ok=True)

    gt_lines: list[str] = []
    stats = {"всего": 0, "в эталоне eval — пропущено": 0, "в эталон eval-v2 — отложено": 0,
             "проверено OCR": 0, "  из них трудных (1–2 ошибки)": 0,
             "бокс без проверки": 0, "пластина не найдена": 0,
             "прицеп: подтверждён чтением": 0, "прицеп: не подтверждён": 0,
             "мото/спец: подтверждён чтением": 0, "мото/спец: не подтверждён": 0,
             "прицеп: по рамкам человека": 0, "мото/спец: по рамкам человека": 0}
    by_class: dict[str, int] = {}
    hold_pids, hold_texts = _eval_holdout()
    hand = _hand_boxes()  # размеченные руками рамки: эталонные метки для мото/спец

    def emit(pid: str, img, box, crop, gt: str, others=()) -> None:
        """Записать пример в оба датасета: фото + все пластины для YOLO, кроп + текст для OCR.

        others — прочие пластины кадра: без них YOLO учился бы не видеть номер тягача.
        """
        cls = plate_rules.match(gt).plate_class
        by_class[cls] = by_class.get(cls, 0) + 1
        part = _split(gt)
        _link(DATA / items[pid]["file"], ds / f"images/{part}/{pid}.jpg")
        h, w = img.shape[:2]
        rows = []
        for x1, y1, x2, y2 in [list(box), *others]:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            rows.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (ds / f"labels/{part}/{pid}.txt").write_text("\n".join(rows) + "\n")
        if crop.size:
            name = f"images/{pid}_{gt}.jpg"
            cv2.imwrite(str(recog / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gt_lines.append(f"{name}\t{gt}")

    for pid, v in items.items():
        gt = v["gt"]
        if pid in hold_pids or gt in hold_texts:
            stats["в эталоне eval — пропущено"] += 1
            continue
        if _split(gt) == "test":
            stats["в эталон eval-v2 — отложено"] += 1
            continue
        img = cv2.imread(str(DATA / v["file"]))
        if img is None:
            continue
        stats["всего"] += 1

        m = plate_rules.match(gt)
        if m.plate_class in SQUARE:  # прицепы, мото, спецтехника — квадратные
            kind = "прицеп" if m.plate_class == "trailer" else "мото/спец"
            if pid in hand:  # рамки человека точнее авто-проверки
                found = _from_hand(pipe, img, hand[pid], m)
                if found is not None:
                    det_box, crop, others = found
                    stats[f"{kind}: по рамкам человека"] += 1
                    emit(pid, img, det_box, crop, gt, others)
                    continue
            found = _square_crop(pipe, img, m, verifier)
            if found is None:
                stats[f"{kind}: не подтверждён"] += 1
                _link(DATA / v["file"], missed / f"{m.plate_class}_{gt}_{pid}.jpg")
                continue
            det, crop, others = found
            stats[f"{kind}: подтверждён чтением"] += 1
            emit(pid, img, (det.x1, det.y1, det.x2, det.y2), crop, gt, others)
            continue

        res = pipe.process_frame(img)
        if not res.plates:
            stats["пластина не найдена"] += 1
            _link(DATA / v["file"], missed / f"{gt}_{pid}.jpg")
            continue

        # Бокс подтверждён, если модель прочитала номер с сайта точно или почти —
        # с одной-двумя ошибками. «Почти» и есть трудные примеры, ради которых
        # дообучают: метка всё равно берётся с сайта, а не из чтения.
        dist, hit = min(((_lev(plate_rules.normalize(p.text), gt), p) for p in res.plates),
                        key=lambda t: t[0])
        verified = dist <= 2
        if not verified:
            hit = None
        if hit is None:
            if not include_unverified or len(res.plates) != 1:
                stats["бокс без проверки"] += 1
                _link(DATA / v["file"], missed / f"unverified_{gt}_{pid}.jpg")
                continue
            hit = res.plates[0]  # одна пластина на фото — бокс скорее всего верный

        if verified:
            stats["проверено OCR"] += 1
            if dist:
                stats["  из них трудных (1–2 ошибки)"] += 1
        else:
            stats["бокс без проверки"] += 1
        # кроп + НАСТОЯЩИЙ текст с сайта, бокс — от проверенного чтения
        x1, y1, x2, y2 = hit.box
        det = Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=hit.det_conf)
        others = [p.box for p in res.plates
                  if p is not hit and p.det_conf >= settings.det_conf]
        emit(pid, img, hit.box, crop_plate(img, det, settings.crop_padding), gt, others)

    (recog / "gt.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""))
    write_benchmark(items)
    write_manifest(verifier_weights, stats)
    (ds / "data.yaml").write_text(DATA_YAML.format(path=ds.resolve()))

    print("\n=== итог")
    for k, val in stats.items():
        print(f"  {k:22s} {val}")
    print(f"  кропов для распознавателя: {len(gt_lines)}")
    print("  по классам номеров:", by_class or "—")
    n_train = len(list((ds / "images/train").glob("*.jpg")))
    n_val = len(list((ds / "images/val").glob("*.jpg")))
    print(f"  YOLO: train={n_train}, val={n_val}  ->  {ds}")
    print(f"  распознаватель: {recog}/gt.txt")
    if stats["пластина не найдена"] or stats["бокс без проверки"]:
        print(f"  на ручную разметку: {missed}")

    print(
        "\n=== дальше\n"
        "  распознаватель: python -m tools.kaggle_bundle -> notebooks/train_kaggle.ipynb\n"
        "  детектор:       python -m tools.kaggle_bundle --detector\n"
        "                  -> notebooks/train_detector_kaggle.ipynb\n"
        "  эталон детектора (разметка руками): python -m tools.label_boxes"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fetch", type=int, metavar="N", help="скачать N фото с текстами")
    ap.add_argument("--build", action="store_true", help="разметить и разложить датасет")
    ap.add_argument(
        "--include-unverified",
        action="store_true",
        help="брать бокс и без совпадения OCR, если на фото одна пластина",
    )
    ap.add_argument(
        "--formats",
        help="качать только эти форматы, напр. uz_trailer — иначе прицепы (5.8% галереи)\n"
        "набираются недопустимо медленно",
    )
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--verifier", type=Path, metavar="PTH",
                    help="дообученный распознаватель: подтверждает прицепы чтением целиком")
    ap.add_argument("--dedupe", action="store_true",
                    help="копии фото в dataset/ и missed/ превратить в жёсткие ссылки")
    ap.add_argument("--manifest", action="store_true",
                    help="только записать data/uz/manifest.json по готовым файлам")
    ap.add_argument("--benchmark", action="store_true",
                    help="только пересобрать эталон data/eval-v2 из собранного")
    args = ap.parse_args()

    if args.benchmark:
        write_benchmark(json.loads(INDEX.read_text()))
        return 0
    if args.dedupe:
        print(f"освобождено {dedupe() / 1e9:.1f} ГБ")
        return 0
    if args.manifest:
        print(write_manifest(args.verifier))
        return 0

    if args.fetch:
        fmts = {f.strip() for f in args.formats.split(",")} if args.formats else None
        fetch(args.fetch, out_dir=DATA, formats=fmts, max_pages=args.max_pages)
    if args.build:
        return build(args.include_unverified, args.verifier)
    if not args.fetch:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

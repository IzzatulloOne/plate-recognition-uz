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
import shutil
from pathlib import Path

import cv2

from app.config import settings
from app.core import plate_rules
from tools.eval_uz import fetch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "uz"
INDEX = DATA / "index.json"

DATA_YAML = """# Датасет номеров Узбекистана, разметка получена автоматически
# (бокс от YOLO, проверен совпадением OCR с текстом номера с platesmania).
path: {path}
train: images/train
val: images/val
names:
  0: plate
"""


def _split(pid: str, val_share: float = 0.2) -> str:
    """Детерминированное разбиение train/val — чтобы повторные прогоны совпадали."""
    h = int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16)
    return "val" if (h % 100) / 100.0 < val_share else "train"


def build(include_unverified: bool = False, val_share: float = 0.2) -> int:
    from app.core.detector import Detection, crop_plate
    from app.core.pipeline import ANPRPipeline

    if not INDEX.exists():
        print("нет выборки — сначала: python -m tools.build_dataset --fetch 300")
        return 2
    items = json.loads(INDEX.read_text())
    pipe = ANPRPipeline(settings)

    ds, recog, missed = DATA / "dataset", DATA / "recog", DATA / "missed"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    (recog / "images").mkdir(parents=True, exist_ok=True)
    missed.mkdir(parents=True, exist_ok=True)

    gt_lines: list[str] = []
    stats = {"всего": 0, "проверено OCR": 0, "бокс без проверки": 0, "пластина не найдена": 0}
    by_class: dict[str, int] = {}

    for pid, v in items.items():
        img = cv2.imread(str(DATA / v["file"]))
        if img is None:
            continue
        stats["всего"] += 1
        gt = v["gt"]
        res = pipe.process_frame(img)
        if not res.plates:
            stats["пластина не найдена"] += 1
            shutil.copy(DATA / v["file"], missed / f"{gt}_{pid}.jpg")
            continue

        # ищем бокс, чей текст совпал с сайтом — такой бокс и кроп размечены точно
        hit = next((p for p in res.plates if plate_rules.normalize(p.text) == gt), None)
        verified = hit is not None
        if hit is None:
            if not include_unverified or len(res.plates) != 1:
                stats["бокс без проверки"] += 1
                shutil.copy(DATA / v["file"], missed / f"unverified_{gt}_{pid}.jpg")
                continue
            hit = res.plates[0]  # одна пластина на фото — бокс скорее всего верный

        stats["проверено OCR" if verified else "бокс без проверки"] += 1
        cls = plate_rules.match(gt).plate_class
        by_class[cls] = by_class.get(cls, 0) + 1

        # --- YOLO: изображение + бокс в нормализованных xywh
        part = _split(pid, val_share)
        shutil.copy(DATA / v["file"], ds / f"images/{part}/{pid}.jpg")
        h, w = img.shape[:2]
        x1, y1, x2, y2 = hit.box
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        (ds / f"labels/{part}/{pid}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )

        # --- распознаватель: кроп + НАСТОЯЩИЙ текст с сайта
        det = Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=hit.det_conf)
        crop = crop_plate(img, det, settings.crop_padding)
        if crop.size:
            name = f"images/{pid}_{gt}.jpg"
            cv2.imwrite(str(recog / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gt_lines.append(f"{name}\t{gt}")

    (recog / "gt.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""))
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
        "\n=== дообучение детектора (нужно >= несколько сотен фото)\n"
        f"  .venv/bin/yolo detect train model=models/yolo11n-plate.pt data={ds}/data.yaml \\\n"
        "      epochs=60 imgsz=960 batch=8 device=cpu name=uz-plate\n"
        "  затем: ANPR_DETECTOR_WEIGHTS=runs/detect/uz-plate/weights/best.pt\n"
        "\n=== дообучение распознавателя (deep-text-recognition-benchmark)\n"
        "  git clone https://github.com/clovaai/deep-text-recognition-benchmark\n"
        f"  python create_lmdb_dataset.py --inputPath {recog} --gtFile {recog}/gt.txt \\\n"
        "      --outputPath lmdb/uz\n"
        "  python train.py --train_data lmdb/uz --valid_data lmdb/uz \\\n"
        "      --Transformation TPS --FeatureExtraction ResNet \\\n"
        "      --SequenceModeling BiLSTM --Prediction CTC \\\n"
        "      --saved_model best_accuracy.pth --FT \\\n"
        "      --character '0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ'\n"
        "  ВАЖНО: --character меняет число классов, поэтому голову придётся\n"
        "  переинициализировать (или оставить исходные 94 символа, чтобы\n"
        "  чекпоинт грузился целиком и работал этот сервис без правок)."
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
    ap.add_argument("--val-share", type=float, default=0.2)
    args = ap.parse_args()

    if args.fetch:
        fetch(args.fetch, out_dir=DATA)
    if args.build:
        return build(args.include_unverified, args.val_share)
    if not args.fetch:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

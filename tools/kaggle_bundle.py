"""Архив для Kaggle: кропы с метками + снимок кода, без единого полного фото.

    python -m tools.kaggle_bundle              # распознаватель -> data/kaggle/anpr-uz.zip
    python -m tools.kaggle_bundle --detector   # детектор -> data/kaggle/anpr-uz-det.zip

Загружать на kaggle.com/datasets только с доступом Private: кропы вырезаны
из фотографий с чужого сайта, публиковать их нельзя.

Код едет внутри архива, чтобы ноутбук не зависел от того, что лежит в GitHub:
обучение проверяется на той версии, что собрана здесь.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECOG = ROOT / "data" / "uz" / "recog"
OUT = ROOT / "data" / "kaggle" / "anpr-uz.zip"
CODE = ("app", "tools", "pyproject.toml")


DETSET = ROOT / "data" / "uz" / "dataset"
DET_OUT = ROOT / "data" / "kaggle" / "anpr-uz-det.zip"
BASE_DETECTOR = ROOT / "models" / "yolo11n-plate.pt"


def detector_bundle(max_side: int = 1024, max_other: int = 2500, seed: int = 0) -> int:
    """Архив для дообучения детектора: все фото с прицепами + столько же прочих.

    Все 8 тысяч фото в оригинале — около 2 ГБ. Прицепы — то, чему учим, их берём
    все; однострочных столько, чтобы детектор их не разучился видеть. Фото
    ужимаются до max_side по длинной стороне: учим на 960, детали не теряются,
    а разметка YOLO в долях кадра от масштаба не зависит.
    """
    import random

    import cv2

    from app.core import plate_rules

    plates_of = {}
    for ln in (RECOG / "gt.txt").read_text().splitlines():
        if ln.strip():
            rel, text = ln.split("\t")
            plates_of[Path(rel).stem.split("_")[0]] = text
    rng = random.Random(seed)
    picked: list[Path] = []
    for part in ("train", "val"):
        imgs = sorted((DETSET / "images" / part).glob("*.jpg"))
        trailers = [f for f in imgs
                    if plate_rules.match(plates_of.get(f.stem, "")).plate_class == "trailer"]
        is_trailer = set(trailers)
        other = [f for f in imgs if f not in is_trailer]
        share = max_other if part == "train" else max_other // 8
        picked += trailers + rng.sample(other, min(share, len(other)))

    DET_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DET_OUT, "w", zipfile.ZIP_STORED) as z:  # JPEG уже сжат
        for f in picked:
            part = f.parent.name
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            k = max_side / max(h, w)
            if k < 1:
                img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            z.writestr(f"det/images/{part}/{f.name}", buf.tobytes())
            z.write(DETSET / "labels" / part / f"{f.stem}.txt", f"det/labels/{part}/{f.stem}.txt")
        z.writestr("det/data.yaml", "path: .\ntrain: images/train\nval: images/val\n"
                   "names:\n  0: plate\n")
        z.write(BASE_DETECTOR, "det/yolo11n-plate.pt")
    n_tr = sum(plate_rules.match(plates_of.get(f.stem, "")).plate_class == "trailer"
               for f in picked)
    print(f"{DET_OUT}  {DET_OUT.stat().st_size / 1e6:.0f} МБ, фото {len(picked)}, "
          f"из них с прицепами {n_tr}")
    print("Загрузка: kaggle.com/datasets -> New Dataset -> этот zip, доступ Private")
    return 0


def main() -> int:
    import sys

    if "--detector" in sys.argv:
        return detector_bundle()
    gt = RECOG / "gt.txt"
    if not gt.exists():
        print("нет data/uz/recog/gt.txt — сначала: python -m tools.build_dataset --build")
        return 2
    lines = [ln for ln in gt.read_text().splitlines() if ln.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("recog/gt.txt", "\n".join(lines) + "\n")
        for ln in lines:  # только то, что в gt.txt: старые кропы прошлых сборок не едут
            rel = ln.split("\t")[0]
            z.write(RECOG / rel, f"recog/{rel}")
        for name in CODE:
            src = ROOT / name
            files = [src] if src.is_file() else sorted(src.rglob("*"))
            for f in files:
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f"code/{f.relative_to(ROOT)}")
    mb = OUT.stat().st_size / 1e6
    print(f"{OUT}  {mb:.1f} МБ, кропов {len(lines)}")
    print("Загрузка: kaggle.com/datasets -> New Dataset -> этот zip, доступ Private")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

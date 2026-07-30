"""Скачать веса YOLO-детектора номерных пластин.

Два источника:

  lpr   — license_plate_detector.pt из репозитория
          Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
          (YOLOv8n). Детектор по умолчанию в этом проекте.
  n/s/m/l/x — morsetechlab/yolov11-license-plate-detection (YOLO11 разных размеров).

Замеры на выборке из 40 узбекских номеров (CPU, оригиналы ~1500 px, imgsz=640,
без второго прохода) — подробности в README:

    lpr-yolov8n-plate  37/40 найдено,  60 мс/кадр
    yolo11n-plate      37/40 найдено,  66 мс/кадр
    yolo11s-plate      37/40 найдено, 167 мс/кадр   <- медленнее, не точнее

    python -m tools.fetch_yolo              # lpr (по умолчанию)
    python -m tools.fetch_yolo -m lpr n     # несколько
    python -m tools.fetch_yolo -m s --onnx  # только для yolo11*
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

HF_REPO = "morsetechlab/yolov11-license-plate-detection"
LPR_URL = (
    "https://github.com/Muhammad-Zeerak-Khan/"
    "Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
)
MODELS = ("lpr", "n", "s", "m", "l", "x")
OUT_DIR = Path(__file__).resolve().parent.parent / "models"


def fetch(name: str, onnx: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if name == "lpr":
        if onnx:
            raise SystemExit("для lpr есть только .pt")
        dst = OUT_DIR / "lpr-yolov8n-plate.pt"
        with urllib.request.urlopen(LPR_URL, timeout=120) as resp:
            dst.write_bytes(resp.read())
        return dst

    from huggingface_hub import hf_hub_download

    ext = "onnx" if onnx else "pt"
    src = hf_hub_download(HF_REPO, f"license-plate-finetune-v1{name}.{ext}")
    dst = OUT_DIR / f"yolo11{name}-plate.{ext}"
    shutil.copy(src, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-m", "--models", nargs="+", default=["lpr"], choices=MODELS)
    ap.add_argument("--onnx", action="store_true", help="скачать .onnx вместо .pt (только yolo11*)")
    args = ap.parse_args()

    for name in args.models:
        dst = fetch(name, args.onnx)
        print(f"{dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    print(
        "\nВыбор весов: ANPR_DETECTOR_WEIGHTS=models/lpr-yolov8n-plate.pt в .env"
        "\nСравнить на своих данных: python -m tools.eval_uz --compare models/*.pt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

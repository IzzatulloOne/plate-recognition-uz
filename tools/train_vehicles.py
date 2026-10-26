"""Классификатор моделей машин (Nexia, Cobalt, Schmitz...) на кропах build_vehicles.

    python -m tools.train_vehicles                    # обучить на CPU, замер на test
    python -m tools.train_vehicles --epochs 40 --device 0
    python -m tools.train_vehicles --eval-only runs/vehicles/cls/weights/best.pt

YOLO11n-cls, предобученный на ImageNet: 1.5 млн параметров, на CPU обучается за
десятки минут и экспортируется в ONNX для слабого железа.

Итог — точность по каждому классу на test (часть эталона eval-v2), а не одна
общая цифра: Schmitz с сотнями фото утопил бы в среднем провал на редком классе.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.build_vehicles import OUT

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "yolo11n-cls.pt"
RUNS = ROOT / "runs" / "vehicles"


def per_class(weights: Path, split: str = "test") -> dict:
    """Точность по классам: доля верных top-1 на каждой папке части split."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    ok, tot, confused = Counter(), Counter(), Counter()
    for cls_dir in sorted((OUT / split).iterdir()):
        files = [str(f) for f in sorted(cls_dir.glob("*.jpg"))]
        for i in range(0, len(files), 64):
            for r in model.predict(files[i:i + 64], imgsz=224, verbose=False):
                pred = names[int(r.probs.top1)]
                tot[cls_dir.name] += 1
                ok[cls_dir.name] += pred == cls_dir.name
                if pred != cls_dir.name:
                    confused[cls_dir.name, pred] += 1
    total_ok, total = sum(ok.values()), sum(tot.values())
    rows = {c: f"{ok[c]}/{tot[c]}" for c in sorted(tot, key=lambda c: -tot[c]) if tot[c]}
    return {"top1": f"{total_ok}/{total}", "per_class": rows,
            "частые путаницы": [f"{a} -> {b}: {n}" for (a, b), n in confused.most_common(8)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-only", type=Path, help="только замер готовых весов на test")
    args = ap.parse_args()

    weights = args.eval_only
    if weights is None:
        from ultralytics import YOLO

        model = YOLO(str(BASE))
        model.train(data=str(OUT), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, project=str(RUNS), name="cls", exist_ok=True,
                    patience=10, plots=False, verbose=False)
        weights = RUNS / "cls" / "weights" / "best.pt"
        YOLO(str(weights)).export(format="onnx", imgsz=args.imgsz)

    report = per_class(weights)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    (weights.parent / "test_report.json").write_text(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

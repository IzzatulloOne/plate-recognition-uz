"""Замер детекторов на эталоне, размеченном руками (tools.label_boxes).

    python -m tools.eval_detector models/yolo11n-plate.pt runs/det/weights/best.pt

Каждое фото проходит так же, как в сервисе: проход в 640, пусто — второй в 1536.
Попадание — рамка с IoU >= 0.5 к размеченной, пары один к одному. Отдельно —
квадратные (двухстрочные) пластины: ради них детектор и дообучается.

Мерить детектор только на автоматической разметке нельзя: её подтверждала наша
же модель, и то, чего не видит она, не попало бы и в эталон.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from app.config import settings
from app.core.detector import PlateDetector
from tools.label_boxes import ITEMS, _load_boxes

TWO_LINE_MAX_AR = 2.6


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list, pred: list, thr: float = 0.5) -> set[int]:
    """Индексы размеченных рамок, нашедших пару: жадно по убыванию IoU, один к одному."""
    pairs = sorted(((_iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
                   reverse=True)
    used_g, used_p = set(), set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
    return used_g


def evaluate(weights: Path, items: dict) -> dict:
    det = PlateDetector(weights=weights, imgsz=settings.det_imgsz, conf=settings.det_conf,
                        iou=settings.det_iou, max_det=settings.det_max_det,
                        device=settings.device)
    n_gt = n_sq = hit = hit_sq = n_pred = all_found = 0
    t0 = time.time()
    for pid, v in items.items():
        gt = _load_boxes(pid, v["w"], v["h"]) or []
        img = cv2.imread(v["file"])
        dets = det.detect(img)
        if not dets and settings.det_retry_imgsz:
            dets = det.detect(img, imgsz=settings.det_retry_imgsz, conf=settings.det_retry_conf)
        pred = [[d.x1, d.y1, d.x2, d.y2] for d in dets]
        found = match(gt, pred)
        square = {i for i, g in enumerate(gt)
                  if (g[2] - g[0]) / max(g[3] - g[1], 1) <= TWO_LINE_MAX_AR}
        n_gt += len(gt)
        n_sq += len(square)
        hit += len(found)
        hit_sq += len(found & square)
        n_pred += len(pred)
        all_found += len(found) == len(gt)
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "—"  # noqa: E731
    return {"детектор": Path(weights).name, "все пластины": pct(hit, n_gt),
            "двухстрочные": pct(hit_sq, n_sq), "точность": pct(hit, n_pred),
            "фото без пропусков": pct(all_found, len(items)),
            "с/фото": round((time.time() - t0) / max(len(items), 1), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+", type=Path)
    args = ap.parse_args()
    items = {p: v for p, v in json.loads(ITEMS.read_text()).items() if v["done"]}
    if not items:
        print("эталон пуст — сначала разметка: python -m tools.label_boxes")
        return 2
    print(f"размеченных фото: {len(items)}")
    for w in args.weights:
        print(json.dumps(evaluate(w, items), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

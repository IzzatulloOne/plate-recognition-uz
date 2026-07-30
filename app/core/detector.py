"""Детектор номерных пластин на YOLO (ultralytics).

Веса по умолчанию: models/yolo11n-plate.pt — YOLO11n, дообученный на номерах
(morsetechlab/yolov11-license-plate-detection). На CPU ~42 мс/кадр @640.
Замена на 's'/'m' — одна переменная окружения ANPR_DETECTOR_WEIGHTS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "false")


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class PlateDetector:
    def __init__(
        self,
        weights: str | Path,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        max_det: int = 20,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"нет весов детектора: {weights}. Запустите: python -m tools.fetch_yolo"
            )
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device
        self.names = self.model.names

    def detect(
        self, frame: np.ndarray, imgsz: int | None = None, conf: float | None = None
    ) -> list[Detection]:
        res = self.model.predict(
            frame,
            imgsz=imgsz or self.imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]
        h, w = frame.shape[:2]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(
                Detection(
                    x1=max(0, int(x1)),
                    y1=max(0, int(y1)),
                    x2=min(w, int(x2)),
                    y2=min(h, int(y2)),
                    conf=float(b.conf[0]),
                )
            )
        out.sort(key=lambda d: -d.conf)
        return out


def crop_plate(frame: np.ndarray, det: Detection, pad: float = 0.06) -> np.ndarray:
    """Кроп с относительным отступом (TPS в распознавателе доедает перспективу)."""
    h, w = frame.shape[:2]
    px = int(det.width * pad)
    py = int(det.height * pad)
    x1 = max(0, det.x1 - px)
    y1 = max(0, det.y1 - py)
    x2 = min(w, det.x2 + px)
    y2 = min(h, det.y2 + py)
    return frame[y1:y2, x1:x2]

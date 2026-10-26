"""Модель машины по кадру: «Nexia», «Cobalt», «Schmitz».

COCO-детектор находит машины, из них берётся та, на которой стоит номер, —
самая тесная рамка с центром пластины внутри. Её кроп называет классификатор
(tools.train_vehicles). Сборка датасета (tools.build_vehicles) выбирает машину
этим же кодом: модель должна видеть в работе то же, на чём училась.

Включается только с весами классификатора (ANPR_VEHICLE_WEIGHTS): без них
пайплайн работает как раньше. Цена — +70 мс медианы на фото (590 против 522,
CPU, 40 фото eval-v2), на медленном хвосте до +250 мс.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COCO_WEIGHTS = ROOT / "models" / "yolo11n.pt"
VEHICLE_CLASSES = (2, 5, 7)  # COCO: car, bus, truck — прицеп COCO видит грузовиком
TRUCK = 7
OTHER = "другое"  # класс «не из известных моделей» — наружу не отдаётся


def _area(b: list[int]) -> int:
    return (b[2] - b[0]) * (b[3] - b[1])


def pick_vehicle(cars: list[list[int]], plate: list[int] | None) -> list[int] | None:
    """Машина, на которой стоит пластина: самая тесная рамка с её центром внутри."""
    if plate is None:
        return None
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    holders = [b for b in cars if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
    return min(holders, key=_area) if holders else None


def candidates(cars: list[tuple[list[int], int]], trailer: bool) -> list[list[int]]:
    """Рамки, среди которых искать машину номера.

    Пластина прицепа стоит на прицепе, а прицеп COCO видит грузовиком. Без этого
    правила на снимке с легковушкой перед прицепом вырезалась легковушка: её
    однострочный номер детектор находит увереннее квадратного номера прицепа.
    """
    return [b for b, c in cars if not trailer or c == TRUCK]


def crop_vehicle(frame: np.ndarray, box: list[int], pad: float = 0.05) -> np.ndarray:
    x1, y1, x2, y2 = box
    p = int(pad * max(x2 - x1, y2 - y1))
    h, w = frame.shape[:2]
    return frame[max(0, y1 - p):min(h, y2 + p), max(0, x1 - p):min(w, x2 + p)]


class VehicleDetector:
    """COCO-детектор машин — общий для сборки датасета и для сервиса."""

    def __init__(self, weights: str | Path = COCO_WEIGHTS, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def detect(self, frame: np.ndarray) -> list[tuple[list[int], int]]:
        """[(рамка, класс COCO)] всех машин кадра."""
        res = self.model.predict(frame, classes=list(VEHICLE_CLASSES), conf=0.25,
                                 device=self.device, verbose=False)[0]
        return [([int(v) for v in b], int(c))
                for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), strict=True)]


class VehicleClassifier:
    """По каждой пластине кадра — (модель, уверенность); ("", 0.0), если не уверен."""

    def __init__(self, weights: str | Path, device: str = "cpu", min_conf: float = 0.5,
                 imgsz: int = 224):
        from ultralytics import YOLO

        self.detector = VehicleDetector(device=device)
        self.model = YOLO(str(weights))
        self.min_conf, self.imgsz, self.device = min_conf, imgsz, device

    def classify(
        self, frame: np.ndarray, plates: list[list[int]], trailers: list[bool] | None = None
    ) -> list[tuple[str, float]]:
        if not plates:
            return []
        cars = self.detector.detect(frame)
        trailers = trailers or [False] * len(plates)
        boxes = [pick_vehicle(candidates(cars, t), p)
                 for p, t in zip(plates, trailers, strict=True)]
        crops = [crop_vehicle(frame, b) for b in boxes if b is not None]
        preds = iter(self.model.predict(crops, imgsz=self.imgsz, device=self.device,
                                        verbose=False) if crops else [])
        out: list[tuple[str, float]] = []
        for b in boxes:
            if b is None:
                out.append(("", 0.0))
                continue
            r = next(preds)
            name, conf = self.model.names[int(r.probs.top1)], float(r.probs.top1conf)
            # «другое» и неуверенный ответ наружу не отдаём: пустая модель честнее
            # правдоподобной ошибки — как и с мелкими пластинами
            out.append((name, conf) if name != OTHER and conf >= self.min_conf else ("", conf))
        return out

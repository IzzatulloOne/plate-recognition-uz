"""IoU-трекер + голосование по кадрам.

На видео одна пластина видна десятки кадров. Голосование по взвешенной
уверенности поднимает точность заметно выше one-shot распознавания — особенно
для жёлтых/зелёных номеров, где модель слабее.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    first_ts: float
    last_ts: float
    age: int = 0  # кадров без обновления
    hits: int = 0
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    best_conf: float = 0.0
    best_crop_jpeg: bytes | None = None
    type_code: str = ""
    color: str = "unknown"
    emitted_ts: float = 0.0
    emitted_text: str = ""  # что уже отдали наружу — чтобы прислать уточнение при смене

    def vote(self, text: str, conf: float, valid: bool) -> None:
        if not text:
            return
        # валидный формат весит больше — так шум не перебивает правильное чтение
        self.votes[text] += conf * (1.6 if valid else 1.0)
        self.counts[text] += 1

    @property
    def stable_text(self) -> str:
        if not self.votes:
            return ""
        return max(self.votes.items(), key=lambda kv: kv[1])[0]

    @property
    def stable_score(self) -> float:
        if not self.votes:
            return 0.0
        total = sum(self.votes.values())
        return max(self.votes.values()) / total if total else 0.0

    @property
    def stable_votes(self) -> int:
        return self.counts.get(self.stable_text, 0)

    def leader_margin(self, over: str) -> float:
        """Во сколько раз текущий лидер перевешивает указанный текст.

        Нужно, чтобы событие-уточнение не выпускалось при болтанке 50/50 между
        двумя похожими чтениями (типично для 0/8, 6/8, B/8).
        """
        if not self.votes:
            return 0.0
        top = max(self.votes.values())
        other = self.votes.get(over, 0.0)
        return top / other if other > 0 else float("inf")


class PlateTracker:
    """Простой IoU-трекер: пластины двигаются предсказуемо, этого достаточно."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 20):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, boxes: list[tuple[int, int, int, int]], ts: float | None = None) -> list[int]:
        """Сопоставляет боксы с треками, возвращает track_id по порядку boxes."""
        ts = ts if ts is not None else time.time()
        assigned: list[int] = []
        used: set[int] = set()

        for box in boxes:
            best_id, best_iou = -1, 0.0
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                v = iou(box, tr.box)
                if v > best_iou:
                    best_id, best_iou = tid, v
            if best_id != -1 and best_iou >= self.iou_threshold:
                tr = self.tracks[best_id]
                tr.box = box
                tr.last_ts = ts
                tr.age = 0
                tr.hits += 1
                used.add(best_id)
                assigned.append(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(
                    track_id=tid, box=box, first_ts=ts, last_ts=ts, hits=1
                )
                used.add(tid)
                assigned.append(tid)

        for tid, tr in list(self.tracks.items()):
            if tid not in used:
                tr.age += 1
                if tr.age > self.max_age:
                    del self.tracks[tid]
        return assigned

    def get(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)

    def reset(self) -> None:
        self.tracks.clear()

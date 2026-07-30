"""Пайплайн: кадр -> YOLO -> кропы -> распознаватель -> правила -> результат.

ANPRPipeline    — без состояния, потокобезопасен на уровне «один вызов за раз»
                  (инференс сериализуется локом, параллелизм даёт batch и потоки torch).
TrackedSession  — состояние на источник: трекинг + голосование + события.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from app.config import Settings
from app.core import plate_rules
from app.core.detector import Detection, PlateDetector, crop_plate
from app.core.recognizer import PlateRecognizer
from app.core.tracker import PlateTracker


@dataclass
class PlateResult:
    box: list[int]
    det_conf: float
    text: str
    conf: float
    raw_text: str
    format: str | None
    plate_class: str
    color: str
    color_conf: float
    type_code: str
    type_conf: float
    valid: bool
    corrected: bool
    track_id: int | None = None
    stable_text: str | None = None
    votes: int = 0
    stable_score: float = 0.0
    confirmed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    plates: list[PlateResult] = field(default_factory=list)
    detect_ms: float = 0.0
    recognize_ms: float = 0.0
    total_ms: float = 0.0
    frame_size: tuple[int, int] = (0, 0)
    events: list[dict] = field(default_factory=list)
    retried: bool = False  # понадобился второй проход детектора в большем разрешении

    def to_dict(self) -> dict:
        return {
            "plates": [p.to_dict() for p in self.plates],
            "timings": {
                "detect_ms": round(self.detect_ms, 1),
                "recognize_ms": round(self.recognize_ms, 1),
                "total_ms": round(self.total_ms, 1),
                "retried": self.retried,
            },
            "frame_size": list(self.frame_size),
            "events": self.events,
        }


def _is_valid_plate(text: str) -> bool:
    """Арбитр для выбора раскладки пластины (одна строка / две)."""
    return plate_rules.match(text).valid


class ANPRPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = PlateDetector(
            weights=settings.detector_weights,
            imgsz=settings.det_imgsz,
            conf=settings.det_conf,
            iou=settings.det_iou,
            max_det=settings.det_max_det,
            device=settings.device,
        )
        self.recognizer = PlateRecognizer(
            weights=settings.recognizer_weights,
            device=settings.device,
            charset=settings.charset,
            preprocess=settings.preprocess,
            contrast_boost=settings.contrast_boost,
            num_threads=settings.torch_threads,
            text_head=settings.text_head,
            type_head=settings.type_head,
        )
        if settings.quantize:
            import torch

            self.recognizer.model = torch.quantization.quantize_dynamic(
                self.recognizer.model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
            )
        self._type_head_colors = settings.type_head_color_set
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ инференс
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Кропы прикладываются к результату как res._crops (для снапшотов)."""
        t_start = time.perf_counter()
        res = FrameResult(frame_size=(frame.shape[1], frame.shape[0]))

        with self._lock:
            t0 = time.perf_counter()
            dets = self.detector.detect(frame)
            if not dets and self.settings.det_retry_imgsz > self.settings.det_imgsz:
                # пустой кадр — пробуем ещё раз крупнее: мелкие и косые пластины
                # (грузовики, прицепы) на 640 часто теряются
                dets = self.detector.detect(
                    frame,
                    imgsz=self.settings.det_retry_imgsz,
                    conf=self.settings.det_retry_conf,
                )
                res.retried = True
            res.detect_ms = (time.perf_counter() - t0) * 1000

            candidates: list[Detection] = [
                d for d in dets if d.width >= self.settings.min_plate_width and d.height >= 8
            ]
            # фильтруем детекции и кропы вместе, иначе списки разъедутся при zip
            pairs = [(d, crop_plate(frame, d, self.settings.crop_padding)) for d in candidates]
            pairs = [(d, c) for d, c in pairs if c.size > 0]
            usable: list[Detection] = [d for d, _ in pairs]
            crops: list[np.ndarray] = [c for _, c in pairs]

            t0 = time.perf_counter()
            recs = self.recognizer.read(
                crops,
                topk=self.settings.beam_topk,
                beam=self.settings.beam_width,
                validate=_is_valid_plate,
            )
            if not self.settings.format_constraint:
                recs = [(r, []) for r, _ in recs]
            res.recognize_ms = (time.perf_counter() - t0) * 1000

        for det, crop, (rec, alts) in zip(usable, crops, recs, strict=True):
            raw = rec.text
            color, color_conf = plate_rules.classify_color(crop)

            # Гипотеза о второй голове: её алфавит — только цифры, 'H' и 'M', то есть
            # она обучалась на номерах другого набора (по словам автора модели —
            # жёлтые/зелёные). Обе головы считаются одним форвардом, так что для таких
            # номеров можно бесплатно добавить её вывод в кандидаты — победит он только
            # если уложится в формат УЗ. По умолчанию выключено: проверьте на своих данных
            # (ANPR_TYPE_HEAD_COLORS=yellow,green).
            if rec.type_code and color in self._type_head_colors:
                alts = [*alts, (rec.type_code, rec.type_conf)]

            if alts and self.settings.format_constraint:
                text, conf, m = plate_rules.pick_best(alts, min_conf=0.0)
                # если формат не найден — доверяем greedy, он честнее по вероятности
                if not m.valid:
                    m = plate_rules.repair(raw)
                    text, conf = m.text, rec.conf
            else:
                m = plate_rules.repair(raw)
                text, conf = m.text, rec.conf

            pr = PlateResult(
                box=[det.x1, det.y1, det.x2, det.y2],
                det_conf=round(det.conf, 4),
                text=text,
                conf=round(float(conf), 4),
                raw_text=raw,
                format=m.format,
                plate_class=m.plate_class,
                color=color,
                color_conf=round(color_conf, 3),
                type_code=rec.type_code,
                type_conf=round(rec.type_conf, 4),
                valid=m.valid,
                corrected=m.corrected,
            )
            res.plates.append(pr)

        res.total_ms = (time.perf_counter() - t_start) * 1000
        res._crops = crops  # type: ignore[attr-defined]
        return res


class TrackedSession:
    """Трекинг + голосование + генерация событий для одного источника."""

    def __init__(self, pipeline: ANPRPipeline, settings: Settings, source: str = "ws"):
        self.pipeline = pipeline
        self.settings = settings
        self.source = source
        self.tracker = PlateTracker(
            iou_threshold=settings.track_iou, max_age=settings.track_max_age
        )
        self.frames = 0
        self._fps_window: list[float] = []

    @property
    def fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    def process(self, frame: np.ndarray, on_event=None) -> FrameResult:
        ts = time.time()
        res = self.pipeline.process_frame(frame)
        crops = getattr(res, "_crops", [])
        self.frames += 1
        self._fps_window.append(ts)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)

        ids = self.tracker.update([tuple(p.box) for p in res.plates], ts=ts)
        for i, (pr, tid) in enumerate(zip(res.plates, ids, strict=True)):
            tr = self.tracker.get(tid)
            if tr is None:
                continue
            tr.vote(pr.text, pr.conf, pr.valid)
            if pr.color != "unknown":
                tr.color = pr.color
            if pr.type_code:
                tr.type_code = pr.type_code
            if pr.conf > tr.best_conf and i < len(crops):
                tr.best_conf = pr.conf
                if self.settings.save_snapshots:
                    ok, buf = cv2.imencode(
                        ".jpg", crops[i], [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    if ok:
                        tr.best_crop_jpeg = buf.tobytes()

            pr.track_id = tid
            pr.stable_text = tr.stable_text
            pr.votes = tr.stable_votes
            pr.stable_score = round(tr.stable_score, 3)

            ready = (
                tr.stable_votes >= self.settings.min_votes
                and tr.best_conf >= self.settings.rec_min_conf
                and bool(tr.stable_text)
            )
            fresh = (ts - tr.emitted_ts) >= self.settings.event_cooldown_s
            # уточнение: голосование передумало (типично для первых кадров, когда
            # номер ещё мелкий) — отдаём новое событие с пометкой updated, но только
            # если новый лидер уверенно перевесил старый, иначе получим болтанку
            changed = (
                bool(tr.emitted_text)
                and tr.stable_text != tr.emitted_text
                and tr.leader_margin(tr.emitted_text) >= self.settings.event_update_margin
                and (ts - tr.emitted_ts) >= self.settings.event_update_min_interval
            )
            if ready and (fresh or changed):
                previous = tr.emitted_text or None
                tr.emitted_ts = ts
                tr.emitted_text = tr.stable_text
                pr.confirmed = True
                m = plate_rules.match(tr.stable_text)
                event = {
                    "ts": ts,
                    "source": self.source,
                    "track_id": tid,
                    "text": tr.stable_text,
                    "pretty": m.pretty(),
                    "conf": round(tr.best_conf, 4),
                    "votes": tr.stable_votes,
                    "format": m.format,
                    "plate_class": m.plate_class,
                    "color": tr.color,
                    "type_code": tr.type_code,
                    "box": pr.box,
                    "updated": changed,
                    "previous": previous if changed else None,
                }
                if on_event is not None:
                    on_event(event, tr.best_crop_jpeg)
                res.events.append(event)
        return res

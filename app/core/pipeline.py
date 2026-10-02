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
from app.core.tracker import PlateTracker, near_text


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
    #: False — в этом кадре пластину не читали, текст взят из трека. Нужно клиенту,
    #: чтобы отличать свежее чтение от показа уже известного номера.
    recognized: bool = True

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
            two_line=settings.two_line,
        )
        if settings.quantize:
            import torch

            self.recognizer.model = torch.quantization.quantize_dynamic(
                self.recognizer.model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
            )
        self._type_head_colors = settings.type_head_color_set
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ инференс
    # Детекция и распознавание разделены намеренно. На видео номер виден десятки
    # кадров, и читать его каждый раз незачем: замер показал, что 90% вызовов
    # распознавателя — это повторное чтение уже решённого номера. TrackedSession
    # вклинивается между этими двумя шагами и решает, кого читать. Для одиночного
    # фото порядок прежний — process_frame делает всё сразу.
    def detect_plates(
        self,
        frame: np.ndarray,
        allow_retry: bool = True,
        imgsz: int | None = None,
        min_width: int | None = None,
    ) -> tuple[list[Detection], list[np.ndarray], float, bool]:
        """Детекция и кропы без распознавания -> (детекции, кропы, мс, был_второй_проход).

        allow_retry=False отключает дорогой второй проход детектора — для видео,
        где кадр без пластины обычное дело, а следующий шанс придёт через 150 мс.
        """
        t0 = time.perf_counter()
        imgsz = imgsz or self.settings.det_imgsz
        min_width = self.settings.min_plate_width if min_width is None else min_width
        retried = False

        with self._lock:
            dets = self.detector.detect(frame, imgsz=imgsz)
            if allow_retry and not dets and self.settings.det_retry_imgsz > imgsz:
                # пустой кадр — пробуем ещё раз крупнее: мелкие и косые пластины
                # (грузовики, прицепы) на 640 часто теряются
                dets = self.detector.detect(
                    frame,
                    imgsz=self.settings.det_retry_imgsz,
                    conf=self.settings.det_retry_conf,
                )
                retried = True
        detect_ms = (time.perf_counter() - t0) * 1000

        candidates = [d for d in dets if d.width >= min_width and d.height >= 8]
        # фильтруем детекции и кропы вместе, иначе списки разъедутся при zip
        pairs = [(d, crop_plate(frame, d, self.settings.crop_padding)) for d in candidates]
        pairs = [(d, c) for d, c in pairs if c.size > 0]
        return [d for d, _ in pairs], [c for _, c in pairs], detect_ms, retried

    def recognize_crops(
        self, dets: list[Detection], crops: list[np.ndarray]
    ) -> tuple[list[PlateResult], float]:
        """Распознать готовые кропы -> (результаты, мс). Пустой вход стоит ноль."""
        if not crops:
            return [], 0.0

        t0 = time.perf_counter()
        with self._lock:
            recs = self.recognizer.read(
                crops,
                topk=self.settings.beam_topk,
                beam=self.settings.beam_width,
                validate=_is_valid_plate,
            )
            if not self.settings.format_constraint:
                recs = [(r, []) for r, _ in recs]
        recognize_ms = (time.perf_counter() - t0) * 1000

        out = [
            self._build_result(det, crop, rec, alts)
            for det, crop, (rec, alts) in zip(dets, crops, recs, strict=True)
        ]
        return out, recognize_ms

    def _build_result(self, det: Detection, crop: np.ndarray, rec, alts) -> PlateResult:
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

        return PlateResult(
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

    def process_frame(self, frame: np.ndarray, allow_retry: bool = True) -> FrameResult:
        """Кадр целиком: детекция + распознавание всего найденного.

        Путь одиночного фото (REST). Кропы прикладываются как res._crops — их
        использует TrackedSession для снапшотов.
        """
        t_start = time.perf_counter()
        res = FrameResult(frame_size=(frame.shape[1], frame.shape[0]))
        dets, crops, res.detect_ms, res.retried = self.detect_plates(frame, allow_retry)
        res.plates, res.recognize_ms = self.recognize_crops(dets, crops)
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
            iou_threshold=settings.track_iou,
            max_age=settings.track_max_age,
            by_text=settings.track_by_text,
        )
        self.frames = 0
        self._fps_window: list[float] = []
        #: track_id -> (площадь пластины при последнем чтении, номер кадра).
        #: По этим двум числам решается, пора ли перечитывать подтверждённый номер.
        self._read_state: dict[int, tuple[int, int]] = {}
        #: последний раз, когда номер уходил наружу — общий на сессию, а не на трек.
        #: На подвижной камере трек может порваться, и обрывки выдали бы одну машину
        #: несколько раз; кулдаун по тексту это ловит независимо от track_id.
        self._emitted: dict[str, float] = {}

    def _recently_emitted(self, text: str, ts: float) -> bool:
        """Этот же номер (или он же с одной перепутанной буквой) уже уходил недавно?"""
        window = self.settings.event_cooldown_s
        for seen, when in list(self._emitted.items()):
            if ts - when > window:
                del self._emitted[seen]
                continue
            if seen == text or near_text(seen, text):
                return True
        return False

    @property
    def fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    def _needs_read(self, tid: int, det: Detection) -> bool:
        """Читать ли эту пластину в этом кадре.

        Пока трек не набрал голосов — читаем всегда, иначе номер нечем определить.
        Дальше только по делу: пластина заметно выросла (машина ближе, кадр лучше
        прежнего) либо давно не перечитывали. Без проверки на рост схема теряет
        номера, закрепившиеся по дальним кадрам — на замере два из девяти.
        """
        s = self.settings
        if not s.skip_confirmed:
            return True
        tr = self.tracker.get(tid)
        if tr is None or tr.stable_votes < s.min_votes:
            return True
        last_area, last_frame = self._read_state.get(tid, (0, -(10**9)))
        if det.width * det.height > last_area * s.recheck_growth:
            return True
        return self.frames - last_frame >= s.recheck_frames

    def _from_track(self, det: Detection, tr) -> PlateResult:
        """Пластина, которую в этом кадре не читали: всё, что знаем, — из трека."""
        m = plate_rules.match(tr.stable_text) if tr.stable_text else None
        return PlateResult(
            box=[det.x1, det.y1, det.x2, det.y2],
            det_conf=round(det.conf, 4),
            text=tr.stable_text,
            conf=round(tr.best_conf, 4),
            raw_text="",
            format=m.format if m else None,
            plate_class=m.plate_class if m else "",
            color=tr.color,
            color_conf=0.0,
            type_code=tr.type_code,
            type_conf=0.0,
            valid=bool(m and m.valid),
            corrected=False,
            recognized=False,
        )

    def process(self, frame: np.ndarray, on_event=None) -> FrameResult:
        ts = time.time()
        t_start = time.perf_counter()
        s = self.settings
        res = FrameResult(frame_size=(frame.shape[1], frame.shape[0]))

        dets, crops, res.detect_ms, res.retried = self.pipeline.detect_plates(
            frame,
            allow_retry=s.det_retry_in_stream,
            imgsz=s.stream_imgsz,
            min_width=s.stream_min_width,
        )
        self.frames += 1
        self._fps_window.append(ts)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)

        # Трекинг идёт до распознавания и опирается только на геометрию: текста,
        # по которому раньше доклеивались рваные треки, здесь ещё нет. Склейка по
        # тексту выполняется ниже, через absorb_by_text, когда текст появится.
        ids = self.tracker.update([(d.x1, d.y1, d.x2, d.y2) for d in dets], ts=ts)

        todo = [i for i, tid in enumerate(ids) if self._needs_read(tid, dets[i])]
        read, res.recognize_ms = self.pipeline.recognize_crops(
            [dets[i] for i in todo], [crops[i] for i in todo]
        )
        # имя не `fresh`: ниже в цикле уже есть флаг с таким именем для кулдауна событий
        read_now = dict(zip(todo, read, strict=True))

        for i, pr in read_now.items():
            self._read_state[ids[i]] = (dets[i].width * dets[i].height, self.frames)
            merged = self.tracker.absorb_by_text(ids[i], pr.text)
            if merged != ids[i]:
                self._read_state.pop(ids[i], None)
                self._read_state[merged] = (dets[i].width * dets[i].height, self.frames)
                ids[i] = merged

        # состояние чтения живёт ровно столько, сколько сам трек
        for dead in self._read_state.keys() - self.tracker.tracks.keys():
            del self._read_state[dead]

        for i, tid in enumerate(ids):
            tr = self.tracker.get(tid)
            if tr is None:
                continue
            pr = read_now.get(i)
            if pr is None:
                pr = self._from_track(dets[i], tr)
            else:
                tr.vote(pr.text, pr.conf, pr.valid)
                if pr.color != "unknown":
                    tr.color = pr.color
                if pr.type_code:
                    tr.type_code = pr.type_code
                if pr.conf > tr.best_conf:
                    tr.best_conf = pr.conf
                    if self.settings.save_snapshots:
                        ok, buf = cv2.imencode(
                            ".jpg", crops[i], [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        if ok:
                            tr.best_crop_jpeg = buf.tobytes()
            res.plates.append(pr)

            pr.track_id = tid
            pr.stable_text = tr.stable_text
            pr.votes = tr.stable_votes
            pr.stable_score = round(tr.stable_score, 3)

            ready = (
                tr.stable_votes >= self.settings.min_votes
                and tr.best_conf >= self.settings.rec_min_conf
                and bool(tr.stable_text)
            )
            fresh = (ts - tr.emitted_ts) >= self.settings.event_cooldown_s and not (
                self._recently_emitted(tr.stable_text, ts) and not tr.emitted_text
            )
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
                self._emitted[tr.stable_text] = ts
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

        res.total_ms = (time.perf_counter() - t_start) * 1000
        res._crops = crops  # type: ignore[attr-defined]
        return res

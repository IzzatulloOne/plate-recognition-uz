"""Серверные источники видео: RTSP / файл / вебкамера.

Каждый источник — отдельный поток: читает кадры, гонит их через TrackedSession,
рассылает результаты подписчикам WebSocket и держит последний annotated-кадр
для MJPEG-превью. Устаревшие кадры отбрасываются (cap.grab без decode), так что
задержка не накапливается даже если CPU не успевает.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.core.pipeline import ANPRPipeline, TrackedSession
from app.core.store import EventStore

log = logging.getLogger("anpr.streams")


@dataclass
class StreamStats:
    frames_read: int = 0
    frames_processed: int = 0
    events: int = 0
    fps: float = 0.0
    last_error: str = ""
    started_at: float = field(default_factory=time.time)
    connected: bool = False


class StreamWorker:
    def __init__(
        self,
        stream_id: str,
        source: str,
        pipeline: ANPRPipeline,
        settings: Settings,
        store: EventStore | None,
        loop: asyncio.AbstractEventLoop,
        target_fps: float = 0.0,
        draw: bool = True,
        repeat: bool = True,
    ):
        self.id = stream_id
        self.source = source
        self.settings = settings
        self.store = store
        self.loop = loop
        self.target_fps = target_fps
        self.draw = draw
        self.repeat = repeat
        #: файл читается «как есть» (быстрее реального времени), поток/камера — по мере поступления
        self.is_file = not source.isdigit() and Path(source).exists()
        self.finished = False
        self.session = TrackedSession(pipeline, settings, source=stream_id)
        self.stats = StreamStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._sub_lock = threading.Lock()
        self._preview: bytes | None = None
        self._preview_lock = threading.Lock()

    # --------------------------------------------------------------- подписчики
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        with self._sub_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._sub_lock:
            self._subscribers.discard(q)

    def _broadcast(self, payload: dict) -> None:
        with self._sub_lock:
            queues = list(self._subscribers)
        for q in queues:
            def put(q=q, payload=payload):
                if q.full():  # подписчик не успевает — выбрасываем самый старый
                    with contextlib.suppress(asyncio.QueueEmpty):
                        q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(payload)

            self.loop.call_soon_threadsafe(put)

    @property
    def preview_jpeg(self) -> bytes | None:
        with self._preview_lock:
            return self._preview

    # ------------------------------------------------------------------ рантайм
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _open(self) -> cv2.VideoCapture:
        src: int | str = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.stats.connected = False
                self.stats.last_error = f"не открылся источник: {self.source}"
                log.warning("[%s] %s, retry in %.0fs", self.id, self.stats.last_error, backoff)
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 30.0)
                continue

            self.stats.connected = True
            self.stats.last_error = ""
            backoff = 1.0
            min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
            last_proc = 0.0

            eof = False
            while not self._stop.is_set():
                # для файла ждём до следующего слота, иначе сожжём CPU на grab();
                # для камеры/RTSP наоборот — жадно вычитываем, чтобы не копить задержку
                if self.is_file and min_interval:
                    sleep_for = min_interval - (time.time() - last_proc)
                    if sleep_for > 0 and self._stop.wait(sleep_for):
                        break
                ok = cap.grab()
                if not ok:
                    eof = True
                    self.stats.last_error = (
                        "конец файла" if self.is_file else "поток оборвался"
                    )
                    break
                self.stats.frames_read += 1
                now = time.time()
                if min_interval and (now - last_proc) < min_interval:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                last_proc = now
                try:
                    self._handle_frame(frame)
                except Exception as exc:  # не роняем поток из-за одного кадра
                    self.stats.last_error = f"{type(exc).__name__}: {exc}"
                    log.exception("[%s] ошибка обработки кадра", self.id)

            cap.release()
            self.stats.connected = False
            if self._stop.is_set():
                break
            if self.is_file and eof:
                if not self.repeat:  # файл дочитан и повтор не нужен — выходим
                    self.finished = True
                    self.stats.last_error = "файл обработан полностью"
                    log.info("[%s] файл обработан, поток завершён", self.id)
                    break
                self.session.tracker.reset()  # новый проход — новые треки
                continue  # перечитываем файл сразу, без штрафной секунды
            if self._stop.wait(1.0):  # RTSP/камера: пауза перед реконнектом
                break
        self.stats.connected = False

    def _handle_frame(self, frame: np.ndarray) -> None:
        def on_event(event: dict, jpeg: bytes | None) -> None:
            self.stats.events += 1
            if self.store is not None:
                event["id"] = self.store.add(event, jpeg)
            self._broadcast({"type": "event", "stream": self.id, "event": event})

        res = self.session.process(frame, on_event=on_event)
        self.stats.frames_processed += 1
        self.stats.fps = round(self.session.fps, 2)

        payload = res.to_dict()
        payload.update({"type": "result", "stream": self.id, "fps": self.stats.fps})
        self._broadcast(payload)

        if self.draw:
            annotated = draw_overlay(frame, res)
            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.settings.ws_jpeg_quality]
            )
            if ok:
                with self._preview_lock:
                    self._preview = buf.tobytes()


def draw_overlay(frame: np.ndarray, res) -> np.ndarray:
    """Рисует боксы и текст — для MJPEG-превью и отладки."""
    out = frame.copy()
    for p in res.plates:
        x1, y1, x2, y2 = p.box
        color = (0, 200, 0) if p.valid else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{p.stable_text or p.text} {p.conf:.2f}"
        if p.track_id:
            label = f"#{p.track_id} {label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


class StreamManager:
    def __init__(self, pipeline: ANPRPipeline, settings: Settings, store: EventStore | None):
        self.pipeline = pipeline
        self.settings = settings
        self.store = store
        self.workers: dict[str, StreamWorker] = {}
        self._lock = threading.Lock()

    def start(
        self,
        stream_id: str,
        source: str,
        target_fps: float = 0.0,
        draw: bool = True,
        repeat: bool = True,
    ) -> StreamWorker:
        with self._lock:
            if stream_id in self.workers:
                raise ValueError(f"поток {stream_id!r} уже запущен")
            worker = StreamWorker(
                stream_id=stream_id,
                source=source,
                pipeline=self.pipeline,
                settings=self.settings,
                store=self.store,
                loop=asyncio.get_running_loop(),
                target_fps=target_fps,
                draw=draw,
                repeat=repeat,
            )
            self.workers[stream_id] = worker
        worker.start()
        return worker

    def stop(self, stream_id: str) -> bool:
        with self._lock:
            worker = self.workers.pop(stream_id, None)
        if worker is None:
            return False
        worker.stop()
        return True

    def get(self, stream_id: str) -> StreamWorker | None:
        return self.workers.get(stream_id)

    def list(self) -> list[dict]:
        return [
            {
                "id": w.id,
                "source": w.source,
                "target_fps": w.target_fps,
                "connected": w.stats.connected,
                "finished": w.finished,
                "frames_read": w.stats.frames_read,
                "frames_processed": w.stats.frames_processed,
                "events": w.stats.events,
                "fps": w.stats.fps,
                "uptime_s": round(time.time() - w.stats.started_at, 1),
                "last_error": w.stats.last_error,
                "subscribers": len(w._subscribers),
            }
            for w in list(self.workers.values())
        ]

    def stop_all(self) -> None:
        for sid in list(self.workers):
            self.stop(sid)

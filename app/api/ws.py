"""WebSocket API.

/ws/recognize        — клиент льёт кадры, сервер отвечает распознаванием.
                       Кадр: бинарное сообщение с JPEG/PNG, либо JSON
                       {"type":"frame","image":"<base64|data-url>","frame_id":N}.
                       Есть backpressure: держим ровно один кадр в обработке,
                       новые кадры вытесняют необработанный (для live-видео это
                       правильное поведение — лучше свежий кадр, чем очередь).

/ws/streams/{id}     — подписка на серверный RTSP/файловый поток: сервер сам
                       читает видео и пушит результаты и события.

/ws/events           — только подтверждённые события со всех потоков.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import check_ws_api_key, get_ws_state
from app.config import settings
from app.core.pipeline import TrackedSession

log = logging.getLogger("anpr.ws")
router = APIRouter()

CLOSE_UNAUTHORIZED = 4401
CLOSE_NOT_FOUND = 4404


def _decode(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_b64(data: str) -> np.ndarray | None:
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        return _decode(base64.b64decode(data, validate=False))
    except (binascii.Error, ValueError):
        return None


@router.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    if not check_ws_api_key(websocket):
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="bad api key")
        return
    await websocket.accept()
    st = get_ws_state(websocket)
    loop = asyncio.get_running_loop()

    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    session = TrackedSession(st.pipeline, settings, source=f"ws:{peer}")
    log.info("ws/recognize подключён: %s", peer)

    # слот на один кадр: новый кадр вытесняет необработанный
    slot: dict = {"frame": None, "frame_id": 0, "client_ts": None}
    slot_event = asyncio.Event()
    stop = asyncio.Event()
    track_events = True
    return_overlay = False

    def on_event(event: dict, jpeg: bytes | None) -> None:
        if st.store is not None:
            event["id"] = st.store.add(event, jpeg)

    async def worker() -> None:
        while not stop.is_set():
            await slot_event.wait()
            slot_event.clear()
            frame, frame_id, client_ts = slot["frame"], slot["frame_id"], slot["client_ts"]
            slot["frame"] = None
            if frame is None:
                continue
            t0 = time.perf_counter()
            try:
                res = await loop.run_in_executor(
                    st.executor,
                    session.process,
                    frame,
                    on_event if track_events else None,
                )
            except Exception as exc:
                log.exception("ошибка инференса")
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                    )
                continue
            payload = res.to_dict()
            payload.update(
                {
                    "type": "result",
                    "frame_id": frame_id,
                    "client_ts": client_ts,
                    "server_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "fps": round(session.fps, 2),
                }
            )
            if return_overlay:
                from app.core.streams import draw_overlay

                ok, buf = cv2.imencode(
                    ".jpg",
                    draw_overlay(frame, res),
                    [cv2.IMWRITE_JPEG_QUALITY, settings.ws_jpeg_quality],
                )
                if ok:
                    payload["overlay_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode()
            with contextlib.suppress(Exception):
                await websocket.send_json(payload)

    worker_task = asyncio.create_task(worker())
    frame_counter = 0
    dropped = 0

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            frame = None
            frame_id = None
            client_ts = None

            if (data := msg.get("bytes")) is not None:
                frame = _decode(data)
            elif (text := msg.get("text")) is not None:
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "битый JSON"})
                    continue
                kind = obj.get("type", "frame")
                if kind == "config":
                    track_events = bool(obj.get("track_events", track_events))
                    return_overlay = bool(obj.get("overlay", return_overlay))
                    if obj.get("reset"):
                        session.tracker.reset()
                    await websocket.send_json(
                        {
                            "type": "config",
                            "track_events": track_events,
                            "overlay": return_overlay,
                        }
                    )
                    continue
                if kind == "ping":
                    await websocket.send_json({"type": "pong", "ts": obj.get("ts")})
                    continue
                if kind == "frame":
                    frame = _decode_b64(obj.get("image", ""))
                    frame_id = obj.get("frame_id")
                    client_ts = obj.get("ts")

            if frame is None:
                await websocket.send_json(
                    {"type": "error", "message": "кадр не декодировался (ожидаю JPEG/PNG)"}
                )
                continue

            frame_counter += 1
            if slot["frame"] is not None:
                dropped += 1
            slot["frame"] = frame
            slot["frame_id"] = frame_id if frame_id is not None else frame_counter
            slot["client_ts"] = client_ts
            slot_event.set()
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        slot_event.set()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        log.info(
            "ws/recognize отключён: %s (кадров %d, вытеснено %d)", peer, frame_counter, dropped
        )


@router.websocket("/ws/streams/{stream_id}")
async def ws_stream(websocket: WebSocket, stream_id: str):
    if not check_ws_api_key(websocket):
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="bad api key")
        return
    st = get_ws_state(websocket)
    worker = st.streams.get(stream_id)
    if worker is None:
        await websocket.close(code=CLOSE_NOT_FOUND, reason=f"нет потока {stream_id}")
        return
    await websocket.accept()
    queue = worker.subscribe()
    try:
        await websocket.send_json({"type": "hello", "stream": stream_id, "source": worker.source})
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        worker.unsubscribe(queue)


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """Все подтверждённые события со всех серверных потоков."""
    if not check_ws_api_key(websocket):
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="bad api key")
        return
    st = get_ws_state(websocket)
    await websocket.accept()
    queues: dict[str, asyncio.Queue] = {}
    last_ping = time.time()
    try:
        while True:
            # подписываемся на потоки, появившиеся после коннекта
            for sid, worker in list(st.streams.workers.items()):
                if sid not in queues:
                    queues[sid] = worker.subscribe()
            for sid in list(queues):
                if sid not in st.streams.workers:
                    queues.pop(sid, None)

            idle = True
            for q in list(queues.values()):
                while not q.empty():
                    payload = q.get_nowait()
                    idle = False
                    if payload.get("type") == "event":
                        await websocket.send_json(payload)
            if idle:
                # keepalive: без него мёртвый коннект не обнаружить (мы только пишем)
                if time.time() - last_ping > 15:
                    last_ping = time.time()
                    await websocket.send_json({"type": "ping", "ts": last_ping})
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    finally:
        for sid, q in queues.items():
            if (w := st.streams.get(sid)) is not None:
                w.unsubscribe(q)

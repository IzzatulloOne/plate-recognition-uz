"""REST: распознавание картинок, управление потоками, журнал событий."""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from app.api.deps import get_state, require_api_key
from app.config import settings
from app.schemas import (
    EventOut,
    HealthResponse,
    RecognizeResponse,
    StreamCreate,
    StreamOut,
)

router = APIRouter()


def _decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "не удалось декодировать изображение")
    return img


# ------------------------------------------------------------------ служебное
@router.get("/healthz", response_model=HealthResponse, tags=["service"])
async def healthz(request: Request):
    st = get_state(request)
    rec = st.pipeline.recognizer
    return HealthResponse(
        status="ok",
        device=settings.device,
        detector=str(settings.detector_weights.name),
        recognizer=str(settings.recognizer_weights.name),
        heads=list(rec.info.heads),
        num_class=rec.info.num_class,
        charset_len=len(settings.charset),
        streams=len(st.streams.workers),
        uptime_s=round(time.time() - st.started_at, 1),
    )


@router.get("/v1/stats", tags=["service"])
async def stats(request: Request):
    st = get_state(request)
    return {"events": st.store.stats() if st.store else None, "streams": st.streams.list()}


# --------------------------------------------------------------- распознавание
@router.post(
    "/v1/recognize",
    response_model=RecognizeResponse,
    dependencies=[Depends(require_api_key)],
    tags=["recognize"],
)
async def recognize(request: Request, file: UploadFile = File(..., description="кадр JPEG/PNG")):
    """Распознать номера на одном изображении (без трекинга)."""
    st = get_state(request)
    img = _decode_image(await file.read())
    res = await asyncio.get_running_loop().run_in_executor(
        st.executor, st.pipeline.process_frame, img
    )
    return RecognizeResponse(**res.to_dict())


@router.post(
    "/v1/recognize/plate",
    dependencies=[Depends(require_api_key)],
    tags=["recognize"],
)
async def recognize_plate(
    request: Request,
    file: UploadFile = File(..., description="уже вырезанная пластина"),
):
    """Прогнать распознаватель напрямую по кропу, минуя YOLO."""
    st = get_state(request)
    img = _decode_image(await file.read())
    rec = st.pipeline.recognizer

    def run():
        results = rec.recognize_with_alternatives(
            [img], topk=settings.beam_topk, beam=settings.beam_width
        )
        from app.core import plate_rules

        r, alts = results[0]
        text, conf, m = (
            plate_rules.pick_best(alts) if alts else (r.text, r.conf, plate_rules.repair(r.text))
        )
        if not m.valid:  # формат не найден — доверяем greedy, он честнее по вероятности
            m = plate_rules.repair(r.text)
            text, conf = m.text, r.conf
        color, color_conf = plate_rules.classify_color(img)
        return {
            "text": text,
            "conf": round(float(conf), 4),
            "raw_text": r.text,
            "raw_conf": round(r.conf, 4),
            "chars": [{"char": c, "p": round(p, 4)} for c, p in r.chars],
            "type_code": r.type_code,
            "type_conf": round(r.type_conf, 4),
            "format": m.format,
            "plate_class": m.plate_class,
            "valid": m.valid,
            "corrected": m.corrected,
            "pretty": m.pretty(),
            "color": color,
            "color_conf": round(color_conf, 3),
            "alternatives": [{"text": t, "conf": round(c, 4)} for t, c in alts[:8]],
        }

    return await asyncio.get_running_loop().run_in_executor(st.executor, run)


@router.post("/v1/annotate", dependencies=[Depends(require_api_key)], tags=["recognize"])
async def annotate(request: Request, file: UploadFile = File(...)):
    """То же, что /v1/recognize, но возвращает JPEG с нарисованными боксами."""
    from app.core.streams import draw_overlay

    st = get_state(request)
    img = _decode_image(await file.read())

    def run():
        res = st.pipeline.process_frame(img)
        ok, buf = cv2.imencode(".jpg", draw_overlay(img, res), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(500, "не удалось закодировать JPEG")
        return buf.tobytes()

    jpeg = await asyncio.get_running_loop().run_in_executor(st.executor, run)
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


# ---------------------------------------------------------------------- потоки
@router.get("/v1/streams", response_model=list[StreamOut], tags=["streams"])
async def list_streams(request: Request):
    return get_state(request).streams.list()


@router.post(
    "/v1/streams",
    response_model=StreamOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
    tags=["streams"],
)
async def create_stream(request: Request, body: StreamCreate):
    """Запустить обработку RTSP/файла/камеры на сервере."""
    st = get_state(request)
    try:
        st.streams.start(
            body.id,
            body.source,
            target_fps=body.target_fps,
            draw=body.draw,
            repeat=body.repeat,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await asyncio.sleep(0.4)  # дать потоку шанс подключиться, чтобы отдать статус
    info = next((s for s in st.streams.list() if s["id"] == body.id), None)
    if info is None:
        raise HTTPException(500, "поток не запустился")
    return info


@router.delete("/v1/streams/{stream_id}", dependencies=[Depends(require_api_key)], tags=["streams"])
async def delete_stream(request: Request, stream_id: str):
    if not get_state(request).streams.stop(stream_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"поток {stream_id!r} не найден")
    return {"stopped": stream_id}


@router.get("/v1/streams/{stream_id}/preview.mjpg", tags=["streams"])
async def stream_preview(request: Request, stream_id: str, fps: float = Query(10.0, ge=1, le=30)):
    """MJPEG-превью с боксами — открывается прямо в браузере."""
    st = get_state(request)
    worker = st.streams.get(stream_id)
    if worker is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"поток {stream_id!r} не найден")

    async def gen():
        delay = 1.0 / fps
        while st.streams.get(stream_id) is not None:
            jpeg = worker.preview_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(
                    len(jpeg)
                ).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(delay)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------- события
@router.get("/v1/events", response_model=list[EventOut], tags=["events"])
async def list_events(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    text: str | None = None,
    source: str | None = None,
    since: float | None = Query(None, description="unix-время, включительно"),
    plate_class: str | None = None,
    color: str | None = None,
):
    st = get_state(request)
    if st.store is None:
        return []
    return st.store.list(
        limit=limit,
        offset=offset,
        text=text,
        source=source,
        since=since,
        plate_class=plate_class,
        color=color,
    )


@router.get("/v1/events/{event_id}/snapshot", tags=["events"])
async def event_snapshot(request: Request, event_id: int):
    st = get_state(request)
    if st.store is None:
        raise HTTPException(404, "хранилище отключено")
    row = st.store.get(event_id)
    if row is None or not row.get("snapshot"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "снапшот не найден")
    path = settings.snapshot_dir / row["snapshot"]
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"файл отсутствует: {row['snapshot']}")
    return FileResponse(str(path), media_type="image/jpeg")

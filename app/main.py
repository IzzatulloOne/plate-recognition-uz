"""FastAPI-приложение ANPR: YOLO + best_accuracy.pth, REST и WebSocket."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import rest, ws
from app.config import settings
from app.core.pipeline import ANPRPipeline
from app.core.store import EventStore
from app.core.streams import StreamManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("anpr")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class AppState:
    pipeline: ANPRPipeline
    streams: StreamManager
    store: EventStore | None
    executor: ThreadPoolExecutor
    started_at: float = field(default_factory=time.time)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("загружаю детектор %s", settings.detector_weights)
    log.info("загружаю распознаватель %s", settings.recognizer_weights)
    t0 = time.time()
    pipeline = ANPRPipeline(settings)
    store = EventStore(
        settings.db_path,
        settings.snapshot_dir,
        settings.save_snapshots,
        keep=settings.events_keep,
    )
    executor = ThreadPoolExecutor(max_workers=max(1, settings.workers), thread_name_prefix="infer")
    app.state.anpr = AppState(
        pipeline=pipeline,
        streams=StreamManager(pipeline, settings, store),
        store=store,
        executor=executor,
    )
    log.info(
        "готово за %.1fс | головы=%s classes=%d device=%s",
        time.time() - t0,
        pipeline.recognizer.info.heads,
        pipeline.recognizer.info.num_class,
        settings.device,
    )
    try:
        yield
    finally:
        app.state.anpr.streams.stop_all()
        executor.shutdown(wait=False, cancel_futures=True)
        if store is not None:
            store.close()
        log.info("остановлено")


app = FastAPI(
    title="ANPR Uzbekistan",
    version="1.0.0",
    summary="Распознавание автомобильных номеров УЗ: YOLO11 + TPS-ResNet-BiLSTM-CTC",
    description=(
        "Детектор пластин — YOLO11n (дообученный на номерах), распознавание — "
        "`best_accuracy.pth` (две CTC-головы: `anpr` — текст, `ctype` — служебный код). "
        "Поверх: правила форматов УЗ, определение цвета фона, трекинг с голосованием "
        "по кадрам, журнал событий, WebSocket для живого видео."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest.router)
app.include_router(ws.router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    page = STATIC_DIR / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ANPR</h1><p>Документация: <a href='/docs'>/docs</a></p>")

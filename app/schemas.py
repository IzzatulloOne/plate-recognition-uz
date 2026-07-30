"""Pydantic-схемы запросов/ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlateOut(BaseModel):
    box: list[int] = Field(..., description="[x1, y1, x2, y2] в пикселях кадра")
    det_conf: float = Field(..., description="уверенность YOLO")
    text: str = Field(..., description="номер после нормализации и правил")
    conf: float = Field(..., description="уверенность распознавателя (0..1)")
    raw_text: str = Field(..., description="сырой greedy-выход CTC, без правил")
    format: str | None = Field(None, description="имя совпавшего формата, напр. uz_private")
    plate_class: str = Field(
        ..., description="private | legal | moto | trailer | special | unknown"
    )
    color: str = Field(..., description="цвет фона: white | yellow | green | blue | red | black")
    color_conf: float
    type_code: str = Field("", description="вывод второй головы (head_ctype), как есть")
    type_conf: float
    valid: bool = Field(..., description="текст подошёл под известный формат")
    corrected: bool = Field(..., description="применялось исправление путаницы символов")
    track_id: int | None = None
    stable_text: str | None = Field(None, description="результат голосования по кадрам трека")
    votes: int = 0
    stable_score: float = 0.0
    confirmed: bool = False


class Timings(BaseModel):
    detect_ms: float
    recognize_ms: float
    total_ms: float
    retried: bool = False


class RecognizeResponse(BaseModel):
    plates: list[PlateOut]
    timings: Timings
    frame_size: list[int]
    events: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    device: str
    detector: str
    recognizer: str
    heads: list[str]
    num_class: int
    charset_len: int
    streams: int
    uptime_s: float


class StreamCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    source: str = Field(..., description="RTSP URL, путь к файлу или индекс камеры ('0')")
    target_fps: float = Field(0.0, ge=0, le=60, description="0 = максимально быстро")
    draw: bool = Field(True, description="готовить MJPEG-превью")
    repeat: bool = Field(True, description="файл: читать по кругу (для RTSP не важно)")


class StreamOut(BaseModel):
    id: str
    source: str
    target_fps: float
    connected: bool
    finished: bool = False
    frames_read: int
    frames_processed: int
    events: int
    fps: float
    uptime_s: float
    last_error: str
    subscribers: int


class EventOut(BaseModel):
    id: int
    ts: float
    source: str
    track_id: int | None
    text: str
    pretty: str | None
    conf: float | None
    votes: int | None
    format: str | None
    plate_class: str | None
    color: str | None
    type_code: str | None
    box: str | None
    snapshot: str | None
    updated: int | None = 0
    previous: str | None = None

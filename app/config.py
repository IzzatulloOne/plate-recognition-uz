"""Конфигурация через переменные окружения / .env (префикс ANPR_)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.model import DEFAULT_CHARSET

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANPR_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------------------------------------------------------- распознавание
    recognizer_weights: Path = ROOT / "best_accuracy.pth"
    charset: str = DEFAULT_CHARSET
    preprocess: str = Field("resize", description="resize | pad")
    contrast_boost: bool = False
    quantize: bool = Field(False, description="dynamic int8 для LSTM/Linear (быстрее на CPU)")
    text_head: str = "anpr"
    type_head: str = "ctype"
    type_head_colors: str = Field(
        "yellow,green",
        description="цвета фона, для которых вывод второй головы идёт в кандидаты: "
        "её алфавит (цифры + H + M) совпадает с алфавитом цветных номеров УЗ. "
        "Кандидат побеждает только при совпадении с форматом, вреда нет.",
    )

    # ---------------------------------------------------------------- детектор
    detector_weights: Path = ROOT / "models" / "yolo11n-plate.pt"
    det_imgsz: int = 640
    det_conf: float = 0.35
    det_iou: float = 0.5
    det_max_det: int = 20
    #: второй проход в большем разрешении, если в первом ничего не нашлось.
    #: Дёшево (платим только за пустые кадры), но вытаскивает мелкие и косые пластины.
    det_retry_imgsz: int = Field(1536, description="0 = выключить второй проход")
    det_retry_conf: float = 0.10

    # ---------------------------------------------------------------- пайплайн
    device: str = "cpu"
    torch_threads: int = 4
    workers: int = Field(2, description="потоков инференса (event loop не блокируется)")
    crop_padding: float = 0.06
    min_plate_width: int = 24
    rec_min_conf: float = 0.30
    format_constraint: bool = Field(True, description="выбор варианта по формату номера УЗ")
    beam_topk: int = 3
    beam_width: int = 12

    # ---------------------------------------------------------------- трекинг
    track_iou: float = 0.25
    track_max_age: int = 20
    min_votes: int = 3
    event_cooldown_s: float = 8.0
    event_update_margin: float = Field(
        1.25, description="во сколько раз новый лидер должен перевесить уже отданный текст"
    )
    event_update_min_interval: float = 1.0

    # ---------------------------------------------------------------- хранение
    db_path: Path = ROOT / "data" / "anpr.db"
    save_snapshots: bool = True
    snapshot_dir: Path = ROOT / "data" / "snapshots"
    events_keep: int = 200_000

    # ---------------------------------------------------------------- сервер
    api_key: str = ""
    cors_origins: str = "*"
    ws_jpeg_quality: int = 70

    @property
    def type_head_color_set(self) -> set[str]:
        return {c.strip() for c in self.type_head_colors.split(",") if c.strip()}

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

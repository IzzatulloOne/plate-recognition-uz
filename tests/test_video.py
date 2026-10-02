"""Регрессия на размеченном видео: не потеряли ли номера, ускоряя обработку.

Видео в репозиторий не кладём — тест пропускается, пока не указан путь:

    ANPR_TEST_VIDEO=~/Downloads/video_2026-08-07_11-26-52.mp4 pytest tests/test_video.py -v -s

Смысл теста именно в сравнении двух схем на одних кадрах: быстрая обязана найти
всё то же, что и полная. Абсолютный FPS здесь не проверяется — он зависит от
машины и от того, насколько она успела нагреться.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

GT_FILE = Path(__file__).with_name("video_ground_truth.json")


def _load_frames(path: Path, step: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


@pytest.fixture(scope="module")
def video():
    raw = os.environ.get("ANPR_TEST_VIDEO")
    if not raw:
        pytest.skip("нет ANPR_TEST_VIDEO — размеченное видео не подключено")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"нет файла {path}")
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    frames = _load_frames(path, gt["frame_step"])
    if not frames:
        pytest.skip(f"не удалось прочитать кадры из {path}")
    return frames, gt


def _run(frames, **overrides) -> tuple[set[str], float]:
    """Прогон видео через сессию -> (тексты подтверждённых событий, мс на кадр)."""
    from app.config import Settings
    from app.core.pipeline import ANPRPipeline, TrackedSession

    settings = Settings(save_snapshots=False, **overrides)
    session = TrackedSession(ANPRPipeline(settings), settings, source="test")
    session.process(frames[0])  # прогрев, в статистику не идёт

    seen: set[str] = set()
    t0 = time.perf_counter()
    for frame in frames:
        session.process(frame, on_event=lambda e, _jpeg: seen.add(e["text"]))
    return seen, (time.perf_counter() - t0) * 1000 / len(frames)


def test_fast_scheme_keeps_every_plate(video):
    """Пропуск распознавания не должен стоить ни одного номера."""
    frames, gt = video
    expected = set(gt["plates"])

    fast, fast_ms = _run(frames)
    full, full_ms = _run(
        frames, skip_confirmed=False, stream_det_imgsz=640, stream_min_plate_width=24
    )

    print(f"\n  быстрая: {fast_ms:5.1f} мс/кадр, номеров {len(fast)}")
    print(f"  полная:  {full_ms:5.1f} мс/кадр, номеров {len(full)}")
    print(f"  ускорение: {full_ms / fast_ms:.1f}x")

    assert expected <= fast, f"быстрая схема потеряла: {sorted(expected - fast)}"
    assert expected <= full, f"полная схема потеряла: {sorted(expected - full)}"

    # Ложные срабатывания: отсев мелких пластин должен их сокращать, а не плодить.
    assert len(fast - expected) <= len(full - expected), (
        f"быстрая схема даёт больше мусора: {sorted(fast - expected)} "
        f"против {sorted(full - expected)}"
    )


def test_unreadable_plate_stays_silent(video):
    """Номер размером 58x28 нечитаем — система обязана промолчать, а не угадать.

    Проверяем, что порог stream_min_plate_width действительно его отсекает:
    прежде он выдавался с уверенностью 0.965 и валидным форматом.
    """
    frames, gt = video
    approx = gt["unreadable"]["approx_text"]
    prefix = approx[:4]  # '60M3' — общая часть всех вариантов чтения

    fast, _ = _run(frames)
    guesses = [t for t in fast if t.startswith(prefix)]
    assert not guesses, f"выдан нечитаемый номер: {guesses}"

"""Бенчмарк пайплайна: где уходит время и сколько FPS вытянет машина.

    python -m tools.bench                      # синтетический кадр 1280x720
    python -m tools.bench --image photo.jpg    # реальный кадр
    python -m tools.bench --video clip.mp4 -n 100
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from app.config import settings
from app.core.pipeline import ANPRPipeline


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("-n", "--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    print(
        f"детектор={settings.detector_weights.name} imgsz={settings.det_imgsz} "
        f"threads={settings.torch_threads} quantize={settings.quantize} "
        f"format_constraint={settings.format_constraint}"
    )
    t0 = time.time()
    pipe = ANPRPipeline(settings)
    print(f"загрузка моделей: {time.time() - t0:.1f}с")

    frames: list[np.ndarray] = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while len(frames) < args.iters:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"не читается: {args.image}")
        frames = [img]
    else:
        frames = [(np.random.rand(720, 1280, 3) * 255).astype("uint8")]
    if not frames:
        raise SystemExit("нет кадров")

    for i in range(args.warmup):
        pipe.process_frame(frames[i % len(frames)])

    det, rec, tot, npl = [], [], [], []
    for i in range(args.iters):
        r = pipe.process_frame(frames[i % len(frames)])
        det.append(r.detect_ms)
        rec.append(r.recognize_ms)
        tot.append(r.total_ms)
        npl.append(len(r.plates))

    def line(name, xs):
        print(
            f"  {name:12s} медиана {statistics.median(xs):7.1f} мс | "
            f"среднее {statistics.mean(xs):7.1f} | min {min(xs):6.1f} | max {max(xs):6.1f}"
        )

    print(f"\nкадров={args.iters}, номеров в кадре в среднем {statistics.mean(npl):.2f}")
    line("детекция", det)
    line("распознав.", rec)
    line("всего", tot)
    print(f"\n  => {1000 / statistics.median(tot):.1f} FPS на одном потоке инференса")
    parallel_fps = settings.workers * 1000 / statistics.median(tot)
    print(f"  => ~{parallel_fps:.1f} FPS при workers={settings.workers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

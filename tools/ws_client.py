"""Клиент для /ws/recognize: гонит видео/камеру/картинки на сервер и печатает ответы.

    python -m tools.ws_client --video clip.mp4 --fps 8
    python -m tools.ws_client --camera 0
    python -m tools.ws_client --images "photos/*.jpg"
    python -m tools.ws_client --video clip.mp4 --url ws://10.0.0.5:8000/ws/recognize --save out/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import time
from pathlib import Path

import cv2

try:
    import websockets
except ImportError:  # websockets тянется вместе с uvicorn[standard]
    raise SystemExit("нужен пакет websockets: uv pip install websockets") from None


def frames_from_args(args):
    """Генератор кадров (BGR)."""
    if args.images:
        for p in sorted(glob.glob(args.images)):
            img = cv2.imread(p)
            if img is not None:
                yield img, Path(p).name
        return
    src = int(args.camera) if args.camera is not None else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"не открылся источник: {src}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            yield frame, f"frame{i}"
    finally:
        cap.release()


async def run(args) -> None:
    headers = {"x-api-key": args.api_key} if args.api_key else None
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(args.url, additional_headers=headers, max_size=None) as ws:
        if args.overlay:
            await ws.send(json.dumps({"type": "config", "overlay": True}))

        stop = asyncio.Event()
        seen: dict[str, int] = {}

        async def receiver():
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    return
                msg = json.loads(raw)
                if msg.get("type") != "result":
                    if msg.get("type") == "error":
                        print("  ! сервер:", msg["message"])
                    continue
                plates = msg.get("plates", [])
                if plates:
                    parts = []
                    for p in plates:
                        txt = p.get("stable_text") or p["text"]
                        seen[txt] = seen.get(txt, 0) + 1
                        flag = "OK " if p["valid"] else "?  "
                        parts.append(f"{flag}{txt} {p['conf']:.2f} [{p['color']}] #{p['track_id']}")
                    print(
                        f"кадр {msg.get('frame_id')}: {msg.get('server_ms')}мс "
                        f"fps={msg.get('fps')} | " + " | ".join(parts)
                    )
                for ev in msg.get("events", []):
                    print(f"  >>> СОБЫТИЕ: {ev['pretty']} (голосов {ev['votes']}, {ev['color']})")
                if save_dir and msg.get("overlay_jpeg_b64"):
                    out = save_dir / f"{msg.get('frame_id')}.jpg"
                    out.write_bytes(base64.b64decode(msg["overlay_jpeg_b64"]))

        recv_task = asyncio.create_task(receiver())
        interval = 1.0 / args.fps if args.fps > 0 else 0.0
        sent = 0
        t0 = time.time()
        for frame, _name in frames_from_args(args):
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue
            await ws.send(buf.tobytes())
            sent += 1
            if interval:
                await asyncio.sleep(interval)
        await asyncio.sleep(1.5)  # дать серверу дообработать хвост
        stop.set()
        recv_task.cancel()

        dt = time.time() - t0
        print(f"\nотправлено {sent} кадров за {dt:.1f}с ({sent / max(dt, 1e-6):.1f} к/с)")
        if seen:
            print("уникальные номера:")
            for txt, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                print(f"  {txt:12s} x{n}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--camera")
    src.add_argument("--images", help="glob, напр. 'photos/*.jpg'")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/recognize")
    ap.add_argument("--fps", type=float, default=8.0, help="0 = без пауз")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--overlay", action="store_true", help="просить кадры с боксами")
    ap.add_argument("--save", help="куда сохранять overlay-кадры")
    args = ap.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

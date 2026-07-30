"""Диагностика распознавателя на своих картинках.

Зачем: в чекпоинте нет ни charset, ни настроек препроцесса — они восстановлены
по весам. Этот скрипт прогоняет ваши кропы всеми вариантами препроцесса и
печатает вывод ОБЕИХ голов, чтобы вы могли:
  1) убедиться, что charset верный (номера читаются, а не мусор);
  2) увидеть, что реально предсказывает голова ctype, и заполнить её смысл.

    python -m tools.probe_model crops/*.jpg
    python -m tools.probe_model crops/*.jpg --expect 01A123BC   # проверить вариант
    python -m tools.probe_model --detect photos/*.jpg           # сначала YOLO, потом OCR
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2

from app.config import settings
from app.core import plate_rules
from app.core.recognizer import PlateRecognizer

VARIANTS = [
    ("resize", False),
    ("resize", True),
    ("pad", False),
    ("pad", True),
]


def expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend(Path(x) for x in sorted(glob.glob(p)))
    return [p for p in out if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("images", nargs="+", help="кропы пластин (или кадры с --detect)")
    ap.add_argument("--detect", action="store_true", help="сначала найти пластины YOLO")
    ap.add_argument("--expect", default="", help="ожидаемый номер — покажет, какой вариант прав")
    ap.add_argument("--all-variants", action="store_true", help="перебрать все препроцессы")
    ap.add_argument("--chars", action="store_true", help="печатать уверенность по символам")
    args = ap.parse_args()

    files = expand(args.images)
    if not files:
        print("файлы не найдены", file=sys.stderr)
        return 2

    variants = VARIANTS if args.all_variants else [(settings.preprocess, settings.contrast_boost)]
    detector = None
    if args.detect:
        from app.core.detector import PlateDetector, crop_plate

        detector = PlateDetector(
            settings.detector_weights, settings.det_imgsz, settings.det_conf, device=settings.device
        )

    for prep, boost in variants:
        rec = PlateRecognizer(
            settings.recognizer_weights,
            device=settings.device,
            charset=settings.charset,
            preprocess=prep,
            contrast_boost=boost,
            num_threads=settings.torch_threads,
            text_head=settings.text_head,
            type_head=settings.type_head,
        )
        print(f"\n=== препроцесс={prep} clahe={boost} | головы={rec.info.heads} ===")
        hits = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"{f.name}: не читается")
                continue
            crops = [img]
            if detector is not None:
                dets = detector.detect(img)
                crops = [crop_plate(img, d, settings.crop_padding) for d in dets] or [img]
                print(f"{f.name}: YOLO нашёл {len(dets)} шт.")

            for i, crop in enumerate(crops):
                (r, alts) = rec.recognize_with_alternatives([crop])[0]
                m = plate_rules.repair(r.text)
                best_text, best_conf, bm = plate_rules.pick_best(alts)
                color, cfrac = plate_rules.classify_color(crop)
                tag = f"{f.name}" + (f"[{i}]" if len(crops) > 1 else "")
                print(
                    f"  {tag:28s} anpr={r.text!r:14s} conf={r.conf:.3f} "
                    f"-> норм={m.text!r:12s} формат={m.format} "
                    f"| по-формату={best_text!r:12s}({best_conf:.3f},{bm.format}) "
                    f"| ctype={r.type_code!r} ({r.type_conf:.3f}) | цвет={color}:{cfrac:.2f}"
                )
                if args.chars:
                    print("      " + "  ".join(f"{c}:{p:.2f}" for c, p in r.chars))
                if alts:
                    print("      альт: " + ", ".join(f"{t}({c:.2f})" for t, c in alts[:5]))
                if args.expect:
                    exp = plate_rules.normalize(args.expect)
                    if exp in (m.text, best_text, plate_rules.normalize(r.text)):
                        hits += 1
        if args.expect:
            print(f"  --> совпадений с {args.expect!r}: {hits}/{len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

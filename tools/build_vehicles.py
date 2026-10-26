"""Датасет моделей машин: вырезанная машина + класс уровня «Nexia, Cobalt».

    python -m tools.build_vehicles            # -> data/uz/vehicles/{train,val,test}/<класс>/
    python -m tools.build_vehicles --min 30   # класс — от 30 фото, остальное в «другое»

Машину вырезаем, а не учим на целом фото: на снимке бывают соседние машины, и
классификатор выучил бы фон. Цель — машина, на которой стоит номер: детектор
номеров даёт пластину, COCO-детектор — машины, берём машину с пластиной внутри.
Пластины не нашлось — самую крупную машину: platesmania кадрирует снимок по
своей машине.

Разбиение то же, что у номеров (build_dataset._split, по тексту номера): test —
часть эталона eval-v2 и в обучение не попадает.

Найденные боксы кэшируются: после добора данных пересчитываются только новые
фото, а пороги классов можно менять без повторной детекции.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

from app.core import plate_rules
from app.core.vehicle import (
    OTHER,
    VehicleDetector,
    candidates,
    crop_vehicle,
    pick_vehicle,
)
from app.core.vehicle_models import canonical
from tools.build_dataset import DATA, INDEX, _split

ROOT = Path(__file__).resolve().parent.parent
OUT = DATA / "vehicles"
BOXES = DATA / "vehicle_boxes.json"
#: длинная сторона кропа машины: в полном размере кропы занимали 3 ГБ, а учим на 224
CROP_SIDE = 320


def find_vehicle(img, coco: VehicleDetector, plates, trailer: bool) -> list[int] | None:
    """Машина с номером (общий выбор app.core.vehicle.pick_vehicle); иначе — самая крупная.

    Запасной вариант с самой крупной машиной — только для сборки датасета: на
    platesmania снимок кадрирован по своей машине. Сервис так не делает — на
    живом кадре самая крупная машина вполне может оказаться чужой.
    """
    cars = candidates(coco.detect(img), trailer)
    if not cars:
        return None
    if plates:
        p = max(plates, key=lambda d: d.conf)
        box = pick_vehicle(cars, [p.x1, p.y1, p.x2, p.y2])
        if box:
            return box
    return max(cars, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=int, default=30, help="минимум фото на класс")
    args = ap.parse_args()

    from app.config import settings
    from app.core.detector import PlateDetector

    items = json.loads(INDEX.read_text())
    labeled = {}
    for pid, v in items.items():
        label = canonical(v.get("model", ""), plate_rules.match(v["gt"]).plate_class == "trailer")
        if label:
            labeled[pid] = (label, v)

    boxes: dict = json.loads(BOXES.read_text()) if BOXES.exists() else {}
    todo = [pid for pid in labeled if pid not in boxes]
    if todo:
        coco = VehicleDetector()
        plates = PlateDetector(weights=settings.detector_weights, imgsz=settings.det_imgsz,
                               conf=settings.det_conf, iou=settings.det_iou,
                               max_det=settings.det_max_det, device=settings.device)
        for n, pid in enumerate(todo, 1):
            img = cv2.imread(str(DATA / labeled[pid][1]["file"]))
            trailer = plate_rules.match(labeled[pid][1]["gt"]).plate_class == "trailer"
            boxes[pid] = (find_vehicle(img, coco, plates.detect(img), trailer)
                          if img is not None else None)
            if n % 200 == 0:
                print(f"  детекция {n}/{len(todo)}", flush=True)
                BOXES.write_text(json.dumps(boxes))
        BOXES.write_text(json.dumps(boxes))

    # Классы — по числу фото в обучении: эталон не должен влиять даже на их выбор
    train_counts = Counter(lab for pid, (lab, v) in labeled.items()
                           if boxes.get(pid) and _split(v["gt"]) == "train")
    classes = sorted(c for c, n in train_counts.items() if n >= args.min)
    rest = sum(n for c, n in train_counts.items() if c not in classes)
    if rest >= args.min:
        classes.append(OTHER)

    if OUT.exists():
        shutil.rmtree(OUT)  # производный артефакт: собирается заново из кэша боксов
    for part in ("train", "val", "test"):
        for c in classes:  # пустые папки нужны: номера классов во всех частях обязаны совпасть
            (OUT / part / c).mkdir(parents=True, exist_ok=True)

    written: Counter = Counter()
    no_car = 0
    for pid, (label, v) in labeled.items():
        box = boxes.get(pid)
        if not box:
            no_car += 1
            continue
        cls = label if label in classes else (OTHER if OTHER in classes else None)
        if cls is None:
            continue
        crop = crop_vehicle(cv2.imread(str(DATA / v["file"])), box)
        h, w = crop.shape[:2]
        if max(h, w) > CROP_SIDE:  # классификатор смотрит в 224: больше хранить незачем
            k = CROP_SIDE / max(h, w)
            crop = cv2.resize(crop, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        part = _split(v["gt"])
        cv2.imwrite(str(OUT / part / cls / f"{pid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written[part, cls] += 1

    print(f"\nклассов {len(classes)}: {', '.join(classes)}")
    for part in ("train", "val", "test"):
        row = {c: written[part, c] for c in classes}
        print(f"  {part:5s} {sum(row.values()):5d}  {row}")
    print(f"машина не найдена: {no_car}  ->  {OUT}")
    from tools import provenance

    provenance.write(OUT / "manifest.json", {
        "что": "кропы машин по классам моделей (tools.build_vehicles)",
        "min на класс": args.min, "классы": classes,
        "по частям": {part: {c: written[part, c] for c in classes}
                      for part in ("train", "val", "test")},
        "детектор машин": {"файл": "models/yolo11n.pt",
                           "sha256": provenance.sha256(ROOT / "models" / "yolo11n.pt")},
        "правила": "app/core/vehicle_models.canonical; прицеп — только грузовики COCO",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

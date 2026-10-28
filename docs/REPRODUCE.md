# Как восстановить модели

Всё, что нужно, чтобы повторить датасет и каждую модель в другой сессии или на
другом компьютере. Что уже обучено и с какими параметрами — `models/registry.json`:
там у каждой модели файл, sha256, источник, параметры и замеры.

Каждый прогон сам пишет своё происхождение (`tools/provenance.py`: коммит, дата,
версии, суммы):

| Что | Где |
|---|---|
| датасет номеров | `data/uz/manifest.json` |
| датасет машин | `data/uz/vehicles/manifest.json` |
| дообучение распознавателя | `<out>/params.json` рядом с `best_accuracy.pth` |
| классификатор машин, детектор | `args.yaml` ultralytics в папке прогона |

Пометка `-dirty` у коммита значит, что код был не закоммичен — по одному хешу такой
прогон не повторить.

## 1. Установка

```bash
uv venv
# Без видеокарты NVIDIA — сначала CPU-сборка torch. Иначе pip притащит ~5 ГБ библиотек
# CUDA: на этом ноутбуке (только Intel UHD) окружение весило 7.5 ГБ вместо 1.7.
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install -e ".[dev,scrape]"                 # scrape — selenium для сбора
./.venv/bin/python -m tools.fetch_yolo            # детектор номеров -> models/
./.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; \
d('IzzatulloOne/anpr-uz', 'best_accuracy.pth', local_dir='.')"
./.venv/bin/python -c "from ultralytics.utils.downloads import attempt_download_asset as a; \
a('models/yolo11n.pt'); a('models/yolo11n-cls.pt')"
sha256sum best_accuracy.pth models/*.pt           # сверить с models/registry.json
```

## 2. Данные (platesmania, через живой браузер)

Капчу решает человек при первом запуске, профиль браузера сохраняется в
`data/chromium-profile`. Ночью — с `--unattended`: на капче этап выждет и закончится.

```bash
B="./.venv/bin/python -m tools.fetch_browser --unattended"
$B --add 5000 --ctype 1 --slice 1 --per-slice 500   # частные, по 500 с региона
$B --add 1500 --ctype 2 --slice 1 --per-slice 150   # юрлица
$B --add 3300 --ctype 6 --formats uz_trailer --slice 1
$B --add 600  --ctype 4                              # электро (M)
$B --add 300  --ctype 3                              # иностранные (H)
```

Что выяснено о сайте: ищет по номеру только от двух символов; параметр региона
`r` игнорирует; глубже сотой страницы повторяет последнюю. Подробнее — CLAUDE.md.

## 3. Датасеты

```bash
./.venv/bin/python -m tools.build_dataset --build \
    --verifier data/kaggle/run1/run/best_accuracy.pth   # прицепы подтверждает дообученная модель
./.venv/bin/python -m tools.build_vehicles
```

Разбиение детерминированное, по тексту номера: те же данные дают те же train/val/test.
Эталон eval-v2 (часть test) в обучение не попадает никогда.

## 4. Распознаватель (Kaggle, бесплатная T4)

```bash
./.venv/bin/python -m tools.kaggle_bundle      # -> data/kaggle/anpr-uz.zip
```

Загрузить как **приватный** датасет, ноутбук `notebooks/train_kaggle.ipynb`
(Settings → Internet включён, Accelerator GPU). Результат забрать:

```bash
./.venv/bin/kaggle kernels output <user>/<notebook> -p data/kaggle/runN --file-pattern '^run/'
```

Принимать, только если: эталон 80 не ниже 72/80, частные не ниже 60/64, прицепы выше
58/224. Замер:

```bash
ANPR_RECOGNIZER_WEIGHTS=data/kaggle/runN/run/best_accuracy.pth ./.venv/bin/python -m tools.eval_uz --run
ANPR_RECOGNIZER_WEIGHTS=... ./.venv/bin/python -m tools.eval_uz --run --out data/eval-v2
```

## 5. Детектор

```bash
./.venv/bin/python -m tools.label_boxes --prepare && ./.venv/bin/python -m tools.label_boxes
./.venv/bin/python -m tools.kaggle_bundle --detector    # -> data/kaggle/anpr-uz-det.zip
# Kaggle: notebooks/train_detector_kaggle.ipynb -> runs/det/weights/best.pt
./.venv/bin/python -m tools.eval_detector models/yolo11n-plate.pt <best.pt>
```

## 6. Модель машины

```bash
./.venv/bin/python -m tools.train_vehicles --epochs 30    # CPU, часы; --device 0 на GPU
ANPR_VEHICLE_WEIGHTS=models/vehicle-cls-v2.pt ./.venv/bin/python -m uvicorn app.main:app
```

## 7. Перенос на другой компьютер

Код — `git clone`. Веса и данные в git не лежат, их копировать:

| Что | Размер | Нужно для |
|---|---|---|
| `best_accuracy.pth` (распознаватель v2), `models/anpr-det-v1.pt` (детектор) | дефолтные веса сервиса; как и `best_accuracy.pth`, обученные нами — не из `fetch_yolo`, копировать или залить на HF |
| `models/yolo11n-plate.pt`, `models/yolo11n*.pt` | базы для дообучения (`fetch_yolo`, ultralytics) |
| `data/kaggle/run*/run/best_accuracy.pth` | 188 МБ каждый | проверяльщик прицепов |
| `data/uz/index.json` + `data/uz/photos/` | ~6 ГБ | пересборки датасетов |
| `data/eval*`, `data/det-bench/` | небольшие | замеров |

После копирования — `sha256sum` и сверка с `models/registry.json`. Данные с
platesmania публиковать нельзя: только приватные копии.

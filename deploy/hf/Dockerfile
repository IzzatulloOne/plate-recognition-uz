# Docker-образ для Hugging Face Spaces (бесплатный CPU-тариф: 2 ядра, 16 ГБ RAM).
#
# Space хранит только этот файл и README.md — код берётся из GitHub на сборке,
# веса распознавателя из репозитория моделей HF. Так Space остаётся лёгким, а
# обновление сводится к перезапуску сборки.

FROM python:3.12-slim

# libgomp1 нужен torch, libglib2.0-0 — opencv. Остальное opencv-headless не тянет.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces запускают контейнер от uid 1000. Пишем всё в его домашний каталог,
# иначе ultralytics и matplotlib упрутся в права при первом запуске.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# CPU-сборка torch: обычный индекс притащил бы ~4.9 ГБ пакетов nvidia и triton,
# бесполезных без видеокарты. На замере venv худеет с 7.3 до 2.4 ГБ.
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision

ARG REPO=https://github.com/IzzatulloOne/plate-recognition-uz.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} . \
    && pip install --no-cache-dir --user -e .

# Веса детектора (~5 МБ) и распознавателя (188 МБ) кладём в слой образа, а не
# качаем на старте: Spaces кешируют собранный образ, а холодный старт и так
# занимает около минуты.
ARG WEIGHTS_REPO=IzzatulloOne/anpr-uz
RUN python -m tools.fetch_yolo \
    && python -c "from huggingface_hub import hf_hub_download as d; \
d(repo_id='${WEIGHTS_REPO}', filename='best_accuracy.pth', local_dir='.')"

# База событий и снапшоты. На бесплатном тарифе диск эфемерный — при перезапуске
# Space всё обнуляется, поэтому снапшоты выключены (см. ANPR_SAVE_SNAPSHOTS ниже).
RUN mkdir -p data/snapshots

ENV ANPR_SAVE_SNAPSHOTS=false \
    ANPR_WORKERS=2 \
    ANPR_TORCH_THREADS=2

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

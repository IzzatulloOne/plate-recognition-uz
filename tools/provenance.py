"""Откуда взялась модель или датасет: коммит кода, дата, версии, контрольные суммы.

Каждый прогон обучения и каждая сборка датасета пишут это рядом с результатом,
чтобы его можно было повторить в другой сессии или на другом компьютере.
Сводный реестр — models/registry.json, порядок действий — docs/REPRODUCE.md.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Короткий хеш коммита; «-dirty», если в рабочей копии есть незакоммиченные правки.

    С пометкой -dirty прогон по одному хешу не повторить: код был не тот, что в коммите.
    """
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "app", "tools"],
                                        cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "нет git"  # например, код приехал на Kaggle архивом


def stamp() -> dict:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {"git": git_commit(), "date": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "torch": torch.__version__, "device": gpu}


def write(path: str | Path, payload: dict) -> Path:
    """payload + stamp() -> JSON. Пути превращаются в строки."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**payload, "происхождение": stamp()}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    return path

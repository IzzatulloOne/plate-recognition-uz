"""Модель распознавания из best_accuracy.pth.

Определено по чекпоинту (246 тензоров):

    Transformation      : TPS_SpatialTransformerNetwork, F=20, I_r_size=(32,100)
    FeatureExtraction   : ResNet (DTRB), 1 канал (grayscale) -> 512
    SequenceModeling    : 2 x BidirectionalLSTM(hidden=256) -> 256
    Prediction          : ДВЕ CTC-головы, обе Linear(256 -> 95):
                            head_anpr  -> текст номера
                            head_ctype -> вспомогательный код (тип номера/ТС)

95 классов = 1 blank (CTC) + 94 символа => charset = string.printable[:94]
(цифры + строчные + ЗАГЛАВНЫЕ + пунктуация).

Анализ bias/‖w‖ по строкам голов подтверждает набор:
    head_anpr : живые классы = blank, 0-9, A-Z (кроме 'I'); строчные и пунктуация мертвы
    head_ctype: живые классы = blank, 0-9, 'H', 'M'
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from app.core.modules.feature_extraction import ResNet_FeatureExtractor
from app.core.modules.sequence_modeling import BidirectionalLSTM
from app.core.modules.transformation import TPS_SpatialTransformerNetwork

#: порядок символов DTRB по умолчанию; индекс класса = позиция + 1 (класс 0 = CTC blank)
DEFAULT_CHARSET = string.printable[:94]

IMG_H = 32
IMG_W = 100
NUM_FIDUCIAL = 20
HIDDEN_SIZE = 256
OUTPUT_CHANNEL = 512


@dataclass(frozen=True)
class ModelInfo:
    num_class: int
    heads: tuple[str, ...]
    img_h: int
    img_w: int


class ANPRNet(nn.Module):
    """TPS-ResNet-BiLSTM-CTC с двумя головами."""

    def __init__(self, num_class: int, heads: tuple[str, ...] = ("anpr", "ctype")):
        super().__init__()
        self.head_names = heads
        self.Transformation = TPS_SpatialTransformerNetwork(
            F_num=NUM_FIDUCIAL,
            I_size=(IMG_H, IMG_W),
            I_r_size=(IMG_H, IMG_W),
            I_channel_num=1,
        )
        self.FeatureExtraction = ResNet_FeatureExtractor(1, OUTPUT_CHANNEL)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(OUTPUT_CHANNEL, HIDDEN_SIZE, HIDDEN_SIZE),
            BidirectionalLSTM(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE),
        )
        for name in heads:
            setattr(self, f"head_{name}", nn.Linear(HIDDEN_SIZE, num_class))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [b, 1, 32, 100] в диапазоне [-1, 1] -> {head: logits [b, T, num_class]}"""
        x = self.Transformation(x)
        visual = self.FeatureExtraction(x)  # [b, C, H', W']
        visual = self.AdaptiveAvgPool(visual.permute(0, 3, 1, 2))  # [b, W', C, 1]
        visual = visual.squeeze(3)  # [b, T, C]
        contextual = self.SequenceModeling(visual)  # [b, T, 256]
        return {name: getattr(self, f"head_{name}")(contextual) for name in self.head_names}


def _strip_prefix(state: dict, prefix: str = "module.") -> dict:
    return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state.items()}


def inspect_checkpoint(path: str | Path) -> ModelInfo:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_prefix(state)
    heads = tuple(
        sorted(
            k[len("head_") : -len(".weight")]
            for k in state
            if k.startswith("head_") and k.endswith(".weight")
        )
    )
    if not heads:
        raise ValueError(f"в {path} не найдено ни одной головы head_*.weight")
    num_class = state[f"head_{heads[0]}.weight"].shape[0]
    return ModelInfo(num_class=num_class, heads=heads, img_h=IMG_H, img_w=IMG_W)


def load_model(path: str | Path, device: str = "cpu") -> tuple[ANPRNet, ModelInfo]:
    info = inspect_checkpoint(path)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_prefix(state)

    model = ANPRNet(num_class=info.num_class, heads=info.heads)
    missing, unexpected = model.load_state_dict(state, strict=False)
    hard_missing = [k for k in missing if "num_batches_tracked" not in k]
    if hard_missing or unexpected:
        raise RuntimeError(
            "чекпоинт не совпал с моделью.\n"
            f"missing={hard_missing[:10]}\nunexpected={unexpected[:10]}"
        )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, info

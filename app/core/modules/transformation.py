"""TPS spatial transformer (RARE), совместим с clovaai/deep-text-recognition-benchmark.

Ключи в best_accuracy.pth:
    Transformation.LocalizationNetwork.conv.{0,1,4,5,8,9,12,13}
    Transformation.LocalizationNetwork.localization_fc1.0 / localization_fc2  (out=40 -> F=20)
    Transformation.GridGenerator.inv_delta_C (23,23)  # F+3
    Transformation.GridGenerator.P_hat (3200,23)      # 32*100 x F+3
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TPS_SpatialTransformerNetwork(nn.Module):
    """Rectification network of RARE, namely TPS based STN."""

    def __init__(self, F_num: int, I_size, I_r_size, I_channel_num: int = 1):
        super().__init__()
        self.F = F_num
        self.I_size = I_size
        self.I_r_size = I_r_size  # (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I: torch.Tensor) -> torch.Tensor:
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape(
            [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2]
        )
        return F.grid_sample(
            batch_I, build_P_prime_reshape, padding_mode="border", align_corners=True
        )


class LocalizationNetwork(nn.Module):
    """Predict C' (K x 2) from input image I."""

    def __init__(self, F_num: int, I_channel_num: int):
        super().__init__()
        self.F = F_num
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(self.I_channel_num, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128 x I_h/4 x I_w/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 256 x I_h/8 x I_w/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # 512
        )
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # init fc2 to identity fiducial points (важно только для обучения)
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F_num / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F_num / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F_num / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I: torch.Tensor) -> torch.Tensor:
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        return self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)


class GridGenerator(nn.Module):
    """Grid Generator of RARE: P_prime = P_hat @ T."""

    def __init__(self, F_num: int, I_r_size):
        super().__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F_num
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        # buffers, чтобы совпадать с чекпоинтом (обучали с DataParallel)
        self.register_buffer(
            "inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float()
        )
        self.register_buffer(
            "P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float()
        )

    def _build_C(self, F_num):
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F_num / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F_num / 2))
        ctrl_pts_y_bottom = np.ones(int(F_num / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        return np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)

    def _build_inv_delta_C(self, F_num, C):
        hat_C = np.zeros((F_num, F_num), dtype=float)
        for i in range(0, F_num):
            for j in range(i, F_num):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F_num, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, F_num))], axis=1),
            ],
            axis=0,
        )
        return np.linalg.inv(delta_C)

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        # n (= I_r_width x I_r_height) x 2
        return np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2).reshape([-1, 2])

    def _build_P_hat(self, F_num, C, P):
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F_num, 1))  # n x F x 2
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        return np.concatenate([np.ones((n, 1)), P, rbf], axis=1)

    def build_P_prime(self, batch_C_prime: torch.Tensor) -> torch.Tensor:
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        zeros = torch.zeros(
            batch_size, 3, 2, dtype=batch_C_prime.dtype, device=batch_C_prime.device
        )
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, zeros), dim=1)
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        return torch.bmm(batch_P_hat, batch_T)

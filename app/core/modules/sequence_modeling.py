"""BiLSTM sequence modeling (DTRB)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: visual feature [b, T, input_size] -> [b, T, output_size]"""
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)

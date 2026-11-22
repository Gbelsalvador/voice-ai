from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class STTModel(nn.Module):
    """Baseline STT character-level : CNN de sous-échantillonnage + BiGRU + CTC."""
    def __init__(self, vocab_size: int, n_mels: int = 80, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        reduced_freq = (n_mels + 3) // 4
        self.projection = nn.Linear(64 * reduced_freq, hidden)
        self.encoder = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(hidden * 2, vocab_size)

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths + 3, 4, rounding_mode="floor")

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        # features: [B, T, F]
        x = features.unsqueeze(1)                  # [B, 1, T, F]
        x = self.cnn(x)                           # [B, C, T/4, F/4]
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = torch.relu(self.projection(x))
        out_lengths = self.output_lengths(lengths).clamp(min=1, max=t)
        packed = pack_padded_sequence(x, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=t)
        return self.classifier(x), out_lengths

"""
NanoVox lightweight vocoder: mel spectrogram -> waveform.

Implements a stripped-down HiFi-GAN-style generator optimized for CPU.
Parameter count: ~3M (nano) / ~5M (small) - kept tiny intentionally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import VocoderConfig


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d, padding=self._pad(kernel_size, d)
                ),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            )
            for d in dilation
        ])

    @staticmethod
    def _pad(k: int, d: int) -> int:
        return (k - 1) * d // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class MRF(nn.Module):
    """Multi-Receptive Field Fusion."""

    def __init__(
        self,
        channels: int,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, k, d)
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for block in self.blocks:
            y = block(x)
            out = y if out is None else out + y
        return out / len(self.blocks)


class HiFiGANGenerator(nn.Module):
    """
    Lightweight HiFi-GAN generator.

    Upsamples mel spectrograms to raw waveforms via transposed convolutions
    followed by multi-receptive-field fusion residual blocks.
    """

    def __init__(self, cfg: VocoderConfig):
        super().__init__()
        ch = cfg.initial_channels
        self._n_mels = cfg.n_mels

        self.pre = nn.Conv1d(cfg.n_mels, ch, 7, padding=3)

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        for i, (u, k) in enumerate(
            zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.ConvTranspose1d(
                    ch // (2 ** i),
                    ch // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            self.mrfs.append(
                MRF(
                    ch // (2 ** (i + 1)),
                    cfg.resblock_kernel_sizes,
                    cfg.resblock_dilation_sizes,
                )
            )

        final_ch = ch // (2 ** len(cfg.upsample_rates))
        self.post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(final_ch, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T_mel) or (B, T_mel, n_mels) - we handle both
        Returns:
            audio: (B, 1, T_audio)
        """
        # Accept (B, T_mel, n_mels) and convert to (B, n_mels, T_mel)
        # n_mels is always 80 per config; use that to detect layout
        if mel.dim() == 3 and mel.shape[1] != self._n_mels:
            mel = mel.transpose(1, 2)  # -> (B, n_mels, T)

        x = self.pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        return self.post(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vocoder(cfg: VocoderConfig) -> HiFiGANGenerator:
    return HiFiGANGenerator(cfg)

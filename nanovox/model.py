"""
NanoVox model architectures.

Two variants:
  - NanoVoxNano  (~14M params): for edge/embedded, fastest
  - NanoVoxSmall (~40M params): better quality, still CPU-native

Architecture:
  Text -> [TextEncoder (Transformer)] -> hidden states
       -> [LengthRegulator] -> expanded states
       -> [MelDecoder (Transformer)] -> mel spectrogram
       -> [Vocoder] -> waveform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import NanoVoxConfig, TextEncoderConfig, MelDecoderConfig, VocoderConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = query.shape

        def split_heads(x):
            x = x.view(B, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)  # (B, H, T, dk)

        q = split_heads(self.q_proj(query))
        k = split_heads(self.k_proj(key))
        v = split_heads(self.v_proj(value))

        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, mem_mask)))
        x = self.norm3(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Encodes tokenized text into continuous representations.
    """

    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pe = SinusoidalPE(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.d_model = cfg.d_model

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pe(self.embed(tokens))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Duration / Length Regulator
# ---------------------------------------------------------------------------

class DurationPredictor(nn.Module):
    """Predicts phoneme durations for length regulation."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h = x.transpose(1, 2)  # (B, d_model, T)
        h = self.net[0](h)
        h = self.net[1](h)
        h = self.net[2](h.transpose(1, 2)).transpose(1, 2)
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        h = self.net[6](h.transpose(1, 2)).transpose(1, 2)
        h = self.net[7](h)
        h = h.transpose(1, 2)  # (B, T, d_model)
        return self.net[8](h).squeeze(-1)  # (B, T)


class LengthRegulator(nn.Module):
    """Repeats encoder outputs according to predicted durations."""

    def __init__(self, d_model: int, speed: float = 1.0):
        super().__init__()
        self.predictor = DurationPredictor(d_model)
        self.speed = speed

    def forward(
        self,
        x: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_durations = self.predictor(x)
        durations = torch.clamp(
            torch.round(torch.exp(log_durations) / self.speed).long(), min=1
        )

        if target_durations is not None:
            durations = target_durations

        # Repeat each token embedding by its duration
        expanded = []
        for b in range(x.size(0)):
            frames = []
            for t in range(x.size(1)):
                d = int(durations[b, t].item())
                frames.append(x[b, t : t + 1].expand(d, -1))
            expanded.append(torch.cat(frames, dim=0))

        # Pad to same length
        max_len = max(e.size(0) for e in expanded)
        padded = torch.zeros(x.size(0), max_len, x.size(2), device=x.device)
        for b, e in enumerate(expanded):
            padded[b, : e.size(0)] = e

        return padded, durations


# ---------------------------------------------------------------------------
# Mel Decoder
# ---------------------------------------------------------------------------

class MelDecoder(nn.Module):
    """Transformer decoder that produces mel spectrogram frames."""

    def __init__(self, cfg: MelDecoderConfig, encoder_d_model: int):
        super().__init__()
        self.input_proj = nn.Linear(encoder_d_model, cfg.d_model)
        self.pe = SinusoidalPE(cfg.d_model, cfg.max_mel_len, cfg.dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mel_proj = nn.Linear(cfg.d_model, cfg.n_mels)
        self.n_mels = cfg.n_mels

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pe(self.input_proj(x))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.mel_proj(x)  # (B, T_mel, n_mels)


# ---------------------------------------------------------------------------
# Full TTS Model
# ---------------------------------------------------------------------------

class NanoVoxModel(nn.Module):
    """
    End-to-end TTS model: text tokens -> mel spectrogram.

    The vocoder (mel -> waveform) is a separate module (see vocoder.py).
    """

    def __init__(self, cfg: NanoVoxConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg.encoder)
        self.length_regulator = LengthRegulator(cfg.encoder.d_model, cfg.speed)
        self.decoder = MelDecoder(cfg.decoder, cfg.encoder.d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        target_durations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, T_text) token IDs
            src_mask: optional padding mask
            target_durations: (B, T_text) durations (training only)
        Returns:
            mel: (B, T_mel, n_mels)
            durations: (B, T_text) predicted durations
        """
        enc = self.encoder(tokens, src_mask)
        expanded, durations = self.length_regulator(enc, target_durations)
        mel = self.decoder(expanded)
        return mel, durations

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: NanoVoxConfig) -> "NanoVoxModel":
        return cls(cfg)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_nano_model() -> NanoVoxModel:
    """Build the ~14M parameter nano model."""
    from .config import NANO_CONFIG
    return NanoVoxModel(NANO_CONFIG)


def build_small_model() -> NanoVoxModel:
    """Build the ~40M parameter small model."""
    from .config import SMALL_CONFIG
    return NanoVoxModel(SMALL_CONFIG)

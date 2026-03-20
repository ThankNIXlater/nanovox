"""
NanoVox model configurations.

Two variants:
  - nano  : ~14M parameters, fastest inference, smallest footprint
  - small : ~40M parameters, higher quality, still CPU-friendly
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TextEncoderConfig:
    vocab_size: int = 512
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1


@dataclass
class MelDecoderConfig:
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 1024
    n_mels: int = 80
    max_mel_len: int = 1024
    dropout: float = 0.1


@dataclass
class VocoderConfig:
    n_mels: int = 80
    upsample_rates: tuple = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    initial_channels: int = 128


@dataclass
class NanoVoxConfig:
    """Complete model configuration."""
    model_name: str = "nano"
    encoder: TextEncoderConfig = None
    decoder: MelDecoderConfig = None
    vocoder: VocoderConfig = None

    # Audio params
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0

    # Inference params
    speed: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = TextEncoderConfig()
        if self.decoder is None:
            self.decoder = MelDecoderConfig()
        if self.vocoder is None:
            self.vocoder = VocoderConfig()


# Preset configs

NANO_CONFIG = NanoVoxConfig(
    model_name="nano",
    encoder=TextEncoderConfig(
        vocab_size=512,
        d_model=192,
        num_heads=4,
        num_layers=3,
        d_ff=768,
        max_seq_len=512,
        dropout=0.1,
    ),
    decoder=MelDecoderConfig(
        d_model=192,
        num_heads=4,
        num_layers=3,
        d_ff=768,
        n_mels=80,
        max_mel_len=1024,
        dropout=0.1,
    ),
    vocoder=VocoderConfig(
        initial_channels=64,
        upsample_rates=(8, 8, 2, 2),
    ),
)

SMALL_CONFIG = NanoVoxConfig(
    model_name="small",
    encoder=TextEncoderConfig(
        vocab_size=512,
        d_model=384,
        num_heads=6,
        num_layers=6,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1,
    ),
    decoder=MelDecoderConfig(
        d_model=384,
        num_heads=6,
        num_layers=6,
        d_ff=1536,
        n_mels=80,
        max_mel_len=1024,
        dropout=0.1,
    ),
    vocoder=VocoderConfig(
        initial_channels=128,
        upsample_rates=(8, 8, 2, 2),
    ),
)

MODEL_CONFIGS = {
    "nano": NANO_CONFIG,
    "small": SMALL_CONFIG,
}


def get_config(model_name: str) -> NanoVoxConfig:
    """Get config by model name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]

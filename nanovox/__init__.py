"""NanoVox - Lightweight TTS. CPU-only. Ship voice anywhere."""

__version__ = "0.2.1"

from .inference import speak, synthesize

__all__ = ["speak", "synthesize", "__version__"]

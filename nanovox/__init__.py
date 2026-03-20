"""
NanoVox - Lightweight CPU-native text-to-speech.

Quick start:
    from nanovox import speak
    speak("Hello world", output="hello.wav")

Or with more control:
    from nanovox import NanoVoxTTS
    tts = NanoVoxTTS(model="small")
    tts.speak("Hello world", output="hello.wav", speed=0.9)
"""

__version__ = "0.1.0"
__author__ = "NanoVox Contributors"
__license__ = "MIT"

from .inference import speak, NanoVoxTTS
from .config import get_config, MODEL_CONFIGS, NANO_CONFIG, SMALL_CONFIG
from .tokenizer import get_tokenizer, CharTokenizer
from .model import NanoVoxModel, build_nano_model, build_small_model
from .vocoder import HiFiGANGenerator, build_vocoder

__all__ = [
    "speak",
    "NanoVoxTTS",
    "get_config",
    "MODEL_CONFIGS",
    "NANO_CONFIG",
    "SMALL_CONFIG",
    "get_tokenizer",
    "CharTokenizer",
    "NanoVoxModel",
    "build_nano_model",
    "build_small_model",
    "HiFiGANGenerator",
    "build_vocoder",
]

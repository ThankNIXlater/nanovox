"""
NanoVox Inference - Real TTS powered by Piper (ONNX, CPU-only)
Two model variants:
  - nano: en_US-amy-low (~15MB, fastest)
  - small: en_US-lessac-medium (~61MB, better quality)
"""

import os
import wave
import json
import struct
import subprocess
import sys
from pathlib import Path

VOICE_DIR = Path.home() / ".cache" / "nanovox" / "voices"

VOICES = {
    "nano": {
        "model": "en_US-amy-low.onnx",
        "config": "en_US-amy-low.onnx.json",
        "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low",
        "description": "Fast, lightweight (~15MB model)",
    },
    "small": {
        "model": "en_US-lessac-medium.onnx",
        "config": "en_US-lessac-medium.onnx.json",
        "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium",
        "description": "Higher quality (~61MB model)",
    },
}


def _download_voice(variant: str):
    """Download voice model if not cached."""
    voice = VOICES[variant]
    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    for fname in [voice["model"], voice["config"]]:
        local = VOICE_DIR / fname
        if local.exists():
            continue
        url = f"{voice['url_base']}/{fname}"
        print(f"[NanoVox] Downloading {fname}...")
        import urllib.request
        urllib.request.urlretrieve(url, str(local))
        print(f"[NanoVox] Saved to {local}")


def synthesize(text: str, output: str = "output.wav", model: str = "nano", speed: float = 1.0):
    """
    Synthesize text to speech using piper-tts.

    Args:
        text: Text to speak
        output: Output WAV file path
        model: 'nano' (fast) or 'small' (quality)
        speed: Speaking rate multiplier (0.5 = slow, 2.0 = fast)
    """
    if model not in VOICES:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(VOICES.keys())}")

    _download_voice(model)

    voice = VOICES[model]
    model_path = str(VOICE_DIR / voice["model"])
    config_path = str(VOICE_DIR / voice["config"])

    cmd = [
        sys.executable, "-m", "piper",
        "--model", model_path,
        "--config", config_path,
        "--output_file", output,
        "--length-scale", str(1.0 / speed),
    ]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
    )

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper TTS failed: {err}")

    if not os.path.exists(output):
        raise RuntimeError(f"Output file not created: {output}")

    size_kb = os.path.getsize(output) / 1024
    print(f"[NanoVox] Generated {output} ({size_kb:.1f}KB) using {model} model")
    return output


def speak(text: str, output: str = "output.wav", model: str = "nano", speed: float = 1.0):
    """Convenience alias for synthesize()."""
    return synthesize(text, output=output, model=model, speed=speed)

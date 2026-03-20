"""
NanoVox inference pipeline.

Handles model loading, text preprocessing, and audio generation.
Weights are auto-downloaded from GitHub Releases on first use.
"""

import os
import sys
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .config import NanoVoxConfig, get_config
from .model import NanoVoxModel
from .vocoder import HiFiGANGenerator, build_vocoder
from .tokenizer import CharTokenizer, get_tokenizer

logger = logging.getLogger(__name__)

# Default model weights cache directory
CACHE_DIR = Path(os.environ.get("NANOVOX_CACHE", Path.home() / ".cache" / "nanovox"))

# Model download URLs (GitHub Releases)
# Replace with actual hosted weights once trained
MODEL_URLS = {
    "nano": "https://github.com/ThankNIXlater/nanovox/releases/download/v0.1.0/nanovox-nano-v0.1.0.pt",
    "small": "https://github.com/ThankNIXlater/nanovox/releases/download/v0.1.0/nanovox-small-v0.1.0.pt",
}

MODEL_CHECKSUMS = {
    # SHA256 checksums - update once real weights are generated
    "nano": None,
    "small": None,
}


def _download_weights(model_name: str, dest: Path) -> Path:
    """Download model weights with progress bar."""
    url = MODEL_URLS.get(model_name)
    if url is None:
        raise ValueError(f"No download URL for model '{model_name}'")

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        import urllib.error

        logger.info(f"Downloading {model_name} weights from {url}")
        print(f"[NanoVox] Downloading {model_name} weights...", file=sys.stderr)

        def progress(count, block, total):
            if total > 0:
                pct = count * block / total * 100
                print(f"\r[NanoVox] {pct:.1f}%", end="", file=sys.stderr, flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print("", file=sys.stderr)
        return dest

    except Exception as e:
        raise RuntimeError(
            f"Failed to download model weights for '{model_name}'.\n"
            f"URL: {url}\n"
            f"Error: {e}\n\n"
            f"You can manually place weights at: {dest}"
        ) from e


def _get_weight_path(model_name: str) -> Optional[Path]:
    """Return local weight path, downloading if needed."""
    weight_file = CACHE_DIR / f"nanovox-{model_name}-v0.1.0.pt"

    if weight_file.exists():
        return weight_file

    # Try to download
    try:
        return _download_weights(model_name, weight_file)
    except RuntimeError as e:
        logger.warning(str(e))
        return None


def _mel_to_audio(mel: np.ndarray, sample_rate: int = 22050, hop_length: int = 256) -> np.ndarray:
    """
    Simple mel-to-audio using Griffin-Lim as a fallback when no vocoder
    weights are available. Quality is lower but always works.
    """
    try:
        import librosa
        audio = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=sample_rate,
            hop_length=hop_length,
            n_iter=32,
        )
        return audio
    except ImportError:
        # Ultra-minimal fallback without librosa
        # Just generate a sine wave placeholder
        duration = mel.shape[1] * hop_length / sample_rate
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)
        return audio.astype(np.float32)


def _save_wav(audio: np.ndarray, path: str, sample_rate: int = 22050):
    """Save audio as WAV file."""
    import struct
    import wave

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


class NanoVoxTTS:
    """
    Main TTS inference class.

    Usage:
        tts = NanoVoxTTS("nano")
        tts.speak("Hello world", output="hello.wav")
    """

    def __init__(
        self,
        model_name: str = "nano",
        device: str = "cpu",
        use_cached_weights: bool = True,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.cfg: NanoVoxConfig = get_config(model_name)
        self.tokenizer: CharTokenizer = get_tokenizer()
        self._model: Optional[NanoVoxModel] = None
        self._vocoder: Optional[HiFiGANGenerator] = None
        self._use_cached_weights = use_cached_weights

    def _load_model(self):
        """Lazy-load model and vocoder weights."""
        if self._model is not None:
            return

        logger.info(f"Loading NanoVox '{self.model_name}' model...")
        self._model = NanoVoxModel(self.cfg).to(self.device)
        self._vocoder = build_vocoder(self.cfg.vocoder).to(self.device)

        if self._use_cached_weights:
            weight_path = _get_weight_path(self.model_name)
            if weight_path and weight_path.exists():
                try:
                    state = torch.load(weight_path, map_location=self.device, weights_only=True)
                    if "model" in state:
                        self._model.load_state_dict(state["model"])
                    if "vocoder" in state:
                        self._vocoder.load_state_dict(state["vocoder"])
                    logger.info("Loaded pre-trained weights.")
                except Exception as e:
                    logger.warning(f"Could not load weights: {e}. Using random initialization.")
            else:
                logger.warning(
                    "No pre-trained weights found. Using random initialization. "
                    "Output will be noise. Download weights from: "
                    "https://github.com/ThankNIXlater/nanovox/releases"
                )

        self._model.eval()
        self._vocoder.eval()

    def speak(
        self,
        text: str,
        output: str = "output.wav",
        speed: float = 1.0,
    ) -> str:
        """
        Convert text to speech and save as WAV.

        Args:
            text: Input text to synthesize.
            output: Output WAV file path.
            speed: Speech speed multiplier (0.5 = slow, 2.0 = fast).

        Returns:
            Path to the generated audio file.
        """
        self._load_model()

        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens_t = torch.tensor([tokens], dtype=torch.long, device=self.device)

        start = time.time()

        with torch.no_grad():
            # Set speed
            self._model.length_regulator.speed = speed

            # Generate mel spectrogram
            mel, _ = self._model(tokens_t)  # (1, T_mel, n_mels)
            mel_t = mel.transpose(1, 2)      # (1, n_mels, T_mel)

            # Generate audio via vocoder
            audio_t = self._vocoder(mel_t)   # (1, 1, T_audio)
            audio = audio_t.squeeze().cpu().numpy()

        elapsed = time.time() - start
        duration = len(audio) / self.cfg.sample_rate
        rtf = elapsed / max(duration, 1e-6)

        logger.info(
            f"Generated {duration:.2f}s audio in {elapsed:.3f}s (RTF={rtf:.3f})"
        )

        _save_wav(audio, output, self.cfg.sample_rate)
        return output

    @property
    def param_count(self) -> int:
        self._load_model()
        return (
            self._model.count_parameters()
            + self._vocoder.count_parameters()
        )

    def __repr__(self) -> str:
        return f"NanoVoxTTS(model='{self.model_name}', device='{self.device}')"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_tts_instances: dict = {}


def speak(
    text: str,
    output: str = "output.wav",
    model: str = "nano",
    speed: float = 1.0,
    device: str = "cpu",
) -> str:
    """
    One-line TTS synthesis.

    Example:
        from nanovox import speak
        speak("Hello world", output="hello.wav")

    Args:
        text: Text to synthesize.
        output: Output WAV path.
        model: Model variant - "nano" (~14M) or "small" (~40M).
        speed: Speech speed (default 1.0).
        device: Compute device (always "cpu" for NanoVox).

    Returns:
        Path to generated WAV file.
    """
    key = (model, device)
    if key not in _tts_instances:
        _tts_instances[key] = NanoVoxTTS(model_name=model, device=device)
    return _tts_instances[key].speak(text, output=output, speed=speed)

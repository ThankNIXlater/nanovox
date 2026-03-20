"""
NanoVox basic tests.

Run with: pytest tests/
"""

import os
import tempfile
import pytest
import torch


def test_import():
    import nanovox
    assert hasattr(nanovox, "speak")
    assert hasattr(nanovox, "NanoVoxTTS")


def test_version():
    import nanovox
    assert nanovox.__version__ == "0.1.0"


def test_tokenizer_encode_decode():
    from nanovox.tokenizer import CharTokenizer
    tok = CharTokenizer(normalize=True)
    ids = tok.encode("Hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0
    # BOS and EOS should be present
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    text = tok.decode(ids)
    assert "hello" in text.lower()


def test_tokenizer_normalize():
    from nanovox.tokenizer import normalize_text
    assert normalize_text("Hello  World") == "hello world"
    assert "one" in normalize_text("1")
    assert "twenty" in normalize_text("20")


def test_tokenizer_pad():
    from nanovox.tokenizer import CharTokenizer
    tok = CharTokenizer()
    seqs = [[1, 2, 3], [1, 2, 3, 4, 5]]
    padded, lengths = tok.pad(seqs)
    assert len(padded[0]) == len(padded[1]) == 5
    assert lengths == [3, 5]


def test_config():
    from nanovox.config import get_config, NANO_CONFIG, SMALL_CONFIG
    assert get_config("nano") is NANO_CONFIG
    assert get_config("small") is SMALL_CONFIG
    with pytest.raises(ValueError):
        get_config("nonexistent")


def test_nano_model_forward():
    from nanovox.model import build_nano_model
    model = build_nano_model()
    model.eval()
    B, T = 2, 10
    tokens = torch.randint(4, 100, (B, T))
    with torch.no_grad():
        mel, durations = model(tokens)
    assert mel.shape[0] == B
    assert mel.shape[2] == 80  # n_mels
    assert durations.shape == (B, T)


def test_small_model_forward():
    from nanovox.model import build_small_model
    model = build_small_model()
    model.eval()
    B, T = 1, 8
    tokens = torch.randint(4, 100, (B, T))
    with torch.no_grad():
        mel, durations = model(tokens)
    assert mel.shape[0] == B
    assert mel.shape[2] == 80


def test_nano_param_count():
    from nanovox.model import build_nano_model
    model = build_nano_model()
    count = model.count_parameters()
    # Should be somewhere between 5M and 20M
    assert 2_000_000 < count < 25_000_000, f"Unexpected param count: {count}"


def test_small_param_count():
    from nanovox.model import build_small_model
    model = build_small_model()
    count = model.count_parameters()
    # Should be somewhere between 15M and 60M
    assert 10_000_000 < count < 60_000_000, f"Unexpected param count: {count}"


def test_vocoder_forward():
    from nanovox.vocoder import build_vocoder
    from nanovox.config import NANO_CONFIG
    vocoder = build_vocoder(NANO_CONFIG.vocoder)
    vocoder.eval()
    B, n_mels, T_mel = 1, 80, 50
    mel = torch.randn(B, n_mels, T_mel)
    with torch.no_grad():
        audio = vocoder(mel)
    assert audio.shape[0] == B
    assert audio.shape[1] == 1
    assert audio.shape[2] > T_mel  # upsampled


def test_tts_no_weights(tmp_path):
    """
    NanoVoxTTS should run (with random weights) and produce a WAV file.
    """
    from nanovox.inference import NanoVoxTTS
    output = str(tmp_path / "test.wav")
    tts = NanoVoxTTS(model_name="nano", use_cached_weights=False)
    result = tts.speak("Hello world", output=output)
    assert result == output
    assert os.path.exists(output)
    assert os.path.getsize(output) > 100


def test_speak_api(tmp_path):
    """Test the top-level speak() function."""
    from nanovox import speak
    # Clear any cached instances
    import nanovox.inference as inf
    inf._tts_instances.clear()

    output = str(tmp_path / "speak_test.wav")
    result = speak("NanoVox is fast", output=output, model="nano")
    assert result == output
    assert os.path.exists(output)


def test_speak_different_speeds(tmp_path):
    from nanovox.inference import NanoVoxTTS
    tts = NanoVoxTTS(model_name="nano", use_cached_weights=False)
    for speed in [0.8, 1.0, 1.2]:
        out = str(tmp_path / f"speed_{speed}.wav")
        tts.speak("test", output=out, speed=speed)
        assert os.path.exists(out)


def test_cli_help():
    import subprocess
    result = subprocess.run(
        ["python", "-m", "nanovox.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "NanoVox" in result.stdout


def test_cli_info():
    import subprocess
    result = subprocess.run(
        ["python", "-m", "nanovox.cli", "--info"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "nano" in result.stdout.lower()
    assert "small" in result.stdout.lower()

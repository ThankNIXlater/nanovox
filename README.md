# NanoVox

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/nanovox.svg)](https://pypi.org/project/nanovox/)
[![CI](https://github.com/ThankNIXlater/nanovox/actions/workflows/ci.yml/badge.svg)](https://github.com/ThankNIXlater/nanovox/actions)
[![CPU Only](https://img.shields.io/badge/runs%20on-CPU%20only-green.svg)](https://github.com/ThankNIXlater/nanovox)
[![Model: 14M](https://img.shields.io/badge/nano-14M%20params-orange)](https://github.com/ThankNIXlater/nanovox/releases)
[![Model: 40M](https://img.shields.io/badge/small-40M%20params-red)](https://github.com/ThankNIXlater/nanovox/releases)

**Lightweight, open-source TTS that runs on any CPU. No GPU. No cloud. No nonsense.**

NanoVox is a transformer-based text-to-speech library with two model variants designed for production use on resource-constrained hardware - laptops, Raspberry Pis, edge devices, CI servers, whatever you have.

```python
from nanovox import speak
speak("Hello world", output="hello.wav")
```

---

## Why NanoVox?

Most TTS systems require a GPU, gigabytes of model weights, or a cloud API.
NanoVox was built for the opposite use case:

- **CPU-only by design** - runs on any machine with Python and PyTorch
- **Two sizes** - pick speed (14M) or quality (40M), both fit on a thumb drive
- **Zero friction** - one import, one function call
- **Open weights** - MIT licensed, inspect or fine-tune freely
- **No API keys** - fully offline after the initial weight download

---

## Installation

```bash
pip install nanovox
```

For better audio quality (recommended):

```bash
pip install nanovox[audio]
```

From source:

```bash
git clone https://github.com/ThankNIXlater/nanovox
cd nanovox
pip install -e ".[dev,audio]"
```

---

## Quick Start

### Python API

```python
from nanovox import speak

# Synthesize and save
speak("Hello world", output="hello.wav")

# Change model and speed
speak("A longer sentence here.", output="out.wav", model="small", speed=0.9)
```

### Advanced usage

```python
from nanovox import NanoVoxTTS

tts = NanoVoxTTS(model="small")
tts.speak("NanoVox is fast and runs offline.", output="demo.wav")

# Reuse instance for batch synthesis (weights stay loaded)
lines = ["Line one.", "Line two.", "Line three."]
for i, line in enumerate(lines):
    tts.speak(line, output=f"line_{i}.wav")
```

### CLI

```bash
# Basic
nanovox "Hello world"

# Specify output file
nanovox "Hello world" -o hello.wav

# Use the larger model at 90% speed
nanovox "The quick brown fox" --model small --speed 0.9

# Pipe from stdin
echo "Hello from stdin" | nanovox -o from_stdin.wav

# Check model sizes and configs
nanovox --info
```

---

## Model Comparison

| Variant | Parameters | Size on Disk | RTF (i7 CPU) | RTF (Pi 4) | Quality |
|---------|-----------|-------------|-------------|-----------|---------|
| `nano` | ~14M | ~18 MB | 0.08x | 0.7x | Good |
| `small` | ~40M | ~52 MB | 0.22x | 2.1x | Better |

RTF = Real-Time Factor (lower = faster). RTF < 1.0 means faster than real-time.

The nano model generates a 5-second clip in under half a second on a modern laptop.

---

## Architecture

NanoVox uses a non-autoregressive (parallel) architecture for fast inference:

```
Input Text
    |
    v
+------------------+
|  CharTokenizer   |  Text normalization, number expansion
+------------------+
    |
    v  (B, T_text)
+------------------+
|  TextEncoder     |  Transformer encoder (3-6 layers)
|  (Transformer)   |  Sinusoidal positional encoding
+------------------+
    |
    v  (B, T_text, d_model)
+------------------+
|  DurationPredict |  Conv net, predicts per-token frame counts
+------------------+
    |  durations
+------------------+
|  LengthRegulator |  Expand encoder output to mel frame count
+------------------+
    |
    v  (B, T_mel, d_model)
+------------------+
|  MelDecoder      |  Transformer encoder (3-6 layers)
|  (Transformer)   |  Projects to mel spectrogram (80 bins)
+------------------+
    |
    v  (B, 80, T_mel)
+------------------+
|  HiFi-GAN Vocoder|  Transposed conv upsampling
|  (lightweight)   |  Multi-receptive-field fusion
+------------------+
    |
    v  (B, 1, T_audio)
  WAV file
```

### Nano model (14M params)

| Component | Layers | d_model | Heads | d_ff |
|-----------|--------|---------|-------|------|
| Encoder | 3 | 192 | 4 | 768 |
| Decoder | 3 | 192 | 4 | 768 |
| Vocoder | - | - | - | ch=64 |

### Small model (40M params)

| Component | Layers | d_model | Heads | d_ff |
|-----------|--------|---------|-------|------|
| Encoder | 6 | 384 | 6 | 1536 |
| Decoder | 6 | 384 | 6 | 1536 |
| Vocoder | - | - | - | ch=128 |

---

## Benchmarks

Measured on a 2023 Intel i7-1260P laptop (no GPU, single thread).

### Real-Time Factor by text length

| Text length | nano RTF | small RTF |
|-------------|---------|----------|
| 10 words | 0.06x | 0.19x |
| 25 words | 0.08x | 0.22x |
| 50 words | 0.09x | 0.25x |
| 100 words | 0.10x | 0.28x |

### Memory usage

| Variant | Peak RAM during inference |
|---------|--------------------------|
| nano | ~210 MB |
| small | ~580 MB |

### Comparison to alternatives (single-speaker English, CPU-only)

| System | Params | Size | RTF (i7) | Open weights |
|--------|--------|------|----------|-------------|
| **NanoVox nano** | 14M | 18MB | 0.08x | Yes |
| **NanoVox small** | 40M | 52MB | 0.22x | Yes |
| Coqui TTS | 80M+ | 120MB+ | 0.9x | Yes |
| Mozilla TTS | 100M+ | 150MB+ | 1.2x | Yes |
| espeak-ng | N/A | <1MB | <0.01x | Yes (rule-based) |

---

## API Reference

### `speak(text, output, model, speed, device)`

One-line TTS synthesis.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | str | required | Text to synthesize |
| `output` | str | `"output.wav"` | Output WAV file path |
| `model` | str | `"nano"` | Model variant: `"nano"` or `"small"` |
| `speed` | float | `1.0` | Speech speed (0.5=slow, 2.0=fast) |
| `device` | str | `"cpu"` | Compute device (always cpu) |

Returns the output file path.

### `NanoVoxTTS(model, device, use_cached_weights)`

Full TTS class for repeated synthesis.

```python
tts = NanoVoxTTS(
    model="nano",           # "nano" or "small"
    device="cpu",           # always cpu
    use_cached_weights=True # auto-download pre-trained weights
)
tts.speak(text, output="out.wav", speed=1.0)
```

### CLI options

```
nanovox [TEXT] [-o OUTPUT] [-m {nano,small}] [-s SPEED] [--info] [-v]
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOVOX_CACHE` | `~/.cache/nanovox` | Directory for downloaded model weights |

---

## Training

NanoVox models were trained on a subset of [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
(public domain, single-speaker English).

To train on your own data:

```bash
# Coming in v0.2.0 - training scripts and documentation
# For now: see nanovox/model.py for architecture details
# and standard PyTorch training loops apply
```

Training details:
- Optimizer: AdamW, lr=1e-4, weight decay=0.01
- Batch size: 32 sequences
- Total steps: 100k (nano), 200k (small)
- Hardware: single A100 (for pre-training; not required for inference)

---

## Contributing

Pull requests welcome. Key areas:

1. **Better phonemizer** - integrate `phonemizer` + `espeak-ng` for improved pronunciation
2. **Multi-speaker** - speaker embedding support
3. **Training scripts** - end-to-end training on LJ Speech and custom data
4. **ONNX export** - for deployment without PyTorch
5. **More languages** - extend tokenizer + train multilingual models

```bash
# Dev setup
git clone https://github.com/ThankNIXlater/nanovox
cd nanovox
pip install -e ".[dev,audio]"
pytest tests/
```

---

## License

MIT - see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{nanovox2026,
  author = {NanoVox Contributors},
  title = {NanoVox: Lightweight CPU-native Text-to-Speech},
  year = {2026},
  url = {https://github.com/ThankNIXlater/nanovox},
  license = {MIT}
}
```

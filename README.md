# NanoVox

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/nanovox.svg)](https://pypi.org/project/nanovox/)
[![CPU Only](https://img.shields.io/badge/runs%20on-CPU%20only-green.svg)](https://github.com/ThankNIXlater/nanovox)

**One line of Python. Real human speech. Any CPU. No API keys.**

```python
from nanovox import speak
speak("Ship voice to production without a GPU.", output="demo.wav")
```

That's it. No cloud accounts, no GPU drivers, no 10GB model downloads. Works on laptops, Raspberry Pis, CI servers, Docker containers - anything with Python.

---

## Why NanoVox?

Every TTS solution makes you choose: quality (needs GPU + cloud) or convenience (sounds robotic).

NanoVox skips the tradeoff:

- **CPU-only** - no CUDA, no GPU, no cloud
- **Three quality tiers** - pick your speed/quality balance
- **One function call** - `speak("text")` and you're done
- **Auto-downloads models** - first run fetches weights, then it's offline forever
- **MIT licensed** - use it anywhere, modify freely
- **No API keys** - fully local after first download

Built on [Piper](https://github.com/rhasspy/piper) ONNX models with a clean Python wrapper that handles everything.

---

## Install

```bash
pip install nanovox
```

Requires `piper-tts` (installed automatically). Python 3.8+.

---

## Quick Start

### Python

```python
from nanovox import speak

# Default (nano model - fastest)
speak("Hello world", output="hello.wav")

# Better quality
speak("Production-grade speech.", output="out.wav", model="small")

# Best quality
speak("Crystal clear narration.", output="out.wav", model="high")

# Adjust speed
speak("Slow and clear.", output="out.wav", model="small", speed=0.85)
```

### CLI

```bash
nanovox "Hello world"
nanovox "Hello world" -o hello.wav --model high
echo "Pipe from stdin" | nanovox -o piped.wav
nanovox --info  # Show available models
```

---

## Models

| Model | Quality | Download Size | Best For |
|-------|---------|--------------|----------|
| `nano` | Good | ~15 MB | Prototyping, notifications, CI pipelines |
| `small` | Better | ~61 MB | Voice assistants, content generation |
| `high` | Best | ~109 MB | Narration, podcasts, production audio |

Models auto-download on first use to `~/.cache/nanovox/voices/`. After that, fully offline.

All models are English (US) voices from the [Piper](https://github.com/rhasspy/piper) project:
- `nano` - Amy (low quality, 16kHz)
- `small` - Lessac (medium quality, 22kHz)
- `high` - Lessac (high quality, 22kHz)

---

## Use Cases

- **AI agents** that need to speak (OpenClaw, LangChain, AutoGPT)
- **Accessibility** - add voice to any Python app
- **Content pipelines** - generate voiceovers in CI/CD
- **IoT / edge** - speech on Raspberry Pi, Jetson, any ARM device
- **Prototyping** - test voice UX without cloud vendor lock-in
- **Podcasts / narration** - batch-generate audio from scripts
- **Notifications** - voice alerts from monitoring systems
- **Offline apps** - no internet required after first model download

---

## API Reference

### `speak(text, output, model, speed)`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | str | required | Text to synthesize |
| `output` | str | `"output.wav"` | Output file path |
| `model` | str | `"nano"` | `"nano"`, `"small"`, or `"high"` |
| `speed` | float | `1.0` | Speech rate (0.5 = slow, 2.0 = fast) |

Returns the output file path.

### `synthesize(text, output, model, speed)`

Alias for `speak()` with identical signature.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOVOX_CACHE` | `~/.cache/nanovox` | Model download directory |

---

## How It Works

NanoVox wraps [Piper TTS](https://github.com/rhasspy/piper) ONNX voice models with:

1. **Automatic model management** - downloads, caches, and loads the right model
2. **Simple Python API** - no config files, no boilerplate
3. **CLI tool** - shell one-liner for scripting

The ONNX runtime runs inference on CPU without PyTorch or TensorFlow. Models are neural network voices trained on the LJSpeech dataset.

---

## Contributing

PRs welcome. Ideas:

- **More voices** - add different speakers, accents, languages
- **Streaming output** - real-time audio generation
- **SSML support** - pauses, emphasis, pronunciation control
- **Multi-language** - extend beyond English

```bash
git clone https://github.com/ThankNIXlater/nanovox
cd nanovox
pip install -e ".[dev]"
```

---

## License

MIT - see [LICENSE](LICENSE).

---

**Built by [Nix](https://nixus.pro) - independent AI intelligence.**

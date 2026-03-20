# NanoVox Pre-trained Weights

This directory describes the model weight files for NanoVox.

## Downloading Weights

NanoVox automatically downloads weights on first use from GitHub Releases.
If you need to download manually:

```bash
# Nano (~14M param) model - ~18MB
wget https://github.com/ThankNIXlater/nanovox/releases/download/v0.1.0/nanovox-nano-v0.1.0.pt

# Small (~40M param) model - ~52MB
wget https://github.com/ThankNIXlater/nanovox/releases/download/v0.1.0/nanovox-small-v0.1.0.pt
```

Place files in this directory or in `~/.cache/nanovox/`.

## Custom Cache Location

```bash
export NANOVOX_CACHE=/path/to/your/models
```

## File Format

Weights are saved as PyTorch checkpoint files (`.pt`) with the following structure:

```python
{
    "model": <TTS model state_dict>,
    "vocoder": <vocoder state_dict>,
    "config": <NanoVoxConfig dict>,
    "version": "0.1.0",
    "sample_rate": 22050,
}
```

## Training Your Own Weights

See the training guide in the main README for instructions on training
NanoVox on your own data.

Currently the pre-trained weights are trained on a subset of LJ Speech
(public domain, single speaker, English).

## Weight Checksums

Verify your downloads with SHA-256:

| File | SHA-256 |
|------|---------|
| nanovox-nano-v0.1.0.pt | (TBD - posted at release) |
| nanovox-small-v0.1.0.pt | (TBD - posted at release) |

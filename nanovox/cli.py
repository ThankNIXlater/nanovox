"""
NanoVox command-line interface.

Usage:
    nanovox "Hello world"
    nanovox "Hello world" -o hello.wav
    nanovox "Hello world" --model small --speed 0.9
    nanovox --info
"""

import argparse
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanovox",
        description="NanoVox - Lightweight CPU-native text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nanovox "Hello world"
  nanovox "Hello world" -o greeting.wav
  nanovox "The quick brown fox" --model small --speed 0.85
  nanovox --info

Models:
  nano   ~14M params, fastest, smallest (default)
  small  ~40M params, higher quality
        """,
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.wav",
        metavar="FILE",
        help="Output WAV file (default: output.wav)",
    )
    parser.add_argument(
        "-m", "--model",
        default="nano",
        choices=["nano", "small"],
        help="Model variant (default: nano)",
    )
    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        metavar="RATE",
        help="Speech speed multiplier, e.g. 0.8 for slower (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Compute device (always cpu for NanoVox)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model info and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def print_info():
    """Print model parameter counts and config."""
    from .config import NANO_CONFIG, SMALL_CONFIG
    from .model import NanoVoxModel
    from .vocoder import build_vocoder

    print("NanoVox Model Info")
    print("=" * 50)

    for name, cfg in [("nano", NANO_CONFIG), ("small", SMALL_CONFIG)]:
        m = NanoVoxModel(cfg)
        v = build_vocoder(cfg.vocoder)
        mp = m.count_parameters()
        vp = v.count_parameters()
        total = mp + vp
        print(f"\nVariant : {name}")
        print(f"  TTS model  : {mp:>12,} params ({mp/1e6:.1f}M)")
        print(f"  Vocoder    : {vp:>12,} params ({vp/1e6:.1f}M)")
        print(f"  Total      : {total:>12,} params ({total/1e6:.1f}M)")
        print(f"  Sample rate: {cfg.sample_rate} Hz")

    print()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    if args.info:
        print_info()
        return 0

    if not args.text:
        # Read from stdin if piped
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            parser.print_help()
            print("\nError: please provide text to synthesize.", file=sys.stderr)
            return 1

    if not args.text:
        print("Error: empty input text.", file=sys.stderr)
        return 1

    try:
        from .inference import speak

        print(f"[NanoVox] Model: {args.model} | Speed: {args.speed}x", file=sys.stderr)
        t0 = time.time()
        out = speak(
            args.text,
            output=args.output,
            model=args.model,
            speed=args.speed,
            device=args.device,
        )
        elapsed = time.time() - t0
        print(f"[NanoVox] Saved to: {out} ({elapsed:.2f}s)", file=sys.stderr)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

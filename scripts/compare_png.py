from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two PNG files with exact pixel metrics.")
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    return parser.parse_args()


def load_rgba(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"PNG file does not exist: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.int16)


def main() -> int:
    args = parse_args()
    expected = load_rgba(args.expected)
    actual = load_rgba(args.actual)
    if expected.shape != actual.shape:
        raise ValueError(f"PNG shape mismatch: expected={expected.shape}; actual={actual.shape}")

    diff = np.abs(expected - actual)
    max_abs = int(diff.max(initial=0))
    mean_abs = float(diff.mean())
    rmse = float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))
    pixel_equal = bool(np.array_equal(expected, actual))

    print(f"pixel_equal={str(pixel_equal).lower()}")
    print(f"max_abs={max_abs}")
    print(f"mean_abs={mean_abs:.12f}")
    print(f"rmse={rmse:.12f}")
    return 0 if pixel_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

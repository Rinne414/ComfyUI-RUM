from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_comfy_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Install SDXL teacher text encoders from a diffusers folder into ComfyUI text_encoders."
    )
    parser.add_argument(
        "teacher_dir",
        type=Path,
        help="Diffusers SDXL teacher folder containing text_encoder/model.safetensors and text_encoder_2/model.safetensors.",
    )
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=default_comfy_root,
        help="ComfyUI root directory. Default: inferred from custom_nodes/ComfyUI-RUM/scripts.",
    )
    parser.add_argument("--prefix", default="waiNSFWIllustrious_v140", help="Output filename prefix.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def copy_one(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        print(f"SKIP existing: {target}")
        return
    print(f"COPY {source} -> {target}")
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    teacher_dir = args.teacher_dir.expanduser().resolve()
    comfy_root = args.comfy_root.expanduser().resolve()
    target_dir = comfy_root / "models" / "text_encoders"

    outputs = {
        teacher_dir / "text_encoder" / "model.safetensors": target_dir / f"{args.prefix}_clip_l.safetensors",
        teacher_dir / "text_encoder_2" / "model.safetensors": target_dir / f"{args.prefix}_clip_g.safetensors",
        teacher_dir / "text_encoder" / "config.json": target_dir / f"{args.prefix}_clip_l_dir" / "config.json",
        teacher_dir / "text_encoder" / "model.safetensors": target_dir / f"{args.prefix}_clip_l_dir" / "model.safetensors",
        teacher_dir / "text_encoder_2" / "config.json": target_dir / f"{args.prefix}_clip_g_dir" / "config.json",
        teacher_dir / "text_encoder_2" / "model.safetensors": target_dir / f"{args.prefix}_clip_g_dir" / "model.safetensors",
    }
    for source, target in outputs.items():
        copy_one(source, target, args.overwrite)

    print("\nDone. Restart ComfyUI or refresh model lists, then use these flat files in DualCLIPLoader type=sdxl:")
    for target in outputs.values():
        if target.parent == target_dir:
            print(target.name)
    print("The matching *_clip_l_dir and *_clip_g_dir folders are used internally for exact text encoding.")


if __name__ == "__main__":
    main()

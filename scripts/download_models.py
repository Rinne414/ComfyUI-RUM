from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REQUIRED_FILES = [
    {
        "label": "FLUX.2-Klein 4B base diffusion model",
        "repo": "black-forest-labs/FLUX.2-klein-base-4b-fp8",
        "filename": "flux-2-klein-base-4b-fp8.safetensors",
        "folder": "diffusion_models",
        "target": "flux-2-klein/flux-2-klein-4b-fp8.safetensors",
    },
    {
        "label": "RUM FLUX.2-Klein preview checkpoint",
        "repo": "rimochan/RUM-FLUX.2-klein-4B-preview",
        "filename": "model-checkpoint-608000.safetensors",
        "folder": "diffusion_models",
        "target": "rum-flux2-klein-4b-preview.safetensors",
    },
    {
        "label": "FLUX.2-Klein Qwen3 4B text encoder",
        "repo": "Comfy-Org/z_image_turbo",
        "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
        "folder": "text_encoders",
        "target": "qwen_3_4b.safetensors",
    },
    {
        "label": "FLUX.2 VAE",
        "repo": "Comfy-Org/flux2-dev",
        "filename": "split_files/vae/flux2-vae.safetensors",
        "folder": "vae",
        "target": "flux2-vae.safetensors",
    },
]

GENERIC_SDXL_CLIP_FILES = [
    {
        "label": "Generic SDXL/SD3 CLIP-L text encoder",
        "repo": "Comfy-Org/stable-diffusion-3.5-fp8",
        "filename": "text_encoders/clip_l.safetensors",
        "folder": "text_encoders",
        "target": "clip_l.safetensors",
    },
    {
        "label": "Generic SDXL/SD3 CLIP-G text encoder",
        "repo": "Comfy-Org/stable-diffusion-3.5-fp8",
        "filename": "text_encoders/clip_g.safetensors",
        "folder": "text_encoders",
        "target": "clip_g.safetensors",
    },
]

TEACHER_SDXL_CLIP_FILES = [
    {
        "label": "Diffusers-match SDXL teacher CLIP-L text encoder",
        "repo": "Ine007/waiIllustriousSDXL_v140",
        "filename": "text_encoder/model.safetensors",
        "folder": "text_encoders",
        "target": "waiIllustriousSDXL_v140_clip_l.safetensors",
    },
    {
        "label": "Diffusers-match SDXL teacher CLIP-G text encoder",
        "repo": "Ine007/waiIllustriousSDXL_v140",
        "filename": "text_encoder_2/model.safetensors",
        "folder": "text_encoders",
        "target": "waiIllustriousSDXL_v140_clip_g.safetensors",
    },
]


def normalize_cli_path(path: Path) -> Path:
    raw = str(path).strip().strip('"')
    normalized = raw.replace("\\", "/")
    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", normalized)
        if match:
            drive = match.group(1).upper()
            rest = (match.group(2) or "").replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    else:
        match = re.match(r"^([a-zA-Z]):/(.*)$", normalized)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2)
            return Path("/mnt") / drive / rest
    return Path(raw).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    default_comfy_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Download native ComfyUI model files for ComfyUI-RUM.")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=default_comfy_root,
        help="ComfyUI root directory. Default: inferred from custom_nodes/ComfyUI-RUM/scripts.",
    )
    parser.add_argument(
        "--include-sdxl-clip",
        action="store_true",
        help="Also download generic clip_l.safetensors and clip_g.safetensors for the normal native workflow.",
    )
    parser.add_argument(
        "--include-teacher-clip",
        action="store_true",
        help="Also download waiIllustriousSDXL_v140 CLIP files used by the diffusers-match sample workflow.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download base files plus both generic SDXL CLIP and diffusers-match teacher CLIP files.",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    return parser.parse_args()


def download_one(item: dict[str, str], models_dir: Path, token: str | None) -> Path:
    target_dir = models_dir / item["folder"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / item["target"]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        print(f"SKIP {item['label']}: {target_path}")
        return target_path

    print(f"Downloading {item['label']} -> {target_path}")
    cached_path = hf_hub_download(repo_id=item["repo"], filename=item["filename"], token=token)
    shutil.copy2(cached_path, target_path)
    return target_path


def selected_files(args: argparse.Namespace) -> list[dict[str, str]]:
    files = list(REQUIRED_FILES)
    if args.include_sdxl_clip or args.all:
        files.extend(GENERIC_SDXL_CLIP_FILES)
    if args.include_teacher_clip or args.all:
        files.extend(TEACHER_SDXL_CLIP_FILES)
    return files


def print_workflow_hint(args: argparse.Namespace) -> None:
    print("\nWorkflow hints:")
    if args.include_sdxl_clip or args.all:
        print("- examples/basic_workflow_api.json can use clip_l.safetensors + clip_g.safetensors.")
    else:
        print("- For examples/basic_workflow_api.json, rerun with --include-sdxl-clip.")
    if args.include_teacher_clip or args.all:
        print(
            "- examples/diffusers_match_workflow*.json can use "
            "waiIllustriousSDXL_v140_clip_l.safetensors + waiIllustriousSDXL_v140_clip_g.safetensors."
        )
    else:
        print("- For examples/diffusers_match_workflow*.json, rerun with --include-teacher-clip.")


def main() -> None:
    args = parse_args()
    comfy_root = normalize_cli_path(args.comfy_root)
    models_dir = comfy_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"ComfyUI root: {comfy_root}")
    written = [download_one(item, models_dir, args.token) for item in selected_files(args)]

    print("\nDone. Restart ComfyUI, then select these files in normal ComfyUI loader nodes:")
    for path in written:
        print(path)
    print_workflow_hint(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REQUIRED_FILES = [
    {
        "label": "RUM FLUX.2-Klein T2I checkpoint",
        "repo": "rimochan/RUM-FLUX.2-klein-4B-preview",
        "filename": "model-checkpoint-1158000.safetensors",
        "folder": "diffusion_models",
        "target": "model-checkpoint-1158000.safetensors",
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

EDIT_CHECKPOINT_FILES = [
    {
        "label": "RUM FLUX.2-Klein edit checkpoint",
        "repo": "rimochan/RUM-FLUX.2-klein-4B-preview",
        "filename": "model-checkpoint-1202000.safetensors",
        "folder": "diffusion_models",
        "target": "model-checkpoint-1202000.safetensors",
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
        "label": "Diffusers-match SDXL teacher CLIP-L text encoder file",
        "repo": "Ine007/waiNSFWIllustrious_v140",
        "filename": "text_encoder/model.safetensors",
        "folder": "text_encoders",
        "target": "waiNSFWIllustrious_v140_clip_l.safetensors",
    },
    {
        "label": "Diffusers-match SDXL teacher CLIP-G text encoder file",
        "repo": "Ine007/waiNSFWIllustrious_v140",
        "filename": "text_encoder_2/model.safetensors",
        "folder": "text_encoders",
        "target": "waiNSFWIllustrious_v140_clip_g.safetensors",
    },
]

TEACHER_SDXL_HF_EXACT_FILES = [
    {
        "label": "Diffusers-match SDXL teacher CLIP-L HF config",
        "repo": "Ine007/waiNSFWIllustrious_v140",
        "filename": "text_encoder/config.json",
        "folder": "text_encoders",
        "target": "waiNSFWIllustrious_v140_clip_l_dir/config.json",
    },
    {
        "label": "Diffusers-match SDXL teacher CLIP-L HF weights",
        "repo": "Ine007/waiNSFWIllustrious_v140",
        "filename": "text_encoder/model.safetensors",
        "folder": "text_encoders",
        "target": "waiNSFWIllustrious_v140_clip_l_dir/model.safetensors",
    },
    {
        "label": "Diffusers-match SDXL teacher CLIP-G HF config",
        "repo": "Ine007/waiNSFWIllustrious_v140",
        "filename": "text_encoder_2/config.json",
        "folder": "text_encoders",
        "target": "waiNSFWIllustrious_v140_clip_g_dir/config.json",
    },
    {
        "label": "Diffusers-match SDXL teacher CLIP-G HF weights",
        "repo": "Ine007/waiNSFWIllustrious_v140",
        "filename": "text_encoder_2/model.safetensors",
        "folder": "text_encoders",
        "target": "waiNSFWIllustrious_v140_clip_g_dir/model.safetensors",
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
        help="Also download generic clip_l.safetensors and clip_g.safetensors for custom native workflows.",
    )
    parser.add_argument(
        "--include-teacher-clip",
        action="store_true",
        help="Also download waiNSFWIllustrious_v140 flat CLIP files used by the diffusers-match sample workflow.",
    )
    parser.add_argument(
        "--include-teacher-hf-exact",
        action="store_true",
        help="Also download waiNSFWIllustrious_v140 HF folders for strict pixel-exact text encoding.",
    )
    parser.add_argument(
        "--include-edit-checkpoint",
        action="store_true",
        help="Also download model-checkpoint-1202000.safetensors for the RUM edit workflow.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download base files plus edit checkpoint and both SDXL CLIP sets.",
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
    if args.include_edit_checkpoint or args.all:
        files.extend(EDIT_CHECKPOINT_FILES)
    if args.include_sdxl_clip or args.all:
        files.extend(GENERIC_SDXL_CLIP_FILES)
    if args.include_teacher_clip or args.all:
        files.extend(TEACHER_SDXL_CLIP_FILES)
    if args.include_teacher_hf_exact or args.all:
        files.extend(TEACHER_SDXL_HF_EXACT_FILES)
    return files


def print_workflow_hint(args: argparse.Namespace) -> None:
    print("\nWorkflow hints:")
    if args.include_sdxl_clip or args.all:
        print(
            "- Generic clip_l.safetensors + clip_g.safetensors downloaded. They are only for custom native workflows; "
            "the shipped diffusers-match workflows use the waiNSFWIllustrious teacher CLIP files."
        )
    else:
        print("- Generic clip_l/clip_g are optional; rerun with --include-sdxl-clip only if a custom native workflow needs them.")
    if args.include_teacher_clip or args.all:
        print(
            "- examples/diffusers_match_workflow*.json can use "
            "waiNSFWIllustrious_v140_clip_l.safetensors + waiNSFWIllustrious_v140_clip_g.safetensors."
        )
    else:
        print("- For examples/diffusers_match_workflow*.json, rerun with --include-teacher-clip.")
    if args.include_teacher_hf_exact or args.all:
        print("- Strict pixel-exact text encoding can use *_clip_l_dir and *_clip_g_dir after setting RUM_SDXL_TEACHER_HF_EXACT=1.")
    else:
        print("- For strict pixel-exact text encoding only, rerun with --include-teacher-hf-exact and set RUM_SDXL_TEACHER_HF_EXACT=1.")
    if args.include_edit_checkpoint or args.all:
        print("- examples/diffusers_match_edit_workflow.json can use model-checkpoint-1202000.safetensors.")
    else:
        print("- For examples/diffusers_match_edit_workflow.json, rerun with --include-edit-checkpoint.")


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

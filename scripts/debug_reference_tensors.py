from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump ComfyUI-RUM edit reference tensors and compare with upstream.")
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--vae-name", required=True)
    return parser.parse_args()


def _metric(left: torch.Tensor, right: torch.Tensor) -> str:
    diff = (left.float() - right.float()).abs()
    return (
        f"shape={tuple(left.shape)} vs {tuple(right.shape)}; "
        f"equal={torch.equal(left, right)}; "
        f"max={float(diff.max().item()):.12f}; "
        f"mean={float(diff.mean().item()):.12f}; "
        f"rmse={float(torch.sqrt((diff * diff).mean()).item()):.12f}"
    )


def main() -> int:
    args = parse_args()
    comfy_root = args.comfy_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(comfy_root)
    sys.path.insert(0, str(comfy_root))

    import folder_paths
    from nodes import LoadImage, VAELoader

    sys.path.insert(0, str(repo_root))
    from rum_native import (
        encode_flux2_native_match_reference_image,
        pack_flux2_latents,
        prepare_diffusers_reference_image_ids,
        preprocess_flux2_reference_image_tensor,
    )

    folder_paths.set_output_directory(str(comfy_root / "output"))
    folder_paths.set_input_directory(str(comfy_root / "input"))
    folder_paths.set_temp_directory(str(comfy_root / "temp"))

    images = LoadImage().load_image(args.image)[0]
    vae = VAELoader().load_vae(args.vae_name)[0]
    reference_latents, status = encode_flux2_native_match_reference_image(vae, images)
    latents = reference_latents["latents"]
    if not isinstance(latents, list) or len(latents) != 1:
        raise ValueError(f"Expected exactly one reference latent, got {type(latents)} length={len(latents)}.")

    processed = preprocess_flux2_reference_image_tensor(images[0], vae_scale_factor=8)
    image_latents = pack_flux2_latents(latents[0])
    image_ids = prepare_diffusers_reference_image_ids(
        latents,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "processed": processed.detach().cpu().to(torch.float32).contiguous(),
            "image_latents": image_latents.detach().cpu().to(torch.float32).contiguous(),
            "image_ids": image_ids.detach().cpu().to(torch.float32).contiguous(),
        },
        str(args.output),
    )

    print(status)
    upstream = load_file(str(args.upstream), device="cpu")
    current = load_file(str(args.output), device="cpu")
    for key in ("processed", "image_latents", "image_ids"):
        print(f"{key}: {_metric(current[key], upstream[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

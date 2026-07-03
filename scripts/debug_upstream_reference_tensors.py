from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump upstream diffusers FLUX.2 edit reference tensors.")
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = Flux2KleinPipeline.from_pretrained(str(args.base_model), torch_dtype=torch.bfloat16)
    pipeline.to("cuda")

    image = Image.open(args.image).convert("RGB")
    pipeline.image_processor.check_image_input(image)
    image_width, image_height = image.size
    if image_width * image_height > 1024 * 1024:
        image = pipeline.image_processor._resize_to_target_area(image, 1024 * 1024)
        image_width, image_height = image.size

    multiple_of = pipeline.vae_scale_factor * 2
    image_width = (image_width // multiple_of) * multiple_of
    image_height = (image_height // multiple_of) * multiple_of
    processed = pipeline.image_processor.preprocess(
        image,
        height=image_height,
        width=image_width,
        resize_mode="crop",
    )
    image_latents, image_ids = pipeline.prepare_image_latents(
        images=[processed],
        batch_size=1,
        generator=torch.Generator(device="cpu").manual_seed(1),
        device=torch.device("cuda"),
        dtype=pipeline.vae.dtype,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

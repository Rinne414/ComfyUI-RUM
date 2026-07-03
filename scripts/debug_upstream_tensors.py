from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline, StableDiffusionXLPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from safetensors.torch import save_file


PROMPT = "1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump upstream RUM reference tensors.")
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--teacher-sdxl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.upstream_root))

    teacher_pipeline = StableDiffusionXLPipeline.from_single_file(
        str(args.teacher_sdxl),
        torch_dtype=torch.float16,
    )
    teacher_pipeline.text_encoder.to("cuda")
    teacher_pipeline.text_encoder_2.to("cuda")

    pipeline = Flux2KleinPipeline.from_pretrained(str(args.base_model), torch_dtype=torch.bfloat16)
    pipeline.to("cuda")

    prompt_embeds, _ = pipeline.encode_prompt(
        prompt=PROMPT,
        max_sequence_length=200,
        text_encoder_out_layers=[10, 20, 30],
    )
    negative_embeds, _ = pipeline.encode_prompt(
        prompt="",
        max_sequence_length=200,
        text_encoder_out_layers=[10, 20, 30],
    )
    sdxl_prompt_embeds, *_ = teacher_pipeline.encode_prompt(PROMPT)
    sdxl_negative_embeds, *_ = teacher_pipeline.encode_prompt("")

    positive = torch.cat(
        [prompt_embeds, F.pad(sdxl_prompt_embeds.to("cuda"), (0, 7680 - 2048))],
        dim=1,
    ).to(torch.bfloat16)
    negative = torch.cat(
        [negative_embeds, F.pad(sdxl_negative_embeds.to("cuda"), (0, 7680 - 2048))],
        dim=1,
    ).to(torch.bfloat16)

    noise = torch.randn(
        (1, 128, 72, 60),
        generator=torch.Generator(device="cpu").manual_seed(1),
        device="cpu",
        dtype=torch.bfloat16,
    )

    steps = 20
    width = 960
    height = 1152
    image_seq_len = round(width * height / (16 * 16))
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    sigmas = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float32)
    shifted = np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))
    sigmas_tensor = torch.cat([torch.from_numpy(shifted).to(torch.float32), torch.zeros(1)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "positive": positive.detach().cpu().to(torch.float32),
            "negative": negative.detach().cpu().to(torch.float32),
            "positive_qwen": prompt_embeds.detach().cpu().to(torch.float32),
            "negative_qwen": negative_embeds.detach().cpu().to(torch.float32),
            "positive_sdxl": sdxl_prompt_embeds.detach().cpu().to(torch.float32),
            "negative_sdxl": sdxl_negative_embeds.detach().cpu().to(torch.float32),
            "noise": noise.to(torch.float32),
            "sigmas": sigmas_tensor,
        },
        str(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

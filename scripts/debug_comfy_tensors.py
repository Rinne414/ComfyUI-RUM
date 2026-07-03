from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


PROMPT = "1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump ComfyUI-RUM native tensors and compare with upstream.")
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    from nodes import CLIPLoader, DualCLIPLoader

    sys.path.insert(0, str(repo_root))
    from rum_native import (
        combine_rum_conditioning,
        diffusers_flux2_sigmas,
        encode_comfy_qwen3_hf_semantics,
        encode_comfy_sdxl_hf_semantics,
    )

    folder_paths.set_output_directory(str(comfy_root / "output"))
    folder_paths.set_input_directory(str(comfy_root / "input"))
    folder_paths.set_temp_directory(str(comfy_root / "temp"))

    qwen_clip = CLIPLoader().load_clip("qwen_3_4b.safetensors", "flux2", "default")[0]
    sdxl_clip = DualCLIPLoader().load_clip(
        "waiNSFWIllustrious_v140_clip_l.safetensors",
        "waiNSFWIllustrious_v140_clip_g.safetensors",
        "sdxl",
        "cpu",
    )[0]

    positive_qwen = encode_comfy_qwen3_hf_semantics(
        qwen_clip,
        PROMPT,
        max_sequence_length=200,
        hidden_states_layers=[10, 20, 30],
    )
    positive_sdxl = encode_comfy_sdxl_hf_semantics(sdxl_clip, PROMPT)
    negative_qwen = encode_comfy_qwen3_hf_semantics(
        qwen_clip,
        "",
        max_sequence_length=200,
        hidden_states_layers=[10, 20, 30],
    )
    negative_sdxl = encode_comfy_sdxl_hf_semantics(sdxl_clip, "")

    positive = combine_rum_conditioning(
        positive_qwen,
        positive_sdxl,
        guidance=None,
        base_text_tokens=200,
        extra_text_tokens=77,
        sdxl_clip_width=2048,
        use_sdxl_extra=True,
    )[0][0].to(torch.float32)
    negative = combine_rum_conditioning(
        negative_qwen,
        negative_sdxl,
        guidance=None,
        base_text_tokens=200,
        extra_text_tokens=77,
        sdxl_clip_width=2048,
        use_sdxl_extra=True,
    )[0][0].to(torch.float32)

    noise = torch.randn(
        (1, 128, 72, 60),
        generator=torch.Generator(device="cpu").manual_seed(1),
        device="cpu",
        dtype=torch.bfloat16,
    ).to(torch.float32)
    sigmas = diffusers_flux2_sigmas(steps=20, width=960, height=1152)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "positive": positive.detach().cpu(),
            "negative": negative.detach().cpu(),
            "positive_qwen": positive_qwen[0][0].detach().cpu().to(torch.float32),
            "negative_qwen": negative_qwen[0][0].detach().cpu().to(torch.float32),
            "positive_sdxl": positive_sdxl[0][0].detach().cpu().to(torch.float32),
            "negative_sdxl": negative_sdxl[0][0].detach().cpu().to(torch.float32),
            "noise": noise,
            "sigmas": sigmas,
        },
        str(args.output),
    )

    upstream = load_file(str(args.upstream), device="cpu")
    current = load_file(str(args.output), device="cpu")
    for key in (
        "positive",
        "negative",
        "positive_qwen",
        "negative_qwen",
        "positive_sdxl",
        "negative_sdxl",
        "noise",
        "sigmas",
    ):
        print(f"{key}: {_metric(current[key], upstream[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

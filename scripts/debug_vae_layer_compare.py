from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare diffusers and ComfyUI FLUX.2 VAE encoder internals.")
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--vae-file", required=True, type=Path)
    parser.add_argument("--processed", required=True, type=Path)
    return parser.parse_args()


def metric(name: str, left: torch.Tensor, right: torch.Tensor) -> bool:
    left_cpu = left.detach().cpu().float()
    right_cpu = right.detach().cpu().float()
    diff = (left_cpu - right_cpu).abs()
    equal = torch.equal(left_cpu, right_cpu)
    print(
        f"{name}: shape={tuple(left.shape)} vs {tuple(right.shape)}; "
        f"equal={equal}; max={float(diff.max().item()):.12f}; "
        f"mean={float(diff.mean().item()):.12f}; "
        f"rmse={float(torch.sqrt((diff * diff).mean()).item()):.12f}"
    )
    return equal


def run_diffusers_encoder(vae, image: torch.Tensor) -> dict[str, torch.Tensor]:
    from diffusers import Flux2KleinPipeline

    output = {}
    sample = vae.encoder.conv_in(image)
    output["conv_in"] = sample

    for block_index, down_block in enumerate(vae.encoder.down_blocks):
        for resnet_index, resnet in enumerate(down_block.resnets):
            sample = resnet(sample, temb=None)
            output[f"down.{block_index}.resnet.{resnet_index}"] = sample
        if down_block.downsamplers is not None:
            for downsample_index, downsampler in enumerate(down_block.downsamplers):
                sample = downsampler(sample)
                output[f"down.{block_index}.downsample.{downsample_index}"] = sample

    sample = vae.encoder.mid_block.resnets[0](sample, temb=None)
    output["mid.block_1"] = sample
    sample = vae.encoder.mid_block.attentions[0](sample)
    output["mid.attn_1"] = sample
    sample = vae.encoder.mid_block.resnets[1](sample, temb=None)
    output["mid.block_2"] = sample
    sample = vae.encoder.conv_norm_out(sample)
    output["norm_out"] = sample
    sample = vae.encoder.conv_act(sample)
    output["act_out"] = sample
    sample = vae.encoder.conv_out(sample)
    output["conv_out"] = sample
    sample = vae.quant_conv(sample)
    output["quant_conv"] = sample
    sample = vae.encode(image).latent_dist.mode()
    output["mode"] = sample
    sample = Flux2KleinPipeline._patchify_latents(sample)
    output["patchified"] = sample
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(sample.device, sample.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    sample = (sample - bn_mean) / bn_std
    output["bn"] = sample
    return output


def run_comfy_encoder(first_stage, image: torch.Tensor, patch_attention: bool) -> dict[str, torch.Tensor]:
    from rum_native import _patch_flux2_vae_attention_for_diffusers_math, _restore_module_forwards, patchify_flux2_latents

    patched_forwards = _patch_flux2_vae_attention_for_diffusers_math(first_stage) if patch_attention else []
    output = {}
    try:
        sample = first_stage.encoder.conv_in(image)
        output["conv_in"] = sample

        for block_index, down in enumerate(first_stage.encoder.down):
            for resnet_index, block in enumerate(down.block):
                sample = block(sample, None)
                output[f"down.{block_index}.resnet.{resnet_index}"] = sample
                if len(down.attn) > 0:
                    sample = down.attn[resnet_index](sample)
                    output[f"down.{block_index}.attn.{resnet_index}"] = sample
            if hasattr(down, "downsample"):
                sample = down.downsample(sample)
                output[f"down.{block_index}.downsample.0"] = sample

        sample = first_stage.encoder.mid.block_1(sample, None)
        output["mid.block_1"] = sample
        sample = first_stage.encoder.mid.attn_1(sample)
        output["mid.attn_1"] = sample
        sample = first_stage.encoder.mid.block_2(sample, None)
        output["mid.block_2"] = sample
        sample = first_stage.encoder.norm_out(sample)
        output["norm_out"] = sample
        sample = torch.nn.functional.silu(sample)
        output["act_out"] = sample
        sample = first_stage.encoder.conv_out(sample)
        output["conv_out"] = sample
        sample = first_stage.quant_conv(sample)
        output["quant_conv"] = sample
        sample, _ = first_stage.regularization(sample)
        output["mode"] = sample
        sample = patchify_flux2_latents(sample)
        output["patchified"] = sample
        bn_mean = first_stage.bn.running_mean.view(1, -1, 1, 1).to(device=sample.device, dtype=sample.dtype)
        bn_std = torch.sqrt(first_stage.bn.running_var.view(1, -1, 1, 1).to(device=sample.device, dtype=sample.dtype) + first_stage.bn_eps)
        sample = (sample - bn_mean) / bn_std
        output["bn"] = sample
    finally:
        _restore_module_forwards(patched_forwards)
    return output


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.path.insert(0, str(repo_root))
    os.chdir(args.comfy_root.resolve())

    import comfy.sd
    from diffusers.models import AutoencoderKLFlux2

    processed = load_file(str(args.processed), device="cpu")["processed"].to(device="cuda", dtype=torch.bfloat16)

    diffusers_vae = AutoencoderKLFlux2.from_pretrained(
        str(args.base_model),
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    comfy_vae = comfy.sd.VAE(sd=load_file(str(args.vae_file), device="cpu"))
    first_stage = comfy_vae.first_stage_model.to("cuda").eval()

    diffusers_outputs = run_diffusers_encoder(diffusers_vae, processed)
    comfy_outputs = run_comfy_encoder(first_stage, processed, patch_attention=True)

    for key in diffusers_outputs:
        if key not in comfy_outputs:
            print(f"{key}: missing in comfy outputs")
            continue
        if not metric(key, comfy_outputs[key], diffusers_outputs[key]):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

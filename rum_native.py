from __future__ import annotations

import re
import math
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


_DIFFUSION_PREFIX = "diffusion_model."

class RUMDualTextProjection(nn.Module):
    def __init__(self, base_weight: torch.Tensor, extra_weight: torch.Tensor, base_text_tokens: int):
        super().__init__()
        if base_weight.ndim != 2:
            raise ValueError(f"context_embedder.weight 必须是 2D tensor，实际是 {tuple(base_weight.shape)}。")
        if extra_weight.ndim != 2:
            raise ValueError(f"context_embedder_2.weight 必须是 2D tensor，实际是 {tuple(extra_weight.shape)}。")
        if base_weight.shape[0] != extra_weight.shape[0]:
            raise ValueError(
                "context_embedder.weight 和 context_embedder_2.weight 输出维度不同："
                f"{base_weight.shape[0]} != {extra_weight.shape[0]}。"
            )
        if base_text_tokens <= 0:
            raise ValueError("base_text_tokens 必须大于 0。")

        self.base_text_tokens = int(base_text_tokens)
        self.base_in_features = int(base_weight.shape[1])
        self.extra_in_features = int(extra_weight.shape[1])
        self.out_features = int(base_weight.shape[0])
        self.base_weight = nn.Parameter(base_weight.detach().contiguous(), requires_grad=False)
        self.extra_weight = nn.Parameter(extra_weight.detach().contiguous(), requires_grad=False)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        base_tokens = min(self.base_text_tokens, text.shape[1])
        base_text = _fit_last_dim(text[:, :base_tokens], self.base_in_features)
        base_weight = self.base_weight.to(device=base_text.device, dtype=base_text.dtype)
        base_out = F.linear(base_text, base_weight)

        extra_text = text[:, base_tokens:]
        if extra_text.shape[1] == 0:
            return base_out

        extra_text = _fit_last_dim(extra_text, self.extra_in_features)
        extra_weight = self.extra_weight.to(device=extra_text.device, dtype=extra_text.dtype)
        extra_out = F.linear(extra_text, extra_weight)
        return torch.cat([base_out, extra_out], dim=1)


def combine_rum_conditioning(
    flux2_conditioning,
    sdxl_conditioning,
    *,
    guidance: float | None,
    base_text_tokens: int,
    extra_text_tokens: int,
    sdxl_clip_width: int,
    use_sdxl_extra: bool = True,
):
    if not flux2_conditioning:
        raise ValueError("flux2_conditioning 不能为空。")
    if not sdxl_conditioning:
        raise ValueError("sdxl_conditioning 不能为空。")
    if base_text_tokens <= 0:
        raise ValueError("base_text_tokens 必须大于 0；FLUX.2-Klein 原生 ComfyUI 通常是 512。")
    if extra_text_tokens <= 0:
        raise ValueError("extra_text_tokens 必须大于 0；SDXL CLIP 通常是 77。")
    if sdxl_clip_width <= 0:
        raise ValueError("sdxl_clip_width 必须大于 0；SDXL CLIP-L + CLIP-G 通常是 2048。")

    output = []
    for index, flux_item in enumerate(flux2_conditioning):
        sdxl_item = sdxl_conditioning[min(index, len(sdxl_conditioning) - 1)]
        flux_embeds, flux_meta = flux_item
        sdxl_embeds, _ = sdxl_item

        if flux_embeds.ndim != 3 or sdxl_embeds.ndim != 3:
            raise ValueError(
                "RUM conditioning 需要 3D tensor：[batch, tokens, channels]。"
                f"当前 FLUX={tuple(flux_embeds.shape)}，SDXL={tuple(sdxl_embeds.shape)}。"
            )

        flux_base = flux_embeds[:, :base_text_tokens]
        if use_sdxl_extra:
            sdxl_extra = sdxl_embeds.to(device=flux_embeds.device, dtype=flux_embeds.dtype)
            sdxl_extra = sdxl_extra[:, :extra_text_tokens]
            sdxl_extra = _fit_last_dim(sdxl_extra, sdxl_clip_width)
            sdxl_extra = _fit_last_dim(sdxl_extra, flux_base.shape[-1])
        else:
            sdxl_extra = flux_base.new_zeros((flux_base.shape[0], extra_text_tokens, flux_base.shape[-1]))

        combined = torch.cat([flux_base, sdxl_extra], dim=1).to(dtype=torch.bfloat16).to(dtype=torch.float32)
        meta = flux_meta.copy()
        if guidance is None:
            meta.pop("guidance", None)
        else:
            meta["guidance"] = float(guidance)
        meta["rum_base_text_tokens"] = int(flux_base.shape[1])
        meta["rum_extra_text_tokens"] = int(sdxl_extra.shape[1])
        meta["pooled_output"] = flux_base.new_zeros((flux_base.shape[0], flux_base.shape[-1])).to(dtype=torch.float32)
        meta.pop("attention_mask", None)
        output.append([combined, meta])
    return output


class RUMDiffusersMatchTokenPolicy:
    def __init__(self, base_text_tokens: int, extra_text_tokens: int):
        if base_text_tokens <= 0:
            raise ValueError("diffusers match base_text_tokens 必须大于 0。")
        if extra_text_tokens <= 0:
            raise ValueError("diffusers match extra_text_tokens 必须大于 0。")
        self.base_text_tokens = int(base_text_tokens)
        self.extra_text_tokens = int(extra_text_tokens)

    @property
    def total_tokens(self) -> int:
        return self.base_text_tokens + self.extra_text_tokens

    def select(self, cross_attn: torch.Tensor, branch: str | None) -> torch.Tensor:
        if branch == "negative":
            return self.negative(cross_attn)
        if branch == "positive":
            return self.positive(cross_attn)
        return self.legacy(cross_attn)

    def positive(self, cross_attn: torch.Tensor) -> torch.Tensor:
        if cross_attn.shape[1] == self.total_tokens:
            return cross_attn
        if cross_attn.shape[1] <= 512:
            return cross_attn[:, -self.total_tokens :]
        base = cross_attn[:, : self.base_text_tokens]
        extra = cross_attn[:, -self.extra_text_tokens :]
        return torch.cat([base, extra], dim=1)

    def negative(self, cross_attn: torch.Tensor) -> torch.Tensor:
        if cross_attn.shape[1] <= 512:
            return cross_attn
        return cross_attn[:, :512]

    def legacy(self, cross_attn: torch.Tensor) -> torch.Tensor:
        if cross_attn.shape[1] <= self.total_tokens:
            return cross_attn
        if cross_attn.shape[1] == 512:
            return cross_attn
        if cross_attn.shape[1] < 512:
            return cross_attn[:, -self.total_tokens :]
        base = cross_attn[:, : self.base_text_tokens]
        extra = cross_attn[:, -self.extra_text_tokens :]
        return torch.cat([base, extra], dim=1)


class RUMDiffusersMatchExtraConds:
    def __init__(
        self,
        original_extra_conds,
        token_policy: RUMDiffusersMatchTokenPolicy,
        *,
        disable_guidance: bool,
        reference_latents=None,
    ):
        self.original_extra_conds = original_extra_conds
        self.token_policy = token_policy
        self.disable_guidance = bool(disable_guidance)
        self.reference_latents = reference_latents

    def __call__(self, **kwargs):
        output = self.original_extra_conds(**kwargs)
        if self.disable_guidance:
            output = output.copy()
            output.pop("guidance", None)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is None:
            return output

        branch = kwargs.get("rum_diffusers_cfg_branch")
        if branch is None:
            branch = kwargs.get("prompt_type")
        if branch is None:
            branch = kwargs.get("transformer_options", {}).get("rum_diffusers_cfg_branch")
        selected = self.token_policy.select(cross_attn, branch)

        try:
            import comfy.conds
        except Exception:
            return output

        output = output.copy()
        output["c_crossattn"] = comfy.conds.CONDRegular(selected)
        if self.reference_latents is not None:
            output["rum_reference_latents"] = RUMReferenceLatentsCond(self.reference_latents)
        return output


class RUMReferenceLatentsCond:
    def __init__(self, reference_latents):
        self.reference_latents = reference_latents

    def process_cond(self, batch_size, **kwargs):
        return self

    def can_concat(self, other):
        return False

    def concat(self, others):
        raise ValueError("RUM reference latents 不支持 concat；请使用单个全图 conditioning。")

    def size(self):
        latents = self.reference_latents.get("latents") if isinstance(self.reference_latents, dict) else None
        if not isinstance(latents, list):
            return [1]
        return [sum(int(latent.numel()) for latent in latents)]


def apply_diffusers_match_model_wrapper(
    model,
    *,
    base_text_tokens: int,
    extra_text_tokens: int,
    disable_guidance: bool = True,
):
    import comfy.patcher_extension

    patched = model.clone()
    patched.rum_diffusers_match_config = {
        "base_text_tokens": int(base_text_tokens),
        "extra_text_tokens": int(extra_text_tokens),
        "disable_guidance": bool(disable_guidance),
    }
    patched.add_wrapper(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        RUMDiffusersTimestepWrapper(),
    )
    return patched


def _linear_split(module: nn.Module, tensor: torch.Tensor, start: int, width: int) -> torch.Tensor:
    weight = module.weight.narrow(0, start, width)
    bias = getattr(module, "bias", None)
    if bias is not None:
        bias = bias.narrow(0, start, width)
    if weight.device != tensor.device or weight.dtype != tensor.dtype:
        weight = weight.to(device=tensor.device, dtype=tensor.dtype)
    if bias is not None and (bias.device != tensor.device or bias.dtype != tensor.dtype):
        bias = bias.to(device=tensor.device, dtype=tensor.dtype)
    return F.linear(tensor, weight, bias)


def _diffusers_rope_from_comfy_pe(pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pe.ndim != 6 or pe.shape[-2:] != (2, 2):
        raise ValueError(f"RUM diffusers-match RoPE 需要 ComfyUI FLUX pe tensor，当前形状是 {tuple(pe.shape)}。")
    matrix = pe[0, 0]
    cos = matrix[..., 0, 0]
    sin = matrix[..., 1, 0]
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def _diffusers_apply_rotary_emb(tensor: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rope
    cos = cos[None, :, None, :].to(device=tensor.device)
    sin = sin[None, :, None, :].to(device=tensor.device)
    real, imag = tensor.reshape(*tensor.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(3)
    return (tensor.float() * cos + rotated.float() * sin).to(tensor.dtype)


def _diffusers_rms_norm(tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if weight.device != tensor.device or weight.dtype != tensor.dtype:
        weight = weight.to(device=tensor.device, dtype=tensor.dtype)
    return F.rms_norm(tensor, (tensor.shape[-1],), weight, eps)


def _rms_norm_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight):
        return weight

    scale = getattr(module, "scale", None)
    if torch.is_tensor(scale):
        return scale

    raise AttributeError(f"RUM diffusers-match expected RMSNorm weight/scale on {type(module).__name__}.")


def _diffusers_native_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    try:
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=False,
        )
    except TypeError:
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    return output.permute(0, 2, 1, 3)


def patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"FLUX.2 latent 必须是 4D tensor，当前形状是 {tuple(latents.shape)}。")
    batch_size, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"FLUX.2 latent 高宽必须能被 2 整除，当前形状是 {tuple(latents.shape)}。")
    latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, channels * 4, height // 2, width // 2)


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"FLUX.2 patchified latent 必须是 4D tensor，当前形状是 {tuple(latents.shape)}。")
    batch_size, channels, height, width = latents.shape
    if channels % 4 != 0:
        raise ValueError(f"FLUX.2 patchified latent channel 必须能被 4 整除，当前形状是 {tuple(latents.shape)}。")
    latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"FLUX.2 latent pack 需要 4D tensor，当前形状是 {tuple(latents.shape)}。")
    batch_size, channels, height, width = latents.shape
    return latents.reshape(batch_size, channels, height * width).permute(0, 2, 1)


def prepare_diffusers_latent_ids(
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"FLUX.2 latent ids 需要 4D tensor，当前形状是 {tuple(latents.shape)}。")
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)


def prepare_diffusers_text_ids(
    batch_size: int,
    token_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("FLUX.2 text ids batch_size 必须大于 0。")
    if token_count <= 0:
        raise ValueError("FLUX.2 text ids token_count 必须大于 0。")
    text_ids = torch.zeros((int(batch_size), int(token_count), 4), device=device, dtype=dtype)
    text_ids[:, :, 3] = torch.arange(int(token_count), device=device, dtype=dtype)
    return text_ids


def prepare_diffusers_reference_image_ids(
    reference_latents: list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(reference_latents, list):
        raise ValueError(f"RUM reference_latents 必须是 list，当前类型是 {type(reference_latents)}。")
    if len(reference_latents) == 0:
        raise ValueError("RUM reference_latents 至少需要 1 张参考图 latent。")
    if batch_size <= 0:
        raise ValueError("RUM reference_latents batch_size 必须大于 0。")

    image_ids = []
    for index, latent in enumerate(reference_latents):
        if latent.ndim != 4:
            raise ValueError(f"RUM reference latent 必须是 4D tensor，当前形状是 {tuple(latent.shape)}。")
        if latent.shape[0] != 1:
            raise ValueError(f"RUM reference latent 单项 batch 必须为 1，当前形状是 {tuple(latent.shape)}。")
        _, _, height, width = latent.shape
        t = torch.tensor([10 * (index + 1)])
        ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_ids.append(ids)

    combined = torch.cat(image_ids, dim=0).unsqueeze(0)
    return combined.expand(int(batch_size), -1, -1).to(device=device, dtype=dtype)


def pack_rum_reference_latents(
    reference_latents,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = validate_rum_reference_latents(reference_latents)

    packed_latents = []
    normalized_latents = []
    for latent in latents:
        current = latent.to(device=device, dtype=dtype)
        normalized_latents.append(current)
        packed_latents.append(pack_flux2_latents(current).squeeze(0))

    reference_tokens = torch.cat(packed_latents, dim=0).unsqueeze(0).expand(int(batch_size), -1, -1)
    reference_ids = prepare_diffusers_reference_image_ids(
        normalized_latents,
        batch_size=int(batch_size),
        device=device,
        dtype=torch.float32,
    )
    return reference_tokens, reference_ids


def append_reference_tokens(
    denoise_tokens: torch.Tensor,
    denoise_ids: torch.Tensor,
    reference_tokens: torch.Tensor | None,
    reference_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if denoise_tokens.ndim != 3:
        raise ValueError(f"denoise tokens 必须是 3D tensor，当前形状是 {tuple(denoise_tokens.shape)}。")
    if denoise_ids.ndim != 3:
        raise ValueError(f"denoise ids 必须是 3D tensor，当前形状是 {tuple(denoise_ids.shape)}。")
    denoise_token_count = int(denoise_tokens.shape[1])
    if reference_tokens is None and reference_ids is None:
        return denoise_tokens, denoise_ids, denoise_token_count
    if reference_tokens is None or reference_ids is None:
        raise ValueError("RUM reference tokens 和 ids 必须同时提供。")
    if reference_tokens.ndim != 3 or reference_ids.ndim != 3:
        raise ValueError(
            "RUM reference tokens/ids 必须是 3D tensor："
            f"tokens={tuple(reference_tokens.shape)}, ids={tuple(reference_ids.shape)}。"
        )
    if denoise_tokens.shape[0] != reference_tokens.shape[0] or denoise_ids.shape[0] != reference_ids.shape[0]:
        raise ValueError(
            "RUM reference batch 与 denoise batch 不一致："
            f"denoise={tuple(denoise_tokens.shape)}, reference={tuple(reference_tokens.shape)}。"
        )
    if denoise_tokens.shape[2] != reference_tokens.shape[2]:
        raise ValueError(
            "RUM reference channel 与 denoise channel 不一致："
            f"denoise={tuple(denoise_tokens.shape)}, reference={tuple(reference_tokens.shape)}。"
        )
    if denoise_ids.shape[2] != reference_ids.shape[2]:
        raise ValueError(
            "RUM reference id 维度与 denoise id 维度不一致："
            f"denoise={tuple(denoise_ids.shape)}, reference={tuple(reference_ids.shape)}。"
        )
    return (
        torch.cat([denoise_tokens, reference_tokens], dim=1),
        torch.cat([denoise_ids, reference_ids], dim=1),
        denoise_token_count,
    )


def crop_noise_to_latent_tokens(noise_pred: torch.Tensor, latent_token_count: int) -> torch.Tensor:
    if noise_pred.ndim != 3:
        raise ValueError(f"RUM raw noise 必须是 3D token tensor，当前形状是 {tuple(noise_pred.shape)}。")
    if latent_token_count <= 0:
        raise ValueError("latent_token_count 必须大于 0。")
    if noise_pred.shape[1] < latent_token_count:
        raise ValueError(
            f"RUM raw noise token 数不足：noise={noise_pred.shape[1]}, latent={latent_token_count}。"
        )
    return noise_pred[:, : int(latent_token_count)]


def _pil_from_comfy_image(image: torch.Tensor):
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"ComfyUI IMAGE 单张输入应为 [H,W,C>=3]，当前形状是 {tuple(image.shape)}。")
    import numpy as np
    from PIL import Image

    array = torch.round(image[..., :3].detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def preprocess_flux2_reference_image_tensor(image: torch.Tensor, *, vae_scale_factor: int) -> torch.Tensor:
    if vae_scale_factor <= 0:
        raise ValueError("vae_scale_factor 必须大于 0。")
    pil_image = _pil_from_comfy_image(image)
    width, height = pil_image.size
    if width < 64 or height < 64:
        raise ValueError(f"参考图尺寸过小：{width}x{height}；宽高都必须至少 64px。")
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > 8:
        raise ValueError(f"参考图宽高比过大：{width}x{height}，ratio={aspect_ratio:.1f}:1；最大允许 8:1。")

    target_area = 1024 * 1024
    if width * height > target_area:
        from PIL import Image

        scale = math.sqrt(target_area / float(width * height))
        width = int(width * scale)
        height = int(height * scale)
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

    multiple_of = int(vae_scale_factor) * 2
    cropped_width = (pil_image.size[0] // multiple_of) * multiple_of
    cropped_height = (pil_image.size[1] // multiple_of) * multiple_of
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError(
            f"参考图裁剪到 {multiple_of} 倍数后无效：原始处理尺寸={pil_image.size[0]}x{pil_image.size[1]}。"
        )

    left = (pil_image.size[0] - cropped_width) // 2
    top = (pil_image.size[1] - cropped_height) // 2
    pil_image = pil_image.crop((left, top, left + cropped_width, top + cropped_height))

    import numpy as np

    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).permute(0, 3, 1, 2)
    return tensor * 2.0 - 1.0


def encode_flux2_vae_mode_latent(first_stage: nn.Module, image: torch.Tensor) -> torch.Tensor:
    encoded = first_stage.encoder(image)
    encoded = first_stage.quant_conv(encoded)
    latent, _ = first_stage.regularization(encoded)
    latent = patchify_flux2_latents(latent)
    bn_mean = first_stage.bn.running_mean.view(1, -1, 1, 1).to(device=latent.device, dtype=latent.dtype)
    bn_std = torch.sqrt(
        first_stage.bn.running_var.view(1, -1, 1, 1).to(device=latent.device, dtype=latent.dtype) + first_stage.bn_eps
    )
    return (latent - bn_mean) / bn_std


@torch.inference_mode()
def encode_flux2_native_match_reference_image(vae, images) -> tuple[dict, str]:
    import comfy.model_management

    validate_comfy_reference_image_batch(images)

    first_stage = _resolve_vae_first_stage(vae)
    if not hasattr(first_stage, "bn"):
        raise ValueError("当前 VAE 不是 FLUX.2 VAE：找不到 batch norm 参数 bn。")

    vae_scale_factor = 8
    processed = [preprocess_flux2_reference_image_tensor(image, vae_scale_factor=vae_scale_factor) for image in images]
    dtype = getattr(vae, "vae_dtype", torch.bfloat16)
    device = comfy.model_management.vae_device()
    offload_device = comfy.model_management.vae_offload_device()
    largest = max(processed, key=lambda tensor: tensor.shape[-2] * tensor.shape[-1])
    memory_used = vae.memory_used_encode(largest.shape, dtype=dtype)
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory_used)

    first_stage = first_stage.to(device)
    patched_forwards = _patch_flux2_vae_attention_for_diffusers_math(first_stage)
    latents = []
    try:
        for image in processed:
            encoded = encode_flux2_vae_mode_latent(first_stage, image.to(device=device, dtype=dtype)).to(torch.float32)
            latents.append(encoded.to(device="cpu", dtype=torch.float32))
    finally:
        _restore_module_forwards(patched_forwards)
        first_stage.to(offload_device)

    status_parts = [f"{latent.shape[-1] * 16}x{latent.shape[-2] * 16}->{tuple(latent.shape)}" for latent in latents]
    return {"latents": latents}, "RUM reference latents 已编码：" + "; ".join(status_parts)


def _rum_diffusers_double_block_forward(
    self,
    img: torch.Tensor,
    txt: torch.Tensor,
    vec,
    pe: torch.Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    transformer_options=None,
):
    if modulation_dims_img is not None or modulation_dims_txt is not None:
        raise ValueError("RUM diffusers-match exact forward 暂不支持 reference/timestep-zero modulation。")
    transformer_options = transformer_options or {}
    if transformer_options.get("patches"):
        raise ValueError("RUM diffusers-match exact forward 暂不支持 attention patches / LoRA patches。")

    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    norm_img = self.img_norm1(img)
    norm_img = (1 + img_mod1.scale) * norm_img + img_mod1.shift
    norm_txt = self.txt_norm1(txt)
    norm_txt = (1 + txt_mod1.scale) * norm_txt + txt_mod1.shift

    hidden_size = self.hidden_size
    img_query = _linear_split(self.img_attn.qkv, norm_img, 0, hidden_size)
    img_key = _linear_split(self.img_attn.qkv, norm_img, hidden_size, hidden_size)
    img_value = _linear_split(self.img_attn.qkv, norm_img, hidden_size * 2, hidden_size)
    txt_query = _linear_split(self.txt_attn.qkv, norm_txt, 0, hidden_size)
    txt_key = _linear_split(self.txt_attn.qkv, norm_txt, hidden_size, hidden_size)
    txt_value = _linear_split(self.txt_attn.qkv, norm_txt, hidden_size * 2, hidden_size)

    img_query = img_query.unflatten(-1, (self.num_heads, -1))
    img_key = img_key.unflatten(-1, (self.num_heads, -1))
    img_value = img_value.unflatten(-1, (self.num_heads, -1))
    txt_query = txt_query.unflatten(-1, (self.num_heads, -1))
    txt_key = txt_key.unflatten(-1, (self.num_heads, -1))
    txt_value = txt_value.unflatten(-1, (self.num_heads, -1))

    img_query = _diffusers_rms_norm(img_query, _rms_norm_weight(self.img_attn.norm.query_norm))
    img_key = _diffusers_rms_norm(img_key, _rms_norm_weight(self.img_attn.norm.key_norm))
    txt_query = _diffusers_rms_norm(txt_query, _rms_norm_weight(self.txt_attn.norm.query_norm))
    txt_key = _diffusers_rms_norm(txt_key, _rms_norm_weight(self.txt_attn.norm.key_norm))

    query = torch.cat((txt_query, img_query), dim=1)
    key = torch.cat((txt_key, img_key), dim=1)
    value = torch.cat((txt_value, img_value), dim=1)
    rope = _diffusers_rope_from_comfy_pe(pe)
    query = _diffusers_apply_rotary_emb(query, rope)
    key = _diffusers_apply_rotary_emb(key, rope)

    attn = _diffusers_native_attention(query, key, value, attn_mask=attn_mask).flatten(2, 3).to(query.dtype)
    txt_attn, img_attn = attn.split_with_sizes((txt.shape[1], img.shape[1]), dim=1)
    txt_attn = self.txt_attn.proj(txt_attn)
    img_attn = self.img_attn.proj(img_attn)

    img = img + img_mod1.gate * img_attn
    norm_img = self.img_norm2(img)
    norm_img = norm_img * (1 + img_mod2.scale) + img_mod2.shift
    img = img + img_mod2.gate * self.img_mlp(norm_img)

    txt = txt + txt_mod1.gate * txt_attn
    norm_txt = self.txt_norm2(txt)
    norm_txt = norm_txt * (1 + txt_mod2.scale) + txt_mod2.shift
    txt = txt + txt_mod2.gate * self.txt_mlp(norm_txt)
    return img, txt


def _rum_diffusers_single_block_forward(
    self,
    x: torch.Tensor,
    vec,
    pe: torch.Tensor,
    attn_mask=None,
    modulation_dims=None,
    transformer_options=None,
) -> torch.Tensor:
    if modulation_dims is not None:
        raise ValueError("RUM diffusers-match exact forward 暂不支持 reference/timestep-zero modulation。")
    transformer_options = transformer_options or {}
    if transformer_options.get("patches"):
        raise ValueError("RUM diffusers-match exact forward 暂不支持 attention patches / LoRA patches。")

    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec

    hidden_states = self.pre_norm(x)
    hidden_states = (1 + mod.scale) * hidden_states + mod.shift
    projected = self.linear1(hidden_states)
    qkv, mlp = torch.split(projected, [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)
    query, key, value = qkv.chunk(3, dim=-1)

    query = query.unflatten(-1, (self.num_heads, -1))
    key = key.unflatten(-1, (self.num_heads, -1))
    value = value.unflatten(-1, (self.num_heads, -1))
    query = _diffusers_rms_norm(query, _rms_norm_weight(self.norm.query_norm))
    key = _diffusers_rms_norm(key, _rms_norm_weight(self.norm.key_norm))

    rope = _diffusers_rope_from_comfy_pe(pe)
    query = _diffusers_apply_rotary_emb(query, rope)
    key = _diffusers_apply_rotary_emb(key, rope)
    hidden_states = _diffusers_native_attention(query, key, value, attn_mask=attn_mask).flatten(2, 3).to(query.dtype)

    mlp = self.mlp_act(mlp)
    output = self.linear2(torch.cat((hidden_states, mlp), dim=-1))
    return x + mod.gate * output


def _rum_diffusers_final_layer_forward(self, x: torch.Tensor, vec: torch.Tensor, modulation_dims=None) -> torch.Tensor:
    if modulation_dims is not None:
        raise ValueError("RUM diffusers-match exact forward 暂不支持 reference/timestep-zero modulation。")
    if vec.ndim != 2:
        vec = vec.reshape(vec.shape[0], -1)

    modulation = self.adaLN_modulation[1](self.adaLN_modulation[0](vec).to(x.dtype))
    shift, scale = modulation.chunk(2, dim=1)
    x = self.norm_final(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    return self.linear(x)


@contextmanager
def _diffusers_match_forward_patch(diffusion_model):
    originals = []
    try:
        for block in getattr(diffusion_model, "double_blocks", []):
            originals.append((block, block.forward))
            block.forward = MethodType(_rum_diffusers_double_block_forward, block)
        for block in getattr(diffusion_model, "single_blocks", []):
            originals.append((block, block.forward))
            block.forward = MethodType(_rum_diffusers_single_block_forward, block)
        final_layer = getattr(diffusion_model, "final_layer", None)
        if final_layer is not None:
            originals.append((final_layer, final_layer.forward))
            final_layer.forward = MethodType(_rum_diffusers_final_layer_forward, final_layer)
        yield
    finally:
        for block, original_forward in reversed(originals):
            block.forward = original_forward


def _call_diffusers_match_forward_orig(
    diffusion_model,
    x_in: torch.Tensor,
    timestep_in: torch.Tensor,
    *,
    context: torch.Tensor,
    transformer_options: dict,
    control,
    guidance,
    reference_latents,
) -> torch.Tensor:
    if x_in.ndim != 4:
        raise ValueError(f"RUM diffusers-match 需要 4D latent，当前形状是 {tuple(x_in.shape)}。")

    batch_size, _, height, width = x_in.shape
    img_tokens = pack_flux2_latents(x_in)
    img_ids = prepare_diffusers_latent_ids(
        x_in,
        device=x_in.device,
        dtype=torch.float32,
    )
    if reference_latents is not None:
        reference_tokens, reference_ids = pack_rum_reference_latents(
            reference_latents,
            batch_size=batch_size,
            device=x_in.device,
            dtype=x_in.dtype,
        )
        img_tokens, img_ids, denoise_token_count = append_reference_tokens(
            img_tokens,
            img_ids,
            reference_tokens,
            reference_ids,
        )
    else:
        denoise_token_count = int(img_tokens.shape[1])

    txt_ids = prepare_diffusers_text_ids(
        batch_size,
        context.shape[1],
        device=x_in.device,
        dtype=torch.float32,
    )
    model_output = diffusion_model.forward_orig(
        img=img_tokens,
        img_ids=img_ids,
        txt=context,
        txt_ids=txt_ids,
        timesteps=timestep_in,
        y=None,
        guidance=guidance,
        control=control,
        transformer_options=transformer_options,
        attn_mask=None,
    )
    model_output = crop_noise_to_latent_tokens(model_output, denoise_token_count)
    return model_output.permute(0, 2, 1).reshape(batch_size, model_output.shape[-1], height, width)


class RUMDiffusersTimestepWrapper:
    def __call__(self, apply_model, x, timestep, context, *args, **kwargs):
        diffusion_model = apply_model.class_obj
        original_time_in = diffusion_model.time_in
        diffusion_dtype = diffusion_model.dtype

        class DiffusersTimeIn(nn.Module):
            def forward(self, ignored_embedding):
                from comfy.ldm.flux.layers import timestep_embedding

                diffusers_timestep = (timestep.to(torch.float32) * 1000.0).to(dtype=diffusion_dtype)
                embedding = timestep_embedding(diffusers_timestep, 256, time_factor=1.0).to(
                    device=diffusers_timestep.device,
                    dtype=diffusion_dtype,
                )
                return original_time_in(embedding)

        diffusion_model.time_in = DiffusersTimeIn()
        try:
            with _diffusers_match_forward_patch(diffusion_model):
                return apply_model(x, timestep, context, *args, **kwargs)
        finally:
            diffusion_model.time_in = original_time_in



def _condition_tensor(model_cond):
    value = getattr(model_cond, "cond", model_cond)
    return value


def _condition_reference_latents(model_cond):
    value = getattr(model_cond, "reference_latents", model_cond)
    return value


def _model_inference_dtype(model, fallback: torch.dtype) -> torch.dtype:
    get_dtype_inference = getattr(model, "get_dtype_inference", None)
    if callable(get_dtype_inference):
        return get_dtype_inference()

    get_dtype = getattr(model, "get_dtype", None)
    if callable(get_dtype):
        return get_dtype()

    diffusion_model = getattr(model, "diffusion_model", None)
    dtype = getattr(diffusion_model, "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else fallback


def _predict_raw_noise(model, x, timestep, cond, model_options, branch: str):
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.samplers

    area_cond = comfy.samplers.get_area_and_mult(cond, x, timestep)
    if area_cond is None:
        raise ValueError(f"RUM diffusers-match {branch} conditioning 当前 timestep 没有可用区域。")
    if area_cond.area is not None:
        raise ValueError("RUM diffusers-match raw-noise 路径暂不支持 area/mask conditioning；请用全图 conditioning 做对齐验证。")

    diffusion_model = model.diffusion_model
    diffusion_dtype = _model_inference_dtype(model, x.dtype)
    x_in = model.model_sampling.calculate_input(timestep, area_cond.input_x).to(diffusion_dtype)
    device = x_in.device

    conditioning = area_cond.conditioning
    context = _condition_tensor(conditioning["c_crossattn"])
    context = comfy.model_management.cast_to_device(context, device, diffusion_dtype)

    transformer_options = model.current_patcher.apply_hooks(hooks=area_cond.hooks)
    if "transformer_options" in model_options:
        transformer_options = comfy.patcher_extension.merge_nested_dicts(
            transformer_options,
            model_options["transformer_options"],
            copy_dict1=False,
        )
    if area_cond.patches is not None:
        transformer_options["patches"] = comfy.patcher_extension.merge_nested_dicts(
            transformer_options.get("patches", {}),
            area_cond.patches,
        )
    transformer_options["cond_or_uncond"] = [0]
    transformer_options["uuids"] = [area_cond.uuid]
    transformer_options["sigmas"] = timestep
    transformer_options["rum_diffusers_cfg_branch"] = branch

    guidance = _condition_tensor(conditioning["guidance"]) if "guidance" in conditioning else None
    reference_latents = _condition_reference_latents(conditioning["rum_reference_latents"]) if "rum_reference_latents" in conditioning else None
    timestep_in = ((timestep.to(torch.float32) * 1000.0).to(diffusion_dtype) / 1000.0).to(diffusion_dtype)

    with torch.inference_mode(), _diffusers_match_forward_patch(diffusion_model):
        model_output = _call_diffusers_match_forward_orig(
            diffusion_model,
            x_in,
            timestep_in,
            context=context,
            transformer_options=transformer_options,
            control=area_cond.control,
            guidance=guidance,
            reference_latents=reference_latents,
        )

    if len(model_output) > 1 and not torch.is_tensor(model_output):
        from comfy import utils

        model_output, _ = utils.pack_latents(model_output)
    return model_output



def create_diffusers_cfg_guider(model, positive, negative, cfg: float, reference_latents=None):
    import comfy.model_patcher
    import comfy.samplers

    if reference_latents is not None:
        validate_rum_reference_latents(reference_latents)

    def branch_options(model_options, branch: str):
        options = comfy.model_patcher.create_model_options_clone(model_options)
        transformer_options = options.setdefault("transformer_options", {})
        transformer_options["rum_diffusers_cfg_branch"] = branch
        options["disable_cfg1_optimization"] = True
        return options

    class RUMDiffusersCFGGuider(comfy.samplers.CFGGuider):
        def inner_sample(
            self,
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=None,
        ):
            match_config = getattr(self.model_patcher, "rum_diffusers_match_config", None)
            if match_config is None:
                return super().inner_sample(
                    noise,
                    latent_image,
                    device,
                    sampler,
                    sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )

            original_extra_conds = self.inner_model.extra_conds
            token_policy = RUMDiffusersMatchTokenPolicy(
                match_config["base_text_tokens"],
                match_config["extra_text_tokens"],
            )
            self.inner_model.extra_conds = RUMDiffusersMatchExtraConds(
                original_extra_conds,
                token_policy,
                disable_guidance=match_config.get("disable_guidance", True),
                reference_latents=reference_latents,
            )
            try:
                return super().inner_sample(
                    noise,
                    latent_image,
                    device,
                    sampler,
                    sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )
            finally:
                self.inner_model.extra_conds = original_extra_conds

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            positive_cond = self.conds.get("positive", None)
            negative_cond = self.conds.get("negative", None)

            if model_options.get("rum_diffusers_return_raw_noise", False):
                positive_out = _predict_raw_noise(
                    self.inner_model,
                    x,
                    timestep,
                    positive_cond[0],
                    model_options,
                    "positive",
                )
                if negative_cond is None or self.cfg <= 1.0:
                    return positive_out
                negative_out = _predict_raw_noise(
                    self.inner_model,
                    x,
                    timestep,
                    negative_cond[0],
                    model_options,
                    "negative",
                )
                return negative_out + self.cfg * (positive_out - negative_out)

            positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [positive_cond],
                x,
                timestep,
                branch_options(model_options, "positive"),
            )[0]
            if negative_cond is None or self.cfg <= 1.0:
                return positive_out

            negative_out = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond],
                x,
                timestep,
                branch_options(model_options, "negative"),
            )[0]
            return comfy.samplers.cfg_function(
                self.inner_model,
                positive_out,
                negative_out,
                self.cfg,
                x,
                timestep,
                model_options=model_options,
                cond=positive_cond,
                uncond=negative_cond,
            )

    guider = RUMDiffusersCFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(float(cfg))
    return guider


class RUMDiffusersEulerSampler:
    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        import math

        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        import torch
        from tqdm.auto import trange

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if denoise_mask is not None:
            raise ValueError("RUM diffusers-match sampler 暂不支持 denoise_mask；请用全图采样来做对齐验证。")

        model_dtype = getattr(model_wrap.inner_model, "get_dtype_inference", lambda: noise.dtype)()
        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0],
            noise,
            latent_image,
            self.max_denoise(model_wrap, sigmas),
        ).to(dtype=model_dtype)

        model_options = extra_args.get("model_options", {})
        seed = extra_args.get("seed", None)
        total_steps = len(sigmas) - 1
        step_iter = trange(total_steps, disable=disable_pbar)
        for i in step_iter:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_in = sigma.expand(x.shape[0]).to(device=x.device)
            step_model_options = model_options.copy()
            step_model_options["rum_diffusers_return_raw_noise"] = True
            noise_pred = model_wrap(x, sigma_in, model_options=step_model_options, seed=seed)

            dt = (sigma_next - sigma).to(device=x.device, dtype=torch.float32)
            x = (x.float() + dt * noise_pred).to(dtype=noise_pred.dtype)

            if callback is not None:
                denoised = x.float() - noise_pred.float() * sigma.to(device=x.device, dtype=torch.float32)
                callback(i, denoised, x, total_steps)

        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x.float())


def create_diffusers_euler_sampler():
    return RUMDiffusersEulerSampler()


def diffusers_flux2_sigmas(*, steps: int, width: int, height: int) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps 必须大于 0。")
    if width <= 0 or height <= 0:
        raise ValueError("width/height 必须大于 0。")

    import numpy as np

    image_seq_len = round(width * height / (16 * 16))
    mu = _diffusers_compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    sigmas = np.linspace(1.0, 1.0 / int(steps), int(steps), dtype=np.float32)
    sigmas = np.exp(mu) / (np.exp(mu) + (1 / sigmas - 1))
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _diffusers_compute_empirical_mu(*, image_seq_len: int, num_steps: int) -> float:
    # Matches diffusers Flux2KleinPipeline sigma/time-shift policy, not ComfyUI's Flux2Scheduler approximation.
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _diffusers_time_shift_exponential(mu: float, sigma: float, timesteps: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.tensor(mu, dtype=timesteps.dtype, device=timesteps.device)) / (
        torch.exp(torch.tensor(mu, dtype=timesteps.dtype, device=timesteps.device))
        + (1 / timesteps - 1) ** sigma
    )



def _unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def _resolve_vae_first_stage(vae):
    first_stage = getattr(vae, "first_stage_model", None)
    if first_stage is None:
        raise ValueError("当前 VAE 对象没有 first_stage_model，不能执行 RUM native-match decode。")
    return first_stage


def _patch_flux2_vae_attention_for_diffusers_math(first_stage):
    import comfy.ldm.modules.diffusionmodules.model as comfy_vae_model

    originals = []
    for module in first_stage.modules():
        if isinstance(module, comfy_vae_model.AttnBlock):
            originals.append((module, module.forward))
            module.forward = MethodType(_flux2_vae_attention_forward_diffusers_math, module)
    return originals


def _restore_module_forwards(originals):
    for module, forward in originals:
        module.forward = forward


def _vae_linear(hidden_states: torch.Tensor, module: nn.Module, channels: int) -> torch.Tensor:
    weight = module.weight.reshape(channels, channels).to(dtype=hidden_states.dtype)
    bias = module.bias.to(dtype=hidden_states.dtype) if module.bias is not None else None
    return F.linear(hidden_states, weight, bias)


def _flux2_vae_attention_forward_diffusers_math(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    batch_size, channels, height, width = x.shape
    sequence_length = height * width

    hidden_states = x.view(batch_size, channels, sequence_length).transpose(1, 2)
    hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = _vae_linear(hidden_states, self.q, channels)
    key = _vae_linear(hidden_states, self.k, channels)
    value = _vae_linear(hidden_states, self.v, channels)

    query = query.view(batch_size, 1, sequence_length, channels)
    key = key.view(batch_size, 1, sequence_length, channels)
    value = value.view(batch_size, 1, sequence_length, channels)

    hidden_states = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, channels)
    hidden_states = _vae_linear(hidden_states, self.proj_out, channels)
    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channels, height, width)
    return hidden_states + residual


@torch.inference_mode()
def decode_flux2_native_match_vae_latent(vae, samples):
    import comfy.model_management

    latent = samples["samples"] if isinstance(samples, dict) else samples
    if latent.is_nested:
        latent = latent.unbind()[0]

    first_stage = _resolve_vae_first_stage(vae)
    if not hasattr(first_stage, "bn"):
        raise ValueError("当前 VAE 不是 FLUX.2 VAE：找不到 batch norm 参数 bn。")

    dtype = getattr(vae, "vae_dtype", torch.bfloat16)
    device = comfy.model_management.vae_device()
    offload_device = comfy.model_management.vae_offload_device()
    memory_used = vae.memory_used_decode(latent.shape, dtype=dtype)
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory_used)

    first_stage = first_stage.to(device)
    patched_forwards = _patch_flux2_vae_attention_for_diffusers_math(first_stage)
    try:
        latents = latent.to(device=device, dtype=dtype)
        decoded = first_stage.decode(latents)
        images = (decoded * 0.5 + 0.5).clamp(0.0, 1.0).movedim(1, -1).to(torch.float32)
    finally:
        _restore_module_forwards(patched_forwards)
        first_stage.to(offload_device)

    return images.detach().cpu()

def load_rum_native_model(checkpoint_path: str | Path, *, base_text_tokens: int = 200):
    import comfy.sd

    checkpoint = _resolve_checkpoint_path(checkpoint_path)
    state_dict = load_file(str(checkpoint), device="cpu")
    base_weight = _require_key(state_dict, "context_embedder.weight")
    extra_weight = _require_key(state_dict, "context_embedder_2.weight")
    converted = convert_rum_diffusers_to_comfy(state_dict, output_prefix="")
    converted_count = len(converted)
    model = comfy.sd.load_diffusion_model_state_dict(converted)
    if model is None:
        raise ValueError("无法把 RUM checkpoint 加载为 ComfyUI native MODEL。")

    dual_projection = RUMDualTextProjection(base_weight, extra_weight, base_text_tokens=base_text_tokens)
    model.add_object_patch(f"{_DIFFUSION_PREFIX}txt_in", dual_projection)
    return model, converted_count, str(checkpoint)


def apply_rum_model_patch(model, checkpoint_path: str | Path, *, base_text_tokens: int, strict: bool):
    checkpoint = _resolve_checkpoint_path(checkpoint_path)
    state_dict = load_file(str(checkpoint), device="cpu")
    base_weight = _require_key(state_dict, "context_embedder.weight")
    extra_weight = _require_key(state_dict, "context_embedder_2.weight")
    converted = convert_rum_diffusers_to_comfy(state_dict, output_prefix=_DIFFUSION_PREFIX)
    converted.pop(f"{_DIFFUSION_PREFIX}txt_in.weight", None)

    patched = model.clone()
    dual_projection = RUMDualTextProjection(base_weight, extra_weight, base_text_tokens=base_text_tokens)
    patched.add_object_patch(f"{_DIFFUSION_PREFIX}txt_in", dual_projection)

    model_keys = set(patched.model_state_dict().keys())
    converted = _adapt_converted_keys_to_model(converted, model_keys)
    missing = sorted(key for key in converted if key not in model_keys)
    shape_errors = []
    for key, tensor in converted.items():
        if key not in model_keys:
            continue
        current = patched.get_model_object(key)
        current_shape = tuple(current.shape)
        if current_shape != tuple(tensor.shape):
            shape_errors.append(f"{key}: checkpoint {tuple(tensor.shape)} != model {current_shape}")

    if strict and (missing or shape_errors):
        details = []
        if missing:
            details.append("缺少目标权重：" + ", ".join(missing[:20]))
        if shape_errors:
            details.append("形状不匹配：" + "; ".join(shape_errors[:20]))
        raise ValueError("RUM checkpoint 无法严格套用到当前 MODEL。" + "\n".join(details))

    patches = {
        key: ("set", (tensor,))
        for key, tensor in converted.items()
        if key in model_keys and tuple(patched.get_model_object(key).shape) == tuple(tensor.shape)
    }
    if not patches:
        raise ValueError("没有任何 RUM 权重能套用到当前 MODEL；请确认底模是 FLUX.2-Klein 4B。")
    patched.add_patches(patches, strength_patch=1.0, strength_model=1.0)
    return patched, len(patches), str(checkpoint)


def _adapt_converted_keys_to_model(
    converted: dict[str, torch.Tensor],
    model_keys: set[str],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, tensor in converted.items():
        if key in model_keys:
            output[key] = tensor
            continue

        alternative = _alternate_norm_key(key)
        if alternative is not None and alternative in model_keys:
            output[alternative] = tensor
            continue

        output[key] = tensor
    return output


def _alternate_norm_key(key: str) -> str | None:
    suffixes = (
        ".img_attn.norm.query_norm.",
        ".img_attn.norm.key_norm.",
        ".txt_attn.norm.query_norm.",
        ".txt_attn.norm.key_norm.",
        ".norm.query_norm.",
        ".norm.key_norm.",
    )
    if not any(suffix in key for suffix in suffixes):
        return None
    if key.endswith(".weight"):
        return f"{key[:-len('.weight')]}.scale"
    if key.endswith(".scale"):
        return f"{key[:-len('.scale')]}.weight"
    return None


def convert_rum_diffusers_to_comfy(state_dict: dict[str, torch.Tensor], output_prefix: str = "") -> dict[str, torch.Tensor]:
    keys = set(state_dict)
    hidden_size = _infer_hidden_size(state_dict)
    double_blocks = _count_blocks(keys, "transformer_blocks")
    single_blocks = _count_blocks(keys, "single_transformer_blocks")
    if double_blocks == 0 or single_blocks == 0:
        raise ValueError("RUM checkpoint 里找不到 transformer_blocks / single_transformer_blocks。")

    key_map = _flux2_diffusers_to_comfy_key_map(
        double_blocks=double_blocks,
        single_blocks=single_blocks,
        hidden_size=hidden_size,
        output_prefix=output_prefix,
    )
    output: dict[str, torch.Tensor] = {}
    for source_key, target in key_map.items():
        if source_key not in state_dict:
            continue
        weight = state_dict[source_key]
        if isinstance(target, str):
            output[target] = weight
            continue

        target_key, offset, transform = target
        weight = transform(weight)
        if offset is None:
            output[target_key] = weight
            continue

        old_weight = output.get(target_key)
        if old_weight is None:
            shape = list(weight.shape)
            shape[offset[0]] = offset[1] + offset[2]
            old_weight = torch.zeros(shape, dtype=weight.dtype, device=weight.device)
        elif old_weight.shape[offset[0]] < offset[1] + offset[2]:
            shape = list(old_weight.shape)
            shape[offset[0]] = offset[1] + offset[2]
            expanded = torch.zeros(shape, dtype=old_weight.dtype, device=old_weight.device)
            slices = [slice(None)] * old_weight.ndim
            slices[offset[0]] = slice(0, old_weight.shape[offset[0]])
            expanded[tuple(slices)] = old_weight
            old_weight = expanded

        old_weight.narrow(offset[0], offset[1], offset[2]).copy_(weight)
        output[target_key] = old_weight
    return output


def _flux2_diffusers_to_comfy_key_map(
    *,
    double_blocks: int,
    single_blocks: int,
    hidden_size: int,
    output_prefix: str,
):
    identity = lambda tensor: tensor
    key_map = {}

    for index in range(double_blocks):
        source = f"transformer_blocks.{index}"
        target = f"{output_prefix}double_blocks.{index}"
        for suffix in ("weight",):
            img_qkv = f"{target}.img_attn.qkv.{suffix}"
            key_map[f"{source}.attn.to_q.{suffix}"] = (img_qkv, (0, 0, hidden_size), identity)
            key_map[f"{source}.attn.to_k.{suffix}"] = (img_qkv, (0, hidden_size, hidden_size), identity)
            key_map[f"{source}.attn.to_v.{suffix}"] = (img_qkv, (0, hidden_size * 2, hidden_size), identity)

            txt_qkv = f"{target}.txt_attn.qkv.{suffix}"
            key_map[f"{source}.attn.add_q_proj.{suffix}"] = (txt_qkv, (0, 0, hidden_size), identity)
            key_map[f"{source}.attn.add_k_proj.{suffix}"] = (txt_qkv, (0, hidden_size, hidden_size), identity)
            key_map[f"{source}.attn.add_v_proj.{suffix}"] = (txt_qkv, (0, hidden_size * 2, hidden_size), identity)

        block_map = {
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "ff.linear_in.weight": "img_mlp.0.weight",
            "ff.linear_out.weight": "img_mlp.2.weight",
            "ff_context.linear_in.weight": "txt_mlp.0.weight",
            "ff_context.linear_out.weight": "txt_mlp.2.weight",
            "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
            "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
        }
        for source_suffix, target_suffix in block_map.items():
            key_map[f"{source}.{source_suffix}"] = f"{target}.{target_suffix}"

    for index in range(single_blocks):
        source = f"single_transformer_blocks.{index}"
        target = f"{output_prefix}single_blocks.{index}"
        block_map = {
            "attn.to_qkv_mlp_proj.weight": "linear1.weight",
            "attn.to_out.weight": "linear2.weight",
            "attn.norm_q.weight": "norm.query_norm.scale",
            "attn.norm_k.weight": "norm.key_norm.scale",
        }
        for source_suffix, target_suffix in block_map.items():
            key_map[f"{source}.{source_suffix}"] = f"{target}.{target_suffix}"

    basic_map = {
        "x_embedder.weight": "img_in.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        "context_embedder.weight": "txt_in.weight",
        "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
        "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
        "proj_out.weight": "final_layer.linear.weight",
    }
    for source_key, target_key in basic_map.items():
        key_map[source_key] = f"{output_prefix}{target_key}"
    key_map["norm_out.linear.weight"] = (
        f"{output_prefix}final_layer.adaLN_modulation.1.weight",
        None,
        _swap_scale_shift,
    )
    return key_map


def _fit_last_dim(tensor: torch.Tensor, width: int) -> torch.Tensor:
    if tensor.shape[-1] == width:
        return tensor
    if tensor.shape[-1] > width:
        return tensor[..., :width]
    return F.pad(tensor, (0, width - tensor.shape[-1]))




def _get_qwen_token_key(tokens: dict) -> str:
    if "qwen3_4b" in tokens:
        return "qwen3_4b"
    if "qwen3_8b" in tokens:
        return "qwen3_8b"
    if len(tokens) != 1:
        raise ValueError(f"无法识别 Qwen token key：{list(tokens)}")
    return next(iter(tokens))


def _qwen_ids_and_mask_from_comfy_tokens(tokens: dict, *, max_sequence_length: int, pad_token: int):
    key = _get_qwen_token_key(tokens)
    rows = tokens[key]
    if not rows:
        raise ValueError("Qwen tokenizer 没有生成任何 token。")

    ids = []
    for item in rows[0]:
        if isinstance(item, tuple):
            ids.append(int(item[0]))
        else:
            ids.append(int(item))

    max_sequence_length = int(max_sequence_length)
    if len(ids) > max_sequence_length:
        ids = ids[:max_sequence_length]
    elif len(ids) < max_sequence_length:
        ids.extend([int(pad_token)] * (max_sequence_length - len(ids)))

    attention_mask = [0 if token == int(pad_token) else 1 for token in ids]
    return ids, attention_mask


def _exact_qwen_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    bias = getattr(module, "bias", None)
    weight = module.weight.to(device=x.device, dtype=x.dtype)
    bias = None if bias is None else bias.to(device=x.device, dtype=x.dtype)
    return F.linear(x, weight, bias)


def _exact_qwen_rms_norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    y = x.float()
    variance = y.pow(2).mean(-1, keepdim=True)
    y = y * torch.rsqrt(variance + float(module.eps))
    weight = module.weight.to(device=x.device, dtype=input_dtype)
    if getattr(module, "add", False):
        weight = weight + 1.0
    return weight * y.to(input_dtype)


def _exact_qwen_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _exact_qwen_apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_exact_qwen_rotate_half(q) * sin)
    k_embed = (k * cos) + (_exact_qwen_rotate_half(k) * sin)
    return q_embed, k_embed


def _exact_qwen_repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, kv_heads * repeats, seq_len, head_dim)


def _exact_qwen_causal_mask(attention_mask: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    batch, seq_len = attention_mask.shape
    causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    causal = causal[None, None, :, :].expand(batch, 1, seq_len, seq_len)
    padding = attention_mask[:, None, None, :].to(device=device, dtype=torch.bool).expand(batch, 1, seq_len, seq_len)
    return causal & padding


def _exact_qwen_rope_cos_sin(model, hidden: torch.Tensor, position_ids: torch.Tensor):
    config = model.config
    head_dim = int(config.head_dim)
    theta = float(config.rope_theta)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
    )
    inv_freq = inv_freq.to(device=hidden.device)
    inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids = position_ids[:, None, :].float()

    device_type = hidden.device.type if isinstance(hidden.device.type, str) and hidden.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq.float() @ position_ids.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=hidden.dtype), sin.to(dtype=hidden.dtype)


@torch.inference_mode()
def encode_comfy_qwen3_hf_semantics(
    qwen_clip,
    text: str,
    *,
    max_sequence_length: int,
    hidden_states_layers: Iterable[int],
    dtype: torch.dtype = torch.bfloat16,
):
    layers = tuple(int(layer) for layer in hidden_states_layers)
    if len(layers) != 3:
        raise ValueError("RUM Qwen layers 必须正好包含 3 个层号。")

    cond_stage_model = getattr(qwen_clip, "cond_stage_model", None)
    clip_attr = getattr(cond_stage_model, "clip", None)
    text_model = getattr(cond_stage_model, clip_attr, None) if clip_attr else None
    transformer = getattr(text_model, "transformer", None)
    model = getattr(transformer, "model", None)
    if model is None or not hasattr(model, "layers"):
        raise ValueError("当前 CLIP 不是 ComfyUI FLUX.2/Klein Qwen text encoder。")

    pad_token = int(getattr(text_model, "special_tokens", {}).get("pad", 151643))
    tokens = qwen_clip.tokenize(text, min_length=int(max_sequence_length))
    ids, mask = _qwen_ids_and_mask_from_comfy_tokens(
        tokens,
        max_sequence_length=int(max_sequence_length),
        pad_token=pad_token,
    )

    qwen_clip.load_model(tokens)
    device = qwen_clip.patcher.load_device
    ids_tensor = torch.tensor([ids], device=device, dtype=torch.long)
    attention_mask = torch.tensor([mask], device=device, dtype=torch.long)

    hidden = model.embed_tokens(ids_tensor, out_dtype=dtype)
    seq_len = hidden.shape[1]
    cache_position = torch.arange(0, seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)
    cos, sin = _exact_qwen_rope_cos_sin(model, hidden, position_ids)
    causal_mask = _exact_qwen_causal_mask(attention_mask, dtype=hidden.dtype, device=device)

    layer_set = set(layers)
    collected_by_layer: dict[int, torch.Tensor] = {}
    for index, layer in enumerate(model.layers):
        if index in layer_set:
            collected_by_layer[index] = hidden.clone()

        residual = hidden
        hidden = _exact_qwen_rms_norm(layer.input_layernorm, hidden)

        attn = layer.self_attn
        input_shape = hidden.shape[:-1]
        query = _exact_qwen_linear(attn.q_proj, hidden).view(*input_shape, -1, attn.head_dim)
        key = _exact_qwen_linear(attn.k_proj, hidden).view(*input_shape, -1, attn.head_dim)
        value = _exact_qwen_linear(attn.v_proj, hidden).view(*input_shape, -1, attn.head_dim)

        query = _exact_qwen_rms_norm(attn.q_norm, query).transpose(1, 2)
        key = _exact_qwen_rms_norm(attn.k_norm, key).transpose(1, 2)
        value = value.transpose(1, 2)

        query, key = _exact_qwen_apply_rope(query, key, cos, sin)
        repeats = int(attn.num_heads // attn.num_kv_heads)
        key = _exact_qwen_repeat_kv(key, repeats)
        value = _exact_qwen_repeat_kv(value, repeats)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.head_dim**-0.5,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        hidden = residual + _exact_qwen_linear(attn.o_proj, attn_output)

        residual = hidden
        hidden = _exact_qwen_rms_norm(layer.post_attention_layernorm, hidden)
        mlp = layer.mlp
        hidden = _exact_qwen_linear(
            mlp.down_proj,
            F.silu(_exact_qwen_linear(mlp.gate_proj, hidden)) * _exact_qwen_linear(mlp.up_proj, hidden),
        )
        hidden = residual + hidden

    if len(model.layers) in layer_set:
        collected_by_layer[len(model.layers)] = _exact_qwen_rms_norm(model.norm, hidden)

    missing = [layer for layer in layers if layer not in collected_by_layer]
    if missing:
        raise ValueError(f"Qwen 层号超出范围：{layers}；当前模型层数={len(model.layers)}。")

    stacked = torch.stack([collected_by_layer[layer] for layer in layers], dim=1)
    batch_size, layer_count, seq_len, hidden_dim = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, layer_count * hidden_dim)
    return [[prompt_embeds.to(device="cpu", dtype=torch.float32), {}]]




def _exact_clip_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    bias = getattr(module, "bias", None)
    weight = module.weight.to(device=x.device, dtype=x.dtype)
    bias = None if bias is None else bias.to(device=x.device, dtype=x.dtype)
    return F.linear(x, weight, bias)


def _exact_clip_layer_norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        x,
        module.normalized_shape,
        module.weight.to(device=x.device, dtype=x.dtype),
        module.bias.to(device=x.device, dtype=x.dtype),
        module.eps,
    )


def _exact_clip_causal_mask(seq_len: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device).triu_(1)[None, None, :, :]


def _exact_clip_eager_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor, scale: float) -> torch.Tensor:
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * float(scale)
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn_weights, value).transpose(1, 2).contiguous()


def _clip_ids_from_comfy_tokens(token_rows, *, max_sequence_length: int) -> list[int]:
    if not token_rows:
        raise ValueError("SDXL tokenizer 没有生成任何 token。")
    ids = [int(item[0]) if isinstance(item, tuple) else int(item) for item in token_rows[0]]
    max_sequence_length = int(max_sequence_length)
    if len(ids) > max_sequence_length:
        ids = ids[:max_sequence_length]
    return ids


def _encode_clip_text_hf_semantics(text_model, input_ids: torch.Tensor, *, layer_index: int, dtype: torch.dtype):
    transformer = text_model.transformer
    model = transformer.text_model
    input_ids = input_ids.to(transformer.get_input_embeddings().weight.device)
    hidden = model.embeddings.token_embedding(input_ids, out_dtype=dtype)
    hidden = hidden + model.embeddings.position_embedding.weight.to(device=input_ids.device, dtype=dtype)
    causal_mask = _exact_clip_causal_mask(hidden.shape[1], dtype=hidden.dtype, device=hidden.device)

    collected = None
    for index, layer in enumerate(model.encoder.layers):
        residual = hidden
        normed = _exact_clip_layer_norm(layer.layer_norm1, hidden)
        attn = layer.self_attn
        batch, seq_len, channels = normed.shape
        query = _exact_clip_linear(attn.q_proj, normed).view(batch, seq_len, attn.heads, -1).transpose(1, 2)
        key = _exact_clip_linear(attn.k_proj, normed).view(batch, seq_len, attn.heads, -1).transpose(1, 2)
        value = _exact_clip_linear(attn.v_proj, normed).view(batch, seq_len, attn.heads, -1).transpose(1, 2)
        attn_output = _exact_clip_eager_attention(
            query,
            key,
            value,
            causal_mask,
            query.shape[-1] ** -0.5,
        ).reshape(batch, seq_len, channels)
        hidden = residual + _exact_clip_linear(attn.out_proj, attn_output)

        residual = hidden
        hidden = _exact_clip_layer_norm(layer.layer_norm2, hidden)
        hidden = layer.mlp.activation(_exact_clip_linear(layer.mlp.fc1, hidden))
        hidden = _exact_clip_linear(layer.mlp.fc2, hidden)
        hidden = residual + hidden

        if index == layer_index:
            collected = hidden.clone()

    if collected is None:
        raise ValueError(f"SDXL CLIP layer_index 超出范围：{layer_index}。")
    return collected


_SDXL_TEACHER_CACHE = {}
_SDXL_TEACHER_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def validate_comfy_reference_image_batch(images: torch.Tensor) -> None:
    if not torch.is_tensor(images):
        raise ValueError(f"RUM reference encode 需要 ComfyUI IMAGE tensor [B,H,W,C]，当前类型是 {type(images)}。")
    if images.ndim != 4:
        raise ValueError(f"RUM reference encode 需要 ComfyUI IMAGE tensor [B,H,W,C]，当前形状是 {tuple(images.shape)}。")
    if images.shape[0] <= 0:
        raise ValueError("RUM reference encode 需要至少 1 张图片。")
    if images.shape[1] <= 0 or images.shape[2] <= 0:
        raise ValueError(f"RUM reference encode 图片高宽必须大于 0，当前形状是 {tuple(images.shape)}。")
    if images.shape[-1] < 3:
        raise ValueError(f"RUM reference encode 需要 RGB 图片 [B,H,W,C>=3]，当前 channel={images.shape[-1]}。")
    if not torch.is_floating_point(images):
        raise ValueError(f"RUM reference encode 需要浮点 IMAGE tensor，当前 dtype={images.dtype}。")


def validate_rum_reference_latents(reference_latents) -> list[torch.Tensor]:
    if reference_latents is None:
        raise ValueError("RUM reference_latents 不能为空。")
    if isinstance(reference_latents, dict):
        latents = reference_latents.get("latents")
    else:
        latents = reference_latents
    if not isinstance(latents, list):
        raise ValueError(f"RUM reference_latents 需要包含 latents list，当前类型是 {type(latents)}。")
    if len(latents) == 0:
        raise ValueError("RUM reference_latents list 不能为空。")

    for index, latent in enumerate(latents):
        if not torch.is_tensor(latent):
            raise ValueError(f"RUM reference latent #{index} 必须是 tensor，当前类型是 {type(latent)}。")
        if latent.ndim != 4:
            raise ValueError(f"RUM reference latent #{index} 必须是 4D tensor，当前形状是 {tuple(latent.shape)}。")
        if latent.shape[0] != 1:
            raise ValueError(f"RUM reference latent #{index} 单项 batch 必须为 1，当前形状是 {tuple(latent.shape)}。")
        if latent.shape[1] <= 0 or latent.shape[2] <= 0 or latent.shape[3] <= 0:
            raise ValueError(f"RUM reference latent #{index} 维度必须大于 0，当前形状是 {tuple(latent.shape)}。")
    return latents


def validate_sdxl_teacher_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _SDXL_TEACHER_DTYPES:
        raise ValueError(f"RUM SDXL teacher exact path 只支持 FP16/BF16/FP32，当前 dtype={dtype}。")
    return dtype


def _resolve_comfy_model_dir() -> Path:
    try:
        import folder_paths
    except Exception as exc:
        raise ValueError("无法导入 ComfyUI folder_paths，不能定位 SDXL teacher HF 目录。") from exc

    model_paths = getattr(folder_paths, "folder_names_and_paths", {}).get("text_encoders")
    if not model_paths:
        model_paths = getattr(folder_paths, "folder_names_and_paths", {}).get("clip")
    if not model_paths:
        raise ValueError("ComfyUI folder_paths 没有注册 text_encoders/clip 目录，不能定位 SDXL teacher HF 目录。")

    directories = model_paths[0]
    if not directories:
        raise ValueError("ComfyUI text_encoders/clip 搜索目录为空，不能定位 SDXL teacher HF 目录。")
    return Path(directories[0])


def _find_sdxl_teacher_dirs(text_encoder_dir: Path) -> tuple[Path, Path]:
    l_dirs = sorted(path for path in text_encoder_dir.glob("*_clip_l_dir") if (path / "model.safetensors").is_file())
    g_dirs = sorted(path for path in text_encoder_dir.glob("*_clip_g_dir") if (path / "model.safetensors").is_file())
    pairs = []
    g_by_prefix = {path.name.removesuffix("_clip_g_dir"): path for path in g_dirs}
    for l_dir in l_dirs:
        prefix = l_dir.name.removesuffix("_clip_l_dir")
        g_dir = g_by_prefix.get(prefix)
        if g_dir is not None:
            pairs.append((l_dir, g_dir))

    if len(pairs) == 1:
        return pairs[0]
    if not pairs:
        raise FileNotFoundError(
            "找不到 SDXL teacher HF 目录。请在 ComfyUI models/text_encoders 下准备 "
            "*_clip_l_dir/model.safetensors 和 *_clip_g_dir/model.safetensors。"
        )
    names = ", ".join(f"{left.name} + {right.name}" for left, right in pairs)
    raise ValueError(f"找到多个 SDXL teacher HF 目录，无法确定使用哪一组：{names}")


def _load_sdxl_teacher_hf_pair(device: torch.device, dtype: torch.dtype):
    text_encoder_dir = _resolve_comfy_model_dir()
    l_dir, g_dir = _find_sdxl_teacher_dirs(text_encoder_dir)
    cache_key = (str(l_dir), str(g_dir), str(device), str(dtype))
    cached = _SDXL_TEACHER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    except Exception as exc:
        raise ValueError("RUM SDXL teacher exact path 需要 transformers 可用。") from exc

    tokenizer_dir = Path(__file__).resolve().parents[2] / "comfy" / "sd1_tokenizer"
    if not tokenizer_dir.is_dir():
        tokenizer_dir = text_encoder_dir.parent.parent / "comfy" / "sd1_tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"找不到 ComfyUI SD1 tokenizer 目录：{tokenizer_dir}")

    clip_l = CLIPTextModel.from_pretrained(str(l_dir), torch_dtype=dtype).to(device).eval()
    clip_g = CLIPTextModelWithProjection.from_pretrained(str(g_dir), torch_dtype=dtype).to(device).eval()
    tokenizer_l = CLIPTokenizer.from_pretrained(str(tokenizer_dir))
    tokenizer_g = CLIPTokenizer.from_pretrained(str(tokenizer_dir), pad_token="!")
    cached = (clip_l, clip_g, tokenizer_l, tokenizer_g)
    _SDXL_TEACHER_CACHE[cache_key] = cached
    return cached


@torch.inference_mode()
def _encode_sdxl_teacher_hf_exact(text: str, *, device: torch.device, dtype: torch.dtype):
    dtype = validate_sdxl_teacher_dtype(dtype)
    clip_l, clip_g, tokenizer_l, tokenizer_g = _load_sdxl_teacher_hf_pair(device, dtype)
    embeddings = []
    for tokenizer, model in ((tokenizer_l, clip_l), (tokenizer_g, clip_g)):
        inputs = tokenizer(
            [text],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        output = model(inputs.input_ids.to(device), output_hidden_states=True)
        embeddings.append(output.hidden_states[-2])
    return torch.cat(embeddings, dim=-1)


@torch.inference_mode()
def encode_comfy_sdxl_hf_semantics(sdxl_clip, text: str, *, dtype: torch.dtype = torch.float16):
    cond_stage_model = getattr(sdxl_clip, "cond_stage_model", None)
    clip_l = getattr(cond_stage_model, "clip_l", None)
    clip_g = getattr(cond_stage_model, "clip_g", None)
    if clip_l is None or clip_g is None:
        raise ValueError("当前 CLIP 不是 ComfyUI SDXL DualCLIP。")

    if not torch.cuda.is_available():
        raise RuntimeError("RUM SDXL teacher exact path 需要 CUDA；像素级对齐验证不能在 CPU CLIP/VAE 上执行。")
    device = torch.device("cuda")
    exact = _encode_sdxl_teacher_hf_exact(text, device=device, dtype=dtype)
    return [[exact.to(device="cpu", dtype=torch.float32), {}]]

    tokens = sdxl_clip.tokenize(text, disable_weights=True)
    if "l" not in tokens or "g" not in tokens:
        raise ValueError(f"无法识别 SDXL token key：{list(tokens)}")

    ids_l = _clip_ids_from_comfy_tokens(tokens["l"], max_sequence_length=77)
    ids_g = _clip_ids_from_comfy_tokens(tokens["g"], max_sequence_length=77)
    if len(ids_l) != len(ids_g):
        cut_to = min(len(ids_l), len(ids_g))
        ids_l = ids_l[:cut_to]
        ids_g = ids_g[:cut_to]

    sdxl_clip.load_model(tokens)
    device = sdxl_clip.patcher.load_device
    ids_l_tensor = torch.tensor([ids_l], device=device, dtype=torch.long)
    ids_g_tensor = torch.tensor([ids_g], device=device, dtype=torch.long)
    out_l = _encode_clip_text_hf_semantics(clip_l, ids_l_tensor, layer_index=clip_l.num_layers - 2, dtype=dtype)
    out_g = _encode_clip_text_hf_semantics(clip_g, ids_g_tensor, layer_index=clip_g.num_layers - 2, dtype=dtype)
    cut_to = min(out_l.shape[1], out_g.shape[1])
    output = torch.cat([out_l[:, :cut_to], out_g[:, :cut_to]], dim=-1)
    return [[output.to(device="cpu", dtype=torch.float32), {}]]


def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _count_blocks(keys: Iterable[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indexes = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return max(indexes) + 1 if indexes else 0


def _infer_hidden_size(state_dict: dict[str, torch.Tensor]) -> int:
    if "x_embedder.weight" in state_dict:
        return int(state_dict["x_embedder.weight"].shape[0])
    if "context_embedder.weight" in state_dict:
        return int(state_dict["context_embedder.weight"].shape[0])
    raise ValueError("无法从 RUM checkpoint 推断 hidden_size。")


def _require_key(state_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state_dict:
        raise ValueError(f"RUM checkpoint 缺少必要权重：{key}")
    return state_dict[key]


def _resolve_checkpoint_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"找不到 RUM checkpoint：{raw_path}")
    candidates = sorted(path.rglob("*.safetensors"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"{raw_path} 不是明确的 safetensors 文件，请直接选择/填写 RUM checkpoint 文件。")

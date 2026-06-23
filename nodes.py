from __future__ import annotations

from pathlib import Path

def _diffusion_model_list():
    import folder_paths

    names = folder_paths.get_filename_list("diffusion_models")
    return names if names else ["put_rum_checkpoint_in_models_diffusion_models.safetensors"]


def _validate_rum_checkpoint_override_path(rum_checkpoint_path: str) -> str:
    raw_path = rum_checkpoint_path.strip().strip("\"'")
    checkpoint = Path(raw_path).expanduser()
    if not checkpoint.is_absolute():
        raise ValueError(f"RUM checkpoint override path 必须是绝对路径，当前值：{rum_checkpoint_path!r}。")
    try:
        resolved = checkpoint.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"RUM checkpoint override path 不存在或无法访问：{rum_checkpoint_path!r}。") from exc
    if not resolved.is_file():
        raise ValueError(f"RUM checkpoint override path 必须指向文件：{str(resolved)!r}。")
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError(f"RUM checkpoint override path 必须是 .safetensors 文件：{str(resolved)!r}。")
    return str(resolved)


def _resolve_rum_checkpoint_path(rum_checkpoint_name: str, rum_checkpoint_path: str) -> str:
    import folder_paths

    checkpoint = rum_checkpoint_path.strip()
    if checkpoint:
        return _validate_rum_checkpoint_override_path(checkpoint)
    return folder_paths.get_full_path_or_raise("diffusion_models", rum_checkpoint_name)


class RUMFlux2LoadNativeModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rum_checkpoint_name": (_diffusion_model_list(),),
                "base_text_tokens": ("INT", {"default": 200, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "rum_checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：填写现有 .safetensors 文件的绝对路径时会覆盖 rum_checkpoint_name。推荐直接选择 BF16 RUM checkpoint。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load_model"
    CATEGORY = "RUM/native"

    def load_model(self, rum_checkpoint_name: str, base_text_tokens: int, rum_checkpoint_path: str = ""):
        from .rum_native import load_rum_native_model

        checkpoint = _resolve_rum_checkpoint_path(rum_checkpoint_name, rum_checkpoint_path)
        model, count, resolved = load_rum_native_model(checkpoint, base_text_tokens=base_text_tokens)
        return (model, f"RUM native BF16 MODEL 已加载：{count} 个权重；base={base_text_tokens}；checkpoint={resolved}")


class RUMFlux2ApplyModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rum_checkpoint_name": (_diffusion_model_list(),),
                "base_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "strict": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "rum_checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：填写现有 .safetensors 文件的绝对路径时会覆盖 rum_checkpoint_name。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply_patch"
    CATEGORY = "RUM/native"

    def apply_patch(self, model, rum_checkpoint_name: str, base_text_tokens: int, strict: bool, rum_checkpoint_path: str = ""):
        from .rum_native import apply_rum_model_patch

        checkpoint = _resolve_rum_checkpoint_path(rum_checkpoint_name, rum_checkpoint_path)
        patched, count, resolved = apply_rum_model_patch(
            model,
            checkpoint,
            base_text_tokens=base_text_tokens,
            strict=strict,
        )
        return (patched, f"RUM native patch 已套用：{count} 个权重；checkpoint={resolved}")


class RUMFlux2CombineConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux2_conditioning": ("CONDITIONING",),
                "sdxl_conditioning": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "use_guidance_embedding": ("BOOLEAN", {"default": False}),
                "base_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "sdxl_clip_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "use_sdxl_extra": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "RUM/native"

    def combine(
        self,
        flux2_conditioning,
        sdxl_conditioning,
        guidance: float,
        use_guidance_embedding: bool,
        base_text_tokens: int,
        extra_text_tokens: int,
        sdxl_clip_width: int,
        use_sdxl_extra: bool,
    ):
        from .rum_native import combine_rum_conditioning

        return (
            combine_rum_conditioning(
                flux2_conditioning,
                sdxl_conditioning,
                guidance=guidance if use_guidance_embedding else None,
                base_text_tokens=base_text_tokens,
                extra_text_tokens=extra_text_tokens,
                sdxl_clip_width=sdxl_clip_width,
                use_sdxl_extra=use_sdxl_extra,
            ),
        )



class RUMSDXLDiffusersTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": "1girl, kisaki (blue archive), eating baozi, sitting, indoors"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "RUM/native"

    def encode(self, clip, text: str):
        tokens = clip.tokenize(text, disable_weights=True)
        return (clip.encode_from_tokens_scheduled(tokens),)


def _parse_layers(layers: str) -> list[int]:
    parsed = []
    for part in layers.split(","):
        part = part.strip()
        if part:
            try:
                parsed.append(int(part))
            except ValueError as exc:
                raise ValueError("RUM Qwen layers 只能填写整数，例如 10,20,30。") from exc
    if len(parsed) != 3:
        raise ValueError("RUM Qwen layers 必须正好是 3 个整数，例如 10,20,30。")
    return parsed


def _resolve_qwen_text_model(clip):
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    clip_attr = getattr(cond_stage_model, "clip", None)
    text_model = getattr(cond_stage_model, clip_attr, None) if clip_attr else None
    if text_model is None or not hasattr(text_model, "layer"):
        raise ValueError("当前 CLIP 不是 FLUX.2/Klein Qwen text encoder，不能设置 Qwen 层号。")
    return text_model


def _encode_qwen_with_layers(clip, text: str, *, min_length: int, layers: str):
    parsed = _parse_layers(layers)
    text_model = _resolve_qwen_text_model(clip)
    old_layer = text_model.layer
    old_layer_idx = text_model.layer_idx
    old_options_default = getattr(text_model, "options_default", None)
    old_clip_layer_idx = getattr(clip, "layer_idx", None)

    try:
        projected_pooled = old_options_default[2] if old_options_default is not None and len(old_options_default) > 2 else True
        text_model.layer = parsed
        text_model.layer_idx = None
        if old_options_default is not None:
            text_model.options_default = (parsed, None, projected_pooled)
        clip.layer_idx = None
        tokens = clip.tokenize(text, min_length=min_length)
        return clip.encode_from_tokens_scheduled(tokens), parsed
    finally:
        text_model.layer = old_layer
        text_model.layer_idx = old_layer_idx
        if old_options_default is not None:
            text_model.options_default = old_options_default
        clip.layer_idx = old_clip_layer_idx


def _clone_qwen_clip_with_layers(clip, layers: str):
    parsed = _parse_layers(layers)
    patched = clip.clone()
    _resolve_qwen_text_model(patched)
    patched.clip_layer(parsed)
    return patched, parsed


class RUMFlux2NativeMatchTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_clip": ("CLIP",),
                "sdxl_clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": "1girl, kisaki (blue archive), eating baozi, sitting, indoors"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "use_guidance_embedding": ("BOOLEAN", {"default": False}),
                "base_text_tokens": ("INT", {"default": 200, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "sdxl_clip_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "qwen_layers": ("STRING", {"default": "10,20,30", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "status")
    FUNCTION = "encode"
    CATEGORY = "RUM/native"

    @classmethod
    def IS_CHANGED(cls, qwen_clip, sdxl_clip, prompt, negative_prompt, guidance,
                   use_guidance_embedding, base_text_tokens, extra_text_tokens,
                   sdxl_clip_width, qwen_layers):
        return hash((prompt, negative_prompt, guidance, use_guidance_embedding,
                     base_text_tokens, extra_text_tokens, qwen_layers, sdxl_clip_width))

    def _encode(
        self,
        qwen_clip,
        sdxl_clip,
        prompt: str,
        guidance: float,
        use_guidance_embedding: bool,
        base_text_tokens: int,
        extra_text_tokens: int,
        sdxl_clip_width: int,
        qwen_layers: str,
    ):
        from .rum_native import combine_rum_conditioning, encode_comfy_qwen3_hf_semantics, encode_comfy_sdxl_hf_semantics

        positive_qwen = encode_comfy_qwen3_hf_semantics(
            qwen_clip,
            prompt,
            max_sequence_length=base_text_tokens,
            hidden_states_layers=_parse_layers(qwen_layers),
        )

        positive_sdxl = encode_comfy_sdxl_hf_semantics(sdxl_clip, prompt)

        emb = combine_rum_conditioning(
            positive_qwen,
            positive_sdxl,
            guidance=guidance if use_guidance_embedding else None,
            base_text_tokens=base_text_tokens,
            extra_text_tokens=extra_text_tokens,
            sdxl_clip_width=sdxl_clip_width,
            use_sdxl_extra=True,
        )
        return emb

    def encode(
        self,
        qwen_clip,
        sdxl_clip,
        prompt: str,
        negative_prompt: str,
        guidance: float,
        use_guidance_embedding: bool,
        base_text_tokens: int,
        extra_text_tokens: int,
        sdxl_clip_width: int,
        qwen_layers: str,
    ):
        positive = self._encode(qwen_clip=qwen_clip, sdxl_clip=sdxl_clip, prompt=prompt, guidance=guidance, use_guidance_embedding=use_guidance_embedding, base_text_tokens=base_text_tokens, extra_text_tokens=extra_text_tokens, qwen_layers=qwen_layers, sdxl_clip_width=sdxl_clip_width)
        negative = self._encode(qwen_clip=qwen_clip, sdxl_clip=sdxl_clip, prompt=negative_prompt, guidance=guidance, use_guidance_embedding=use_guidance_embedding, base_text_tokens=base_text_tokens, extra_text_tokens=extra_text_tokens, qwen_layers=qwen_layers, sdxl_clip_width=sdxl_clip_width)
        return (
            positive,
            negative,
            "Native match text 已编码："
            f"pos={base_text_tokens}+{extra_text_tokens}; "
            f"neg={base_text_tokens}+{extra_text_tokens}; ",
        )

class RUMFlux2SetQwenLayers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "layers": ("STRING", {"default": "10,20,30", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "status")
    FUNCTION = "set_layers"
    CATEGORY = "RUM/native"

    @classmethod
    def IS_CHANGED(cls, clip, layers: str):
        return hash(str(layers))

    def set_layers(self, clip, layers: str):
        patched, parsed = _clone_qwen_clip_with_layers(clip, layers)
        return (patched, f"RUM Qwen layers 已设置：{','.join(str(x) for x in parsed)}")


class RUMFlux2DiffusersNoiseSource:
    def __init__(self, seed: int, dtype: str):
        self.seed = int(seed)
        self.dtype = dtype

    def generate_noise(self, input_latent):
        import torch
        import comfy.nested_tensor

        latent_image = input_latent["samples"]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        noise_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        def make_noise(shape, output_dtype, layout):
            noise = torch.randn(shape, generator=generator, device="cpu", dtype=noise_dtype, layout=layout)
            return noise.to(dtype=output_dtype)

        if latent_image.is_nested:
            return comfy.nested_tensor.NestedTensor(
                [make_noise(tensor.shape, tensor.dtype, tensor.layout) for tensor in latent_image.unbind()]
            )
        return make_noise(latent_image.shape, latent_image.dtype, latent_image.layout)


class RUMFlux2DiffusersNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "get_noise"
    CATEGORY = "RUM/native"

    def get_noise(self, noise_seed: int, dtype: str):
        return (RUMFlux2DiffusersNoiseSource(noise_seed, dtype),)


class RUMFlux2DiffusersEulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "RUM/native"

    def get_sampler(self):
        from .rum_native import create_diffusers_euler_sampler

        return (create_diffusers_euler_sampler(),)


class RUMFlux2DiffusersScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 960, "min": 16, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "RUM/native"

    def get_sigmas(self, steps: int, width: int, height: int):
        from .rum_native import diffusers_flux2_sigmas

        return (diffusers_flux2_sigmas(steps=steps, width=width, height=height),)


class RUMRoundImageForSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "round_image"
    CATEGORY = "RUM/native"

    def round_image(self, images):
        import torch

        return (torch.round(images.clamp(0.0, 1.0) * 255.0).clamp(0.0, 255.0) / 255.0,)


class RUMFlux2NativeMatchVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "RUM/native"

    def decode(self, samples, vae):
        from .rum_native import decode_flux2_native_match_vae_latent

        return (decode_flux2_native_match_vae_latent(vae, samples),)


class RUMFlux2NativeMatchReferenceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("RUM_REFERENCE_LATENTS", "STRING")
    RETURN_NAMES = ("reference_latents", "status")
    FUNCTION = "encode"
    CATEGORY = "RUM/native"

    def encode(self, image, vae):
        from .rum_native import encode_flux2_native_match_reference_image

        return encode_flux2_native_match_reference_image(vae, image)


class RUMFlux2DiffusersCFGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "reference_latents": ("RUM_REFERENCE_LATENTS",),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "RUM/native"

    def get_guider(self, model, positive, negative, cfg: float, reference_latents=None):
        from .rum_native import create_diffusers_cfg_guider

        return (create_diffusers_cfg_guider(model, positive, negative, cfg, reference_latents),)


class RUMFlux2DiffusersMatchModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_text_tokens": ("INT", {"default": 200, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "disable_guidance": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply_patch"
    CATEGORY = "RUM/native"

    def apply_patch(self, model, base_text_tokens: int, extra_text_tokens: int, disable_guidance: bool):
        from .rum_native import apply_diffusers_match_model_wrapper

        patched = apply_diffusers_match_model_wrapper(
            model,
            base_text_tokens=base_text_tokens,
            extra_text_tokens=extra_text_tokens,
            disable_guidance=disable_guidance,
        )
        guidance_status = "disabled" if disable_guidance else "unchanged"
        return (
            patched,
            f"Diffusers-match token policy 已启用：base={base_text_tokens}, extra={extra_text_tokens}, guidance={guidance_status}",
        )


NODE_CLASS_MAPPINGS = {
    "RUMFlux2LoadNativeModel": RUMFlux2LoadNativeModel,
    "RUMFlux2ApplyModelPatch": RUMFlux2ApplyModelPatch,
    "RUMFlux2CombineConditioning": RUMFlux2CombineConditioning,
    "RUMSDXLDiffusersTextEncode": RUMSDXLDiffusersTextEncode,
    "RUMFlux2NativeMatchTextEncode": RUMFlux2NativeMatchTextEncode,
    "RUMFlux2SetQwenLayers": RUMFlux2SetQwenLayers,
    "RUMFlux2DiffusersMatchModelPatch": RUMFlux2DiffusersMatchModelPatch,
    "RUMFlux2DiffusersNoise": RUMFlux2DiffusersNoise,
    "RUMFlux2DiffusersEulerSampler": RUMFlux2DiffusersEulerSampler,
    "RUMFlux2DiffusersScheduler": RUMFlux2DiffusersScheduler,
    "RUMRoundImageForSave": RUMRoundImageForSave,
    "RUMFlux2DiffusersCFGuider": RUMFlux2DiffusersCFGuider,
    "RUMFlux2NativeMatchVAEDecode": RUMFlux2NativeMatchVAEDecode,
    "RUMFlux2NativeMatchReferenceEncode": RUMFlux2NativeMatchReferenceEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RUMFlux2LoadNativeModel": "RUM FLUX.2 Load Native Model",
    "RUMFlux2ApplyModelPatch": "RUM FLUX.2 Apply Model Patch",
    "RUMFlux2CombineConditioning": "RUM FLUX.2 Combine Conditioning",
    "RUMSDXLDiffusersTextEncode": "RUM SDXL Diffusers Text Encode",
    "RUMFlux2NativeMatchTextEncode": "RUM FLUX.2 Native Match Text Encode",
    "RUMFlux2SetQwenLayers": "RUM FLUX.2 Set Qwen Layers",
    "RUMFlux2DiffusersMatchModelPatch": "RUM FLUX.2 Diffusers Match Model Patch",
    "RUMFlux2DiffusersNoise": "RUM FLUX.2 Diffusers Noise",
    "RUMFlux2DiffusersEulerSampler": "RUM FLUX.2 Diffusers Euler Sampler",
    "RUMFlux2DiffusersScheduler": "RUM FLUX.2 Diffusers Scheduler",
    "RUMRoundImageForSave": "RUM Round Image For Save",
    "RUMFlux2DiffusersCFGuider": "RUM FLUX.2 Diffusers CFG Guider",
    "RUMFlux2NativeMatchVAEDecode": "RUM FLUX.2 Native Match VAE Decode",
    "RUMFlux2NativeMatchReferenceEncode": "RUM FLUX.2 Native Match Reference Encode",
}

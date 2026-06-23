from __future__ import annotations

import sys
from pathlib import Path
import math

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import RUMFlux2NativeMatchTextEncode, RUMFlux2SetQwenLayers, _validate_rum_checkpoint_override_path
from rum_native import (
    RUMDiffusersMatchTokenPolicy,
    append_reference_tokens,
    crop_noise_to_latent_tokens,
    pack_rum_reference_latents,
    preprocess_flux2_reference_image_tensor,
    prepare_diffusers_reference_image_ids,
    validate_comfy_reference_image_batch,
    validate_rum_reference_latents,
    validate_sdxl_teacher_dtype,
)


def test_diffusers_match_positive_keeps_277_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.randn(1, 277, 8)

    selected = policy.select(cross_attn, "positive")

    assert selected.shape == (1, 277, 8)
    assert selected.data_ptr() == cross_attn.data_ptr()


def test_diffusers_match_positive_recovers_base_and_extra_from_padded_context():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.arange(1 * 589 * 1, dtype=torch.float32).reshape(1, 589, 1)

    selected = policy.select(cross_attn, "positive")

    assert selected.shape == (1, 277, 1)
    assert torch.equal(selected[:, :200], cross_attn[:, :200])
    assert torch.equal(selected[:, 200:], cross_attn[:, -77:])


def test_diffusers_match_negative_keeps_512_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.randn(1, 512, 8)

    selected = policy.select(cross_attn, "negative")

    assert selected.shape == (1, 512, 8)
    assert selected.data_ptr() == cross_attn.data_ptr()


def test_diffusers_match_negative_crops_overlong_context_to_512_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.arange(1 * 589 * 1, dtype=torch.float32).reshape(1, 589, 1)

    selected = policy.select(cross_attn, "negative")

    assert selected.shape == (1, 512, 1)
    assert torch.equal(selected, cross_attn[:, :512])


def test_diffusers_match_legacy_does_not_crop_plain_flux2_512_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    cross_attn = torch.randn(1, 512, 8)

    selected = policy.select(cross_attn, None)

    assert selected.shape == (1, 512, 8)
    assert selected.data_ptr() == cross_attn.data_ptr()


def test_reference_image_ids_use_diffusers_flux2_klein_time_coordinates():
    reference_latents = [
        torch.zeros(1, 128, 2, 3),
        torch.zeros(1, 128, 1, 2),
    ]

    image_ids = prepare_diffusers_reference_image_ids(
        reference_latents,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert image_ids.shape == (2, 8, 4)
    assert torch.equal(image_ids[0, :6, 0], torch.full((6,), 10.0))
    assert torch.equal(image_ids[0, 6:, 0], torch.full((2,), 20.0))
    assert torch.equal(image_ids[0, :6, 1:], torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
    ]))


def test_pack_reference_latents_concatenates_packed_tokens_and_ids():
    reference_latents = [
        torch.arange(1 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2),
        torch.ones(1, 4, 2, 2, dtype=torch.float32),
    ]

    reference_tokens, reference_ids = pack_rum_reference_latents(
        {"latents": reference_latents},
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert reference_tokens.shape == (2, 8, 4)
    assert reference_ids.shape == (2, 8, 4)
    assert reference_tokens.dtype == torch.float32
    assert reference_ids.dtype == torch.float32
    assert torch.equal(reference_tokens[0, :4], reference_latents[0].reshape(1, 4, 4).permute(0, 2, 1)[0])
    assert torch.equal(reference_tokens[0], reference_tokens[1])
    assert torch.equal(reference_ids[0, :4, 0], torch.full((4,), 10.0))
    assert torch.equal(reference_ids[0, 4:, 0], torch.full((4,), 20.0))


def test_reference_tokens_append_and_noise_crop_keep_generated_token_count():
    denoise_tokens = torch.zeros(1, 3, 4)
    denoise_ids = torch.zeros(1, 3, 4)
    reference_tokens = torch.ones(1, 2, 4)
    reference_ids = torch.ones(1, 2, 4)

    model_input, model_ids, denoise_token_count = append_reference_tokens(
        denoise_tokens,
        denoise_ids,
        reference_tokens,
        reference_ids,
    )
    raw_noise = torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    cropped = crop_noise_to_latent_tokens(raw_noise, denoise_token_count)

    assert model_input.shape == (1, 5, 4)
    assert model_ids.shape == (1, 5, 4)
    assert denoise_token_count == 3
    assert torch.equal(model_input[:, :3], denoise_tokens)
    assert torch.equal(model_input[:, 3:], reference_tokens)
    assert torch.equal(cropped, raw_noise[:, :3])


def test_reference_image_preprocess_matches_flux2_klein_crop_size():
    image = torch.zeros(1, 1200, 670, 3)

    processed = preprocess_flux2_reference_image_tensor(image[0], vae_scale_factor=8)

    assert processed.shape == (1, 3, 1200, 656)
    assert processed.min().item() == -1.0
    assert processed.max().item() == -1.0


def test_reference_image_batch_validation_rejects_empty_batch():
    try:
        validate_comfy_reference_image_batch(torch.zeros(0, 64, 64, 3))
    except ValueError as exc:
        assert "至少 1 张图片" in str(exc)
    else:
        raise AssertionError("Expected empty reference image batch to fail.")


def test_reference_image_batch_validation_rejects_non_rgb_tensor():
    try:
        validate_comfy_reference_image_batch(torch.zeros(1, 64, 64, 1))
    except ValueError as exc:
        assert "C>=3" in str(exc)
    else:
        raise AssertionError("Expected non-RGB reference image batch to fail.")


def test_reference_latents_validation_rejects_missing_latents_key():
    try:
        validate_rum_reference_latents({})
    except ValueError as exc:
        assert "latents list" in str(exc)
    else:
        raise AssertionError("Expected missing reference latents list to fail.")


def test_sdxl_teacher_dtype_validation_rejects_integer_dtype():
    try:
        validate_sdxl_teacher_dtype(torch.int64)
    except ValueError as exc:
        assert "FP16/BF16/FP32" in str(exc)
    else:
        raise AssertionError("Expected integer teacher dtype to fail.")


def test_set_qwen_layers_cache_key_is_stable_and_not_nan():
    first = RUMFlux2SetQwenLayers.IS_CHANGED(None, "10,20,30")
    second = RUMFlux2SetQwenLayers.IS_CHANGED(None, "10,20,30")
    changed = RUMFlux2SetQwenLayers.IS_CHANGED(None, "9,18,27")

    assert not (isinstance(first, float) and math.isnan(first))
    assert first == second
    assert first != changed


def test_native_match_text_cache_key_tracks_text_parameters():
    first = RUMFlux2NativeMatchTextEncode.IS_CHANGED(None, None, "a", "", 0.0, False, 200, 77, 2048, "10,20,30")
    changed_prompt = RUMFlux2NativeMatchTextEncode.IS_CHANGED(None, None, "b", "", 0.0, False, 200, 77, 2048, "10,20,30")
    changed_layers = RUMFlux2NativeMatchTextEncode.IS_CHANGED(None, None, "a", "", 0.0, False, 200, 77, 2048, "9,18,27")

    assert first != changed_prompt
    assert first != changed_layers


def test_checkpoint_override_path_validation_accepts_existing_safetensors(tmp_path: Path):
    checkpoint = tmp_path / "rum.safetensors"
    checkpoint.write_bytes(b"")

    resolved = _validate_rum_checkpoint_override_path(str(checkpoint))

    assert resolved == str(checkpoint.resolve())


def test_checkpoint_override_path_validation_rejects_relative_path():
    try:
        _validate_rum_checkpoint_override_path("model-checkpoint-1158000.safetensors")
    except ValueError as exc:
        assert "绝对路径" in str(exc)
    else:
        raise AssertionError("Expected relative checkpoint override path to fail.")


def test_checkpoint_override_path_validation_rejects_non_safetensors(tmp_path: Path):
    checkpoint = tmp_path / "rum.bin"
    checkpoint.write_bytes(b"")

    try:
        _validate_rum_checkpoint_override_path(str(checkpoint))
    except ValueError as exc:
        assert ".safetensors" in str(exc)
    else:
        raise AssertionError("Expected non-safetensors checkpoint override path to fail.")

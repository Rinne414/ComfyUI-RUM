from __future__ import annotations

import sys
from pathlib import Path
import math

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import RUMFlux2NativeMatchTextEncode, RUMFlux2SetQwenLayers, _validate_rum_checkpoint_override_path
from rum_native import (
    RUMDiffusersMatchExtraConds,
    RUMDiffusersMatchTokenPolicy,
    _alternate_norm_key,
    _swap_scale_shift,
    append_reference_tokens,
    convert_rum_diffusers_to_comfy,
    crop_noise_to_latent_tokens,
    diffusers_flux2_sigmas,
    ensure_rum_projection_matches_base_tokens,
    pack_flux2_latents,
    pack_rum_reference_latents,
    patchify_flux2_latents,
    preprocess_flux2_reference_image_tensor,
    prepare_diffusers_reference_image_ids,
    validate_comfy_reference_image_batch,
    validate_rum_reference_latents,
    validate_sdxl_teacher_dtype,
    should_use_sdxl_teacher_hf_exact,
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


def test_sdxl_teacher_exact_auto_mode_allows_missing_hf_dirs(tmp_path: Path):
    assert not should_use_sdxl_teacher_hf_exact(
        tmp_path,
        cuda_available=True,
        transformers_available=True,
        exact_enabled=False,
    )


def test_sdxl_teacher_exact_mode_requires_complete_hf_pair(tmp_path: Path):
    l_dir = tmp_path / "waiNSFWIllustrious_v140_clip_l_dir"
    g_dir = tmp_path / "waiNSFWIllustrious_v140_clip_g_dir"
    l_dir.mkdir()
    g_dir.mkdir()
    (l_dir / "config.json").write_bytes(b"{}")
    (g_dir / "config.json").write_bytes(b"{}")
    (l_dir / "model.safetensors").write_bytes(b"")
    (g_dir / "model.safetensors").write_bytes(b"")

    assert not should_use_sdxl_teacher_hf_exact(
        tmp_path,
        cuda_available=True,
        transformers_available=True,
        exact_enabled=False,
    )
    assert should_use_sdxl_teacher_hf_exact(
        tmp_path,
        cuda_available=True,
        transformers_available=True,
        exact_enabled=True,
    )
    assert not should_use_sdxl_teacher_hf_exact(
        tmp_path,
        cuda_available=False,
        transformers_available=True,
        exact_enabled=True,
    )
    assert not should_use_sdxl_teacher_hf_exact(
        tmp_path,
        cuda_available=True,
        transformers_available=False,
        exact_enabled=True,
    )


def test_token_policy_meta_validation_rejects_mismatched_base_tokens():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)

    try:
        policy.validate_conditioning_meta("positive", 512, 77)
    except ValueError as exc:
        assert "base_text_tokens" in str(exc)
    else:
        raise AssertionError("Expected mismatched base_text_tokens meta to fail.")


def test_token_policy_meta_validation_accepts_matching_or_absent_meta():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)

    policy.validate_conditioning_meta("positive", 200, 77)
    policy.validate_conditioning_meta("positive", None, None)
    policy.validate_conditioning_meta(None, 200, None)
    # Legacy negative wiring may carry other token counts on purpose.
    policy.validate_conditioning_meta("negative", 512, 77)


def test_match_extra_conds_raises_before_touching_comfy_on_mismatch():
    policy = RUMDiffusersMatchTokenPolicy(base_text_tokens=200, extra_text_tokens=77)
    extra_conds = RUMDiffusersMatchExtraConds(
        lambda **kwargs: {},
        policy,
        disable_guidance=True,
    )

    try:
        extra_conds(
            cross_attn=torch.zeros(1, 589, 8),
            prompt_type="positive",
            rum_base_text_tokens=512,
            rum_extra_text_tokens=77,
        )
    except ValueError as exc:
        assert "token 配置不一致" in str(exc)
    else:
        raise AssertionError("Expected mismatched conditioning meta to fail.")


def test_projection_base_tokens_validation():
    class FakeProjection:
        base_text_tokens = 200

    ensure_rum_projection_matches_base_tokens(FakeProjection(), 200)
    ensure_rum_projection_matches_base_tokens(object(), 512)

    try:
        ensure_rum_projection_matches_base_tokens(FakeProjection(), 512)
    except ValueError as exc:
        assert "base_text_tokens" in str(exc)
    else:
        raise AssertionError("Expected projection/base token mismatch to fail.")


def test_diffusers_sigmas_snapshot_for_default_t2i_sample():
    sigmas = diffusers_flux2_sigmas(steps=20, width=960, height=1152)

    expected = torch.tensor([
        1.00000000, 0.98420829, 0.96723682, 0.94894826, 0.92918307,
        0.90775490, 0.88444471, 0.85899317, 0.83109087, 0.80036604,
        0.76636755, 0.72854286, 0.68620741, 0.63850331, 0.58433998,
        0.52231032, 0.45056668, 0.36663318, 0.26711461, 0.14722598,
        0.00000000,
    ])

    assert sigmas.shape == (21,)
    assert sigmas.dtype == torch.float32
    assert torch.all(sigmas[:-1] > sigmas[1:])
    assert torch.allclose(sigmas, expected, atol=1e-6, rtol=0.0)


def test_patchify_and_pack_flux2_latents_expected_layout():
    latent = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)

    patchified = patchify_flux2_latents(latent)
    packed = pack_flux2_latents(patchified)

    assert patchified.shape == (1, 4, 1, 1)
    assert torch.equal(patchified.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert packed.shape == (1, 1, 4)
    assert torch.equal(packed[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_swap_scale_shift_reorders_adaln_weight():
    weight = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

    swapped = _swap_scale_shift(weight)

    assert torch.equal(swapped, torch.tensor([[3.0], [4.0], [1.0], [2.0]]))


def test_alternate_norm_key_swaps_weight_and_scale_suffixes():
    key = "double_blocks.0.img_attn.norm.query_norm.weight"

    assert _alternate_norm_key(key) == "double_blocks.0.img_attn.norm.query_norm.scale"
    assert _alternate_norm_key("single_blocks.1.norm.key_norm.scale") == "single_blocks.1.norm.key_norm.weight"
    assert _alternate_norm_key("double_blocks.0.img_mlp.0.weight") is None


def test_convert_rum_checkpoint_merges_qkv_and_swaps_adaln():
    hidden = 2
    state_dict = {
        "x_embedder.weight": torch.zeros(hidden, 8),
        "context_embedder.weight": torch.zeros(hidden, 3),
        "transformer_blocks.0.attn.to_q.weight": torch.full((hidden, hidden), 1.0),
        "transformer_blocks.0.attn.to_k.weight": torch.full((hidden, hidden), 2.0),
        "transformer_blocks.0.attn.to_v.weight": torch.full((hidden, hidden), 3.0),
        "transformer_blocks.0.attn.add_q_proj.weight": torch.full((hidden, hidden), 4.0),
        "transformer_blocks.0.attn.add_k_proj.weight": torch.full((hidden, hidden), 5.0),
        "transformer_blocks.0.attn.add_v_proj.weight": torch.full((hidden, hidden), 6.0),
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight": torch.zeros(10, hidden),
        "single_transformer_blocks.0.attn.to_out.weight": torch.zeros(hidden, 4),
        "norm_out.linear.weight": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        "proj_out.weight": torch.zeros(4, hidden),
    }

    converted = convert_rum_diffusers_to_comfy(state_dict, output_prefix="")

    qkv = converted["double_blocks.0.img_attn.qkv.weight"]
    txt_qkv = converted["double_blocks.0.txt_attn.qkv.weight"]
    assert qkv.shape == (hidden * 3, hidden)
    assert torch.equal(qkv[:hidden], torch.full((hidden, hidden), 1.0))
    assert torch.equal(qkv[hidden : hidden * 2], torch.full((hidden, hidden), 2.0))
    assert torch.equal(qkv[hidden * 2 :], torch.full((hidden, hidden), 3.0))
    assert torch.equal(txt_qkv[:hidden], torch.full((hidden, hidden), 4.0))
    assert converted["single_blocks.0.linear1.weight"].shape == (10, hidden)
    assert torch.equal(
        converted["final_layer.adaLN_modulation.1.weight"],
        torch.tensor([[3.0, 0.0], [4.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    )
    assert converted["img_in.weight"].shape == (hidden, 8)
    assert converted["txt_in.weight"].shape == (hidden, 3)

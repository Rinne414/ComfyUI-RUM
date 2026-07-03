# Changelog

## 0.2.9

- Adopted upstream PR #5 (RimoChan): the edit example is now a single `examples/diffusers_match_edit_workflow.json`.
- Fixed a misleading crash when `reference_latents` is combined with a non-RUM sampler: the guider now raises a clear error pointing to `RUMFlux2DiffusersEulerSampler` instead of failing inside cond concat.
- Added consistency validation for `base_text_tokens` / `extra_text_tokens` across model load/patch, diffusers-match patch, and conditioning meta; mismatches now raise before sampling instead of silently producing corrupted images.
- Corrected the English README description of negative conditioning: it uses the same Qwen 200-token layers `10,20,30` + SDXL 77-token path as the positive prompt (verified against the upstream reference dump).
- `RUMFlux2NativeMatchVAEDecode` now raises on nested latent batches larger than 1 instead of silently decoding only the first latent.
- Sampler preview callback now reports the denoised prediction of the current step (pre-update), matching ComfyUI preview semantics; final outputs are unchanged.
- Package import no longer falls back to ComfyUI's own top-level `nodes` module when the relative import fails inside a package context.
- Exact Qwen/CLIP text encoders stop at the last requested hidden-state layer (Qwen text encode skips 6 of 36 layers with the default `10,20,30`), with bit-identical outputs.
- The strict SDXL teacher HF cache now keeps at most one loaded model pair.
- Unified the diffusers timestep semantics into a single `diffusers_match_timestep` helper shared by the raw-noise path and the model-patch wrapper; validated pixel-identical on live T2I/edit/wrapper-path runs.
- Removed dead code (`unpatchify` helpers, unused Qwen layer-override encode, unreachable branches), fixed ruff findings, added ruff to CI, refreshed stale install/download hints, and expanded unit tests (sigma snapshot, checkpoint key conversion, token-policy validation).

## 0.2.8

- Made the strict SDXL teacher HF exact text-encode path opt-in via `RUM_SDXL_TEACHER_HF_EXACT=1`, with clearer failure messages when CUDA/transformers/HF folders are missing.
- Added `--include-teacher-hf-exact` download support and install checks for the teacher HF folders.
- Fixed CI pytest collection (importlib mode, top-level node package import in tests).

## 0.2.7

- Synced upstream RUM inference defaults: T2I checkpoint `model-checkpoint-1158000.safetensors`, `960x1152`, 20 steps, CFG 9 samples.
- Added the edit path: `RUMFlux2NativeMatchReferenceEncode` node, optional `reference_latents` input on `RUMFlux2DiffusersCFGuider`, and edit example workflows using `model-checkpoint-1202000.safetensors`.
- Added input validation for reference images/latents and SDXL teacher dtype.
- Improved node cache keys (no more NaN `IS_CHANGED`) and let model-list errors surface instead of being swallowed.

## 0.2.6

- Fixed VAE attention dtype mismatch: `GroupNorm` upcasts BF16 to FP32 but `F.linear` weight stayed BF16, causing ~2-3 pixel-level drift across all output pixels. Added `_vae_linear` helper for dtype alignment.
- Verified all 4 upstream RUM reference images are now pixel-identical (`max_abs=0`) with the native diffusers-match workflow.
- Added workflow preview screenshot to README.
- Updated README to reflect zero unsolved alignment issues.

## 0.2.5

- Documented why native ComfyUI cannot exactly reproduce the old diffusers output.
- Added direct model-source mapping, Windows/Aki download commands, and workflow-specific model requirements to README.
- Added `--include-teacher-clip` and `--all` to the model download helper.
- Expanded install checks so missing sample-workflow model files are visible immediately.
- Added model requirement notes/titles to sample workflows.

## 0.2.4

- Fixed the visual diffusers-match workflow so it uses the same native approximation path as the API workflow.
- Routed the positive Qwen encode through the diffusers-match layer override while keeping negative on default 512-token Qwen conditioning.
- Made `RUM FLUX.2 Set Qwen Layers` safer by temporarily applying the layer override only during encoding and forcing cache invalidation.
- Rewrote README in Simplified Chinese and documented normal ComfyUI model paths.

## 0.2.3

- Fixed diffusers-match token cropping when ComfyUI pads FLUX.2 text conditioning to 512 tokens.
- Updated example workflows to use Windows-compatible nested diffusion model paths.

## 0.2.2

- Added `RUM FLUX.2 Diffusers Match Model Patch` for old diffusers-style token cropping.
- Added `examples/diffusers_match_workflow_api.json` with the old example seed and `base_text_tokens=200`.

## 0.2.1

- Updated the recommended native workflow to use `Flux2Scheduler` + `SamplerCustomAdvanced` instead of plain `KSampler scheduler=simple`.
- Documented the noisy/glitched image failure mode caused by the wrong FLUX.2 sampling chain.

## 0.2.0

- Switched `main` to native ComfyUI adapter mode.
- Added `RUM FLUX.2 Apply Model Patch`.
- Added `RUM FLUX.2 Combine Conditioning`.
- Moved the old diffusers pipeline implementation to the `diffusers-pipeline` branch.

## 0.1.0

- Initial diffusers pipeline wrapper.

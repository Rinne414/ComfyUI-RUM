# ComfyUI-RUM

ComfyUI-RUM 是 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 ComfyUI native 节点适配。目标是把 RUM 的 FLUX.2-Klein + SDXL teacher CLIP 条件路径拆进 ComfyUI，让它既能作为普通 ComfyUI workflow 使用，也能用于像素级验证 ComfyUI 适配是否匹配原始 diffusers 推理。

RUM 的原始想法、模型和参考推理代码来自 RimoChan/RUM。本仓库只提供 ComfyUI 适配代码和 workflow，不包含模型权重。


## 当前状态

- 主分支坚持 ComfyUI native 路线，不暴露直接调用 diffusers pipeline 的节点，也不要求用户在 ComfyUI 环境安装 diffusers。
- `examples/diffusers_match_workflow_api.json` 是严格验证路径：使用 ComfyUI native 模型、CLIP、sampler 和 VAE 权重，同时复刻原始 diffusers 的关键数学细节。
- 已验证 RimoChan/RUM 仓库自带的全部 4 张 reference 图（`output_0.png` ~ `output_3.png`，4 个不同 prompt × seed 组合，20 steps，960x1024）和当前 native diffusers-match 节点链逐像素 0 差：`max_abs=0`、`mean_abs=0`、`rmse=0`。没有已知的未解决对齐问题。

这段状态很重要：**主分支不是 diffusers wrapper；严格对齐靠 native diffusers-match workflow。**

## 为什么需要专门的 diffusers-match 路径

RUM 不是只给 FLUX.2-Klein 加一组普通 LoRA。它把 FLUX.2 的 Qwen 文本条件和 SDXL teacher CLIP 条件拼到同一个 transformer 条件流里。要贴近原始 diffusers 推理，需要同时处理这些细节：

- Qwen 正面 prompt 使用特定 hidden states 层：`10,20,30`。
- 条件长度是 Qwen `200` tokens + SDXL `77` tokens。
- SDXL teacher CLIP 来自 `waiIllustriousSDXL_v140`，不能随便换成普通 `clip_l.safetensors` / `clip_g.safetensors`。
- SDXL teacher CLIP 在严格对齐 workflow 里用 `DualCLIPLoader` 加载 waiIllustriousSDXL teacher 权重。
- 初始 noise 需要复刻 diffusers CPU generator + BF16 行为。
- scheduler sigma 必须使用 Flux2KleinPipeline 的 `np.linspace(1.0, 1.0 / steps, steps)` 再做 time shift。
- RUM transformer 的 timestep 和 CFG 处理需要贴近原始推理路径。
- FLUX.2 VAE decode 需要匹配 diffusers 的 attention 公式、BF16 postprocess 和 PIL round 量化路径。

这些点少一个都会漂。漂移可能不是小数值误差，而是角色、构图、颜色和细节明显不同。

## 已解决过的关键问题

这些是适配过程中已经确认并修过的问题，保留在 README 里是为了后来的人不要重复踩坑：

- **Qwen rotary buffer dtype 问题**
   - 问题：把 Qwen text encoder 整体 `.to(dtype=bfloat16)` 会把 `rotary_emb.inv_freq` 也转成 BF16。
   - 结果：text embedding 会偏，后面 denoise 全部跟着偏。
   - 处理：native match text encode 使用 ComfyUI Qwen 权重，但单独复刻 Qwen 关键 FP32 rotary / attention 语义，避免 buffer dtype 漂移。

- **scheduler / timestep 问题**
   - 问题：ComfyUI FLUX scheduler 和 diffusers Flux2Klein scheduler 不完全一样。
   - 处理：`RUMFlux2DiffusersScheduler` 使用 diffusers 风格 `np.linspace(1.0, 1.0 / steps)` 加 time shift；模型调用时按 diffusers 的 bfloat16 timestep 逻辑处理。

- **noise 问题**
   - 问题：普通 ComfyUI `RandomNoise` 不等于 diffusers `randn_tensor(..., generator=torch.Generator(device="cpu"), dtype=bfloat16)`。
   - 处理：`RUMFlux2DiffusersNoise` 单独复刻 diffusers CPU BF16 noise 生成。

- **VAE decode 问题**
   - 问题：ComfyUI 默认 VAE attention / 后处理路径会让最终 PNG 产生像素级差异；普通 `VAEDecode + SaveImage` 不是原始 PIL round 路径。
   - 处理：`RUMFlux2NativeMatchVAEDecode` 只用 ComfyUI 已加载的 `flux2-vae.safetensors` 权重，但在 VAE attention、BF16 postprocess 和保存前 round 量化上对齐 diffusers 行为；workflow 再接 `RUMRoundImageForSave` 后交给 `SaveImage`。

- **VAE attention dtype mismatch**
   - 问题：ComfyUI 的 `GroupNorm` 会把 BF16 tensor 自动 upcast 到 FP32。后续 `F.linear` 接收 FP32 hidden_states 和 BF16 weight，触发 `expected scalar type BFloat16 but found Float` 或产生不正确的像素值。
   - 影响：全部像素偏移，mean_abs ≈ 2–3，PSNR ≈ 28 dB。看上去是"同一张图但有微小差异"。
   - 处理：`_vae_linear` helper 在做 `F.linear` 前把 weight/bias cast 到和 hidden_states 相同的 dtype。修正后 4 组 reference 图全部 pixel-identical。

- **`--cpu-vae` 导致像素偏移**
   - 问题：部分 ComfyUI 发行版（如秋叶启动器）默认带 `--cpu-vae` 参数，让 VAE decode 在 CPU 上执行。CPU 和 GPU 的浮点运算结果存在微小差异（不同的 SDPA backend、不同的舍入行为）。
   - 影响：全部像素偏移 0–7，mean_abs ≈ 0.3。图片内容完全一致但不是逐像素相同。
   - 处理：做严格像素对齐验证时，不要使用 `--cpu-vae`。RTX 3090 等 24 GB 显卡完全可以同时在 GPU 上跑 model + VAE。

## Workflow 说明

仓库只保留1个示例 workflow，避免误用。

### diffusers-match API workflow

```text
examples/diffusers_match_workflow_api.json
```

用途：验证 ComfyUI 适配是否贴近原始 diffusers 推理。它仍然是 native workflow，不需要安装 diffusers，也不需要原始 `FLUX.2-klein-base-4B` diffusers 目录。

默认参数：

```text
prompt=1girl, kisaki (blue archive), eating baozi, sitting, indoors
negative_prompt=
seed=1
steps=20
guidance_scale=5
width=960
height=1024
```

重要：不同 ComfyUI 发行版的模型下拉名可能不同。例如 Aki 里可能显示成 `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`。公开 workflow 不能写死每个人本地的模型分类路径；如果 API validation 失败，请把 `rum_checkpoint_name` 改成你本机 `/object_info/RUMFlux2LoadNativeModel` 里实际列出的名字。

## 模型文件

diffusers-match workflow 需要的模型：

| 用途 | 推荐文件 | ComfyUI 位置 |
| --- | --- | --- |
| RUM checkpoint | `rum-flux2-klein-4b-preview.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiIllustriousSDXL_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiIllustriousSDXL_v140_clip_g.safetensors` | `models/text_encoders/` |


## 安装

把仓库放到 ComfyUI 的 custom nodes：

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

安装依赖后重启 ComfyUI：

```bash
pip install -r requirements.txt
```


## 数值验证建议

不要只靠看图判断适配是否正确。建议至少比较：

- text embedding：positive Qwen、positive combined、negative Qwen。
- initial noise：CPU generator、dtype、shape。
- scheduler sigmas 和每步 timestep。
- transformer 每步 raw noise / CFG noise / latent after step。
- final latent。
- VAE decode 后 RGB tensor。
- 最终 PNG 像素指标：`pixel_equal`、`max_abs`、`mean_abs`、`rmse`。

如果 PNG 不一致，先找第一处 tensor 非零差异。不要先调 prompt，也不要靠肉眼猜。

## 常见问题


### 为什么同样 prompt/seed 还是不同

常见原因：

- 用了普通 native workflow，而不是 diffusers-match workflow。
- teacher CLIP 文件不一致（必须使用 waiIllustriousSDXL teacher 权重）。
- Qwen 层号或 token 长度不一致。
- noise dtype 或 generator device 不一致。
- scheduler 和 timestep rounding 不一致。
- VAE decode 没有使用 `RUMFlux2NativeMatchVAEDecode`，或保存前没有接 `RUMRoundImageForSave`。
- 模型文件不是同一个权重，或者同源但精度不同。
- ComfyUI 启动时带了 `--cpu-vae`，导致 VAE decode 在 CPU 上执行，浮点结果和 GPU 不同。

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM)：RUM 的原始项目、模型、训练和 diffusers 推理参考。
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：ComfyUI 节点系统和执行环境。
- [Hugging Face diffusers](https://github.com/huggingface/diffusers)：FLUX.2-Klein pipeline、scheduler、VAE decode 参考实现。

## License note

本仓库不包含模型权重。RUM 上游项目在本适配开始时没有明确 LICENSE；模型和代码的再分发请遵守各自上游规则。

---

# ComfyUI-RUM English README

ComfyUI-RUM is a native ComfyUI node adapter for [RimoChan/RUM](https://github.com/RimoChan/RUM). It brings the RUM FLUX.2-Klein + SDXL teacher CLIP conditioning path into ComfyUI for both normal workflows and pixel-level validation against the original diffusers inference behavior.

The original RUM idea, model, training work, and reference inference code come from RimoChan/RUM. This repository only provides ComfyUI adapter code and example workflows. It does not include model weights.

## Current Status

- The main branch stays native-only: it does not expose a node that directly calls a diffusers pipeline, and users do not need to install diffusers into ComfyUI.
- `examples/diffusers_match_workflow_api.json` is the strict validation path. It uses native ComfyUI model, CLIP, sampler, and VAE weights while reproducing the important diffusers math details.
- All 4 upstream RimoChan/RUM reference images (`output_0.png` through `output_3.png`, 4 different prompt/seed combinations, 20 steps, 960x1024) have been verified pixel-identical with the current native diffusers-match node chain: `max_abs=0`, `mean_abs=0`, `rmse=0`. There are no known unsolved alignment issues.

Important: **main is not a diffusers wrapper; exact validation is achieved through the native diffusers-match workflow.**

## Why diffusers-match Exists

RUM is not a simple LoRA on top of FLUX.2-Klein. It combines FLUX.2 Qwen text conditioning with SDXL teacher CLIP conditioning in the transformer condition stream. Matching the original diffusers inference requires several details to line up:

- Positive Qwen prompt uses hidden-state layers `10,20,30`.
- Positive conditioning uses Qwen `200` tokens plus SDXL `77` tokens.
- Negative conditioning uses Qwen `512` tokens and layers `9,18,27`; it does not append the SDXL branch in the same way as the positive path.
- SDXL teacher CLIP comes from `waiIllustriousSDXL_v140`; generic `clip_l.safetensors` and `clip_g.safetensors` are not equivalent.
- The strict workflow loads SDXL teacher CLIP via `DualCLIPLoader` with the waiIllustriousSDXL teacher weights.
- Initial noise must match diffusers CPU generator + BF16 behavior.
- Scheduler sigmas must use Flux2KleinPipeline-style `np.linspace(1.0, 1.0 / steps, steps)` before time shift.
- Timestep and CFG behavior must follow the original inference path.
- FLUX.2 VAE decode must match diffusers attention formula, BF16 postprocess, and PIL round quantization.

If one of these details is wrong, the result can drift across sampling steps. The drift can affect character identity, composition, colors, and fine details.

## Bugs Already Found and Fixed

These notes are kept here so future debugging does not repeat the same mistakes.

- **Qwen rotary buffer dtype**
   - Problem: converting the Qwen text encoder with `.to(dtype=bfloat16)` also converts `rotary_emb.inv_freq` to BF16.
   - Effect: text embeddings drift, then the whole denoise path drifts.
   - Fix: native match text encoding uses ComfyUI Qwen weights but reproduces the key Qwen FP32 rotary / attention semantics to avoid buffer dtype drift.

- **Scheduler and timestep behavior**
   - Problem: ComfyUI's normal FLUX scheduler is not identical to the diffusers Flux2Klein scheduler.
   - Fix: `RUMFlux2DiffusersScheduler` uses diffusers-style `np.linspace(1.0, 1.0 / steps)` plus time shift, and the model call follows diffusers bfloat16 timestep behavior.

- **Initial noise**
   - Problem: ComfyUI `RandomNoise` is not identical to diffusers `randn_tensor(..., generator=torch.Generator(device="cpu"), dtype=bfloat16)`.
   - Fix: `RUMFlux2DiffusersNoise` reproduces the diffusers CPU BF16 noise path.

- **VAE decode path**
   - Problem: ComfyUI's default VAE attention / postprocess path can create pixel-level PNG differences; plain `VAEDecode + SaveImage` is not the original PIL round path.
   - Fix: `RUMFlux2NativeMatchVAEDecode` uses only the loaded ComfyUI `flux2-vae.safetensors` weights, but aligns VAE attention, BF16 postprocess, and pre-save round quantization with diffusers behavior. The workflow then passes the image through `RUMRoundImageForSave` before `SaveImage`.

- **VAE attention dtype mismatch**
   - Problem: ComfyUI's `GroupNorm` automatically upcasts BF16 tensors to FP32. The subsequent `F.linear` call then receives FP32 hidden_states with BF16 weights, causing either a `expected scalar type BFloat16 but found Float` error or silently producing incorrect pixel values.
   - Effect: every pixel shifts by a small amount (mean_abs ~2–3, PSNR ~28 dB). The image looks like "the same picture with slight differences".
   - Fix: `_vae_linear` helper casts weight and bias to match the hidden_states dtype before calling `F.linear`. After this fix, all 4 reference images are pixel-identical.

- **`--cpu-vae` causes pixel drift**
   - Problem: some ComfyUI distributions (e.g. the Aki launcher) start with `--cpu-vae` by default, running VAE decode on CPU. CPU and GPU floating-point results differ slightly due to different SDPA backends and rounding behavior.
   - Effect: every pixel shifts by 0–7, mean_abs ~0.3. The image content is identical but not pixel-equal.
   - Fix: do not use `--cpu-vae` when running strict pixel-alignment validation. GPUs with 24 GB VRAM (e.g. RTX 3090) can run model + VAE on GPU simultaneously.

## Workflows

Only 1 public example workflows are kept to reduce confusion.

### Normal Native Workflow

### diffusers-match API Workflow

```text
examples/diffusers_match_workflow_api.json
```

Use this for validation against the original diffusers inference behavior. It is still a native workflow: it does not require installing diffusers or providing the original `FLUX.2-klein-base-4B` diffusers directory.

Default parameters:

```text
prompt=1girl, kisaki (blue archive), eating baozi, sitting, indoors
negative_prompt=
seed=1
steps=20
guidance_scale=5
width=960
height=1024
```

Different ComfyUI distributions may expose different model names in the dropdown. For example, Aki may list the model as `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`. The public workflow cannot hard-code every local folder category. If API validation fails, read `/object_info/RUMFlux2LoadNativeModel` and set `rum_checkpoint_name` to the exact local name shown there.

## Model Files

Models required by the diffusers-match workflow:

| Purpose | Recommended file | ComfyUI location |
| --- | --- | --- |
| RUM checkpoint | `rum-flux2-klein-4b-preview.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiIllustriousSDXL_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiIllustriousSDXL_v140_clip_g.safetensors` | `models/text_encoders/` |


## Install

Put this repository under:

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

Install dependencies and restart ComfyUI:

```bash
pip install -r requirements.txt
```


## Numeric Validation

Do not rely on visual inspection only. For alignment work, compare at least:

- text embeddings: positive Qwen, combined positive, negative Qwen.
- initial noise: CPU generator, dtype, shape.
- scheduler sigmas and per-step timestep.
- transformer raw noise, CFG noise, and latent after each step.
- final latent.
- RGB tensor after VAE decode.
- final PNG metrics: `pixel_equal`, `max_abs`, `mean_abs`, `rmse`.

When PNGs differ, find the first tensor stage with a non-zero difference before changing prompts or judging by eye.

## FAQ


### Why can the same prompt and seed still differ?

Common causes:

- The normal native workflow was used instead of the diffusers-match workflow.
- Teacher CLIP files differ (must use the waiIllustriousSDXL teacher weights).
- Qwen layers or token lengths differ.
- Noise dtype or generator device differs.
- Scheduler or timestep rounding differs.
- VAE decode is not using `RUMFlux2NativeMatchVAEDecode`, or the image is not passed through `RUMRoundImageForSave` before saving.
- Model weights are not identical, or are from the same source but different precision.
- ComfyUI was started with `--cpu-vae`, causing VAE decode to run on CPU where floating-point results differ from GPU.

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM): original RUM project, model, training, and diffusers inference reference.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI): node system and execution environment.
- [Hugging Face diffusers](https://github.com/huggingface/diffusers): FLUX.2-Klein pipeline, scheduler, and VAE decode reference implementation.

## License Note

This repository does not include model weights. At the time this adapter work started, upstream RUM did not provide a clear LICENSE. Follow upstream rules for model and code redistribution.

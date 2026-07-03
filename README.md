# ComfyUI-RUM

ComfyUI-RUM 是 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 ComfyUI native 节点适配。目标是把 RUM 的 FLUX.2-Klein + SDXL teacher CLIP 条件路径拆进 ComfyUI，让它既能作为普通 ComfyUI workflow 使用，也能用于像素级验证 ComfyUI 适配是否匹配原始 diffusers 推理。

RUM 的原始想法、模型和参考推理代码来自 RimoChan/RUM。本仓库只提供 ComfyUI 适配代码和 workflow，不包含模型权重，也不直接合并上游训练仓库文件。


## 当前状态

- 主分支坚持 ComfyUI native 路线，不暴露直接调用 diffusers pipeline 的节点，也不要求用户在 ComfyUI 环境安装 diffusers。
- 上游 `RimoChan/RUM` 当前默认分支是 `slave`，2026-06-23 的 HEAD 是 `1662918`。本仓库和上游训练仓库没有共同祖先，不能直接 `merge` 或 `cherry-pick`；这里同步的是推理契约、默认参数、权重文件名和验证样本。
- `examples/diffusers_match_workflow_api.json` 是 T2I 严格验证路径，默认使用上游当前推荐的 `model-checkpoint-1158000.safetensors`、`960x1152`、20 steps、CFG 9、seed 1 示例。
- `examples/diffusers_match_edit_workflow.json` 是新增 edit 路径，默认使用 `model-checkpoint-1202000.safetensors` 和参考图输入。
- 新增 `RUMFlux2NativeMatchReferenceEncode` 节点，把 ComfyUI `IMAGE` + `VAE` 编码成专用 `RUM_REFERENCE_LATENTS`；`RUMFlux2DiffusersCFGuider` 增加可选 `reference_latents` 输入，不连接时旧 T2I workflow 行为不变。

这段状态很重要：**主分支不是 diffusers wrapper；严格对齐靠 native diffusers-match workflow。** T2I/edit workflow 已在 `I:\ComfyUI-aki-v1.6\ComfyUI` 生成成功；严格 `pixel_equal=true` 还要求上游 reference 环境、ComfyUI 环境和模型精度完全一致。

## 为什么需要专门的 diffusers-match 路径

RUM 不是只给 FLUX.2-Klein 加一组普通 LoRA。它把 FLUX.2 的 Qwen 文本条件和 SDXL teacher CLIP 条件拼到同一个 transformer 条件流里。要贴近原始 diffusers 推理，需要同时处理这些细节：

- Qwen 正面 prompt 使用特定 hidden states 层：`10,20,30`。
- 条件长度是 Qwen `200` tokens + SDXL `77` tokens。
- 负面 prompt 使用与正面完全相同的编码路径：Qwen `200` tokens、层 `10,20,30`，同样拼接 SDXL teacher `77` tokens（只是文本换成 negative prompt）。
- SDXL teacher CLIP 来自 `waiNSFWIllustrious_v140`，不能随便换成普通 `clip_l.safetensors` / `clip_g.safetensors`。
- SDXL teacher CLIP 在严格对齐 workflow 里用 `DualCLIPLoader` 加载 waiNSFWIllustrious teacher 权重。
- 初始 noise 需要复刻 diffusers CPU generator + BF16 行为。
- scheduler sigma 必须使用 Flux2KleinPipeline 的 `np.linspace(1.0, 1.0 / steps, steps)` 再做 time shift。
- RUM transformer 的 timestep 和 CFG 处理需要贴近原始推理路径。
- FLUX.2 VAE decode 需要匹配 diffusers 的 attention 公式、BF16 postprocess 和 PIL round 量化路径。
- edit 路径还需要复刻上游 `Flux2KleinPipeline(image=...)`：参考图按最大面积 `1024*1024` 缩放，裁到 FLUX.2 latent 需要的倍数，normalize 到 `[-1,1]`，VAE encode 后做 BN 标准化，再把 reference tokens 拼到 denoise tokens 后面。

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

- **环境版本会影响逐像素一致性**
   - 问题：`torch` / `transformers` 版本差异可能改变 Qwen hidden states；FLUX.2 VAE 使用 BF16 或 FP32 权重也会改变最终 RGB/PNG 舍入。
   - 影响：图片可以正常生成，但不是严格 `pixel_equal=true`。
   - 处理：把“可用生成”和“逐像素一致验证”分开看；严格验证时固定上游 reference 环境、ComfyUI 环境、VAE dtype，并让 VAE 在 GPU 上运行。

## Workflow 说明

仓库保留 T2I 和 edit 两组 diffusers-match workflow；API JSON 适合自动验证，GUI JSON 适合直接在 ComfyUI UI 中打开。

![diffusers-match workflow preview](docs/diffusers_match_workflow_preview.png)

### T2I diffusers-match workflow

```text
examples/diffusers_match_workflow_api.json
examples/diffusers_match_workflow_gui.json
```

用途：验证 ComfyUI 适配是否贴近原始 diffusers 推理。它仍然是 native workflow，不需要安装 diffusers，也不需要原始 `FLUX.2-klein-base-4B` diffusers 目录。

默认参数：

```text
checkpoint=model-checkpoint-1158000.safetensors
prompt=1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)
negative_prompt=
seed=1
steps=20
cfg=9
width=960
height=1152
```

上游当前 4 个 T2I prompt/seed：

```text
1. seed=1  1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)
2. seed=2  1girl, momoi (blue archive), typing on keyboard, computer, blue necktie, open coat, multicolored coat, angry, indoors, ameto yuki
3. seed=3  1girl, yuuka (blue archive), holding cup, white jacket, black jacket, blue necktie, indoors, fuzichoco
4. seed=4  1girl, azusa (blue archive), holding pizza, eating, indoors, chen bin
```

### Edit diffusers-match workflow

```text
examples/diffusers_match_edit_workflow.json
```

默认参数：

```text
checkpoint=model-checkpoint-1202000.safetensors
reference=rum_reference.jpg
prompt=将服装改为school uniform, short sleeves
negative_prompt=
seed=1
steps=20
cfg=9
width=656
height=1200
```

`rum_reference.jpg` 是上游仓库 `img/抓人.jpg` 的同图 ASCII 文件名副本。API workflow 使用 ASCII 文件名，避免 Windows/PowerShell 环境把中文文件名转成问号；UI 中也可以重新选择同一张上游参考图。

在 ComfyUI UI 中使用 edit workflow：

1. 把上游参考图 `img/抓人.jpg` 放到 `ComfyUI/input/rum_reference.jpg`，或者在 `LoadImage` 节点里选择你自己的参考图。
2. 打开 `examples/diffusers_match_edit_workflow.json`。
3. 确认 `RUMFlux2LoadNativeModel` 选择 `model-checkpoint-1202000.safetensors`。
4. 确认 `RUMFlux2NativeMatchReferenceEncode` 已连接到 `RUMFlux2DiffusersCFGuider.reference_latents`。
5. `reference_latents` 只在 `RUMFlux2DiffusersEulerSampler` 采样路径下生效；如果把 guider 接到普通 sampler，节点会直接报错提示。
6. 不要使用 `--cpu-vae` 做严格像素验证；只想生成可用图片时 CPU VAE 也可以运行，但最终 PNG 可能不是逐像素一致。

上游当前 6 个编辑 prompt/seed：

```text
1. seed=1  将服装改为school uniform, short sleeves
2. seed=2  add twintails
3. seed=3  1girl, remove shoes, bare foot
4. seed=4  1girl, change background to beach
5. seed=6  add choker
6. seed=5  改为1girl, fuzichoco
```

重要：不同 ComfyUI 发行版的模型下拉名可能不同。例如 Aki 里可能显示成 `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`。公开 workflow 不能写死每个人本地的模型分类路径；如果 API validation 失败，请把 `rum_checkpoint_name` 改成你本机 `/object_info/RUMFlux2LoadNativeModel` 里实际列出的名字。

## 模型文件

diffusers-match workflow 正常生成需要的模型：

| 用途 | 推荐文件 | ComfyUI 位置 |
| --- | --- | --- |
| RUM T2I checkpoint | `model-checkpoint-1158000.safetensors` | `models/diffusion_models/` |
| RUM edit checkpoint | `model-checkpoint-1202000.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiNSFWIllustrious_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiNSFWIllustrious_v140_clip_g.safetensors` | `models/text_encoders/` |

strict pixel exact text encode 的可选模型：

| 用途 | 推荐文件 | ComfyUI 位置 |
| --- | --- | --- |
| SDXL teacher CLIP-L exact HF 目录（strict pixel exact） | `waiNSFWIllustrious_v140_clip_l_dir/config.json` + `model.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G exact HF 目录（strict pixel exact） | `waiNSFWIllustrious_v140_clip_g_dir/config.json` + `model.safetensors` | `models/text_encoders/` |

模型来源和直链：

| 用途 | 提供者 | 链接 |
| --- | --- | --- |
| RUM T2I checkpoint | RimoChan / rimochan | [model-checkpoint-1158000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1158000.safetensors) |
| RUM edit checkpoint | RimoChan / rimochan | [model-checkpoint-1202000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1202000.safetensors) |
| Qwen text encoder | Comfy-Org | [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| FLUX.2 VAE | Comfy-Org | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| SDXL teacher CLIP-L file / HF dir | Ine007 | [text_encoder/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder/model.safetensors) and [text_encoder/config.json](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder/config.json) |
| SDXL teacher CLIP-G file / HF dir | Ine007 | [text_encoder_2/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder_2/model.safetensors) and [text_encoder_2/config.json](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder_2/config.json) |

下载默认 T2I 路径：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip
```

同时下载编辑 checkpoint：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-edit-checkpoint
```

可选：同时下载 strict pixel exact text encode 用的 HF 目录：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-teacher-hf-exact
```

检查安装：

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

`DualCLIPLoader` 使用 flat `.safetensors` 文件。`*_clip_l_dir` 和 `*_clip_g_dir` HF 目录只用于 strict pixel exact text encode；默认不会启用，也不会影响正常生成。只有设置环境变量 `RUM_SDXL_TEACHER_HF_EXACT=1` 时，节点才会尝试从 HF 目录加载 transformers 模型；否则使用 ComfyUI 已加载的 SDXL DualCLIP flat 文件。

## 测试

本仓库的轻量单元测试不下载模型，也不启动 ComfyUI：

```bash
python -m pytest -q
```

要检查本机 ComfyUI 环境、节点加载和模型可见性：

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

## 安装

把仓库放到 ComfyUI 的 custom nodes：

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

安装依赖后重启 ComfyUI：

```bash
pip install -r requirements.txt
```

需要支持 FLUX.2/Klein 的较新 ComfyUI（本仓库在 ComfyUI v0.26.2 上验证）。节点依赖 `CLIP.load_model`、`SamplerCustomAdvanced`、nested latent 等较新的 ComfyUI 内部 API，过旧的 ComfyUI 会在加载或采样时报错。


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
- teacher CLIP 文件不一致（必须使用 waiNSFWIllustrious teacher 权重）。
- Qwen 层号或 token 长度不一致。
- 各节点的 `base_text_tokens` / `extra_text_tokens` 不一致（新版本会在采样前直接报错阻止）。
- noise dtype 或 generator device 不一致。
- scheduler 和 timestep rounding 不一致。
- VAE decode 没有使用 `RUMFlux2NativeMatchVAEDecode`，或保存前没有接 `RUMRoundImageForSave`。
- 模型文件不是同一个权重，或者同源但精度不同。
- `torch` / `transformers` 版本不同，导致 Qwen hidden states 已经在 text embedding 阶段分歧。
- ComfyUI 启动时带了 `--cpu-vae`，导致 VAE decode 在 CPU 上执行，浮点结果和 GPU 不同。

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM)：RUM 的原始项目、模型、训练和 diffusers 推理参考。
- [rimochan/RUM-FLUX.2-klein-4B-preview](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview)：T2I 和 edit RUM checkpoint。
- [Ine007/waiNSFWIllustrious_v140](https://huggingface.co/Ine007/waiNSFWIllustrious_v140)：SDXL teacher CLIP-L / CLIP-G 权重。
- [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)：Qwen text encoder 文件。
- [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev)：FLUX.2 VAE 文件。
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：ComfyUI 节点系统和执行环境。
- [Hugging Face diffusers](https://github.com/huggingface/diffusers)：FLUX.2-Klein pipeline、scheduler、VAE decode 参考实现。

## License note

本仓库不包含模型权重。RUM 上游项目在本适配开始时没有明确 LICENSE；模型和代码的再分发请遵守各自上游规则。

---

# ComfyUI-RUM English README

ComfyUI-RUM is a native ComfyUI node adapter for [RimoChan/RUM](https://github.com/RimoChan/RUM). It brings the RUM FLUX.2-Klein + SDXL teacher CLIP conditioning path into ComfyUI for both normal workflows and pixel-level validation against the original diffusers inference behavior.

The original RUM idea, model, training work, and reference inference code come from RimoChan/RUM. This repository only provides ComfyUI adapter code and example workflows. It does not include model weights, and it does not directly merge upstream training-repository files.

## Current Status

- The main branch stays native-only: it does not expose a node that directly calls a diffusers pipeline, and users do not need to install diffusers into ComfyUI.
- Upstream `RimoChan/RUM` currently uses `slave` as the default branch; the 2026-06-23 HEAD is `1662918`. This repository and the upstream training repository do not share common history, so this update syncs inference contracts, defaults, model filenames, and validation samples instead of using `merge` or `cherry-pick`.
- `examples/diffusers_match_workflow_api.json` is the T2I strict validation path. It now defaults to the current upstream `model-checkpoint-1158000.safetensors`, `960x1152`, 20 steps, CFG 9, seed 1 sample.
- `examples/diffusers_match_edit_workflow.json` adds the edit path, defaulting to `model-checkpoint-1202000.safetensors` plus a reference image input.
- `RUMFlux2NativeMatchReferenceEncode` encodes ComfyUI `IMAGE` + `VAE` into `RUM_REFERENCE_LATENTS`; `RUMFlux2DiffusersCFGuider` now accepts optional `reference_latents`, while disconnected T2I workflows keep the old behavior.

Important: **main is not a diffusers wrapper; exact validation is achieved through the native diffusers-match workflow.** T2I and edit workflows have been manually verified to generate images in `I:\ComfyUI-aki-v1.6\ComfyUI`; strict `pixel_equal=true` still requires the upstream reference environment, the ComfyUI environment, and model precision to match.

## Why diffusers-match Exists

RUM is not a simple LoRA on top of FLUX.2-Klein. It combines FLUX.2 Qwen text conditioning with SDXL teacher CLIP conditioning in the transformer condition stream. Matching the original diffusers inference requires several details to line up:

- Positive Qwen prompt uses hidden-state layers `10,20,30`.
- Positive conditioning uses Qwen `200` tokens plus SDXL `77` tokens.
- Negative conditioning uses the same encoding path as the positive prompt: Qwen `200` tokens with layers `10,20,30` plus the SDXL teacher `77`-token branch, encoded from the negative prompt text.
- SDXL teacher CLIP comes from `waiNSFWIllustrious_v140`; generic `clip_l.safetensors` and `clip_g.safetensors` are not equivalent.
- The strict workflow loads SDXL teacher CLIP via `DualCLIPLoader` with the waiNSFWIllustrious teacher weights.
- Initial noise must match diffusers CPU generator + BF16 behavior.
- Scheduler sigmas must use Flux2KleinPipeline-style `np.linspace(1.0, 1.0 / steps, steps)` before time shift.
- Timestep and CFG behavior must follow the original inference path.
- FLUX.2 VAE decode must match diffusers attention formula, BF16 postprocess, and PIL round quantization.
- The edit path also needs to reproduce upstream `Flux2KleinPipeline(image=...)`: resize the reference image to max area `1024*1024`, crop to the FLUX.2 latent multiple, normalize to `[-1,1]`, VAE encode, apply FLUX.2 BN normalization, then append reference tokens after denoise tokens.

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

- **Environment versions affect pixel identity**
   - Problem: `torch` / `transformers` version differences can change Qwen hidden states; BF16 vs FP32 FLUX.2 VAE weights can also change final RGB/PNG rounding.
   - Effect: image generation can work, but strict `pixel_equal=true` can fail.
   - Fix: treat functional generation and pixel-identical validation as separate checks. For strict validation, pin the upstream reference environment, the ComfyUI environment, VAE dtype, and keep VAE on GPU.

## Workflows

The repository now ships T2I and edit diffusers-match workflows. API JSON files are for automated validation; GUI JSON files can be opened directly in the ComfyUI UI.

![diffusers-match workflow preview](docs/diffusers_match_workflow_preview.png)

### T2I diffusers-match Workflow

```text
examples/diffusers_match_workflow_api.json
examples/diffusers_match_workflow_gui.json
```

Use this for validation against the original diffusers inference behavior. It is still a native workflow: it does not require installing diffusers or providing the original `FLUX.2-klein-base-4B` diffusers directory.

Default parameters:

```text
checkpoint=model-checkpoint-1158000.safetensors
prompt=1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)
negative_prompt=
seed=1
steps=20
cfg=9
width=960
height=1152
```

Current upstream T2I prompt/seed samples:

```text
1. seed=1  1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)
2. seed=2  1girl, momoi (blue archive), typing on keyboard, computer, blue necktie, open coat, multicolored coat, angry, indoors, ameto yuki
3. seed=3  1girl, yuuka (blue archive), holding cup, white jacket, black jacket, blue necktie, indoors, fuzichoco
4. seed=4  1girl, azusa (blue archive), holding pizza, eating, indoors, chen bin
```

### Edit diffusers-match Workflow

```text
examples/diffusers_match_edit_workflow.json
```

Default parameters:

```text
checkpoint=model-checkpoint-1202000.safetensors
reference=rum_reference.jpg
prompt=change clothes to school uniform, short sleeves
negative_prompt=
seed=1
steps=20
cfg=9
width=656
height=1200
```

`rum_reference.jpg` is an ASCII-filename copy of upstream `img/抓人.jpg`. The API workflow uses the ASCII filename to avoid Chinese filename encoding issues on Windows/PowerShell; in the UI you can also select the same upstream reference image manually.

To use the edit workflow in the ComfyUI UI:

1. Put upstream `img/抓人.jpg` into `ComfyUI/input/rum_reference.jpg`, or select your own reference image in the `LoadImage` node.
2. Open `examples/diffusers_match_edit_workflow.json`.
3. Confirm `RUMFlux2LoadNativeModel` is set to `model-checkpoint-1202000.safetensors`.
4. Confirm `RUMFlux2NativeMatchReferenceEncode` is connected to `RUMFlux2DiffusersCFGuider.reference_latents`.
5. `reference_latents` only works on the `RUMFlux2DiffusersEulerSampler` sampling path; connecting the guider to a regular sampler raises a clear error.
6. Do not use `--cpu-vae` for strict pixel validation. CPU VAE can still generate usable images, but final PNGs may not be pixel-identical.

Current upstream edit prompt/seed samples:

```text
1. seed=1  将服装改为school uniform, short sleeves
2. seed=2  add twintails
3. seed=3  1girl, remove shoes, bare foot
4. seed=4  1girl, change background to beach
5. seed=6  add choker
6. seed=5  改为1girl, fuzichoco
```

Different ComfyUI distributions may expose different model names in the dropdown. For example, Aki may list the model as `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`. The public workflow cannot hard-code every local folder category. If API validation fails, read `/object_info/RUMFlux2LoadNativeModel` and set `rum_checkpoint_name` to the exact local name shown there.

## Model Files

Models required for normal diffusers-match generation:

| Purpose | Recommended file | ComfyUI location |
| --- | --- | --- |
| RUM T2I checkpoint | `model-checkpoint-1158000.safetensors` | `models/diffusion_models/` |
| RUM edit checkpoint | `model-checkpoint-1202000.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiNSFWIllustrious_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiNSFWIllustrious_v140_clip_g.safetensors` | `models/text_encoders/` |

Optional models for strict pixel-exact text encoding:

| Purpose | Recommended file | ComfyUI location |
| --- | --- | --- |
| SDXL teacher CLIP-L exact HF folder (strict pixel exact) | `waiNSFWIllustrious_v140_clip_l_dir/config.json` + `model.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G exact HF folder (strict pixel exact) | `waiNSFWIllustrious_v140_clip_g_dir/config.json` + `model.safetensors` | `models/text_encoders/` |

Model sources and direct links:

| Purpose | Provider | Link |
| --- | --- | --- |
| RUM T2I checkpoint | RimoChan / rimochan | [model-checkpoint-1158000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1158000.safetensors) |
| RUM edit checkpoint | RimoChan / rimochan | [model-checkpoint-1202000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1202000.safetensors) |
| Qwen text encoder | Comfy-Org | [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| FLUX.2 VAE | Comfy-Org | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| SDXL teacher CLIP-L file / HF folder | Ine007 | [text_encoder/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder/model.safetensors) and [text_encoder/config.json](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder/config.json) |
| SDXL teacher CLIP-G file / HF folder | Ine007 | [text_encoder_2/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder_2/model.safetensors) and [text_encoder_2/config.json](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder_2/config.json) |

Download the default T2I files:

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip
```

Also download the edit checkpoint:

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-edit-checkpoint
```

Optionally download the HF folders for strict pixel-exact text encoding:

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-teacher-hf-exact
```

Check installation:

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

`DualCLIPLoader` uses the flat `.safetensors` files. The `*_clip_l_dir` and `*_clip_g_dir` HF folders are only used for strict pixel-exact text encoding; they are disabled by default and do not affect normal generation. Set `RUM_SDXL_TEACHER_HF_EXACT=1` only when you want the node to load transformers models from those HF folders. Otherwise the node uses the SDXL DualCLIP flat files already loaded by ComfyUI.

## Tests

The lightweight unit tests do not download models or start ComfyUI:

```bash
python -m pytest -q
```

To check the local ComfyUI environment, node imports, and model visibility:

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```


## Install

Put this repository under:

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

Install dependencies and restart ComfyUI:

```bash
pip install -r requirements.txt
```

A recent ComfyUI with FLUX.2/Klein support is required (this repository is verified on ComfyUI v0.26.2). The nodes rely on newer ComfyUI internals such as `CLIP.load_model`, `SamplerCustomAdvanced`, and nested latents; older ComfyUI versions will fail at load or sampling time.


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
- Teacher CLIP files differ (must use the waiNSFWIllustrious teacher weights).
- Qwen layers or token lengths differ.
- `base_text_tokens` / `extra_text_tokens` differ between nodes (newer versions raise an error before sampling instead).
- Noise dtype or generator device differs.
- Scheduler or timestep rounding differs.
- VAE decode is not using `RUMFlux2NativeMatchVAEDecode`, or the image is not passed through `RUMRoundImageForSave` before saving.
- Model weights are not identical, or are from the same source but different precision.
- `torch` / `transformers` versions differ, causing Qwen hidden states to diverge at the text embedding stage.
- ComfyUI was started with `--cpu-vae`, causing VAE decode to run on CPU where floating-point results differ from GPU.

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM): original RUM project, model, training, and diffusers inference reference.
- [rimochan/RUM-FLUX.2-klein-4B-preview](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview): T2I and edit RUM checkpoints.
- [Ine007/waiNSFWIllustrious_v140](https://huggingface.co/Ine007/waiNSFWIllustrious_v140): SDXL teacher CLIP-L / CLIP-G weights.
- [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo): Qwen text encoder file.
- [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev): FLUX.2 VAE file.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI): node system and execution environment.
- [Hugging Face diffusers](https://github.com/huggingface/diffusers): FLUX.2-Klein pipeline, scheduler, and VAE decode reference implementation.

## License Note

This repository does not include model weights. At the time this adapter work started, upstream RUM did not provide a clear LICENSE. Follow upstream rules for model and code redistribution.

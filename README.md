# ComfyUI-RUM

ComfyUI-RUM 是 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 ComfyUI 原生节点适配。它把 RUM 的 FLUX.2-Klein + SDXL teacher CLIP 条件路径带进 ComfyUI，可以直接当普通 ComfyUI workflow 使用。

- **不需要**安装 diffusers，也不需要原始 `FLUX.2-klein-base-4B` diffusers 目录。
- 本仓库只提供 ComfyUI 适配代码和示例 workflow，**不包含模型权重**。
- 想了解对齐原理、pixel-level 验证或排查漂移？请看 [TECH.md](TECH.md)。

![diffusers-match workflow preview](docs/diffusers_match_workflow_preview.png)

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

## 下载模型

正常生成需要下面这些模型：

| 用途 | 推荐文件 | ComfyUI 位置 |
| --- | --- | --- |
| RUM T2I checkpoint | `model-checkpoint-1158000.safetensors` | `models/diffusion_models/` |
| RUM edit checkpoint | `model-checkpoint-1202000.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiNSFWIllustrious_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiNSFWIllustrious_v140_clip_g.safetensors` | `models/text_encoders/` |

模型来源和直链：

| 用途 | 提供者 | 链接 |
| --- | --- | --- |
| RUM T2I checkpoint | RimoChan / rimochan | [model-checkpoint-1158000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1158000.safetensors) |
| RUM edit checkpoint | RimoChan / rimochan | [model-checkpoint-1202000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1202000.safetensors) |
| Qwen text encoder | Comfy-Org | [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| FLUX.2 VAE | Comfy-Org | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| SDXL teacher CLIP-L | Ine007 | [text_encoder/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder/model.safetensors) |
| SDXL teacher CLIP-G | Ine007 | [text_encoder_2/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder_2/model.safetensors) |

下载默认 T2I 需要的模型：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip
```

同时下载编辑 checkpoint：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-edit-checkpoint
```

检查安装（确认节点能加载、模型可见）：

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

> 做逐像素严格对齐验证时，还有可选的 strict pixel-exact HF 目录模型；正常生成不需要，细节见 [TECH.md](TECH.md)。

## 使用 Workflow

仓库提供 T2I 和 edit 两组 workflow，在 ComfyUI UI 里打开 GUI JSON 即可使用。（自动化验证用的 API 格式 workflow 放在 `examples/validation/`，一般使用者用不到，细节见 [TECH.md](TECH.md)。）

### T2I workflow

```text
examples/diffusers_match_workflow_gui.json
```

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

在 ComfyUI UI 里打开 GUI JSON，确认 `RUMFlux2LoadNativeModel` 选的是 `model-checkpoint-1158000.safetensors`，然后直接运行即可。

### Edit workflow

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

在 ComfyUI UI 中使用 edit workflow：

1. 把参考图放到 `ComfyUI/input/rum_reference.jpg`，或者在 `LoadImage` 节点里选择你自己的参考图。（`rum_reference.jpg` 是上游 `img/抓人.jpg` 的 ASCII 文件名副本，避免 Windows/PowerShell 把中文文件名转成问号。）
2. 打开 `examples/diffusers_match_edit_workflow.json`。
3. 确认 `RUMFlux2LoadNativeModel` 选择 `model-checkpoint-1202000.safetensors`。
4. 确认 `RUMFlux2NativeMatchReferenceEncode` 已连接到 `RUMFlux2DiffusersCFGuider.reference_latents`。
5. `reference_latents` 只在 `RUMFlux2DiffusersEulerSampler` 采样路径下生效；如果把 guider 接到普通 sampler，节点会直接报错提示。

> 提示：不同 ComfyUI 发行版的模型下拉名可能不同（例如 Aki 里可能是 `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`）。如果 API workflow 验证失败，请把 `rum_checkpoint_name` 改成你本机 `/object_info/RUMFlux2LoadNativeModel` 里实际列出的名字。

## 常见问题

**同样 prompt / seed，结果却不一样？**

先确认这几点：

- 用的是本仓库的 diffusers-match workflow，不是普通 native workflow。
- teacher CLIP 是 waiNSFWIllustrious teacher 权重，不是普通 `clip_l` / `clip_g`。
- ComfyUI 启动时**没有**带 `--cpu-vae`（它会让 VAE 在 CPU 上跑，浮点结果和 GPU 不同）。

如果你追求的是逐像素完全一致（`pixel_equal=true`），影响因素更多（Qwen 层号 / token 长度、noise dtype、scheduler rounding、`torch` / `transformers` 版本等），完整清单和排查方法见 [TECH.md](TECH.md)。

**API workflow 验证失败？** 多半是模型下拉名不匹配，见上面的提示；也可以先跑 `python scripts/check_install.py` 确认模型可见。

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

# ComfyUI-RUM (English)

ComfyUI-RUM is a native ComfyUI node adapter for [RimoChan/RUM](https://github.com/RimoChan/RUM). It brings the RUM FLUX.2-Klein + SDXL teacher CLIP conditioning path into ComfyUI so you can use it as a normal ComfyUI workflow.

- You do **not** need to install diffusers or provide the original `FLUX.2-klein-base-4B` diffusers directory.
- This repository only provides ComfyUI adapter code and example workflows. It **does not include model weights**.
- Want to understand the alignment internals, pixel-level validation, or debug drift? See [TECH.md](TECH.md).

![diffusers-match workflow preview](docs/diffusers_match_workflow_preview.png)

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

## Download Models

Normal generation needs these models:

| Purpose | Recommended file | ComfyUI location |
| --- | --- | --- |
| RUM T2I checkpoint | `model-checkpoint-1158000.safetensors` | `models/diffusion_models/` |
| RUM edit checkpoint | `model-checkpoint-1202000.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiNSFWIllustrious_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiNSFWIllustrious_v140_clip_g.safetensors` | `models/text_encoders/` |

Model sources and direct links:

| Purpose | Provider | Link |
| --- | --- | --- |
| RUM T2I checkpoint | RimoChan / rimochan | [model-checkpoint-1158000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1158000.safetensors) |
| RUM edit checkpoint | RimoChan / rimochan | [model-checkpoint-1202000.safetensors](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview/resolve/main/model-checkpoint-1202000.safetensors) |
| Qwen text encoder | Comfy-Org | [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| FLUX.2 VAE | Comfy-Org | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| SDXL teacher CLIP-L | Ine007 | [text_encoder/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder/model.safetensors) |
| SDXL teacher CLIP-G | Ine007 | [text_encoder_2/model.safetensors](https://huggingface.co/Ine007/waiNSFWIllustrious_v140/resolve/main/text_encoder_2/model.safetensors) |

Download the models for the default T2I workflow:

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip
```

Also download the edit checkpoint:

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-edit-checkpoint
```

Check the install (nodes load, models are visible):

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

> For strict pixel-exact alignment there are optional HF-folder models. Normal generation does not need them; see [TECH.md](TECH.md).

## Use the Workflows

The repository ships T2I and edit workflows. Open the GUI JSON in the ComfyUI UI to use them. (The API-format workflow used for automated validation lives in `examples/validation/`; normal users do not need it — see [TECH.md](TECH.md).)

### T2I workflow

```text
examples/diffusers_match_workflow_gui.json
```

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

Open the GUI JSON in the ComfyUI UI, confirm `RUMFlux2LoadNativeModel` is set to `model-checkpoint-1158000.safetensors`, then run it.

### Edit workflow

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

To use the edit workflow in the ComfyUI UI:

1. Put a reference image into `ComfyUI/input/rum_reference.jpg`, or select your own reference image in the `LoadImage` node. (`rum_reference.jpg` is an ASCII-filename copy of upstream `img/抓人.jpg`, to avoid Chinese filename encoding issues on Windows/PowerShell.)
2. Open `examples/diffusers_match_edit_workflow.json`.
3. Confirm `RUMFlux2LoadNativeModel` is set to `model-checkpoint-1202000.safetensors`.
4. Confirm `RUMFlux2NativeMatchReferenceEncode` is connected to `RUMFlux2DiffusersCFGuider.reference_latents`.
5. `reference_latents` only works on the `RUMFlux2DiffusersEulerSampler` sampling path; connecting the guider to a regular sampler raises a clear error.

> Note: different ComfyUI distributions may expose different model names in the dropdown (e.g. Aki may list it as `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`). If API-workflow validation fails, set `rum_checkpoint_name` to the exact name shown in your local `/object_info/RUMFlux2LoadNativeModel`.

## FAQ

**Same prompt and seed, but different result?**

Check these first:

- You are using this repo's diffusers-match workflow, not a plain native workflow.
- Teacher CLIP is the waiNSFWIllustrious teacher weights, not generic `clip_l` / `clip_g`.
- ComfyUI did **not** start with `--cpu-vae` (it runs VAE on CPU, where floating-point results differ from GPU).

If you are after pixel-identical output (`pixel_equal=true`), more factors matter (Qwen layers / token lengths, noise dtype, scheduler rounding, `torch` / `transformers` versions, etc.). The full checklist and debugging method are in [TECH.md](TECH.md).

**API workflow validation fails?** Usually the model dropdown name does not match — see the note above. You can also run `python scripts/check_install.py` first to confirm models are visible.

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

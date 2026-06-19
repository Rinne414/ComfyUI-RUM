from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MODELS = {
    "diffusion_models": [
        "flux-2-klein/flux-2-klein-4b-fp8.safetensors",
        "rum-flux2-klein-4b-preview.safetensors",
    ],
    "text_encoders": [
        "qwen_3_4b.safetensors",
        "clip_l.safetensors",
        "clip_g.safetensors",
        "waiIllustriousSDXL_v140_clip_l.safetensors",
        "waiIllustriousSDXL_v140_clip_g.safetensors",
    ],
    "vae": ["flux2-vae.safetensors"],
}

OPTIONAL_WORKFLOW_FILES = {
    "normal native workflow": ["clip_l.safetensors", "clip_g.safetensors"],
    "diffusers-match workflow": [
        "waiIllustriousSDXL_v140_clip_l.safetensors",
        "waiIllustriousSDXL_v140_clip_g.safetensors",
    ],
}


def normalize_cli_path(path: Path) -> Path:
    raw = str(path).strip().strip('"')
    normalized = raw.replace("\\", "/")
    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", normalized)
        if match:
            drive = match.group(1).upper()
            rest = (match.group(2) or "").replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    else:
        match = re.match(r"^([a-zA-Z]):/(.*)$", normalized)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2)
            return Path("/mnt") / drive / rest
    return Path(raw).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    default_comfy_root = ROOT.parent.parent
    parser = argparse.ArgumentParser(description="Check a ComfyUI-RUM native installation.")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=default_comfy_root,
        help="ComfyUI root directory. Default: inferred from custom_nodes/ComfyUI-RUM/scripts.",
    )
    return parser.parse_args()


def load_node_module(comfy_root: Path):
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_RUM_check",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create module spec for ComfyUI-RUM.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_names(names: list[str]) -> set[str]:
    return {name.replace("\\", "/") for name in names}


def print_model_status(folder_name: str, names: list[str]) -> None:
    visible = normalized_names(names)
    print(f"{folder_name}: {len(names)} file(s) visible to ComfyUI")
    for expected in EXPECTED_MODELS.get(folder_name, []):
        marker = "OK" if expected in visible else "MISS"
        print(f"  {marker} {expected}")


def print_workflow_status(text_encoder_names: list[str]) -> None:
    visible = normalized_names(text_encoder_names)
    print("Workflow model hints:")
    for label, required in OPTIONAL_WORKFLOW_FILES.items():
        missing = [name for name in required if name not in visible]
        if missing:
            print(f"  MISS {label}: {', '.join(missing)}")
        else:
            print(f"  OK {label}")


def main() -> int:
    args = parse_args()
    comfy_root = normalize_cli_path(args.comfy_root)
    print(f"Checking ComfyUI-RUM at: {ROOT}")
    print(f"ComfyUI root: {comfy_root}")
    for package in ("torch", "safetensors"):
        module = __import__(package)
        print(f"{package}: {getattr(module, '__version__', 'ok')}")

    module = load_node_module(comfy_root)
    print("Node classes:", ", ".join(sorted(module.NODE_CLASS_MAPPINGS)))

    import folder_paths

    text_encoder_names: list[str] = []
    for folder_name in ("diffusion_models", "text_encoders", "vae"):
        try:
            names = folder_paths.get_filename_list(folder_name)
        except Exception as exc:
            print(f"{folder_name}: ERROR {exc}")
            continue
        if folder_name == "text_encoders":
            text_encoder_names = names
        print_model_status(folder_name, names)

    print_workflow_status(text_encoder_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

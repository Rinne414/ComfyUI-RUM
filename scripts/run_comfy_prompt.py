from __future__ import annotations

import argparse
import json
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a ComfyUI API workflow and wait for completion.")
    parser.add_argument("--server", required=True)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--prompt-id-out", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def post_prompt(server: str, prompt: dict) -> str:
    payload = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI prompt submission failed: status={exc.code}; body={body}") from exc

    prompt_id = result.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f"ComfyUI prompt response did not include prompt_id: {result}")
    return prompt_id


def read_history(server: str, prompt_id: str) -> dict:
    with urllib.request.urlopen(f"{server}/history/{prompt_id}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_completion(server: str, prompt_id: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + int(timeout_seconds)
    while time.monotonic() < deadline:
        history = read_history(server, prompt_id)
        item = history.get(prompt_id)
        if isinstance(item, dict):
            return item
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for ComfyUI prompt: {prompt_id}")


def main() -> int:
    args = parse_args()
    prompt = read_json(args.workflow)
    prompt_id = post_prompt(args.server.rstrip("/"), prompt)
    args.prompt_id_out.write_text(prompt_id, encoding="ascii")
    print(f"PROMPT_ID={prompt_id}")
    item = wait_for_completion(args.server.rstrip("/"), prompt_id, args.timeout_seconds)
    print(json.dumps(item.get("status", {}), ensure_ascii=False, indent=2))
    print(json.dumps(item.get("outputs", {}), ensure_ascii=False, indent=2))
    status = item.get("status", {})
    return 0 if status.get("status_str") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

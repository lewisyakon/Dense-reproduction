#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the two Qwen3 models")
    parser.add_argument("--root", default="/mnt/hd/lewis/models")
    parser.add_argument("--model", choices=("4B", "8B", "both"), default="both")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    sizes = ("4B", "8B") if args.model == "both" else (args.model,)
    for size in sizes:
        destination = root / f"Qwen3-{size}"
        print(f"Downloading Qwen/Qwen3-{size} to {destination}", flush=True)
        snapshot_download(
            repo_id=f"Qwen/Qwen3-{size}",
            local_dir=destination,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )


if __name__ == "__main__":
    main()


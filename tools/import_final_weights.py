#!/usr/bin/env python3
"""Import the three final Dense Phase-2 adapters and build a checksum manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    "/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/converters/"
    "dense_gsm8k_math_algebra_arc_c"
)
RUNS = [
    ("M1", "m1_sender_qwen3_0.6b__receiver_qwen3_1.7b", "Qwen3-0.6B", "Qwen3-1.7B"),
    ("M2", "m2_sender_qwen3_0.6b__receiver_qwen3_4b", "Qwen3-0.6B", "Qwen3-4B"),
    ("M3", "m3_sender_qwen3_4b__receiver_qwen3_0.6b", "Qwen3-4B", "Qwen3-0.6B"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"existing destination has wrong size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    records = []
    for pair, run, sender, receiver in RUNS:
        source = SOURCE_ROOT / run / "phase2.pt"
        destination = REPO / "weights" / pair / "phase2.pt"
        hardlink_or_copy(source, destination)
        payload = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        state = payload["adapter_state_dict"]
        parameters = sum(tensor.numel() for tensor in state.values())
        tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        records.append(
            {
                "id": pair,
                "pair": pair,
                "sender": sender,
                "receiver": receiver,
                "path": str(destination.relative_to(REPO)),
                "phase": payload["phase"],
                "step": payload["step"],
                "parameters": parameters,
                "tensor_bytes": tensor_bytes,
                "file_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "source_path": str(source),
                "load_api": "dense_kv.checkpoint.load_adapter_checkpoint",
            }
        )
        print(f"imported {destination}")
    manifest = {
        "schema_version": 1,
        "created_at": "2026-11-18",
        "precision_policy": "Original BF16 Phase-2 checkpoint; no conversion",
        "entries": records,
        "total_file_bytes": sum(record["file_bytes"] for record in records),
        "excluded": "Phase-1 and intermediate checkpoints",
    }
    (REPO / "weights/manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from dense_kv.config import load_config
from dense_kv.runtime import adapter_parameter_count, model_config_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a dense-KV experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def check_model(label: str, spec) -> dict:
    path = Path(spec.path)
    result = {"label": label, "path": str(path), "exists": path.exists()}
    if path.exists():
        try:
            config = model_config_from_disk(path)
            actual = {
                "num_layers": config.get("num_hidden_layers"),
                "num_kv_heads": config.get("num_key_value_heads"),
                "head_dim": config.get("head_dim"),
                "rope_theta": config.get("rope_theta"),
            }
            expected = {
                "num_layers": spec.num_layers,
                "num_kv_heads": spec.num_kv_heads,
                "head_dim": spec.head_dim,
                "rope_theta": spec.rope_theta,
            }
            result["config"] = actual
            result["config_matches"] = actual == expected
            metadata_path = path / ".cache/huggingface/download/config.json.metadata"
            if metadata_path.exists():
                local_revision = metadata_path.read_text(encoding="utf-8").splitlines()[0]
                result["revision"] = local_revision
                result["revision_matches"] = (
                    spec.revision is None or local_revision == spec.revision
                )
            else:
                result["revision"] = None
                result["revision_matches"] = spec.revision is None
            index_path = path / "model.safetensors.index.json"
            single_weight = path / "model.safetensors"
            if index_path.exists():
                with index_path.open("r", encoding="utf-8") as handle:
                    index = json.load(handle)
                shards = sorted(set(index.get("weight_map", {}).values()))
                missing = [shard for shard in shards if not (path / shard).is_file()]
                actual_bytes = sum(
                    (path / shard).stat().st_size for shard in shards if (path / shard).is_file()
                )
                expected_bytes = index.get("metadata", {}).get("total_size")
                result["weight_shards"] = len(shards)
                result["missing_weight_shards"] = missing
                result["weight_bytes"] = actual_bytes
                result["expected_weight_bytes"] = expected_bytes
                result["weights_complete"] = not missing and (
                    expected_bytes is None or actual_bytes >= expected_bytes
                )
            elif single_weight.is_file():
                result["weight_shards"] = 1
                result["missing_weight_shards"] = []
                result["weight_bytes"] = single_weight.stat().st_size
                result["expected_weight_bytes"] = single_weight.stat().st_size
                result["weights_complete"] = True
            else:
                result["missing_weight_shards"] = ["model weights"]
                result["weights_complete"] = False
        except Exception as exc:
            result["config_error"] = str(exc)
            result["config_matches"] = False
    return result


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_path = Path(cfg.paths.output_dir)
    storage_probe = output_path if output_path.exists() else output_path.parent
    while not storage_probe.exists() and storage_probe != storage_probe.parent:
        storage_probe = storage_probe.parent
    usage = shutil.disk_usage(storage_probe)
    models = [check_model("sender", cfg.sender), check_model("receiver", cfg.receiver)]
    gpus = []
    cuda_error = None
    try:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_gib": round(properties.total_memory / 2**30, 2),
                }
            )
    except Exception as exc:
        cuda_error = str(exc)
    parameters = adapter_parameter_count(cfg)
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_error": cuda_error,
        "gpus": gpus,
        "models": models,
        "adapter_parameters": parameters,
        "adapter_weight_gib_bfloat16": round(parameters * 2 / 2**30, 3),
        "output_filesystem_free_gib": round(usage.free / 2**30, 2),
        "output_filesystem_total_gib": round(usage.total / 2**30, 2),
    }
    print(json.dumps(report, indent=2))
    models_ok = all(
        item.get("exists")
        and item.get("config_matches")
        and item.get("revision_matches")
        and item.get("weights_complete")
        for item in models
    )
    device_ok = args.allow_cpu or torch.cuda.is_available()
    return 0 if models_ok and device_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

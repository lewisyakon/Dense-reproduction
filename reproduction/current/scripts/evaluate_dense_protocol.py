#!/usr/bin/env python3
"""Evaluate a Stage-1 Dense KV checkpoint under unified Protocol A or C."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("DENSE_PROJECT_ROOT", REPO_ROOT / "reproduction" / "current"))
DENSE_ROOT = Path(os.environ.get("DENSE_REPO_ROOT", REPO_ROOT))
if str(DENSE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DENSE_ROOT / "src"))

from dense_kv.checkpoint import load_adapter_checkpoint  # noqa: E402
from dense_kv.config import load_config  # noqa: E402
from dense_kv.prompts import assistant_prefix_ids, chat_ids, question_batch  # noqa: E402
from dense_kv.runtime import (  # noqa: E402
    build_adapter,
    extract_cache,
    load_model,
    load_tokenizer,
    manual_generate,
    parse_dtype,
    set_seed,
)


TASK_DEFAULTS = {
    "arc_challenge": 8,
    "gsm8k": 512,
    "hotpotqa_bridge": 64,
    "humaneval": 512,
}


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def hotpot_scores(prediction: str, reference: str) -> tuple[float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    exact = float(normalized_prediction == normalized_reference)
    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()
    if not prediction_tokens or not reference_tokens:
        return exact, float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return exact, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def extract_arc_label(text: str) -> str:
    matches = re.findall(r"(?i)(?:answer|option|choice)?\s*[:]?\s*\b([A-D])\b", text)
    return matches[-1].upper() if matches else ""


def normalize_number(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def extract_gsm_answer(text: str) -> str:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(matches[-1]) if matches else ""


def score_record(task: str, prediction: str, record: dict[str, Any]) -> dict[str, float | str | None]:
    if task == "arc_challenge":
        parsed = extract_arc_label(prediction)
        return {"parsed_prediction": parsed, "accuracy": float(parsed == record["answer"].upper())}
    if task == "gsm8k":
        parsed = extract_gsm_answer(prediction)
        reference = extract_gsm_answer(record["answer"])
        return {"parsed_prediction": parsed, "exact_match": float(parsed == reference)}
    if task == "hotpotqa_bridge":
        exact, f1 = hotpot_scores(prediction, record["answer"])
        return {"parsed_prediction": normalize_answer(prediction), "exact_match": exact, "f1": f1}
    if task == "humaneval":
        return {"parsed_prediction": None, "pass_at_1": None}
    raise ValueError(task)


def user_question(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Missing messages for {record.get('record_id')}")
    users = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError(f"Expected exactly one user message for {record.get('record_id')}")
    return users[0]


class ForwardTimer:
    def __init__(self, model):
        self.model = model
        self.original_forward = None
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event, int | None]] = []

    def __enter__(self):
        self.original_forward = self.model.forward

        def timed_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            length = int(input_ids.shape[-1]) if isinstance(input_ids, torch.Tensor) else None
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            output = self.original_forward(*args, **kwargs)
            finished.record()
            self.events.append((started, finished, length))
            return output

        self.model.forward = timed_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.forward = self.original_forward

    def result(self) -> tuple[float, float, list[dict[str, float | int | None]]]:
        calls = [
            {
                "milliseconds": float(start.elapsed_time(finish)),
                "input_tokens": length,
            }
            for start, finish, length in self.events
        ]
        prefill = float(calls[0]["milliseconds"]) if calls else 0.0
        decode = sum(float(call["milliseconds"]) for call in calls[1:])
        return prefill, decode, calls


def timed_cuda(callable_):
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    value = callable_()
    finished.record()
    return value, started, finished


def cache_bytes(cache) -> int:
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in cache
    )


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--protocol", choices=("a", "c"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Optional directory containing Qwen model folders; overrides model paths in the saved config.",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--no-prompt-truncation",
        action="store_true",
        help="Use the complete tokenized sender prompt instead of training.max_prompt_tokens.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_root is not None:
        cfg = replace(
            cfg,
            sender=replace(cfg.sender, path=str(args.model_root / Path(cfg.sender.path).name)),
            receiver=replace(cfg.receiver, path=str(args.model_root / Path(cfg.receiver.path).name)),
        )
    set_seed(cfg.training.seed)
    dtype = parse_dtype(cfg.adapter.dtype)
    sender_tokenizer = load_tokenizer(cfg.sender, local_files_only=True)
    receiver_tokenizer = load_tokenizer(cfg.receiver, local_files_only=True)
    sender = load_model(
        cfg.sender,
        cfg.training.sender_device,
        dtype=dtype,
        local_files_only=True,
    )
    receiver = load_model(
        cfg.receiver,
        cfg.training.receiver_device,
        dtype=dtype,
        local_files_only=True,
    )
    adapter = build_adapter(cfg)
    checkpoint_state = load_adapter_checkpoint(
        args.checkpoint,
        adapter,
        expected_direction=cfg.direction,
    )
    if checkpoint_state.get("phase") != 2:
        raise ValueError(f"Expected Phase-II checkpoint, found phase {checkpoint_state.get('phase')}")
    adapter.eval()
    max_new_tokens = args.max_new_tokens or TASK_DEFAULTS[args.task]
    max_prompt_tokens = None if args.no_prompt_truncation else cfg.training.max_prompt_tokens
    eos_ids = (
        {receiver_tokenizer.eos_token_id}
        if receiver_tokenizer.eos_token_id is not None
        else set()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    completed: set[str] = set()
    total = 0
    processed_this_run = 0
    metric_sums = Counter()
    timing_sums = Counter()
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                previous = json.loads(line)
                if previous["record_id"] in completed:
                    continue
                if previous["protocol"] != args.protocol:
                    raise ValueError("Existing predictions use a different protocol")
                completed.add(previous["record_id"])
                total += 1
                for name, value in previous["metrics"].items():
                    if isinstance(value, (int, float)) and value is not None and math.isfinite(value):
                        metric_sums[name] += float(value)
                for name in (
                    "source_prefill_ms",
                    "adapter_conversion_ms",
                    "receiver_prefill_ms",
                    "receiver_decode_ms",
                    "total_compute_ms",
                ):
                    timing_sums[name] += float(previous["timing"][name])

    adapter_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    adapter_weight_bytes = args.checkpoint.stat().st_size
    del checkpoint_state

    with args.input.open(encoding="utf-8") as source, predictions_path.open("a", encoding="utf-8") as output:
        for ordinal, line in enumerate(source):
            if args.limit is not None and ordinal >= args.limit:
                break
            record = json.loads(line)
            record_id = record["record_id"]
            if record_id in completed:
                continue
            if record["task"] != args.task:
                raise ValueError(f"Expected {args.task}, found {record['task']}")
            question = user_question(record)
            source_batch = question_batch(
                sender_tokenizer,
                [question],
                max_prompt_tokens,
            )
            source_prompt_tokens = int(source_batch.attention_mask.sum().item())
            if args.protocol == "a":
                receiver_prefix = chat_ids(
                    receiver_tokenizer,
                    question,
                    generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                receiver_prefix = assistant_prefix_ids(
                    receiver_tokenizer,
                    question,
                    enable_thinking=False,
                )

            torch.cuda.synchronize()
            total_started = time.perf_counter()
            with torch.inference_mode():
                source_cache, source_start, source_finish = timed_cuda(
                    lambda: extract_cache(
                        sender,
                        source_batch.input_ids,
                        source_batch.attention_mask,
                        source_batch.position_ids,
                    )
                )
                transformed, adapter_start, adapter_finish = timed_cuda(
                    lambda: adapter(source_cache, source_batch.position_ids)
                )
                with ForwardTimer(receiver) as receiver_timer:
                    generated, receiver_usage = manual_generate(
                        receiver,
                        receiver_prefix,
                        past_layers=transformed,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        eos_token_ids=eos_ids,
                    )
            torch.cuda.synchronize()
            total_compute_ms = (time.perf_counter() - total_started) * 1000
            source_prefill_ms = float(source_start.elapsed_time(source_finish))
            adapter_conversion_ms = float(adapter_start.elapsed_time(adapter_finish))
            receiver_prefill_ms, receiver_decode_ms, receiver_calls = receiver_timer.result()
            prediction = receiver_tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
            scores = score_record(args.task, prediction, record)
            source_cache_nbytes = cache_bytes(source_cache)
            transformed_cache_nbytes = cache_bytes(transformed)
            stage_sum = (
                source_prefill_ms
                + adapter_conversion_ms
                + receiver_prefill_ms
                + receiver_decode_ms
            )
            result = {
                "record_id": record_id,
                "task": args.task,
                "direction": cfg.direction,
                "protocol": args.protocol,
                "prediction": prediction,
                "reference": record.get("answer"),
                "metrics": scores,
                "source_prompt_tokens": source_prompt_tokens,
                "receiver_prefill_tokens": receiver_usage["prefill_tokens"],
                "output_tokens": receiver_usage["decode_tokens"],
                "prompt_utf8_bytes": len(question.encode("utf-8")),
                "output_utf8_bytes": len(prediction.encode("utf-8")),
                "source_original_cache_bytes": source_cache_nbytes,
                "transformed_cache_bytes": transformed_cache_nbytes,
                "logical_transmission_bytes": transformed_cache_nbytes,
                "timing": {
                    "source_prefill_ms": source_prefill_ms,
                    "adapter_conversion_ms": adapter_conversion_ms,
                    "receiver_prefill_ms": receiver_prefill_ms,
                    "receiver_decode_ms": receiver_decode_ms,
                    "total_compute_ms": total_compute_ms,
                    "unattributed_orchestration_ms": max(0.0, total_compute_ms - stage_sum),
                    "receiver_forward_calls": receiver_calls,
                },
            }
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            total += 1
            processed_this_run += 1
            for name, value in scores.items():
                if isinstance(value, (int, float)) and value is not None and math.isfinite(value):
                    metric_sums[name] += float(value)
            timing_values = {
                "source_prefill_ms": source_prefill_ms,
                "adapter_conversion_ms": adapter_conversion_ms,
                "receiver_prefill_ms": receiver_prefill_ms,
                "receiver_decode_ms": receiver_decode_ms,
                "total_compute_ms": total_compute_ms,
            }
            for name, value in timing_values.items():
                timing_sums[name] += value

            summary = {
                "task": args.task,
                "protocol": args.protocol,
                "direction": cfg.direction,
                "completed_rows": total,
                "processed_rows_this_run": processed_this_run,
                "checkpoint": str(args.checkpoint),
                "checkpoint_phase": 2,
                "config": str(args.config),
                "input": str(args.input),
                "max_new_tokens": max_new_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "prompt_truncation_enabled": max_prompt_tokens is not None,
                "quality_means": {name: value / total for name, value in metric_sums.items()},
                "timing_means_ms": {name: value / total for name, value in timing_sums.items()},
                "adapter_parameters": adapter_parameters,
                "adapter_weight_bytes": adapter_weight_bytes,
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "dtype": cfg.adapter.dtype,
                "phase2_protocol_a_probability": cfg.training.context_aware_probability,
                "phase2_protocol_c_probability": 1.0 - cfg.training.context_aware_probability,
                "logical_payload": "complete transformed receiver-compatible KV cache",
                "cache_group_selection": False,
                "receiver_thinking_enabled": False,
                "network_time_included": False,
            }
            atomic_write_json(summary_path, summary)
            print(json.dumps({
                "record_id": record_id,
                "protocol": args.protocol,
                "metrics": scores,
                "output_tokens": result["output_tokens"],
            }))
            del source_cache, transformed

    print(json.dumps({
        "completed_rows": total,
        "processed_rows_this_run": processed_this_run,
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()

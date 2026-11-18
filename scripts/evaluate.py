#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dense_kv.config import load_config
from dense_kv.evaluation import VALID_MODES, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense KV communication")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", action="append", choices=VALID_MODES)
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    modes = list(VALID_MODES) if args.all_modes else args.mode
    if not modes:
        parser.error("Pass --all-modes or at least one --mode")
    summary = evaluate(
        load_config(args.config),
        modes,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        checkpoint=args.checkpoint,
        output_path=args.output,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

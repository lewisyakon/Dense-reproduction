#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dense_kv.prepare import prepare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/eval JSONL data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-per-dataset", type=int)
    parser.add_argument("--max-eval-per-dataset", type=int)
    args = parser.parse_args()
    train, evaluation = prepare_datasets(
        args.output_dir,
        max_train_per_dataset=args.max_train_per_dataset,
        max_eval_per_dataset=args.max_eval_per_dataset,
    )
    print(f"Wrote {train}")
    print(f"Wrote {evaluation}")


if __name__ == "__main__":
    main()


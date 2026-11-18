#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dense_kv.config import load_config
from dense_kv.phase1 import train_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase-I cache reconstruction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    metrics = train_phase1(
        load_config(args.config),
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        resume=args.resume,
    )
    print(metrics)


if __name__ == "__main__":
    main()


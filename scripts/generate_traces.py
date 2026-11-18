#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dense_kv.config import load_config
from dense_kv.traces import generate_receiver_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Receiver-self traces")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    count = generate_receiver_traces(
        load_config(args.config),
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Wrote {count} new traces")


if __name__ == "__main__":
    main()


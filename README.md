# Dense Latent Communication 独立复现仓库

> 这是 2026-11-18 的独立可迁移快照，不继承原仓库的 Git 历史或 remote。
> `reproduction/current/` 保存本次 M1/M2/M3 正式训练和统一测评所用的配置与队列；
> 大型资产的迁移方法见 [`MIGRATION.md`](MIGRATION.md)。

三份完成训练的 Phase-2 adapter 已通过 Git LFS 纳入 `weights/`。克隆后执行
`git lfs pull` 即可直接加载；清单见 `weights/manifest.json`。

This directory implements the minimal reproduction described in
`qwen3_4b_8b_dense_kv_minimal_reproduction_typora.md` for the two independent
directions:

- Qwen3-4B to Qwen3-8B
- Qwen3-8B to Qwen3-4B

The implementation includes inverse/forward RoPE, fixed monotonic layer
alignment, direct KV-head alignment, independent per-layer/per-head K and V
MLPs, a shared K/V gate per receiver head, Phase-I cache reconstruction,
Receiver-self trace generation, Phase-II language-model training, cache
injection, and the four requested evaluation modes.

## Environment

The existing environment on this machine can be used directly:

```bash
export DENSE_KV_PYTHON=/mnt/hd/lewis/conda-envs/c2c/bin/python
$DENSE_KV_PYTHON -m pip install -e . --no-deps
```

Large artifacts default to `/mnt/hd/lewis/DenseLatentCommunication`. Code stays
in this directory, so no bind mount is required. Override any path in the JSON
configuration when the models or artifacts live elsewhere.

The server-local and NFS archival layout, including synchronization commands,
is documented in [`STORAGE_LAYOUT.md`](STORAGE_LAYOUT.md).

## Reproduction sequence

Run all commands from this directory.

```bash
# 1. Unit and synthetic integration checks
PYTHONPATH=src $DENSE_KV_PYTHON -m unittest discover -s tests -v

# 2. Check the model files, runtime, GPU, and exact Qwen configuration
PYTHONPATH=src $DENSE_KV_PYTHON scripts/preflight.py \
  --config configs/4b_to_8b.json

# Optional full-width, one-step real-model smoke in each direction
PYTHONPATH=src $DENSE_KV_PYTHON scripts/train_phase1.py \
  --config configs/4b_to_8b_smoke.json
PYTHONPATH=src $DENSE_KV_PYTHON scripts/train_phase1.py \
  --config configs/8b_to_4b_smoke.json

# 3. Download/normalize the requested train and evaluation datasets
PYTHONPATH=src $DENSE_KV_PYTHON scripts/prepare_data.py \
  --output-dir /mnt/hd/lewis/DenseLatentCommunication/data

# 4. Phase I in each direction
PYTHONPATH=src $DENSE_KV_PYTHON scripts/train_phase1.py \
  --config configs/4b_to_8b.json
PYTHONPATH=src $DENSE_KV_PYTHON scripts/train_phase1.py \
  --config configs/8b_to_4b.json

# 5. Generate receiver-self traces in each direction
PYTHONPATH=src $DENSE_KV_PYTHON scripts/generate_traces.py \
  --config configs/4b_to_8b.json
PYTHONPATH=src $DENSE_KV_PYTHON scripts/generate_traces.py \
  --config configs/8b_to_4b.json

# 6. Phase II in each direction (2000 optimizer steps)
PYTHONPATH=src $DENSE_KV_PYTHON scripts/train_phase2.py \
  --config configs/4b_to_8b.json
PYTHONPATH=src $DENSE_KV_PYTHON scripts/train_phase2.py \
  --config configs/8b_to_4b.json

# 7. Receiver-only, T2T, context-aware, and context-unaware evaluation
PYTHONPATH=src $DENSE_KV_PYTHON scripts/evaluate.py \
  --config configs/4b_to_8b.json --all-modes
PYTHONPATH=src $DENSE_KV_PYTHON scripts/evaluate.py \
  --config configs/8b_to_4b.json --all-modes
```

Use `--max-samples`, `--max-steps`, and `--max-new-tokens` for smoke runs. Each
script writes JSONL metrics beside its checkpoint/output, making interrupted
runs auditable.

## Cache semantics

The cached source sequence is the user chat message without the assistant
generation prefix. In context-unaware mode the receiver gets only its assistant
generation prefix after the transformed cache; in context-aware mode it gets
the complete raw question prompt. This makes the distinction testable while
still giving the causal LM a real token from which to begin decoding.

Qwen3 thinking mode is explicitly enabled for Receiver-self trace generation,
Phase II, and evaluation. It can be disabled consistently with the
`training.enable_thinking` configuration flag.

Only keys are inverse-rotated and re-rotated. Qwen applies RoPE to queries and
keys, not values. The position IDs used during extraction are carried into the
adapter, including padded batches.

# Reproduction status

Last updated: 2026-08-21 (Asia/Singapore)

## Scope implemented

- Independent `4B_to_8B` and `8B_to_4B` adapter configurations.
- Fixed monotonic receiver-to-sender layer mapping.
- Direct same-index alignment for the eight Qwen3 KV heads.
- Independent K and V `128 -> 2048 -> 128` GELU MLPs for every receiver
  layer and KV head.
- One sigmoid gate per receiver layer/head, shared by that head's K and V.
- Sender inverse RoPE and receiver forward RoPE; values are not rotated.
- Phase-I cache reconstruction training.
- Receiver-self guided trace generation using the question and gold answer.
- Phase-II training with one adapter and 50/50 context-aware/context-unaware
  sampling.
- Receiver-only, T2T, context-aware, and context-unaware evaluation.
- GSM8K, MATH-500, and ARC-C answer extraction plus token/FLOPs accounting.

## Verified model facts

| Property | Qwen3-4B | Qwen3-8B |
| --- | ---: | ---: |
| Layers | 36 | 36 |
| KV heads | 8 | 8 |
| Head dimension | 128 | 128 |
| RoPE theta | 1,000,000 | 1,000,000 |
| Exact parameters | 4,022,468,096 | 8,190,735,360 |
| Hugging Face revision | `1cfa9a7208912126459214e8b04321603b3df60c` | `b968826d9c46dd6066d109eabc6255188de91218` |

The tokenizer JSON, tokenizer configuration, and vocabulary files have matching
SHA-256 hashes. Sampled chat token IDs also match with Qwen3 thinking both on
and off.

Each directional adapter has exactly 303,243,552 parameters (about 0.565 GiB
of BF16 weights). Full native caches are generated online rather than persisted.

## Verification completed

The test suite currently contains 12 passing tests. It covers:

- exact layer mapping;
- RoPE round-trip and equality with Transformers Qwen3 RoPE;
- adapter shapes, head isolation, gates, gradients, and checkpoint round-trip;
- masked reconstruction MSE;
- answer extraction;
- tiny Qwen3 native cache extraction, transformation, DynamicCache injection,
  decoding, and Phase-II backpropagation into the adapter while receiver model
  weights remain frozen.

Command:

```bash
PYTHONPATH=src /mnt/hd/lewis/conda-envs/c2c/bin/python \
  -m unittest discover -s tests -v
```

Both directional preflights pass against the downloaded model artifacts. They
verify the pinned revisions, model geometry, every indexed safetensors shard,
CUDA visibility, and the 303,243,552-parameter adapter shape.

## Full-width real-model smoke results

The complete 36-layer, 8-KV-head, expansion-16 adapters were exercised with
the real Qwen3-4B and Qwen3-8B weights. These are one-sample, one-optimizer-step
integration checks, not training-quality results.

| Direction | Phase-I reconstruction | Key MSE | Value MSE | Phase-II loss | Phase-II mode |
| --- | ---: | ---: | ---: | ---: | --- |
| 4B to 8B | 11.801210 | 8.655854 | 3.145357 | 0.916590 | context-unaware |
| 8B to 4B | 12.121027 | 9.908597 | 2.212430 | 1.258773 | context-unaware |

- Both Phase-I runs consumed 21 prompt tokens and saved 606,593,646-byte
  checkpoints.
- Each Receiver generated one 128-token self-guided trace from the question
  and gold answer.
- Both Phase-II runs loaded the corresponding Phase-I checkpoint, injected a
  transformed `DynamicCache`, backpropagated generation loss, and saved a
  Phase-II checkpoint.
- Receiver-only, T2T, context-aware, and context-unaware evaluation paths ran
  in both directions on one GSM8K fixture. Dense-KV modes used zero Sender
  decode tokens, as required. Accuracy was 0/1 in all modes because the
  128-token thinking-limited outputs did not reach a final answer; these
  values must not be reported as formal benchmark results.

Smoke artifacts are archived under
`/mnt/nfs/cloud/checkpoints/DenseLatentCommunication/smoke/`. The local
`/mnt/hd` smoke copy was removed after checksum verification to keep only
training-hot files on the nearly full server-local disk.

## Formal dataset preparation

The requested train/evaluation data has been downloaded and normalized to:

- `/mnt/hd/lewis/DenseLatentCommunication/data/train.jsonl`
- `/mnt/hd/lewis/DenseLatentCommunication/data/eval.jsonl`

Record counts:

| File | Records | Dataset breakdown |
| --- | ---: | --- |
| `train.jsonl` | 12,836 | GSM8K 7,473; MATH algebra 4,244; ARC-C 1,119 |
| `eval.jsonl` | 2,118 | GSM8K 1,319; MATH-500 500; ARC-C validation 299 |

All records include the required `id`, `dataset`, `question`, and
`gold_answer` fields. The normalized JSONL files occupy about 9.6 MiB.

The same files have been archived to
`/mnt/nfs/cloud/data/DenseLatentCommunication/`. Line counts and SHA-256 hashes
match the `/mnt/hd` active copies.

## Storage decision

At initial inspection `/home/lewis` had about 69 GiB free and `/mnt/hd` about
47 GiB free. After downloading both models and preparing the formal JSONL data,
`/mnt/hd` had about 29 GiB free. After moving cold smoke artifacts and the raw
Hugging Face cache to NFS, `/mnt/hd` has about 38 GiB free.
A bind mount would not create more free space and would add an unnecessary
privileged dependency. Code therefore remains in `/home/lewis`, while
model/data/formal-checkpoint paths point directly to `/mnt/hd/lewis`.
Only training-hot data should stay on `/mnt/hd`; archival data and raw caches
belong on `/mnt/nfs/cloud`.

## Runtime state

- Qwen3-4B: complete at `/mnt/hd/lewis/models/Qwen3-4B`.
- Qwen3-8B: all five official Hugging Face safetensors shards are complete at
  `/mnt/hd/lewis/models/Qwen3-8B` (16,381,516,776 shard bytes on disk).
- Python: `/mnt/hd/lewis/conda-envs/c2c/bin/python`.
- Transformers: 4.52.4.
- PyTorch: 2.12.0+cu130.
- Hardware visible to the kernel: RTX 3090 24 GiB and RTX PRO 6000 Blackwell
  96 GiB.
- CUDA device nodes are intentionally hidden inside the file sandbox. A
  read-only external query sees both GPUs and reports about 24.0 GiB free on
  the 3090 and 81.6 GiB free on the RTX PRO 6000. Training can use the same
  external execution path; no host `/dev` changes are needed.
- In the training Python environment, PyTorch enumerates `cuda:0` as the 96 GiB
  RTX PRO 6000 and `cuda:1` as the 24 GiB RTX 3090. Formal configs keep Qwen3-8B
  on `cuda:0`, Qwen3-4B on `cuda:1`, and place each adapter with its Receiver.
- Formal datasets are prepared under
  `/mnt/hd/lewis/DenseLatentCommunication/data`.
- Formal datasets are archived under
  `/mnt/nfs/cloud/data/DenseLatentCommunication`.
- Raw Hugging Face cache path `/mnt/hd/lewis/huggingface` is a symlink to
  `/mnt/nfs/cloud/cache/huggingface`.
- `/mnt/nfs/cloud` is mounted as `nfs4` from `lewis:/mnt/nfs/cloud`.

## Remaining execution sequence

1. Run the formal 2000-step Phase I for `4B_to_8B`, then `8B_to_4B`.
2. Generate full Receiver-self traces for each direction.
3. Run 2000 Phase-II optimizer steps for each direction.
4. Evaluate all four methods on GSM8K, MATH-500, and ARC-C.
5. Archive new formal checkpoints to NFS after each stage.

The preflight command validates configuration values, every indexed weight
shard, adapter size, CUDA visibility, and free artifact storage before a run:

```bash
PYTHONPATH=src /mnt/hd/lewis/conda-envs/c2c/bin/python scripts/preflight.py \
  --config configs/4b_to_8b.json
```

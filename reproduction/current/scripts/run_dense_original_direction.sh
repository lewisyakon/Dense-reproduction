#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly DENSE_ROOT="${PROJECT_ROOT}/vendor/DenseLatentCommunication"
readonly PYTHON_BIN="/mnt/hd/lewis/conda-envs/c2c/bin/python"
readonly CONFIG_ROOT="${PROJECT_ROOT}/configs/stage1/converters/dense_gsm8k_math_algebra_arc_c"
readonly OUTPUT_ROOT="/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/converters/dense_gsm8k_math_algebra_arc_c"
readonly LOCK_FILE="/run/user/$(id -u)/subproblem4-stage1-gpu.lock"
readonly PHYSICAL_GPU=0
readonly MIN_FREE_MIB=18000
readonly MODEL_GROUP="${1:?usage: run_dense_original_direction.sh MODEL_GROUP}"

case "${MODEL_GROUP}" in
  m1)
    readonly RUN_NAME="m1_sender_qwen3_0.6b__receiver_qwen3_1.7b"
    ;;
  m2)
    readonly RUN_NAME="m2_sender_qwen3_0.6b__receiver_qwen3_4b"
    ;;
  m3)
    readonly RUN_NAME="m3_sender_qwen3_4b__receiver_qwen3_0.6b"
    ;;
  *)
    echo "Unsupported Dense model group: ${MODEL_GROUP}" >&2
    exit 64
    ;;
esac

readonly CONFIG="${CONFIG_ROOT}/${RUN_NAME}.json"
readonly RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
test -x "${PYTHON_BIN}"
test -f "${CONFIG}"
test -f "/mnt/hd2/lewis/Subproblem4UnifiedReproduction/data/DenseLatentCommunication/train.jsonl"
install -d -m 0750 "${RUN_ROOT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Subproblem-4 GPU job holds ${LOCK_FILE}" >&2
  exit 75
fi

readonly FREE_MIB="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "Physical GPU ${PHYSICAL_GPU} has only ${FREE_MIB} MiB free; ${MIN_FREE_MIB} MiB is required" >&2
  exit 75
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
unset DENSE_KV_CUDA_MEMORY_FRACTION
export PYTHONPATH="${DENSE_ROOT}/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/mnt/hd2/lewis/huggingface-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

checkpoint_complete() {
  local checkpoint="$1"
  local expected_phase="$2"
  local expected_step="$3"
  [[ -f "${checkpoint}" ]] || return 1
  "${PYTHON_BIN}" -c \
    "import json,torch; cfg=json.load(open('${CONFIG}')); s=torch.load('${checkpoint}',map_location='cpu',weights_only=True); assert s['direction']==cfg['direction']; assert s['phase']==${expected_phase}; assert s['step']==${expected_step}; assert s['config']['paths']['train_data']=='/mnt/hd2/lewis/Subproblem4UnifiedReproduction/data/DenseLatentCommunication/train.jsonl'"
}

echo "Dense original-data preflight: ${MODEL_GROUP} ${RUN_NAME}"
"${PYTHON_BIN}" "${DENSE_ROOT}/scripts/preflight.py" --config "${CONFIG}"

if checkpoint_complete "${RUN_ROOT}/phase1.pt" 1 2000; then
  echo "Dense Phase I already complete: ${MODEL_GROUP}"
else
  echo "Dense Phase I: ${MODEL_GROUP}"
  "${PYTHON_BIN}" "${DENSE_ROOT}/scripts/train_phase1.py" --config "${CONFIG}"
  checkpoint_complete "${RUN_ROOT}/phase1.pt" 1 2000
fi

echo "Dense receiver-self traces: ${MODEL_GROUP}"
"${PYTHON_BIN}" "${DENSE_ROOT}/scripts/generate_traces.py" \
  --config "${CONFIG}" \
  --max-new-tokens 768

"${PYTHON_BIN}" -c \
  "from pathlib import Path; train=sum(1 for line in Path('/mnt/hd2/lewis/Subproblem4UnifiedReproduction/data/DenseLatentCommunication/train.jsonl').open() if line.strip()); traces=sum(1 for line in Path('${RUN_ROOT}/receiver_self_traces.jsonl').open() if line.strip()); assert traces==train,(traces,train); print('validated_traces',traces)"

if checkpoint_complete "${RUN_ROOT}/phase2.pt" 2 2000; then
  echo "Dense Phase II already complete: ${MODEL_GROUP}"
else
  echo "Dense Phase II: ${MODEL_GROUP}"
  "${PYTHON_BIN}" "${DENSE_ROOT}/scripts/train_phase2.py" --config "${CONFIG}"
  checkpoint_complete "${RUN_ROOT}/phase2.pt" 2 2000
fi

echo "Dense original-data training complete: ${MODEL_GROUP} ${RUN_NAME}"

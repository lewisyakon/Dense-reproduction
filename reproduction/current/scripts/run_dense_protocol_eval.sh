#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly ASSET_ROOT="/mnt/hd2/lewis/Subproblem4UnifiedReproduction"
readonly DENSE_ROOT="${PROJECT_ROOT}/vendor/DenseLatentCommunication"
readonly PYTHON_BIN="/mnt/hd/lewis/conda-envs/c2c/bin/python"
readonly PHYSICAL_GPU="${DENSE_EVAL_GPU:-0}"
readonly EVAL_SLOT="${DENSE_EVAL_SLOT:-0}"
readonly MIN_FREE_MIB="${DENSE_EVAL_MIN_FREE_MIB:-18000}"
readonly NO_PROMPT_TRUNCATION="${DENSE_EVAL_NO_PROMPT_TRUNCATION:-0}"
readonly LOCK_FILE="/run/user/$(id -u)/subproblem4-dense-eval-gpu-${PHYSICAL_GPU}-slot-${EVAL_SLOT}.lock"
readonly DIRECTION="${1:?usage: run_dense_protocol_eval.sh DIRECTION PROTOCOL TASK}"
readonly PROTOCOL="${2:?usage: run_dense_protocol_eval.sh DIRECTION PROTOCOL TASK}"
readonly TASK="${3:?usage: run_dense_protocol_eval.sh DIRECTION PROTOCOL TASK}"

case "${DIRECTION}" in
  qwen3_0.6b_to_1.7b)
    readonly RUN_NAME="m1_sender_qwen3_0.6b__receiver_qwen3_1.7b"
    ;;
  qwen3_0.6b_to_4b)
    readonly RUN_NAME="m2_sender_qwen3_0.6b__receiver_qwen3_4b"
    ;;
  qwen3_4b_to_0.6b)
    readonly RUN_NAME="m3_sender_qwen3_4b__receiver_qwen3_0.6b"
    ;;
  *) echo "Unsupported direction: ${DIRECTION}" >&2; exit 64 ;;
esac

case "${PROTOCOL}" in
  a|c) ;;
  *) echo "Unsupported protocol: ${PROTOCOL}" >&2; exit 64 ;;
esac

case "${TASK}" in
  arc_challenge)
    readonly INPUT="${ASSET_ROOT}/data/canonical/eval/arc_challenge_test.jsonl"
    ;;
  gsm8k)
    readonly INPUT="${ASSET_ROOT}/data/canonical/eval/gsm8k_test.jsonl"
    ;;
  hotpotqa_bridge)
    readonly INPUT="${ASSET_ROOT}/data/canonical/eval/hotpotqa_bridge_validation.jsonl"
    ;;
  humaneval)
    readonly INPUT="${ASSET_ROOT}/data/canonical/eval/humaneval_test.jsonl"
    ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 64 ;;
esac

readonly CONFIG="${PROJECT_ROOT}/configs/stage1/converters/dense_gsm8k_math_algebra_arc_c/${RUN_NAME}.json"
readonly CHECKPOINT="${ASSET_ROOT}/experiments/stage1/converters/dense_gsm8k_math_algebra_arc_c/${RUN_NAME}/phase2.pt"
readonly DEFAULT_OUTPUT_ROOT="${ASSET_ROOT}/experiments/stage1/evaluations/dense_original_data_v1"
readonly OUTPUT_ROOT="${DENSE_EVAL_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
readonly OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}/${TASK}/protocol_${PROTOCOL}"
test -f "${CONFIG}"
test -f "${CHECKPOINT}"
test -f "${INPUT}"
install -d -m 0750 "${OUTPUT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Subproblem-4 Stage-1 GPU job holds ${LOCK_FILE}" >&2
  exit 75
fi

readonly FREE_MIB="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "GPU ${PHYSICAL_GPU} has only ${FREE_MIB} MiB free; ${MIN_FREE_MIB} MiB required" >&2
  exit 75
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
unset DENSE_KV_CUDA_MEMORY_FRACTION || true
export PYTHONPATH="${DENSE_ROOT}/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

extra_args=()
if [[ "${NO_PROMPT_TRUNCATION}" == "1" ]]; then
  extra_args+=(--no-prompt-truncation)
elif [[ "${NO_PROMPT_TRUNCATION}" != "0" ]]; then
  echo "DENSE_EVAL_NO_PROMPT_TRUNCATION must be 0 or 1" >&2
  exit 64
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_dense_protocol.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --input "${INPUT}" \
  --task "${TASK}" \
  --protocol "${PROTOCOL}" \
  --output-dir "${OUTPUT}" \
  "${extra_args[@]}"

"${PYTHON_BIN}" -c \
  "import json; from pathlib import Path; expected=sum(1 for line in Path('${INPUT}').open() if line.strip()); summary=json.loads(Path('${OUTPUT}/summary.json').read_text()); assert summary['completed_rows']==expected,(summary['completed_rows'],expected); print('validated_rows',expected)"

if [[ "${NO_PROMPT_TRUNCATION}" == "1" ]]; then
  "${PYTHON_BIN}" -c \
    "import json; from pathlib import Path; summary=json.loads(Path('${OUTPUT}/summary.json').read_text()); assert summary['prompt_truncation_enabled'] is False,summary; assert summary['max_prompt_tokens'] is None,summary; print('validated_prompt_truncation_disabled')"
fi

if [[ "${TASK}" == "humaneval" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/score_humaneval.py" \
    --predictions "${OUTPUT}/predictions.jsonl" \
    --input "${INPUT}" \
    --summary "${OUTPUT}/summary.json" \
    --output "${OUTPUT}/humaneval_scores.jsonl"
fi

#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly ASSET_ROOT="/mnt/hd2/lewis/Subproblem4UnifiedReproduction"
readonly SWAT_OUTPUT_ROOT="/mnt/hd2/lewis/SwaTReproduction/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1"
readonly SWAT_STATUS_FILE="${SWAT_OUTPUT_ROOT}/queue.status"
readonly SWAT_QUEUE_PID="${SWAT_QUEUE_PID:-}"
readonly LAUNCHER="${PROJECT_ROOT}/scripts/run_dense_protocol_eval.sh"
readonly OUTPUT_ROOT="${ASSET_ROOT}/experiments/stage1/evaluations/dense_hotpotqa_no_prompt_limit_v1"
readonly STATUS_FILE="${OUTPUT_ROOT}/queue.status"
readonly PHYSICAL_GPU="${DENSE_EVAL_GPU:-0}"
readonly MIN_FREE_MIB="${DENSE_EVAL_MIN_FREE_MIB:-18000}"

install -d -m 0750 "${OUTPUT_ROOT}/logs"

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${STATUS_FILE}"
}

on_signal() {
  write_status "stopped signal_received"
  exit 143
}
trap on_signal INT TERM

wait_for_swat() {
  while true; do
    swat_status="missing"
    if [[ -f "${SWAT_STATUS_FILE}" ]]; then
      swat_status="$(cut -f2- "${SWAT_STATUS_FILE}")"
    fi
    if [[ "${swat_status}" == completed_all* ]]; then
      write_status "swat_completed; waiting_for_gpu gpu=${PHYSICAL_GPU}"
      return 0
    fi
    if [[ -n "${SWAT_QUEUE_PID}" ]] && ! kill -0 "${SWAT_QUEUE_PID}" 2>/dev/null; then
      write_status "failed prerequisite_swat_not_completed swat_status=${swat_status}"
      return 1
    fi
    write_status "waiting_for_swat swat_status=${swat_status}"
    sleep 30
  done
}

wait_for_gpu() {
  while true; do
    if nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits >/dev/null 2>&1; then
      free_mib="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
      if (( free_mib >= MIN_FREE_MIB )); then
        return 0
      fi
      write_status "waiting_for_gpu gpu=${PHYSICAL_GPU} free_mib=${free_mib} required_mib=${MIN_FREE_MIB}"
    else
      write_status "waiting_for_nvidia_driver gpu=${PHYSICAL_GPU}"
    fi
    sleep 30
  done
}

wait_for_swat

for direction in qwen3_0.6b_to_1.7b qwen3_0.6b_to_4b qwen3_4b_to_0.6b; do
  case "${direction}" in
    qwen3_0.6b_to_1.7b) run_name="m1_sender_qwen3_0.6b__receiver_qwen3_1.7b" ;;
    qwen3_0.6b_to_4b) run_name="m2_sender_qwen3_0.6b__receiver_qwen3_4b" ;;
    qwen3_4b_to_0.6b) run_name="m3_sender_qwen3_4b__receiver_qwen3_0.6b" ;;
  esac
  for protocol in a c; do
    output_dir="${OUTPUT_ROOT}/${run_name}/hotpotqa_bridge/protocol_${protocol}"
    summary="${output_dir}/summary.json"
    expected="$(awk 'NF {count++} END {print count+0}' "${ASSET_ROOT}/data/canonical/eval/hotpotqa_bridge_validation.jsonl")"
    completed=0
    if [[ -f "${summary}" ]]; then
      completed="$(/mnt/hd/lewis/conda-envs/c2c/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_rows"])' "${summary}")"
    fi
    if [[ "${completed}" == "${expected}" ]]; then
      prompt_limit="$(/mnt/hd/lewis/conda-envs/c2c/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("max_prompt_tokens", "missing"))' "${summary}")"
      if [[ "${prompt_limit}" == "None" ]]; then
        write_status "skipped_complete direction=${direction} protocol=${protocol} rows=${completed} prompt_limit=none"
        continue
      fi
      write_status "failed existing_complete_output_has_prompt_limit direction=${direction} protocol=${protocol} value=${prompt_limit}"
      exit 65
    fi

    wait_for_gpu
    write_status "running direction=${direction} protocol=${protocol} task=hotpotqa_bridge completed=${completed}/${expected} prompt_limit=none gpu=${PHYSICAL_GPU}"
    set +e
    DENSE_EVAL_GPU="${PHYSICAL_GPU}" \
    DENSE_EVAL_MIN_FREE_MIB="${MIN_FREE_MIB}" \
    DENSE_EVAL_NO_PROMPT_TRUNCATION=1 \
    DENSE_EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
      "${LAUNCHER}" "${direction}" "${protocol}" hotpotqa_bridge \
      >>"${OUTPUT_ROOT}/logs/${direction}__protocol_${protocol}__hotpotqa_bridge.log" 2>&1
    exit_status=$?
    set -e
    if (( exit_status != 0 )); then
      write_status "failed direction=${direction} protocol=${protocol} exit=${exit_status}"
      exit "${exit_status}"
    fi
    write_status "completed direction=${direction} protocol=${protocol} rows=${expected} prompt_limit=none"
  done
done

write_status "completed_all method=Dense dataset=hotpotqa_bridge pairs=M1,M2,M3 protocols=a,c prompt_limit=none"

#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly ASSET_ROOT="/mnt/hd2/lewis/Subproblem4UnifiedReproduction"
readonly LAUNCHER="${PROJECT_ROOT}/scripts/run_dense_protocol_eval.sh"
readonly OUTPUT_ROOT="${ASSET_ROOT}/experiments/stage1/evaluations/dense_original_data_v1"
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
    sleep 60
  done
}

for direction in qwen3_0.6b_to_1.7b qwen3_0.6b_to_4b qwen3_4b_to_0.6b; do
  case "${direction}" in
    qwen3_0.6b_to_1.7b) run_name="m1_sender_qwen3_0.6b__receiver_qwen3_1.7b" ;;
    qwen3_0.6b_to_4b) run_name="m2_sender_qwen3_0.6b__receiver_qwen3_4b" ;;
    qwen3_4b_to_0.6b) run_name="m3_sender_qwen3_4b__receiver_qwen3_0.6b" ;;
  esac
  for protocol in a c; do
    for task in gsm8k hotpotqa_bridge humaneval; do
      output_dir="${OUTPUT_ROOT}/${run_name}/${task}/protocol_${protocol}"
      summary="${output_dir}/summary.json"
      input="${ASSET_ROOT}/data/canonical/eval/${task}_test.jsonl"
      if [[ "${task}" == "hotpotqa_bridge" ]]; then
        input="${ASSET_ROOT}/data/canonical/eval/hotpotqa_bridge_validation.jsonl"
      fi
      expected="$(awk 'NF {count++} END {print count+0}' "${input}")"
      completed=0
      if [[ -f "${summary}" ]]; then
        completed="$(/mnt/hd/lewis/conda-envs/c2c/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_rows"])' "${summary}")"
      fi
      if [[ "${completed}" == "${expected}" ]]; then
        write_status "skipped_complete direction=${direction} protocol=${protocol} task=${task} rows=${completed}"
        continue
      fi

      install -d -m 0750 "${output_dir}"
      while true; do
        wait_for_gpu
        write_status "running direction=${direction} protocol=${protocol} task=${task} completed=${completed}/${expected}"
        set +e
        DENSE_EVAL_GPU="${PHYSICAL_GPU}" DENSE_EVAL_MIN_FREE_MIB="${MIN_FREE_MIB}" \
          "${LAUNCHER}" "${direction}" "${protocol}" "${task}" \
          >>"${OUTPUT_ROOT}/logs/${direction}__protocol_${protocol}__${task}.log" 2>&1
        exit_status=$?
        set -e
        if (( exit_status != 75 )); then
          break
        fi
        write_status "resource_race direction=${direction} protocol=${protocol} task=${task}; retrying_in_60_seconds"
        sleep 60
      done
      if (( exit_status != 0 )); then
        write_status "failed direction=${direction} protocol=${protocol} task=${task} exit=${exit_status}"
        exit "${exit_status}"
      fi
      write_status "completed direction=${direction} protocol=${protocol} task=${task} rows=${expected}"
    done
  done
done

write_status "completed_all methods=Dense datasets=gsm8k,hotpotqa_bridge,humaneval protocols=a,c order=m1,m2,m3"

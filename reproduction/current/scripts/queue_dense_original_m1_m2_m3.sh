#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly LAUNCHER="${PROJECT_ROOT}/scripts/run_dense_original_direction.sh"
readonly OUTPUT_ROOT="/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/converters/dense_gsm8k_math_algebra_arc_c"
readonly PHYSICAL_GPU=0
readonly MIN_FREE_MIB=18000

wait_for_local_assets() {
  while [[ ! -d /mnt/hd2/lewis/Subproblem4UnifiedReproduction ]]; do
    sleep 60
  done
}

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${OUTPUT_ROOT}/queue.status"
}

wait_for_gpu() {
  while true; do
    if nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits >/dev/null 2>&1; then
      free_mib="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
      if (( free_mib >= MIN_FREE_MIB )); then
        return 0
      fi
      write_status "waiting_for_gpu_${PHYSICAL_GPU}_free_memory current_free_mib=${free_mib} required_free_mib=${MIN_FREE_MIB}"
    else
      write_status "waiting_for_nvidia_driver"
    fi
    sleep 60
  done
}

wait_for_local_assets
install -d -m 0750 "${OUTPUT_ROOT}"

for model_group in m1 m2 m3; do
  case "${model_group}" in
    m1) run_name="m1_sender_qwen3_0.6b__receiver_qwen3_1.7b" ;;
    m2) run_name="m2_sender_qwen3_0.6b__receiver_qwen3_4b" ;;
    m3) run_name="m3_sender_qwen3_4b__receiver_qwen3_0.6b" ;;
  esac
  output_dir="${OUTPUT_ROOT}/${run_name}"
  install -d -m 0750 "${output_dir}"

  if [[ -f "${output_dir}/phase2.pt" ]]; then
    write_status "checking_existing_phase2 model_group=${model_group} output=${output_dir}"
  fi

  while true; do
    wait_for_gpu
    write_status "running model_group=${model_group} output=${output_dir}"
    set +e
    "${LAUNCHER}" "${model_group}" >>"${output_dir}/training.log" 2>&1
    exit_status=$?
    set -e
    if (( exit_status != 75 )); then
      break
    fi
    write_status "resource_race model_group=${model_group}; retrying_in_60_seconds"
    sleep 60
  done

  if (( exit_status != 0 )); then
    write_status "failed model_group=${model_group} exit_status=${exit_status} log=${output_dir}/training.log"
    exit "${exit_status}"
  fi
  if [[ ! -f "${output_dir}/phase2.pt" ]]; then
    write_status "failed model_group=${model_group} reason=missing_phase2_checkpoint"
    exit 1
  fi
  write_status "completed model_group=${model_group} output=${output_dir}"
done

write_status "completed_all order=m1,m2,m3"

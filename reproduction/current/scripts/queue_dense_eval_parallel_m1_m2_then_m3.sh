#!/usr/bin/env bash
set -uo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly ASSET_ROOT="/mnt/hd2/lewis/Subproblem4UnifiedReproduction"
readonly GROUP_QUEUE="${PROJECT_ROOT}/scripts/queue_dense_eval_group.sh"
readonly OUTPUT_ROOT="${ASSET_ROOT}/experiments/stage1/evaluations/dense_original_data_v1"
readonly STATUS_FILE="${OUTPUT_ROOT}/orchestrator.status"

install -d -m 0750 "${OUTPUT_ROOT}/logs"

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${STATUS_FILE}"
}

m1_pid=""
m2_pid=""
m3_pid=""

stop_children() {
  for pid in "${m1_pid}" "${m2_pid}" "${m3_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}

trap 'write_status "stopped signal_received"; stop_children; exit 143' INT TERM

write_status "starting_parallel gpu=0 groups=M1,M2 max_total_used_mib=19000"

env \
  DENSE_EVAL_GPU=0 \
  DENSE_EVAL_SLOT=0 \
  DENSE_EVAL_MIN_FREE_MIB=8000 \
  "${GROUP_QUEUE}" M1 \
  >>"${OUTPUT_ROOT}/logs/orchestrator_M1.log" 2>&1 &
m1_pid=$!

env \
  DENSE_EVAL_GPU=0 \
  DENSE_EVAL_SLOT=1 \
  DENSE_EVAL_MIN_FREE_MIB=12000 \
  DENSE_EVAL_GUARDED=1 \
  DENSE_EVAL_MAX_TOTAL_USED_MIB=19000 \
  "${GROUP_QUEUE}" M2 \
  >>"${OUTPUT_ROOT}/logs/orchestrator_M2.log" 2>&1 &
m2_pid=$!

write_status "running_parallel gpu=0 M1_pid=${m1_pid} M2_pid=${m2_pid} max_total_used_mib=19000"

wait "${m1_pid}"
m1_status=$?
m1_pid=""
wait "${m2_pid}"
m2_status=$?
m2_pid=""

if (( m1_status != 0 || m2_status != 0 )); then
  write_status "failed_parallel M1_exit=${m1_status} M2_exit=${m2_status}"
  exit 1
fi

write_status "starting_serial gpu=0 group=M3 min_free_mib=18000"
env \
  DENSE_EVAL_GPU=0 \
  DENSE_EVAL_SLOT=0 \
  DENSE_EVAL_MIN_FREE_MIB=18000 \
  "${GROUP_QUEUE}" M3 \
  >>"${OUTPUT_ROOT}/logs/orchestrator_M3.log" 2>&1 &
m3_pid=$!
wait "${m3_pid}"
m3_status=$?
m3_pid=""

if (( m3_status != 0 )); then
  write_status "failed_serial M3_exit=${m3_status}"
  exit 1
fi

write_status "completed_all groups=M1,M2,M3 datasets=gsm8k,hotpotqa_bridge,humaneval protocols=a,c"

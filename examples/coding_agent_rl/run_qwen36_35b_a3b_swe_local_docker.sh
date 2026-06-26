#!/usr/bin/env bash
# Single-node, no-E2B smoke launcher for Qwen3.6-35B-A3B coding-agent RL.
# It keeps the upstream example's model/task wiring but uses local Docker
# containers for the agent/eval sandboxes and defaults to rollout-only debug.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [[ -z "${ACTOR_NUM_GPUS_PER_NODE:-}" ]]; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  export ACTOR_NUM_GPUS_PER_NODE="${DETECTED_GPUS:-4}"
fi

export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-local_docker}"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-host.docker.internal}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"

export HOSTFILE="${HOSTFILE:-/dev/null}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_NUM_GPUS_PER_NODE}}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-${ROLLOUT_NUM_GPUS}}"
export ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-1}"
export ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-${ROLLOUT_NUM_GPUS}}"

export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-1}"
export ETP_SIZE="${ETP_SIZE:-1}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32768}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"

export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-1}"
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-600}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-300}"
export DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-1}"
export EXP_TAG="${EXP_TAG:-qwen36_35b_a3b_swe_local_docker}"

exec bash "${SCRIPT_DIR}/run_qwen36_35b_a3b_swe_8nodes.sh"

#!/usr/bin/env bash
# Single-node, no-E2B, no-external-agent-CLI training smoke for the
# Qwen3.6-35B-A3B coding-agent example. This intentionally runs one local
# scripted "agent" turn through slime's adapter, then performs one train update.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-local_docker}"
export SWE_AGENT="${SWE_AGENT:-local_script}"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-host.docker.internal}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export MEGATRON_DIR="${MEGATRON_DIR:-/workspace/Megatron-LM}"

export HOSTFILE="${HOSTFILE:-/dev/null}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_NUM_GPUS_PER_NODE}}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-${ROLLOUT_NUM_GPUS}}"
export ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-1}"
export ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-${ROLLOUT_NUM_GPUS}}"

export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-${ACTOR_NUM_GPUS_PER_NODE}}"
export ETP_SIZE="${ETP_SIZE:-1}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
# On the 8-GPU Qwen3.6-35B-A3B smoke, actor DP=8; Megatron requires
# global_batch_size to divide micro_batch_size * data_parallel_size.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-1024}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-32}"
export ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.55}"
export MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
export ENABLE_DEEPEP="${ENABLE_DEEPEP:-0}"
export DISABLE_GRAD_ACCUM_FUSION="${DISABLE_GRAD_ACCUM_FUSION:-1}"

export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-1}"
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-180}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-180}"
export SLIME_AGENT_LOCAL_SCRIPT_MAX_TOKENS="${SLIME_AGENT_LOCAL_SCRIPT_MAX_TOKENS:-8}"
export SLIME_AGENT_LOCAL_SCRIPT_COMMAND="${SLIME_AGENT_LOCAL_SCRIPT_COMMAND:-PYTHONPATH=/workspace/slime python -m pytest -q tests/test_agent/test_trajectory_manager_branching.py}"

export DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/local_docker/slime_train_smoke.jsonl}"
export EXP_TAG="${EXP_TAG:-qwen36_35b_a3b_swe_local_docker_train_smoke}"

exec bash "${SCRIPT_DIR}/run_qwen36_35b_a3b_swe_8nodes.sh"

#!/usr/bin/env bash
# Tiny Moonlight-16B-A3B INT4 MoE smoke.
#
# This is a dependency-light first run for slime's MoE path. It keeps the real
# Megatron + SGLang colocated training loop, but uses a four-row local math
# dataset, one rollout, one sample per prompt, and short generations.

set -euo pipefail

if [[ "${SLIME_SKIP_CLEANUP:-0}" != "1" ]]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 redis || true
  sleep 3
fi

if [[ "${SLIME_TRACE:-1}" == "1" ]]; then
  set -x
fi

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"

source "${SLIME_DIR}/scripts/models/moonlight.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Moonlight-16B-A3B-Instruct-INT4}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/root/Moonlight-16B-A3B-Instruct-INT4_torch_dist}"
LOAD_PATH="${LOAD_PATH:-/root/Moonlight-16B-A3B_smoke_slime}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/scripts/smoke_data/moonlight_math_smoke.jsonl}"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/moonlight_16b_a3b_int4_smoke_$(date +%Y%m%d_%H%M%S)}"

ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-4}"
ETP_SIZE="${ETP_SIZE:-1}"

NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-1024}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.2}"

ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ACTOR_NUM_GPUS_PER_NODE}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.35}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-4}"

for required_path in "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${SAVE_PATH}" "${RUN_ROOT}"

TOPO_OUTPUT="$(nvidia-smi topo -m 2>/dev/null || true)"
NVLINK_COUNT="$(grep -o 'NV[0-9][0-9]*' <<<"${TOPO_OUTPUT}" | wc -l | tr -d ' ' || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} detected_nvlink_refs=${NVLINK_COUNT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
   --load "${LOAD_PATH}"
   --save "${SAVE_PATH}"
   --save-interval 1000000
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size 256
   --train-memory-margin-bytes 268435456
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus "${ACTOR_NUM_GPUS_PER_NODE}"
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-cuda-graph-bs 1 2 4
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --colocate
   --ci-test
)

# scripts/models/moonlight.sh defaults to alltoall; DeepEP is opt-in because it
# depends on the host/container having that communication stack available.
if [[ "${ENABLE_DEEPEP:-0}" == "1" ]]; then
  MISC_ARGS+=(--moe-enable-deepep --moe-token-dispatcher-type flex)
fi

cd "${SLIME_DIR}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="$(
  python3 - <<PY
import json, os
env = {
    "PYTHONPATH": ":".join(p for p in ("${MEGATRON_DIR}", "${SLIME_DIR}", os.environ.get("PYTHONPATH", "")) if p),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NVSHMEM_DISABLE_NCCL": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
}
for key in ("CUDA_HOME", "PATH", "LD_LIBRARY_PATH", "HF_HOME"):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps({"env_vars": env}))
PY
)"

LOG_FILE="${RUN_ROOT}/run.log"
echo "Moonlight smoke log: ${LOG_FILE}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"

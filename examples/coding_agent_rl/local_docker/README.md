# Local Docker Qwen3.6-35B-A3B Train Smoke

This is the dependency-light path for the coding-agent RL example. It keeps the
SLIME/Megatron/SGLang training stack, but removes the external coding-agent
sandbox dependencies:

- no E2B
- no Claude Code CLI tarball
- no Codex CLI tarball
- no external SWE harness service

The sandbox is a local Docker container and the agent is `local_script`, which
talks to SLIME's Anthropic-compatible adapter once, then runs a local eval
command in the container.

## Proven Configuration

The smoke was proved on one `8xH100 80GB` node with:

- model: `Qwen/Qwen3.6-35B-A3B`
- `ACTOR_NUM_GPUS_PER_NODE=8`
- `ROLLOUT_NUM_GPUS=8`
- `ROLLOUT_TP_SIZE=8`
- `ROLLOUT_EP_SIZE=8`
- `GLOBAL_BATCH_SIZE=8`
- `MICRO_BATCH_SIZE=1`
- `ROLLOUT_BATCH_SIZE=8`
- `NUM_ROLLOUT=1`
- `MAX_CONTEXT_LEN=512`
- `MAX_GEN_LEN=32`
- `ROLLOUT_MEM_UTILIZATION=0.20`

`GLOBAL_BATCH_SIZE=1` is not a valid floor for this 8-way actor shape. Megatron
requires the global batch to divide `micro_batch_size * data_parallel_size`, so
the proved minimum here is global batch `8`.

The exact train-smoke commit was:

```bash
2758675c Use dependency-light coding-agent train smoke check
```

Later docs-only commits may sit on top of that commit.

## Prerequisites

Use a normal SLIME GPU training environment with:

- 8 visible GPUs for the proved smoke shape
- CUDA/PyTorch/NCCL suitable for SGLang and Megatron
- Docker available on the Ray head / rollout host
- a local SLIME checkout from this branch
- NVIDIA Megatron-LM available at `MEGATRON_DIR`
- the Hugging Face model downloaded locally
- a converted Megatron `torch_dist` checkpoint

This runbook removes external coding-agent and sandbox services. It does not
remove the need for normal training dependencies or model weights.

## Clone The Fork Branch

```bash
git clone --branch tokenbender/local-docker-coding-agent-rl \
  https://github.com/tokenbender/slime.git
cd slime
```

Install SLIME following the upstream project setup for Megatron + SGLang. The
proved pod used an editable SLIME install and Megatron-LM on `PYTHONPATH`.

## Prepare Megatron-LM

Use NVIDIA Megatron-LM and apply SLIME's Megatron patch:

```bash
git clone https://github.com/NVIDIA/Megatron-LM /workspace/Megatron-LM
cd /workspace/Megatron-LM
git checkout 1dcf0dafa884ad52ffb243625717a3471643e087
git apply /workspace/slime/docker/patch/latest/megatron.patch
```

Then return to SLIME:

```bash
cd /workspace/slime
export MEGATRON_DIR=/workspace/Megatron-LM
export PYTHONPATH="${MEGATRON_DIR}:$(pwd):${PYTHONPATH:-}"
```

## Download And Convert The Model

Download the Hugging Face checkpoint:

```bash
mkdir -p /workspace/models
huggingface-cli download Qwen/Qwen3.6-35B-A3B \
  --local-dir /workspace/models/Qwen3.6-35B-A3B
```

Convert it to Megatron `torch_dist`:

```bash
export HF_CHECKPOINT=/workspace/models/Qwen3.6-35B-A3B
export REF_MODEL_PATH=/workspace/models/Qwen3.6-35B-A3B_torch_dist
export MEGATRON_DIR=/workspace/Megatron-LM
export TORCHRUN_NPROC_PER_NODE=8

bash examples/coding_agent_rl/convert_qwen36_35b_a3b_hf_to_torch_dist.sh
```

The proved pod produced roughly:

- HF checkpoint: `67G`
- Megatron torch_dist checkpoint: `65G`

## Build The Local Sandbox Image

From the SLIME repo root:

```bash
docker build -f examples/coding_agent_rl/local_docker/Dockerfile \
  -t slime-cagent-smoke:latest .
```

The smoke dataset references this image name in `metadata.image`.

## Run The Train Smoke

From the SLIME repo root:

```bash
export HF_CHECKPOINT=/workspace/models/Qwen3.6-35B-A3B
export REF_MODEL_PATH=/workspace/models/Qwen3.6-35B-A3B_torch_dist
export MEGATRON_DIR=/workspace/Megatron-LM
export PYTHONPATH="${MEGATRON_DIR}:$(pwd):${PYTHONPATH:-}"

export ACTOR_NUM_GPUS_PER_NODE=8
export ROLLOUT_NUM_GPUS=8
export ROLLOUT_TP_SIZE=8
export ROLLOUT_EP_SIZE=8
export ROLLOUT_DP_SIZE=1
export EP_SIZE=8
export TP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
export ETP_SIZE=1

export NUM_ROLLOUT=1
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=1
export GLOBAL_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export MAX_CONTEXT_LEN=512
export MAX_GEN_LEN=32
export ROLLOUT_MEM_UTILIZATION=0.20

export SWE_AGENT=local_script
export SLIME_AGENT_SANDBOX_BACKEND=local_docker
export ADAPTER_PUBLIC_HOST=host.docker.internal
export ADAPTER_BIND_HOST=0.0.0.0
export PROMPT_DATA=examples/coding_agent_rl/local_docker/slime_train_smoke.jsonl
export EXTRA_TRAIN_ARGS="--no-offload-train --sglang-max-total-tokens 4096 --sglang-max-running-requests 8 --sglang-server-concurrency 8"

bash examples/coding_agent_rl/run_qwen36_35b_a3b_swe_local_docker_train_smoke.sh
```

Most of these environment variables are already the launcher's defaults; they
are expanded here so a run log can show the exact intended shape.

## Expected Success Markers

A successful run should include:

- SGLang health check succeeds
- Megatron loads `REF_MODEL_PATH`
- first `Update weights` reaches `128/128`
- 8 rollout lines show `reward=1.00 applied=True agent_exit_code=0`
- rollout metrics include `rollout/zero_std/count_1.0: 8`
- `Timer actor_train end` appears
- train metrics include `train/global_batch_size: 8` and `train/step: 0`
- second `Update weights` reaches `128/128`
- Ray reports the job succeeded

The proved smoke had intentionally tiny throughput numbers. Treat it as a
correctness smoke, not a performance profile.

## Extending Beyond The Smoke

To turn this into a real coding-agent experiment, replace
`slime_train_smoke.jsonl` with a real dataset where each row has:

- `prompt`
- `label`
- `metadata.image`
- `metadata.workdir`
- `metadata.problem_statement`
- `metadata.eval_cmd` or a richer evaluator payload

For real SWE tasks you will also want a richer sandbox image with the target
repo and its test dependencies installed. The local Docker backend still avoids
E2B; it just runs those images locally on the rollout host.

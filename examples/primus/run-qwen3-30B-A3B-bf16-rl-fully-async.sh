#!/bin/bash
# Fully-async GRPO on Qwen3-30B-A3B (MoE) with the Primus backend.
# Disaggregated single-node layout: train_async.py rejects --colocate.
# 4 train GPUs: TP=1 PP=1 CP=1 EP=4. 4 rollout GPUs: 2 engines x TP=2.
# DAPO-Math-17k train + AIME-2024 eval. rm-type math grades \\boxed{} on the
# full response (deepscaler requires </think> in the generation, which Qwen3
# does not emit when enable_thinking=false).
# 8k responses, n=8, CUDA graphs. Uses /root/Megatron-LM.
set -euo pipefail

MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-/root/Primus}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang/python}

TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
NUM_GPUS=${NUM_GPUS:-$((TRAIN_GPUS + ROLLOUT_GPUS))}
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/qwen3-30B-A3B-bf16-rl.yaml}
HF_CKPT=${HF_CKPT:-/root/models/Qwen3-30B-A3B}

export PYTHONPATH="${MILES_DIR}:${MEGATRON_DIR}:${PRIMUS_DIR}:${SGLANG_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PRIMUS_PATH="${PRIMUS_DIR}"

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
ROCM_CORE_LIB="${SITE_PACKAGES}/_rocm_sdk_core/lib"
if [[ -d "${ROCM_CORE_LIB}" ]]; then
  export LD_LIBRARY_PATH="${ROCM_CORE_LIB}:${ROCM_CORE_LIB}/llvm/lib:${ROCM_CORE_LIB}/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
fi

export GPU_ARCHS=${GPU_ARCHS_OVERRIDE:-gfx950}
export CU_NUM=${CU_NUM:-256}
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export no_proxy=127.0.0.1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MILES_ALLOW_UNSAFE_PRIMUS_TURBO="${MILES_ALLOW_UNSAFE_PRIMUS_TURBO:-}"

cd "${MILES_DIR}"

pkill -9 sglang || true
ray stop --force || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

source scripts/models/qwen3-30B-A3B.sh

TRAIN_BACKEND=primus
PRIMUS_FLAG=(--primus-config "${PRIMUS_CONFIG}" --primus-path "${PRIMUS_DIR}")
if [[ "${PRIMUS_CONFIG}" == "none" ]]; then
  TRAIN_BACKEND=megatron
  PRIMUS_FLAG=()
fi

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"${PYTHONPATH}\", \"PRIMUS_PATH\": \"${PRIMUS_DIR}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\", \"PATH\": \"${PATH}\", \"GPU_ARCHS\": \"${GPU_ARCHS}\", \"CU_NUM\": \"${CU_NUM}\", \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"MASTER_ADDR\": \"127.0.0.1\", \"no_proxy\": \"127.0.0.1\", \"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES\": \"1\", \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\", \"MILES_ALLOW_UNSAFE_PRIMUS_TURBO\": \"${MILES_ALLOW_UNSAFE_PRIMUS_TURBO:-}\"}}" \
  -- python3 train_async.py \
  "${MODEL_ARGS[@]}" \
  "${PRIMUS_FLAG[@]}" \
  --train-backend "${TRAIN_BACKEND}" \
  --hf-checkpoint "${HF_CKPT}" \
  --megatron-to-hf-mode bridge \
  --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --apply-chat-template-kwargs '{"enable_thinking": false}' \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout "${NUM_ROLLOUT:-100}" \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}" \
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-16384}" \
  --rollout-temperature 1.0 \
  --global-batch-size 32 \
  --eval-interval "${EVAL_INTERVAL:-5}" \
  --eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl \
  --eval-input-key prompt \
  --eval-label-key label \
  --n-samples-per-eval-prompt 1 \
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}" \
  --eval-top-k 1 \
  --advantage-estimator grpo \
  --kl-coef 0.00 \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --fully-async \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "${TRAIN_GPUS}" \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --qkv-format bshd \
  --micro-batch-size 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend unfused \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.5 \
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-32}" \
  --calculate-per-token-loss \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TRAIN_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  "$@"

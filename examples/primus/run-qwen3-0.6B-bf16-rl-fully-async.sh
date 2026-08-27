#!/bin/bash
# Fully-async GRPO on Qwen3-0.6B with the Primus backend.
# Disaggregated single-node layout: train_async.py rejects --colocate.
# Uses the Megatron checkout already in this image (/root/Megatron-LM).
set -euo pipefail

MILES_DIR=${MILES_DIR:-/root/miles}
MEGATRON_DIR=${MEGATRON_DIR:-/root/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-/root/Primus}
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang/python}

TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
NUM_GPUS=${NUM_GPUS:-$((TRAIN_GPUS + ROLLOUT_GPUS))}
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/qwen3-0.6B-bf16-rl.yaml}
HF_CKPT=${HF_CKPT:-/root/models/Qwen3-0.6B}

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
# Required to enable Primus-Turbo attention; empty keeps Miles' default refuse.
export MILES_ALLOW_UNSAFE_PRIMUS_TURBO="${MILES_ALLOW_UNSAFE_PRIMUS_TURBO:-}"

cd "${MILES_DIR}"

pkill -9 sglang || true
ray stop --force || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

source scripts/models/qwen3-0.6B.sh

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
  --prompt-data /root/datasets/gsm8k/train.parquet \
  --input-key messages \
  --label-key label \
  --apply-chat-template \
  --apply-chat-template-kwargs '{"enable_thinking": false}' \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout "${NUM_ROLLOUT:-3}" \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 256 \
  --rollout-temperature 1.0 \
  --global-batch-size 16 \
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
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --qkv-format bshd \
  --micro-batch-size 1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend unfused \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.4 \
  --sglang-disable-cuda-graph \
  --calculate-per-token-loss \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TRAIN_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  "$@"

#!/bin/bash
# GRPO on Qwen3-0.6B with the Primus backend: Megatron, configured through a Primus YAML
# and running Primus' kernels.
# bf16 only, single node, MI300X (gfx942).
set -euo pipefail

MILES_DIR=${MILES_DIR:-/workspace/miles}
MEGATRON_DIR=${MEGATRON_DIR:-/workspace/Megatron-LM}
PRIMUS_DIR=${PRIMUS_DIR:-/workspace/Primus}
SGLANG_DIR=${SGLANG_DIR:-/workspace/sglang/python}

TRAIN_GPUS=${TRAIN_GPUS:-4}
# Colocated puts the rollout engines and the trainer on the same GPUs and offloads
# between them; set COLOCATE=0 to give each its own GPUs instead.
COLOCATE=${COLOCATE:-1}
if [[ "${COLOCATE}" == "1" ]]; then
  NUM_GPUS=${NUM_GPUS:-${TRAIN_GPUS}}
  PLACEMENT_ARGS=(--num-gpus-per-node "${NUM_GPUS}" --colocate)
else
  NUM_GPUS=${NUM_GPUS:-$((TRAIN_GPUS * 2))}
  PLACEMENT_ARGS=(--num-gpus-per-node "${NUM_GPUS}" --rollout-num-gpus "$((NUM_GPUS - TRAIN_GPUS))")
fi
PRIMUS_CONFIG=${PRIMUS_CONFIG:-${MILES_DIR}/examples/primus/qwen3-0.6B-bf16-rl.yaml}
HF_CKPT=${HF_CKPT:-$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)}

export PYTHONPATH="${MILES_DIR}:${MEGATRON_DIR}:${PRIMUS_DIR}:${SGLANG_DIR}"

# This image ships two ROCm trees: _rocm_sdk_core (what torch loads) and
# _rocm_sdk_devel (the compiler, and what LD_LIBRARY_PATH points at). Libraries
# that pull hiprtc/comgr straight from devel - Transformer Engine and aiter among
# them - end up with a second copy of libLLVM in the process and abort with
# "Option 'spirv-expand-step' registered more than once". Preferring core keeps
# one copy for everyone.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
ROCM_CORE_LIB="${SITE_PACKAGES}/_rocm_sdk_core/lib"
export LD_LIBRARY_PATH="${ROCM_CORE_LIB}:${ROCM_CORE_LIB}/llvm/lib:${ROCM_CORE_LIB}/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"

# aiter JIT-compiles its kernels and resolves the target arch by shelling out to
# rocminfo, which is not on the Ray workers' PATH. When that lookup comes back
# empty the generated Makefile carries "--offload-arch=" and hipcc silently falls
# back to gfx906, which then fails to compile. Pin the arch instead.
export GPU_ARCHS=${GPU_ARCHS_OVERRIDE:-$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])")}
export CU_NUM=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)")
export PYTHONUNBUFFERED=16
export MASTER_ADDR=127.0.0.1
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export no_proxy=127.0.0.1

cd "${MILES_DIR}"

pkill -9 sglang || true
ray stop --force || true
sleep 3
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

source scripts/models/qwen3-0.6B.sh

# Set PRIMUS_CONFIG=none to run the identical job on the plain Megatron backend, which
# is how you tell a Primus problem apart from a Miles/ROCm one.
TRAIN_BACKEND=primus
PRIMUS_FLAG=(--primus-config "${PRIMUS_CONFIG}")
if [[ "${PRIMUS_CONFIG}" == "none" ]]; then
  TRAIN_BACKEND=megatron
  PRIMUS_FLAG=()
fi

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"${PYTHONPATH}\", \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\", \"PATH\": \"${PATH}\", \"GPU_ARCHS\": \"${GPU_ARCHS}\", \"CU_NUM\": \"${CU_NUM}\", \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"MASTER_ADDR\": \"127.0.0.1\", \"no_proxy\": \"127.0.0.1\"}}" \
  -- python3 train.py \
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
  --num-rollout "${NUM_ROLLOUT:-2}" \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 1024 \
  --rollout-temperature 1.0 \
  --global-batch-size 64 \
  --advantage-estimator grpo \
  --kl-coef 0.00 \
  --entropy-coef 0.00 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 16384 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend fused \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.4 \
  --use-miles-router \
  --calculate-per-token-loss \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TRAIN_GPUS}" \
  "${PLACEMENT_ARGS[@]}"

#!/bin/bash
# Fully-async GRPO smoke that forces Primus-Turbo flash attention.
# Miles treats use_turbo_attention as unsafe (wrong Megatron-wrapper grads);
# this override is only for kernel/wiring tests.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PRIMUS_CONFIG="${PRIMUS_CONFIG:-${SCRIPT_DIR}/qwen3-0.6B-bf16-rl-turbo-attn.yaml}"
export MILES_ALLOW_UNSAFE_PRIMUS_TURBO=1
exec bash "${SCRIPT_DIR}/run-qwen3-0.6B-bf16-rl-fully-async.sh" "$@"

#!/bin/bash
#SBATCH --job-name=glm53-hip-cc-p0
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/yuankach/glm53/logs/hip-cc-p0-%j.out
# Phase 0 (plan Phase 0a/0b): Claude Code episodes against GLM-5.3-Flash with no trainer.
# Every node runs SGLang TP8, a standalone session server v2, and cc_node.sh (cc-gateway and
# cc-runner with CC_CONTAINERS_PER_NODE CPU containers). Node 0 also runs the episode driver.
# The Crusoe HIP gateways are reached through the relay in $HIP_SF_URL_FILE.
#
#   sbatch -N 1 --export=ALL,CC_STEP=0a,MODEL=...,CC_TASKS=...,CC_TASK_LIMIT=5,CC_SAMPLES=4,... \
#     run_glm53_hip_cc_phase0.sbatch
#   DRY_RUN=1: print the role map (nodes, ports, containers, files) and exit without starting anything.
#
# --export carries file paths only. The runner and gateway tokens are minted here into a 0700
# directory under ~/.hip_cc and shredded at exit; the grader token stays in HIP_SANDBOX_TOKEN_FILE.
set -euo pipefail
HC=${HC:-/home/yuankach/hip-cc/miles/examples/hip_cc}
SNAP=/shared_inference/yuankach/glm53/logs/hip-cc-p0-${SLURM_JOB_ID}/scripts
mkdir -p "${SNAP}"
cp /home/yuankach/Miles-Primus/scripts/{env.sh,node_setup.sh,mhc_rocm_patch.py,kpool_triton_patch.py,kpool_indexer_kernels.py} "${SNAP}/"
cp "${HC}/scripts/cc_node.sh" "${SNAP}/"
export SHARED_ROOT=${SHARED_ROOT:-/shared_inference/yuankach/glm53}
# shellcheck disable=SC1091
source "${SNAP}/env.sh"
export MILES_DIR=${CC_MILES_DIR:-/home/yuankach/hip-cc/miles}

MODEL=${MODEL:-${SHARED_ROOT}/models/GLM-5.3-Flash-bf16}
export SHARED_CKPT=${MODEL} HF_CKPT=${MODEL} CKPT_MODE=nfs ROLLOUT_DUMMY_LOAD=0
CC_STEP=${CC_STEP:-0a}
CC_CONTEXT=${CC_CONTEXT:-131072}
NODES=${CC_ROLLOUT_NODES:-${SLURM_NNODES}}
export CC_CONTAINERS_PER_NODE=${CC_CONTAINERS_PER_NODE:-16}
CC_CONCURRENCY=${CC_CONCURRENCY:-$(( NODES * CC_CONTAINERS_PER_NODE ))}
SGLANG_PORT=${SGLANG_PORT:-30000}
SESSION_PORT=${CC_SESSION_PORT:-30100}
export HIP_SF_URL_FILE=${HIP_SF_URL_FILE:-${SHARED_ROOT}/hip-cc/crusoe_relay_url.txt}
export HIP_SANDBOX_TOKEN_FILE=${HIP_SANDBOX_TOKEN_FILE:-${HOME}/.hip_cc/crusoe_grader_token}
CC_TASKS=${CC_TASKS:?CC_TASKS must name a manifest.jsonl of validated, rendered workspaces}
RUN_DIR=${SHARED_ROOT}/logs/hip-cc-p0-${SLURM_JOB_ID}
export CC_RUN_DIR=${RUN_DIR}/cc
OUT=${CC_OUT:-${SHARED_ROOT}/hip-cc/phase0/${CC_STEP}-${SLURM_JOB_ID}}

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[ "${#ALL_NODES[@]}" -ge "${NODES}" ] || { echo "need ${NODES} nodes"; exit 1; }
ROLL_NODES=("${ALL_NODES[@]:0:${NODES}}")
ROLL_LIST=$(IFS=,; echo "${ROLL_NODES[*]}")
host_ip() { srun --overlap -N1 -n1 -w "$1" bash -c "ip -4 -o addr show ${SOCKET_IFNAME} | awk '{print \$4}' | cut -d/ -f1"; }
export HEAD_IP=$(host_ip "${ROLL_NODES[0]}")
SESSION_URLS=""
for host in "${ROLL_NODES[@]}"; do
  SESSION_URLS+="${SESSION_URLS:+,}http://$(host_ip "${host}"):${SESSION_PORT}"
done

python3 - "${MODEL}/config.json" "${CC_CONTEXT}" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config = config.get("text_config", config)
limit = config.get("max_position_embeddings", 0)
assert limit >= int(sys.argv[2]), f"max_position_embeddings {limit} < CC_CONTEXT {sys.argv[2]}"
print(f"context gate: max_position_embeddings {limit}, rope_scaling {config.get('rope_scaling')}, CC_CONTEXT {sys.argv[2]}")
EOF

echo "== role map (job ${SLURM_JOB_ID}, Phase ${CC_STEP})"
for i in "${!ROLL_NODES[@]}"; do
  echo "node ${i} ${ROLL_NODES[$i]}: SGLang TP8 :${SGLANG_PORT} (${MODEL##*/}, context ${CC_CONTEXT}), session server v2 :${SESSION_PORT}," \
       "cc-gateway :${CC_GATEWAY_PORT:-18400}, cc-runner :${CC_RUNNER_PORT:-18500} x${CC_CONTAINERS_PER_NODE} containers$([ "$i" = 0 ] && echo ', episode driver')"
done
echo "driver: tasks ${CC_TASKS} limit ${CC_TASK_LIMIT:-0} x ${CC_SAMPLES:-4} samples, concurrency ${CC_CONCURRENCY}, out ${OUT}"
echo "grader: URLs from ${HIP_SF_URL_FILE}; token file ${HIP_SANDBOX_TOKEN_FILE}"
echo "budgets: calls ${CC_MAX_CALLS:-40}, wall ${CC_WALL_S:-1200}s, max_out ${CC_MAX_OUT:-16384}, margin ${CC_STOP_MARGIN:-4096}," \
     "correctness ${CC_CHECKS_CORRECTNESS:-10}, performance ${CC_CHECKS_PERF:-4}, counters ${CC_CHECKS_COUNTERS:-1}"
if [ "${DRY_RUN:-0}" = 1 ]; then
  export CC_SECRETS_DIR=/nonexistent
  srun --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 -w "${ROLL_LIST}" /usr/bin/env DRY_RUN=1 bash "${SNAP}/cc_node.sh"
  echo "DRY_RUN: nothing started"; exit 0
fi

mkdir -p "${CC_RUN_DIR}" "${OUT}"
: > "${CC_RUN_DIR}/cc_nodes.txt"
export CC_SECRETS_DIR=${HOME}/.hip_cc/run-${SLURM_JOB_ID}
( umask 077; mkdir -p "${CC_SECRETS_DIR}"
  python3 -c "import secrets,sys; [open(f'{sys.argv[1]}/{n}', 'w').write(secrets.token_urlsafe(32)) for n in ('runner_token', 'gateway_admin_token')]" "${CC_SECRETS_DIR}" )

cleanup() {
  trap - EXIT USR1 TERM
  echo "cleanup $(date -u)"
  srun --overlap --nodes="${NODES}" --ntasks-per-node=1 -w "${ROLL_LIST}" bash -c "docker rm -f ${CNAME} cc_gw_${SLURM_JOB_ID} cc_drv_${SLURM_JOB_ID} >/dev/null 2>&1; docker ps -aq --filter label=cc.job=${SLURM_JOB_ID} | xargs -r docker rm -f >/dev/null 2>&1; true" || true
  shred -u "${CC_SECRETS_DIR}"/* 2>/dev/null || true
  rmdir "${CC_SECRETS_DIR}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'echo "caught signal"; exit 1' USR1 TERM

# Every node must reproduce the published image ID before any container starts on any node.
srun --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 -w "${ROLL_LIST}" \
  bash "${SNAP}/cc_node.sh" verify || { echo "image verification failed; no containers started"; exit 1; }

srun --overlap --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 -w "${ROLL_LIST}" \
  bash "${SNAP}/cc_node.sh" > "${CC_RUN_DIR}/cc_node.log" 2>&1 &
CC_PID=$!

srun --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 -w "${ROLL_LIST}" \
  bash "${SNAP}/node_setup.sh"

SGLANG_ENV="export SGLANG_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1 PYTHONPATH=${SGLANG_DIR:-/home/yuankach/Miles-Primus/sglang-yk/python}:${MILES_DIR}"
SGLANG_EXTRA=()
if [ "${SGLANG_PROFILE:-rocm}" = rocm ]; then
  SGLANG_ENV+=" AITER_ONLINE_TUNE=0 HIP_FORCE_DEV_KERNARG=1 AITER_CONFIG_FMOE=${AITER_CONFIG_FMOE:-${SHARED_ROOT}/aiter_configs/tuned_fmoe_mi300.csv}"
  SGLANG_EXTRA=(--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --moe-runner-backend aiter --cuda-graph-max-bs-decode 128)
fi
SESSION_CONFIG=$(python3 - "${MODEL}" "${SESSION_PORT}" "${SGLANG_PORT}" <<'EOF'
import json, sys
model, port, sglang = sys.argv[1], int(sys.argv[2]), sys.argv[3]
print(json.dumps({
    "host": "0.0.0.0", "port": port, "instance_id": None, "backend_url": f"http://127.0.0.1:{sglang}",
    "timeout": None, "hf_checkpoint": model, "chat_template_path": None, "tito_model": "glm53",
    "apply_chat_template_kwargs": None, "use_rollout_routing_replay": False, "use_rollout_indexer_replay": False,
    "use_sampling_support_replay": False, "sglang_speculative_algorithm": None, "num_layers": None,
    "moe_router_topk": None, "save_debug_trajectory_data": None, "lora_rank": 0, "lora_adapter_path": None,
    "lora_train_only": False, "use_session_server": "v2", "session_message_matcher": "strict",
    "pause_generation_mode": None, "session_sample_picker_path": None, "session_sample_postprocessor_path": None}))
EOF
)
printf '%s' "${SESSION_CONFIG}" > "${RUN_DIR}/session_config.json"
srun --overlap --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 -w "${ROLL_LIST}" bash -c "
  docker exec -d ${CNAME} bash -c '${SGLANG_ENV}; cd ${MILES_DIR}; python3 -m sglang.launch_server --model-path ${MODEL} \
    --tp 8 --host 0.0.0.0 --port ${SGLANG_PORT} --trust-remote-code --context-length ${CC_CONTEXT} \
    --reasoning-parser glm45 --tool-call-parser glm47 --max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS:-64} \
    --chunked-prefill-size 16384 --mem-fraction-static 0.8 ${SGLANG_EXTRA[*]} \
    > ${RUN_DIR}/sglang-\$(hostname).log 2>&1'
  docker exec -d ${CNAME} bash -c 'export PYTHONPATH=${MILES_DIR}; cd ${MILES_DIR}; \
    python3 -m miles.rollout.session.server --config-json \"\$(cat ${RUN_DIR}/session_config.json)\" \
    > ${RUN_DIR}/session-\$(hostname).log 2>&1'"

echo "waiting for SGLang, session servers and cc nodes"
for _ in $(seq 1 360); do
  ready=0
  for url in ${SESSION_URLS//,/ }; do
    ip=${url#http://}; ip=${ip%%:*}
    curl -sf -m 3 "http://${ip}:${SGLANG_PORT}/health" >/dev/null && curl -sf -m 3 "${url}/health" >/dev/null && ready=$(( ready + 1 ))
  done
  nodes_up=$(wc -l < "${CC_RUN_DIR}/cc_nodes.txt")
  [ "${ready}" = "${NODES}" ] && [ "${nodes_up}" = "${NODES}" ] && break
  kill -0 "${CC_PID}" 2>/dev/null || { echo "cc_node step died"; tail -40 "${CC_RUN_DIR}/cc_node.log"; exit 1; }
  sleep 10
done
[ "${ready}" = "${NODES}" ] && [ "${nodes_up}" = "${NODES}" ] || { echo "not ready: sglang+session ${ready}/${NODES}, cc nodes ${nodes_up}/${NODES}"; exit 1; }
echo "ready: ${SESSION_URLS}; cc nodes:"; cat "${CC_RUN_DIR}/cc_nodes.txt"

docker run --rm --name "cc_drv_${SLURM_JOB_ID}" --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH="${HC}" \
  -e CC_NODES_FILE="${CC_RUN_DIR}/cc_nodes.txt" -e CC_RUNNER_TOKEN_FILE="${CC_SECRETS_DIR}/runner_token" \
  -e CC_GATEWAY_ADMIN_TOKEN_FILE="${CC_SECRETS_DIR}/gateway_admin_token" \
  -e HIP_SF_URL_FILE -e HIP_SANDBOX_TOKEN_FILE -e CC_SESSION_URLS="${SESSION_URLS}" -e CC_OUT="${OUT}" \
  -e CC_TASKS="${CC_TASKS}" -e CC_TASK_LIMIT="${CC_TASK_LIMIT:-0}" -e CC_SAMPLES="${CC_SAMPLES:-4}" \
  -e CC_CONCURRENCY="${CC_CONCURRENCY}" -e CC_RESUME="${CC_RESUME:-0}" -e CC_TEMPERATURE="${CC_TEMPERATURE:-1.0}" \
  -e CC_CONTEXT="${CC_CONTEXT}" -e CC_MAX_OUT -e CC_STOP_MARGIN -e CC_MAX_CALLS -e CC_WALL_S \
  -e CC_CHECKS_CORRECTNESS -e CC_CHECKS_PERF -e CC_CHECKS_COUNTERS -e CC_COUNTERS -e CC_HEALTHY_FLOOR \
  -v /home/yuankach:/home/yuankach:ro -v "${OUT}:${OUT}" -v "${CC_RUN_DIR}:${CC_RUN_DIR}:ro" \
  -v "$(dirname "${CC_TASKS}"):$(dirname "${CC_TASKS}"):ro" -v "$(dirname "${HIP_SF_URL_FILE}"):$(dirname "${HIP_SF_URL_FILE}"):ro" \
  "${IMAGE}" python3 "${HC}/hip_cc_phase0.py" 2>&1 | tee "${RUN_DIR}/driver.log"
echo "PHASE 0 ${CC_STEP} FINISHED $(date -u); results ${OUT}/episodes.jsonl"

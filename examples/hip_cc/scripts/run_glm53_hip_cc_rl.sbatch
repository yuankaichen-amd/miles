#!/bin/bash
#SBATCH --job-name=glm53-hip-cc-rl
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/shared_inference/yuankach/glm53/logs/hip-cc-rl-%j.out
# GLM-5.3-Flash HIP-kernel GRPO with Claude Code episodes (plan §3, S1/S2/main run).
# Derived from run_glm53_hip_rl.sbatch with no sandbox nodes: grading is on the Crusoe HIP pool
# behind the relay in HIP_SF_URL_FILE, and every Ray node also runs cc_node.sh (cc-gateway and
# cc-runner with CPU-only rollout containers).
#   nodes 0..TRAINER_NODES-1        trainer (Ray), PP=8, CP_SIZE (default 8), CC_CONTAINERS_TRAINER containers
#   next ROLLOUT_NODES nodes        SGLang rollout (Ray), one TP8 engine per node, CC_CONTAINERS_ROLLOUT containers
# Checkpoint: HF_CHECKPOINT (default the base model). DRY_RUN=1 prints the role map and exits.
# --export carries file paths only; runner/gateway tokens are minted under ~/.hip_cc and shredded at exit.
set -euo pipefail
JOB_START=$(date +%s)
SNAP=/shared_inference/yuankach/glm53/logs/hip-cc-rl-${SLURM_JOB_ID}/scripts
mkdir -p "${SNAP}"
HC=${HC:-/home/yuankach/hip-cc/miles/examples/hip_cc}
cp /home/yuankach/Miles-Primus/scripts/{env.sh,node_setup.sh,mhc_rocm_patch.py,kpool_triton_patch.py,kpool_indexer_kernels.py} "${SNAP}/"
cp "${HC}"/{hip_cc_agent.py,hip_grader_client.py} "${HC}/scripts/cc_node.sh" "${HC}/scripts/run-glm5.3-flash-bf16-hip-cc-rl.sh" "${SNAP}/"
SCRIPTS=${SNAP}
export SHARED_ROOT=${SHARED_ROOT:-/shared_inference/yuankach/glm53}
# shellcheck disable=SC1091
source "${SCRIPTS}/env.sh"
export MILES_DIR=${CC_MILES_DIR:-/home/yuankach/hip-cc/miles}

# Unconditional: env.sh already defaulted SHARED_CKPT to the base model.
export SHARED_CKPT=${HF_CHECKPOINT:-${SHARED_ROOT}/models/GLM-5.3-Flash-bf16}
export STAGED_CKPT=${LOCAL_ROOT}/$(basename "${SHARED_CKPT}")
if [ "${CKPT_MODE}" = local ]; then
  export HF_CKPT=${STAGED_CKPT}
  export ROLLOUT_DUMMY_LOAD=${ROLLOUT_DUMMY_LOAD:-0}
else
  export HF_CKPT=${SHARED_CKPT}
  export ROLLOUT_DUMMY_LOAD=${ROLLOUT_DUMMY_LOAD:-1}
fi

TRAIN_NODES=${TRAINER_NODES:-8}
ROLLOUT_NODES=${ROLLOUT_NODES:-4}
[ "${SANDBOX_NODES:-0}" = 0 ] || { echo "hip-cc runs use no sandbox nodes (SANDBOX_NODES=0)"; exit 1; }
RAY_NODES=$(( TRAIN_NODES + ROLLOUT_NODES ))
[ "${SLURM_NNODES}" -ge "${RAY_NODES}" ] || { echo "need ${RAY_NODES} nodes"; exit 1; }
[ -f "${SGLANG_PATCH}" ] || { echo "missing ${SGLANG_PATCH}"; exit 1; }
[ -n "${TURBO_WHEEL}" ] || { echo "missing turbo wheel"; exit 1; }
[ -f "${SHARED_CKPT}/model.safetensors.index.json" ] || { echo "missing ${SHARED_CKPT}"; exit 1; }
[ -f "${SHARED_CKPT}/chat_template.jinja" ] || { echo "missing chat template"; exit 1; }

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
RAY_LIST=$(IFS=,; echo "${ALL_NODES[*]:0:${RAY_NODES}}")
host_ip() { srun --overlap -N1 -n1 -w "$1" bash -c "ip -4 -o addr show ${SOCKET_IFNAME} | awk '{print \$4}' | cut -d/ -f1"; }
export HEAD_IP=$(host_ip "${ALL_NODES[0]}")

RUN_LOG_DIR=${SHARED_ROOT}/logs/hip-cc-rl-${SLURM_JOB_ID}
export CC_RUN_DIR=${RUN_LOG_DIR}/cc
mkdir -p "${RUN_LOG_DIR}" "${CC_RUN_DIR}"
: > "${CC_RUN_DIR}/cc_nodes.txt"
export HIP_SF_URL_FILE=${HIP_SF_URL_FILE:-${SHARED_ROOT}/hip-cc/crusoe_relay_url.txt}
export HIP_SANDBOX_TOKEN_FILE=${HIP_SANDBOX_TOKEN_FILE:-${HOME}/.hip_cc/crusoe_grader_token}
CC_TRAINER=${CC_CONTAINERS_TRAINER:-8}
CC_ROLLOUT=${CC_CONTAINERS_ROLLOUT:-32}
export CP_SIZE=${CP_SIZE:-8} MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

echo "== role map (job ${SLURM_JOB_ID}, ${RUN_NAME:-glm53-hip-cc})"
for i in $(seq 0 $(( RAY_NODES - 1 ))); do
  if [ "${i}" -lt "${TRAIN_NODES}" ]; then role="trainer PP stage ${i}, CP ${CP_SIZE}x${MAX_TOKENS_PER_GPU}"; cap=${CC_TRAINER}
  else role="SGLang rollout engine $(( i - TRAIN_NODES )) (TP8)"; cap=${CC_ROLLOUT}; fi
  echo "node ${i} ${ALL_NODES[$i]}: ${role}; cc-runner x${cap} + cc-gateway$([ "$i" = 0 ] && echo '; Ray head, rollout driver')"
done
echo "containers: $(( TRAIN_NODES * CC_TRAINER + ROLLOUT_NODES * CC_ROLLOUT )) total; async concurrency ${ASYNC_MAX_CONCURRENT_SAMPLES:-32}"
echo "data: train ${HIP_TRAIN:-unset} val ${HIP_VAL:-unset}; ckpt ${HF_CKPT}; grader URLs ${HIP_SF_URL_FILE}"
if [ "${DRY_RUN:-0}" = 1 ]; then
  CC_SECRETS_DIR=/nonexistent srun --nodes="${RAY_NODES}" --ntasks="${RAY_NODES}" --ntasks-per-node=1 -w "${RAY_LIST}" \
    bash -c "CC_CONTAINERS_PER_NODE=\$( [ \${SLURM_NODEID} -lt ${TRAIN_NODES} ] && echo ${CC_TRAINER} || echo ${CC_ROLLOUT} ) DRY_RUN=1 bash ${SCRIPTS}/cc_node.sh"
  echo "DRY_RUN: nothing started"; exit 0
fi

export CC_SECRETS_DIR=${HOME}/.hip_cc/run-${SLURM_JOB_ID}
( umask 077; mkdir -p "${CC_SECRETS_DIR}"
  python3 -c "import secrets,sys; [open(f'{sys.argv[1]}/{n}', 'w').write(secrets.token_urlsafe(32)) for n in ('runner_token', 'gateway_admin_token')]" "${CC_SECRETS_DIR}" )

cleanup() {
  trap - EXIT USR1 TERM
  echo "cleanup $(date -u)"
  docker exec "${CNAME}" bash -c 'for j in $(ray job list 2>/dev/null | grep -o "raysubmit_[A-Za-z0-9]*" | sort -u); do ray job stop "$j" >/dev/null 2>&1; done' || true
  srun --overlap --ntasks-per-node=1 bash -c "docker rm -f ${CNAME} cc_gw_${SLURM_JOB_ID} >/dev/null 2>&1; docker ps -aq --filter label=cc.job=${SLURM_JOB_ID} | xargs -r docker rm -f >/dev/null 2>&1; true" || true
  shred -u "${CC_SECRETS_DIR}"/* 2>/dev/null || true
  rmdir "${CC_SECRETS_DIR}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'echo "caught signal"; exit 1' USR1 TERM

# Every node must reproduce the published image ID before any container starts on any node.
srun --nodes="${RAY_NODES}" --ntasks="${RAY_NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 -w "${RAY_LIST}" \
  bash "${SCRIPTS}/cc_node.sh" verify || { echo "image verification failed; no containers started"; exit 1; }

# cc nodes come up in parallel with the Ray cluster. The step stays alive on the runners.
srun --overlap --nodes="${RAY_NODES}" --ntasks="${RAY_NODES}" --ntasks-per-node=1 -w "${RAY_LIST}" \
  bash -c "export CC_CONTAINERS_PER_NODE=\$( [ \${SLURM_NODEID} -lt ${TRAIN_NODES} ] && echo ${CC_TRAINER} || echo ${CC_ROLLOUT} ); exec bash ${SCRIPTS}/cc_node.sh" \
  > "${CC_RUN_DIR}/cc_node.log" 2>&1 &
CC_PID=$!

srun --nodes="${RAY_NODES}" --ntasks="${RAY_NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  -w "${RAY_LIST}" bash "${SCRIPTS}/node_setup.sh"
srun --overlap --nodes="${RAY_NODES}" --ntasks="${RAY_NODES}" --ntasks-per-node=1 -w "${RAY_LIST}" \
  docker exec "${CNAME}" pip install -q pydantic requests >/dev/null
# The agent runs as root inside the run container, which may not read 0600 files on the NFS home:
# copy the three tokens over stdin into the container's own filesystem (0600, never argv).
srun --overlap --nodes="${RAY_NODES}" --ntasks="${RAY_NODES}" --ntasks-per-node=1 -w "${RAY_LIST}" bash -c "
  set -e
  for pair in runner_token:${CC_SECRETS_DIR}/runner_token gateway_admin_token:${CC_SECRETS_DIR}/gateway_admin_token \
              grader_token:${HIP_SANDBOX_TOKEN_FILE}; do
    docker exec -i ${CNAME} sh -c \"umask 077; mkdir -p /run/hipcc; cat > /run/hipcc/\${pair%%:*}\" < \${pair#*:}
  done"

for _ in $(seq 1 60); do
  gpus=$(docker exec "${CNAME}" python3 -c "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
  [ "${gpus}" -ge $(( RAY_NODES * 8 )) ] && break
  sleep 5
done
echo "ray cluster GPUs: ${gpus} (sandbox nodes are excluded)"
[ "${gpus}" -ge $(( RAY_NODES * 8 )) ]
for _ in $(seq 1 60); do
  docker exec "${CNAME}" ray job submit --address=http://127.0.0.1:8265 --no-wait -- true >/dev/null 2>&1 && break
  sleep 5
done

echo "waiting for ${RAY_NODES} cc nodes"
for _ in $(seq 1 180); do
  [ "$(wc -l < "${CC_RUN_DIR}/cc_nodes.txt")" -ge "${RAY_NODES}" ] && break
  if ! kill -0 "${CC_PID}" 2>/dev/null; then echo "cc_node step died"; tail -80 "${CC_RUN_DIR}/cc_node.log"; exit 1; fi
  sleep 5
done
[ "$(wc -l < "${CC_RUN_DIR}/cc_nodes.txt")" -ge "${RAY_NODES}" ] || { echo "cc nodes not up"; tail -80 "${CC_RUN_DIR}/cc_node.log"; exit 1; }
cat "${CC_RUN_DIR}/cc_nodes.txt"

NET_ENV="export NCCL_SOCKET_IFNAME=${SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${SOCKET_IFNAME} TP_SOCKET_IFNAME=${SOCKET_IFNAME};"
if [ "${NET_MODE}" = tcp ]; then
  NET_ENV+=" export NCCL_IB_DISABLE=1;"
else
  NET_ENV+=" export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3};"
fi

HIP_TRAIN=${HIP_TRAIN:-${DATA_DIR}/hip/t2h/workspaces/train/manifest.jsonl}
HIP_VAL=${HIP_VAL:-${DATA_DIR}/hip/t2h/workspaces/t2h-eval128/manifest.jsonl}
[ -f "${HIP_TRAIN}" ] && [ -f "${HIP_VAL}" ]

CKPT_DIR=${SHARED_ROOT}/ckpt/${RUN_NAME:-glm53-hip-cc}
LATEST=${CKPT_DIR}/latest_checkpointed_iteration.txt
mkdir -p "${CKPT_DIR}"
if [ -f "${CKPT_DIR}/DONE" ]; then echo "run ${CKPT_DIR} already DONE"; exit 0; fi
CKPT_ARGS="--save ${CKPT_DIR} --save-interval ${SAVE_INTERVAL:-1000000} --save-trigger-sentinel ${CKPT_DIR}/SAVE_NOW"
if [ -f "${LATEST}" ]; then
  CKPT_ARGS+=" --load ${CKPT_DIR}"
  echo "resuming from ${CKPT_DIR} iteration $(cat "${LATEST}")"
fi
rm -f "${CKPT_DIR}/SAVE_NOW" "${CKPT_DIR}/CHUNK_STOPPED"

export PYTHONPATH="${SNAP}${PYTHONPATH:+:${PYTHONPATH}}"

docker exec "${CNAME}" bash -c "
  ${NET_ENV}
  export MASTER_ADDR=${HEAD_IP} no_proxy=${HEAD_IP},127.0.0.1,localhost
  export HF_CKPT=${HF_CKPT} DATA_DIR=${DATA_DIR} TURBO_PYTHONPATH=${TURBO_PYTHONPATH}
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR} PRIMUS_WORKSPACE=${RUN_LOG_DIR}/primus
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export PRIMUS_DSA_DKV_SAFE=${PRIMUS_DSA_DKV_SAFE:-1} PRIMUS_DSA_BWD_NUM_STAGES=${PRIMUS_DSA_BWD_NUM_STAGES:-1}
  export MILES_SGLANG_DUMMY_LOAD=${ROLLOUT_DUMMY_LOAD}
  export CC_NODES_FILE=${CC_RUN_DIR}/cc_nodes.txt CC_RUNNER_TOKEN_FILE=/run/hipcc/runner_token
  export CC_GATEWAY_ADMIN_TOKEN_FILE=/run/hipcc/gateway_admin_token
  export HIP_SF_URL_FILE=${HIP_SF_URL_FILE} HIP_SANDBOX_TOKEN_FILE=/run/hipcc/grader_token
  export CC_CONTEXT=${CC_CONTEXT:-131072} CC_MAX_OUT=${CC_MAX_OUT:-16384} CC_STOP_MARGIN=${CC_STOP_MARGIN:-4096}
  export CC_MAX_CALLS=${CC_MAX_CALLS:-40} CC_WALL_S=${CC_WALL_S:-1200} CC_HEALTHY_FLOOR=${CC_HEALTHY_FLOOR:-8}
  export CC_CHECKS_CORRECTNESS=${CC_CHECKS_CORRECTNESS:-10} CC_CHECKS_PERF=${CC_CHECKS_PERF:-4} CC_CHECKS_COUNTERS=${CC_CHECKS_COUNTERS:-1}
  export CC_EPISODE_LOG=${RUN_LOG_DIR}/episodes.jsonl
  export SGLANG_RADIX_CACHE=${SGLANG_RADIX_CACHE:-1}
  export HIP_TRAIN=${HIP_TRAIN} HIP_VAL=${HIP_VAL}
  export SGLANG_PROFILE=${SGLANG_PROFILE:-rocm}
  export SGLANG_DIR=${SGLANG_DIR:-/home/yuankach/Miles-Primus/sglang-yk/python}
  export AITER_CONFIG_FMOE=${AITER_CONFIG_FMOE:-${SHARED_ROOT}/aiter_configs/tuned_fmoe_mi300.csv}
  export SGLANG_R3_KEEP_ROUTING_ON_RETRACT=${SGLANG_R3_KEEP_ROUTING_ON_RETRACT:-1}
  export PYTHONPATH=${PYTHONPATH}
  export PRIMUS_DIR=${RUN_PRIMUS_DIR:-${PRIMUS_DIR}}
  export CP_SIZE=${CP_SIZE} MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU}
  cd ${MILES_DIR}
  TRAIN_NODES=${TRAIN_NODES} ROLLOUT_NODES=${ROLLOUT_NODES} \
    NUM_ROLLOUT=${NUM_ROLLOUT:-100} EVAL_INTERVAL=${EVAL_INTERVAL:-100000} \
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8} N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8} \
    ASYNC_MAX_CONCURRENT_SAMPLES=${ASYNC_MAX_CONCURRENT_SAMPLES:-32} \
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-65536} \
    ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-65536} \
    bash ${SCRIPTS}/run-glm5.3-flash-bf16-hip-cc-rl.sh ${CKPT_ARGS} ${MILES_EXTRA_ARGS:-}
" > "${RUN_LOG_DIR}/train.log" 2>&1 &
TRAIN_PID=$!
echo "training log: ${RUN_LOG_DIR}/train.log"

save_before_limit() {
  local limit_min=${TIME_LIMIT_MIN:-240} margin=${SAVE_MARGIN_MIN:-75}
  local wait_s=$(( (limit_min - margin) * 60 - ($(date +%s) - JOB_START) ))
  [ "${wait_s}" -gt 0 ] && sleep "${wait_s}"
  echo "$(date -u) time-limit save: touching ${CKPT_DIR}/SAVE_NOW"
  touch "${CKPT_DIR}/SAVE_NOW"
  local before; before=$(stat -c %Y "${LATEST}" 2>/dev/null || echo 0)
  while [ "$(stat -c %Y "${LATEST}" 2>/dev/null || echo 0)" -le "${before}" ]; do sleep 30; done
  echo "$(date -u) checkpoint iteration $(cat "${LATEST}") saved; stopping this chunk"
  touch "${CKPT_DIR}/CHUNK_STOPPED"
  docker exec "${CNAME}" bash -c 'for j in $(ray job list 2>/dev/null | grep -o "raysubmit_[A-Za-z0-9]*" | sort -u); do ray job stop "$j" >/dev/null 2>&1; done' || true
}
save_before_limit &
SAVER_PID=$!

rc=0; wait ${TRAIN_PID} || rc=$?
kill ${SAVER_PID} 2>/dev/null || true
echo "RUN FINISHED rc=${rc} $(date -u)"
if [ -f "${CKPT_DIR}/CHUNK_STOPPED" ]; then
  echo "chunk stopped after a time-limit save; the next job resumes from $(cat "${LATEST}")"
  exit 0
fi
exit ${rc}

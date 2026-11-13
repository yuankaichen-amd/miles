#!/bin/bash
# Run one Phase 0a step inside a held Phase 0 job (CC_HOLD=1), on its node 0, in the base image.
#
#   phase0a/driver.sh <run_dir> <name> <script> [KEY=VALUE ...]
#
# <script> is a path under examples/hip_cc (hip_cc_phase0.py or phase0a/*.py). Output goes to
# $OUT/<name>/ and $RUN_DIR/<name>.log. Only file paths are passed; the tokens stay in their files.
set -euo pipefail
RUN_DIR=$1 NAME=$2 SCRIPT=$3
shift 3
# shellcheck disable=SC1091
source "${RUN_DIR}/job.env"
STEP_OUT=${OUT}/${NAME}
mkdir -p "${STEP_OUT}"
EXTRA=()
for kv in "$@"; do EXTRA+=(-e "${kv}"); done
TASKS_DIR=/shared_inference/yuankach/glm53/datasets/hip/t2h
echo "[$(date -u +%FT%TZ)] ${NAME}: start ${SCRIPT} $*" | tee -a "${RUN_DIR}/steps.log"
set +e
srun --overlap --jobid "${JOB}" -N1 -n1 -w "${NODE}" \
  docker run --rm --name "cc_${NAME}_${JOB}" --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH="${HC}:$(dirname "$(dirname "${HC}")")" \
  -e CC_NODES_FILE="${CC_RUN_DIR}/cc_nodes.txt" -e CC_RUNNER_TOKEN_FILE="${CC_SECRETS_DIR}/runner_token" \
  -e CC_GATEWAY_ADMIN_TOKEN_FILE="${CC_SECRETS_DIR}/gateway_admin_token" \
  -e HIP_SF_URL_FILE="${HIP_SF_URL_FILE}" -e HIP_SANDBOX_TOKEN_FILE="${HIP_SANDBOX_TOKEN_FILE}" \
  -e CC_SESSION_URLS="${SESSION_URLS}" -e SGLANG_URL="${SGLANG_URL}" -e CC_OUT="${STEP_OUT}" -e RUN_DIR="${RUN_DIR}" \
  -e MODEL="${MODEL}" -e CC_CONTEXT="${CC_CONTEXT}" -e CC_MAX_OUT=16384 -e CC_STOP_MARGIN=4096 -e CC_MAX_CALLS=40 \
  -e CC_WALL_S=1200 -e CC_CHECKS_CORRECTNESS=10 -e CC_CHECKS_PERF=4 -e CC_CHECKS_COUNTERS=1 -e CC_COUNTERS=1 \
  "${EXTRA[@]}" \
  -v /home/yuankach:/home/yuankach:ro -v "${OUT}:${OUT}" -v "${RUN_DIR}:${RUN_DIR}" -v "${TASKS_DIR}:${TASKS_DIR}:ro" \
  -v "$(dirname "${HIP_SF_URL_FILE}"):$(dirname "${HIP_SF_URL_FILE}"):ro" -v "${MODEL}:${MODEL}:ro" \
  "${IMAGE}" python3 "${HC}/${SCRIPT}" > "${RUN_DIR}/${NAME}.log" 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] ${NAME}: exit ${rc}" | tee -a "${RUN_DIR}/steps.log"
exit ${rc}

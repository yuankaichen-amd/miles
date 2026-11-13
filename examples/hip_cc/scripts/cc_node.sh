#!/bin/bash
# cc-runner and cc-gateway on one Miles node (plan §4.2, §10 P7). One srun task per node, on the host.
#
#   CC_RUN_DIR        shared run directory; this node appends "<runner_url> <gateway_url>" to
#                     $CC_RUN_DIR/cc_nodes.txt once both answer
#   CC_SECRETS_DIR    0700 directory with runner_token and gateway_admin_token (0600), made by the sbatch
#   CC_CONTAINERS_PER_NODE, CC_IMAGE_TAG (default: the published LATEST), CC_TOKENIZER (HF dir, optional)
#   CC_EXPECTED_IMAGE_ID  default: the published <tag>.image_id
#   HIP_SF_URL_FILE, HIP_SANDBOX_TOKEN_FILE   Crusoe relay URLs and grader token (paths only)
#   DRY_RUN=1         print what would run and exit
#
#   cc_node.sh verify   only build and verify the image; the sbatch runs this on every node and
#                       aborts before any container (SGLang, Ray, gateway, rollouts) starts if one fails
#
# The image is rebuilt from the published build context with no network (build_from_context.sh).
# Its image ID must equal the published one before this node starts any container (gateway,
# runner, rollouts); otherwise the node exits and runs nothing.
# Rollout containers run as uid 10001 with --network none; they reach the gateway and the
# eval-proxy only through the per-episode unix sockets the runner creates under $SOCK_ROOT.
set -euo pipefail
: "${CC_RUN_DIR:?}" "${CC_SECRETS_DIR:?}"
HC=${HC:-/home/yuankach/hip-cc/miles/examples/hip_cc}
IMAGES=${IMAGES:-/shared_inference/yuankach/glm53/images}
BASE=${CC_BASE_IMAGE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}
BASE_TAR=${BASE_TAR:-$IMAGES/sgl-dev_miles-rocm10-mi30x-20261001.tar}
CTX_TAG=${CC_IMAGE_TAG:-$(cat "$IMAGES/hip-cc-rollout/LATEST" 2>/dev/null || echo unpublished)}
CAP=${CC_CONTAINERS_PER_NODE:-16}
RUNNER_PORT=${CC_RUNNER_PORT:-18500}
GW_PORT=${CC_GATEWAY_PORT:-18400}
SERVED=${CC_SERVED_MODEL:-glm-5.3-flash}
JOB=${SLURM_JOB_ID:-manual}
HOST=$(hostname)
NODE_IP=$(ip -4 -o addr show "${SOCKET_IFNAME:-eth0}" | awk '{print $4}' | cut -d/ -f1)
SOCK_ROOT=/tmp/cc-sock-$JOB
NODE_LOG=$CC_RUN_DIR/nodes/$HOST
GW_NAME=cc_gw_$JOB
CTX=$IMAGES/hip-cc-rollout/$CTX_TAG.context.tar
EXPECTED_ID=${CC_EXPECTED_IMAGE_ID:-$(cat "$IMAGES/hip-cc-rollout/$CTX_TAG.image_id" 2>/dev/null || true)}
log() { echo "[cc_node $HOST $(date -u +%T)] $*"; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  log "role: cc-runner http://$NODE_IP:$RUNNER_PORT (capacity $CAP, image hip-cc-rollout:$CTX_TAG)," \
      "cc-gateway http://$NODE_IP:$GW_PORT (container $GW_NAME), sockets $SOCK_ROOT, logs $NODE_LOG," \
      "required image ID ${EXPECTED_ID:-<none published>}"
  exit 0
fi

[ "$CTX_TAG" != unpublished ] || { log "no published image context under $IMAGES/hip-cc-rollout"; exit 1; }
[[ "$EXPECTED_ID" == sha256:* ]] || { log "no published image ID for $CTX_TAG; refusing to start"; exit 1; }
mkdir -p "$NODE_LOG"
for f in runner_token gateway_admin_token; do
  [ "${1:-}" = verify ] && break
  [ "$(stat -c %a "$CC_SECRETS_DIR/$f")" = 600 ] || { log "$CC_SECRETS_DIR/$f must be mode 0600"; exit 1; }
done

(
  flock 9
  docker image inspect "$BASE" >/dev/null 2>&1 || docker load -q -i "$BASE_TAR"
  have=$(docker image inspect -f '{{.Id}}' "hip-cc-rollout:$CTX_TAG" 2>/dev/null || true)
  if [ "$have" != "$EXPECTED_ID" ]; then
    (cd "$(dirname "$CTX")" && sha256sum -c --quiet "$CTX_TAG.context.tar.sha256")
    log "building hip-cc-rollout:$CTX_TAG from $CTX (had ${have:-none})"
    docker image rm -f "hip-cc-rollout:$CTX_TAG" >/dev/null 2>&1 || true
    "$HC/image/build_from_context.sh" "$CTX" "hip-cc-rollout:$CTX_TAG" "$BASE" >/dev/null
  fi
) 9>"/tmp/cc-image-$(id -u).lock"
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "hip-cc-rollout:$CTX_TAG" 2>/dev/null || true)
if [ "$IMAGE_ID" != "$EXPECTED_ID" ]; then
  log "image ID ${IMAGE_ID:-missing} != published $EXPECTED_ID; this node runs no containers"
  exit 1
fi
log "image hip-cc-rollout:$CTX_TAG verified: $IMAGE_ID"
[ "${1:-}" = verify ] && exit 0

cleanup() {
  trap - EXIT TERM INT
  log "cleanup"
  [ -n "${RUNNER_PID:-}" ] && kill "$RUNNER_PID" 2>/dev/null || true
  docker rm -f "$GW_NAME" >/dev/null 2>&1 || true
  docker ps -aq --filter "label=cc.job=$JOB" | xargs -r docker rm -f >/dev/null 2>&1 || true
  rm -rf "$SOCK_ROOT"
}
trap cleanup EXIT TERM INT

docker rm -f "$GW_NAME" >/dev/null 2>&1 || true
TOKENIZER_ARGS=()
[ -n "${CC_TOKENIZER:-}" ] && TOKENIZER_ARGS=(--tokenizer "$CC_TOKENIZER")
docker run -d --name "$GW_NAME" --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e OMP_NUM_THREADS=2 -e PYTHONPATH="$(dirname "$(dirname "$HC")")" \
  -v /home/yuankach:/home/yuankach:ro -v "$NODE_LOG:$NODE_LOG" -v "$CC_SECRETS_DIR:$CC_SECRETS_DIR:ro" \
  ${CC_TOKENIZER:+-v "$CC_TOKENIZER:$CC_TOKENIZER:ro"} \
  "$BASE" python3 "$HC/cc_gateway.py" --host "$NODE_IP" --port "$GW_PORT" --served-model "$SERVED" \
  --admin-token-file "$CC_SECRETS_DIR/gateway_admin_token" --log-dir "$NODE_LOG/gateway" "${TOKENIZER_ARGS[@]}" \
  >/dev/null

export CC_CONTAINER_LABEL="cc.job=$JOB"
python3 "$HC/cc_runner.py" --host "$NODE_IP" --port "$RUNNER_PORT" --image "$IMAGE_ID" \
  --capacity "$CAP" --token-file "$CC_SECRETS_DIR/runner_token" --sock-root "$SOCK_ROOT" \
  --log-root "$NODE_LOG/episodes" > "$NODE_LOG/runner.log" 2>&1 &
RUNNER_PID=$!

for _ in $(seq 1 120); do
  gw=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://$NODE_IP:$GW_PORT/healthz" || true)
  rn=$(python3 - "$NODE_IP" "$RUNNER_PORT" "$CC_SECRETS_DIR/runner_token" <<'EOF'
import sys, urllib.request
ip, port, token_file = sys.argv[1:]
req = urllib.request.Request(f"http://{ip}:{port}/health",
                             headers={"Authorization": "Bearer " + open(token_file).read().strip()})
try:
    print(urllib.request.urlopen(req, timeout=3).status)
except Exception:
    print(0)
EOF
)
  [ "$gw" = 200 ] && [ "$rn" = 200 ] && break
  kill -0 "$RUNNER_PID" 2>/dev/null || { log "runner exited"; tail -20 "$NODE_LOG/runner.log"; exit 1; }
  sleep 2
done
[ "$gw" = 200 ] && [ "$rn" = 200 ] || { log "gateway $gw runner $rn not healthy"; docker logs --tail 20 "$GW_NAME"; exit 1; }
echo "http://$NODE_IP:$RUNNER_PORT http://$NODE_IP:$GW_PORT" >> "$CC_RUN_DIR/cc_nodes.txt"
log "up: runner :$RUNNER_PORT (capacity $CAP) gateway :$GW_PORT image $IMAGE_ID"
wait "$RUNNER_PID"

#!/bin/bash
# Build hip-cc-rollout on a compute node and publish its build context (plan §4.2, §10 P4).
#
#   build_image.sh [--publish]
#
# The base (rocm/sgl-dev, 75 GB) is already saved under $IMAGES and loaded on every Miles node,
# so what is published is the small, deterministic build context, not a second 75 GB save:
# $IMAGES/hip-cc-rollout/<tag>.context.tar (+ .sha256, and LATEST naming the tag). cc_node.sh
# rebuilds the identical image from it on each node; the tag is the context's sha256[:12], and
# <tag>.image_id pins the image ID every node's rebuild must reproduce (build_from_context.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-/shared_inference/yuankach/glm53/images}
BASE=${BASE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}
BASE_TAR=${BASE_TAR:-$IMAGES/sgl-dev_miles-rocm10-mi30x-20261001.tar}
AKA=${AKA:-/home/yuankach/hip-cc/AgentKernelArena}
AKA_COMMIT=${AKA_COMMIT:-ace3ee5f}
CC_VERSION=${CC_VERSION:-2.1.311}
CLAUDE=${CLAUDE:-/home/yuankach/hip-cc/tools/cc-$CC_VERSION/node_modules/@anthropic-ai/claude-code/bin/claude.exe}
KB_DIR=${KB_DIR:-/shared_inference/yuankach/glm53/hip-cc/kb/current}

git -C "$AKA" rev-parse HEAD | grep -q "^$AKA_COMMIT" || { echo "AKA is not at $AKA_COMMIT"; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || docker load -q -i "$BASE_TAR"

CTX=$(mktemp -d /tmp/hipcc-image-XXXXXX)
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Dockerfile" "$HERE/aka_eval.py" "$HERE/uds_relay.py" "$CTX/"
cp "$AKA/src/tools/arena_eval/audit.py" "$CTX/aka_audit.py"
cp "$CLAUDE" "$CTX/claude"
if [ -d "$KB_DIR" ]; then
  cp -r "$(readlink -f "$KB_DIR")" "$CTX/kb"
else
  mkdir "$CTX/kb" && echo "# /kb is not built yet" > "$CTX/kb/INDEX.md"
fi
kb_sha=$(cd "$CTX/kb" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
base_id=$(docker image inspect -f '{{.Id}}' "$BASE")
cat > "$CTX/cc_image.json" <<EOF
{"base": "$BASE", "base_id": "$base_id", "claude_code": "$CC_VERSION",
 "claude_sha256": "$(sha256sum "$CTX/claude" | cut -d' ' -f1)",
 "aka_commit": "$(git -C "$AKA" rev-parse HEAD)", "kb_sha": "$kb_sha"}
EOF
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf "$CTX.tar" -C "$CTX" .
sha=$(sha256sum "$CTX.tar" | cut -d' ' -f1)
TAG=hip-cc-rollout:${sha:0:12}
echo "context sha256 $sha, $(du -h "$CTX.tar" | cut -f1), tag $TAG"
cat "$CTX/cc_image.json"
docker image rm -f "$TAG" >/dev/null 2>&1 || true
IMAGE_ID=$("$HERE/build_from_context.sh" "$CTX.tar" "$TAG" "$BASE")
echo "image id $IMAGE_ID"
docker run --rm --network none --user 10001:10001 --read-only --tmpfs /tmp "$TAG" bash -c \
  'set -e; claude --version; aka-eval 2>&1 | head -1; test -r /kb/INDEX.md; python3 -c "import torch; print(torch.__version__)"; hipcc --version | head -2'
if [ "${1:-}" = "--publish" ]; then
  mkdir -p "$IMAGES/hip-cc-rollout"
  cp "$CTX.tar" "$IMAGES/hip-cc-rollout/${sha:0:12}.context.tar"
  echo "$sha  ${sha:0:12}.context.tar" > "$IMAGES/hip-cc-rollout/${sha:0:12}.context.tar.sha256"
  echo "$IMAGE_ID" > "$IMAGES/hip-cc-rollout/${sha:0:12}.image_id"
  echo "${sha:0:12}" > "$IMAGES/hip-cc-rollout/LATEST"
  echo "published $IMAGES/hip-cc-rollout/${sha:0:12}.context.tar"
fi
rm -f "$CTX.tar"
echo "IMAGE $TAG"

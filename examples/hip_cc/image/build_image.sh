#!/bin/bash
# Build hip-cc-rollout on a compute node and publish its build contexts (plan §4.2, §10 P4/P5).
#
#   build_image.sh [--publish]
#
# Two variants from one staging: hip-cc-rollout (with the frozen /kb) and hip-cc-rollout-nokb
# (no /kb at all, for the Phase 0b /kb-off arms). The base (rocm/sgl-dev, 75 GB) is already saved
# under $IMAGES and loaded on every Miles node, so what is published is each small, deterministic
# build context, not a second 75 GB save: $IMAGES/hip-cc-rollout/<tag>.context.tar (+ .sha256),
# <tag>.image_id pinning the image ID every node's rebuild must reproduce (build_from_context.sh),
# and LATEST / LATEST-nokb naming the tags. Each tag is its context's sha256[:12].
#
# /kb comes only from the frozen build named in $KB_ROOT/FROZEN (kb/build_kb.py output).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-/shared_inference/yuankach/glm53/images}
BASE=${BASE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}
BASE_TAR=${BASE_TAR:-$IMAGES/sgl-dev_miles-rocm10-mi30x-20261001.tar}
AKA=${AKA:-/home/yuankach/hip-cc/AgentKernelArena}
AKA_COMMIT=${AKA_COMMIT:-ace3ee5f}
CC_VERSION=${CC_VERSION:-2.1.311}
CLAUDE=${CLAUDE:-/home/yuankach/hip-cc/tools/cc-$CC_VERSION/node_modules/@anthropic-ai/claude-code/bin/claude.exe}
KB_ROOT=${KB_ROOT:-/shared_inference/yuankach/glm53/hip-cc/kb}
KB_SHA=${KB_SHA:-$(cat "$KB_ROOT/FROZEN" 2>/dev/null || true)}
KB_DIR=$KB_ROOT/$KB_SHA

[ -n "$KB_SHA" ] && [ -f "$KB_DIR/INDEX.md" ] || { echo "no frozen /kb ($KB_ROOT/FROZEN)"; exit 1; }
git -C "$AKA" rev-parse HEAD | grep -q "^$AKA_COMMIT" || { echo "AKA is not at $AKA_COMMIT"; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || docker load -q -i "$BASE_TAR"

WORK=$(mktemp -d /tmp/hipcc-image-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
base_id=$(docker image inspect -f '{{.Id}}' "$BASE")
claude_sha=$(sha256sum "$CLAUDE" | cut -d' ' -f1)
aka_head=$(git -C "$AKA" rev-parse HEAD)

stage() {  # stage <variant> <kb_sha or none>
  local ctx=$WORK/$1
  mkdir -p "$ctx/kb"
  cp "$HERE/Dockerfile" "$HERE/aka_eval.py" "$HERE/uds_relay.py" "$ctx/"
  cp "$AKA/src/tools/arena_eval/audit.py" "$ctx/aka_audit.py"
  cp "$CLAUDE" "$ctx/claude"
  [ "$2" = none ] || cp -r "$KB_DIR/." "$ctx/kb/"
  cat > "$ctx/cc_image.json" <<EOF
{"variant": "$1", "base": "$BASE", "base_id": "$base_id", "claude_code": "$CC_VERSION",
 "claude_sha256": "$claude_sha", "aka_commit": "$aka_head", "kb_sha": "$2"}
EOF
  chmod -R u+w,a+rX "$ctx"
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf "$ctx.tar" -C "$ctx" .
}

build() {  # build <variant> <repo> <kb check>
  local tar=$WORK/$1.tar sha tag id
  sha=$(sha256sum "$tar" | cut -d' ' -f1)
  tag=$2:${sha:0:12}
  echo "[$1] context sha256 $sha, $(du -h "$tar" | cut -f1), tag $tag"
  cat "$WORK/$1/cc_image.json"
  docker image rm -f "$tag" >/dev/null 2>&1 || true
  id=$("$HERE/build_from_context.sh" "$tar" "$tag" "$BASE")
  echo "[$1] image id $id"
  docker run --rm --network none --user 10001:10001 --read-only --tmpfs /tmp "$tag" bash -c \
    "set -e; claude --version; aka-eval 2>&1 | head -1; $3; python3 -c 'import torch; print(torch.__version__)'; hipcc --version | head -1"
  if [ "$PUBLISH" = 1 ]; then
    mkdir -p "$IMAGES/hip-cc-rollout"
    cp "$tar" "$IMAGES/hip-cc-rollout/${sha:0:12}.context.tar"
    echo "$sha  ${sha:0:12}.context.tar" > "$IMAGES/hip-cc-rollout/${sha:0:12}.context.tar.sha256"
    echo "$id" > "$IMAGES/hip-cc-rollout/${sha:0:12}.image_id"
    echo "${sha:0:12}" > "$IMAGES/hip-cc-rollout/$4"
    echo "[$1] published $IMAGES/hip-cc-rollout/${sha:0:12}.context.tar as $4"
  fi
  echo "IMAGE $1 $tag $id"
}

PUBLISH=0
[ "${1:-}" = "--publish" ] && PUBLISH=1
stage kb "$KB_SHA"
stage nokb none
build kb hip-cc-rollout 'test -r /kb/INDEX.md && echo "/kb present"' LATEST
build nokb hip-cc-rollout-nokb 'test ! -e /kb && echo "/kb absent"' LATEST-nokb

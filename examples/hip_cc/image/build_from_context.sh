#!/bin/bash
# Build hip-cc-rollout from a staged context tar, reproducibly and with no network.
#
#   build_from_context.sh <context.tar> <tag> [<base>]     prints the image ID (sha256:...)
#
# Used by build_image.sh (which publishes the ID) and by cc_node.sh (which rebuilds on each node
# and refuses to start anything unless the ID matches). The build sees only the context tar:
# RUN steps have no network, every timestamp is pinned to SOURCE_DATE_EPOCH=0, and layer mtimes
# are rewritten, so the same context on the same base gives the same image ID on every node.
set -euo pipefail
CTX_TAR=$1
TAG=$2
BASE=${3:-${BASE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}}
sha=$(sha256sum "$CTX_TAR" | cut -d' ' -f1)
docker buildx build -q --network none --provenance=false --sbom=false \
  --build-arg BASE="$BASE" --build-arg SOURCE_DATE_EPOCH=0 --label "cc.context_sha256=$sha" \
  --output "type=image,name=$TAG,rewrite-timestamp=true" - < "$CTX_TAR" >/dev/null
docker image inspect -f '{{.Id}}' "$TAG"

# hip-cc-rollout: the Claude Code rollout container (plan §4.2).
# gfx950 hipcc and a ROCm torch for local builds (no GPU), the pinned Claude Code binary,
# the aka-eval shim and the read-only /kb (left out of hip-cc-rollout-nokb: its context has an
# empty kb/, removed below). Built by build_image.sh from the contexts it stages.
ARG BASE=rocm/sgl-dev:miles-rocm10-mi30x-20261001
FROM ${BASE}

COPY claude /opt/cc/bin/claude
COPY aka_eval.py /opt/cc/bin/aka-eval
COPY uds_relay.py aka_audit.py /opt/cc/
COPY kb /kb
COPY cc_image.json /opt/cc/cc_image.json

RUN set -eux; \
    chmod 0755 /opt/cc/bin/claude /opt/cc/bin/aka-eval; \
    chmod -R a+rX,go-w /opt/cc /kb; \
    sed -i '1s|.*|#!/opt/venv/bin/python3|' /opt/cc/bin/aka-eval; \
    bitcode=$(find "$(readlink -f /opt/rocm)/" -type d -name bitcode -path '*amdgcn*' | head -1); \
    test -n "$bitcode"; ln -s "$bitcode" /opt/cc/bitcode; \
    mkdir -p /workspace /home/agent; \
    chmod 0777 /workspace /home/agent; \
    if [ -z "$(ls -A /kb)" ]; then rmdir /kb; fi

# hipcc resolves its install from these instead of /root (unreadable for the non-root agent).
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_DEVICE_LIB_PATH=/opt/cc/bitcode \
    PYTORCH_ROCM_ARCH=gfx950 \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    AKA_KERNEL_CACHE_DIR=/tmp/aka-kernel-cache \
    PATH=/opt/cc/bin:/opt/rocm/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/home/agent \
    MAX_JOBS=2 \
    OMP_NUM_THREADS=1

WORKDIR /workspace

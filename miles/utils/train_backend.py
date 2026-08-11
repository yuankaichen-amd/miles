"""The training backends `--train-backend` accepts.

Primus is a configuration and kernel-patch layer over Megatron rather than a separate
trainer, so it shares Megatron's argument parsing, checkpointing and offload paths.
Code that cares about "is this a Megatron run" should ask `uses_megatron` instead of
comparing against "megatron".
"""

MEGATRON_TRAIN_BACKENDS = ("megatron", "primus")
TRAIN_BACKENDS = ("megatron", "fsdp", "primus")


def uses_megatron(train_backend: str) -> bool:
    return train_backend in MEGATRON_TRAIN_BACKENDS

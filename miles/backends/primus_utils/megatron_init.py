"""The `--custom-megatron-init-path` entry point for `--primus-config` runs.

Miles calls the custom Megatron init at the end of `megatron_utils.initialize.init`:
Megatron's global args are set and parallel state exists, but no model has been
built yet. That is the one window every Primus patch phase needs, so all of them run
here. `PrimusConfigBridge.attach` points the hook at `primus_megatron_init` and keeps
any init the user asked for in `args.primus_chained_megatron_init_path`, which runs first.
"""

from __future__ import annotations

import argparse

from miles.backends.primus_utils.megatron_compat import install_module_name_kwarg_shims
from miles.backends.primus_utils.patches import PrimusPatchRunner

_runner: PrimusPatchRunner | None = None


def get_primus_patch_runner() -> PrimusPatchRunner | None:
    return _runner


def primus_megatron_init(args: argparse.Namespace) -> None:
    global _runner

    chained = getattr(args, "primus_chained_megatron_init_path", None)
    if chained:
        from miles.utils.function_registry import load_function

        load_function(chained)(args)

    _runner = PrimusPatchRunner.create_if_enabled(args)
    if _runner is None:
        return
    _runner.run("setup")
    _runner.run("build_args")
    _runner.bind_megatron_global_args()
    _runner.run("before_train")
    install_module_name_kwarg_shims()

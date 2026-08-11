"""Bridge the small naming gaps between Primus' Megatron fork and Miles'.

Primus is developed against the Megatron it vendors, Miles runs its own, and the two
have drifted in places. Where the drift is only in how a type is spelled, a shim lets
Primus' patches load against Miles' Megatron instead of dying on an import.

The bar for a shim here is deliberately high: it is for types that exist in both forks
under different names or in different modules, never for behaviour Miles' Megatron does
not actually have. Anything else belongs in the incompatibility list in `patches.py`.
"""

from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger(__name__)


def install_megatron_compat_shims() -> list[str]:
    """Install every shim Miles' Megatron needs, returning the names of those applied."""
    installed = []
    if _install_te_grouped_mlp_submodules():
        installed.append("megatron.core.transformer.moe.experts.TEGroupedMLPSubmodules")
    # Depends on the alias above, which is what makes Primus' provider module importable.
    if _install_spec_provider_eager_fallback():
        installed.append("PrimusTurboSpecProvider.fallback_to_eager_attn")

    if installed:
        logger.info(
            "Installed %d Megatron compatibility shim(s) for Primus: %s.",
            len(installed),
            ", ".join(installed),
        )
    return installed


def _install_te_grouped_mlp_submodules() -> bool:
    """Alias `TEGroupedMLPSubmodules` onto `MLPSubmodules`.

    Primus' fork splits a dedicated `TEGroupedMLPSubmodules` dataclass out of
    `MLPSubmodules`; Miles' fork keeps the original and its `TEGroupedMLP` consumes
    `MLPSubmodules` directly. The two carry the same fields and Primus builds them by
    keyword, so this aliases rather than approximates.

    It matters well beyond the MoE path it names. Primus imports the symbol eagerly in
    the module holding `PrimusTurboSpecProvider` but only uses it for grouped GEMM, so
    without the alias every Primus-Turbo kernel is unreachable on a dense model that
    would never touch grouped MLP at all.
    """
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe import experts

    if hasattr(experts, "TEGroupedMLPSubmodules"):
        return False

    # Guard against the forks drifting further: aliasing is only sound while
    # MLPSubmodules still carries the fields Primus builds the grouped spec from.
    fields = {field.name for field in dataclasses.fields(MLPSubmodules)}
    required = {"linear_fc1", "linear_fc2"}
    if not required <= fields:
        logger.warning(
            "Not aliasing TEGroupedMLPSubmodules: MLPSubmodules is missing %s, so Primus' "
            "grouped MLP spec would be built wrong. Primus-Turbo will stay disabled.",
            ", ".join(sorted(required - fields)),
        )
        return False

    experts.TEGroupedMLPSubmodules = MLPSubmodules
    return True


def _install_spec_provider_eager_fallback() -> bool:
    """Teach `PrimusTurboSpecProvider` the `fallback_to_eager_attn` argument.

    Miles' Megatron builds its spec provider as `TESpecProvider(fallback_to_eager_attn=...)`
    and returns eager `DotProductAttention` when the flag is set; Primus' fork has no such
    argument, so the provider Primus swaps in raises TypeError the moment Megatron builds a
    layer spec. Patching the class rather than subclassing keeps Primus' own patch, which
    assigns this exact class into four modules, working untouched.

    Eager attention wins over Turbo when asked for: the caller requests it because
    something in the model needs it, which is not a preference Turbo should override.
    """
    import inspect

    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    if "fallback_to_eager_attn" in inspect.signature(PrimusTurboSpecProvider.__init__).parameters:
        return False

    original_init = PrimusTurboSpecProvider.__init__
    original_core_attention = PrimusTurboSpecProvider.core_attention

    def __init__(self, *args, fallback_to_eager_attn: bool = False, **kwargs):
        original_init(self, *args, **kwargs)
        self.fallback_to_eager_attn = fallback_to_eager_attn

    def core_attention(self):
        if getattr(self, "fallback_to_eager_attn", False):
            return DotProductAttention
        return original_core_attention(self)

    PrimusTurboSpecProvider.__init__ = __init__
    PrimusTurboSpecProvider.core_attention = core_attention
    return True

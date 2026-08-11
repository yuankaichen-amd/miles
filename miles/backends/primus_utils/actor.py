"""Train actor for the Primus backend."""

from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager

from miles.backends.megatron_utils import actor as megatron_actor
from miles.backends.primus_utils.patches import PrimusPatchRunner


class PrimusTrainRayActor(megatron_actor.MegatronTrainRayActor):
    """A Megatron train actor that runs Primus' Megatron patches around init.

    Primus configures Megatron and swaps in Primus-Turbo kernels; it is not a separate
    trainer. So the training step, checkpointing and weight-sync paths are inherited
    unchanged and only the patch phases are added.

    The patches have to be applied here rather than in the driver: monkey patches do
    not cross the Ray boundary.
    """

    def init(self, args: Namespace, role: str, **kwargs) -> int | None:
        self._primus_patches = PrimusPatchRunner.create_if_enabled(args)
        if self._primus_patches is not None:
            self._primus_patches.run("setup")
            self._primus_patches.run("build_args")

        with self._primus_patches_after_megatron_init():
            return super().init(args, role, **kwargs)

    @contextmanager
    def _primus_patches_after_megatron_init(self) -> Iterator[None]:
        """Run the remaining phases between Megatron's init and the model build.

        The before_train set replaces the layer spec provider that the model build
        reads, so it has to land after Megatron's global args exist but before the
        model is built. Both of those happen inside the inherited
        `MegatronTrainRayActor.init`, and there is no method of ours in between, so
        the phases hang off the Megatron init function that init calls.
        """
        if self._primus_patches is None:
            yield
            return

        megatron_init = megatron_actor.init

        def init_then_patch(*args, **kwargs):
            result = megatron_init(*args, **kwargs)
            self._primus_patches.bind_megatron_global_args()
            self._primus_patches.run("before_train")
            return result

        megatron_actor.init = init_then_patch
        try:
            yield
        finally:
            megatron_actor.init = megatron_init

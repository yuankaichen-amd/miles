"""Ordering of the Primus patch phases inside the real PrimusTrainRayActor.init path.

The phases only do the right thing at the right moment: setup and build_args have to
land before Megatron reads its arguments, and before_train has to land after Megatron is
initialised but before the model is built, because it swaps the layer spec provider that
initialize_model_and_optimizer() reads. PrimusTrainRayActor gets in between by wrapping
the Megatron init function, so the wrapping must also come back off afterwards.
"""

import importlib
import inspect
from argparse import Namespace
from unittest.mock import Mock

import pytest


class _StopBeforeModelBuild(Exception):
    """Ends init() at the first statement past the Primus phases."""


@pytest.fixture(scope="module")
def megatron_actor_module():
    return importlib.import_module("miles.backends.megatron_utils.actor")


@pytest.fixture(scope="module")
def primus_actor_module():
    return importlib.import_module("miles.backends.primus_utils.actor")


class _RecordingRunner:
    def __init__(self, calls: list[str]):
        self._calls = calls

    def run(self, phase: str) -> int:
        self._calls.append(phase)
        return 0

    def bind_megatron_global_args(self) -> None:
        self._calls.append("bind_megatron_global_args")


def _run_init(megatron_actor_module, primus_actor_module, monkeypatch, *, with_primus: bool) -> list[str]:
    calls: list[str] = []
    runner = _RecordingRunner(calls) if with_primus else None

    monkeypatch.setattr(megatron_actor_module, "monkey_patch_torch_dist", lambda: None)
    monkeypatch.setattr(megatron_actor_module.TrainRayActor, "init", lambda *a, **k: None)
    monkeypatch.setattr(megatron_actor_module, "all_replay_managers", [])
    monkeypatch.setattr(megatron_actor_module, "routing_replay_manager", Mock())
    # The init decorator defers Timer().start(), which asserts on a second call.
    monkeypatch.setattr(megatron_actor_module, "Timer", Mock())
    monkeypatch.setattr(primus_actor_module.PrimusPatchRunner, "create_if_enabled", staticmethod(lambda args: runner))

    def fake_megatron_init(args, **kwargs):
        calls.append("megatron_init")

    monkeypatch.setattr(megatron_actor_module, "init", fake_megatron_init)

    def stop(*args, **kwargs):
        calls.append("past_primus_phases")
        raise _StopBeforeModelBuild

    monkeypatch.setattr(megatron_actor_module.FTTestActionActorExecutor, "from_args", stop)

    # Normally set by the base class init, which is stubbed out above.
    worker = object.__new__(primus_actor_module.PrimusTrainRayActor)
    worker._indep_dp_store_addr = None
    worker._rank = 0
    with pytest.raises(_StopBeforeModelBuild):
        worker.init(Namespace(), "actor", indep_dp_info=Mock())
    return calls


def test_primus_phases_straddle_megatron_init(megatron_actor_module, primus_actor_module, monkeypatch):
    calls = _run_init(megatron_actor_module, primus_actor_module, monkeypatch, with_primus=True)

    assert calls == [
        "setup",
        "build_args",
        "megatron_init",
        "bind_megatron_global_args",
        "before_train",
        "past_primus_phases",
    ]


def test_before_train_precedes_model_build(megatron_actor_module, primus_actor_module, monkeypatch):
    calls = _run_init(megatron_actor_module, primus_actor_module, monkeypatch, with_primus=True)

    # initialize_model_and_optimizer() runs well past the stop point, so before_train
    # having already fired is what keeps the spec provider swap visible to the model.
    assert calls.index("before_train") < calls.index("past_primus_phases")


def test_megatron_init_is_left_unwrapped_afterwards(megatron_actor_module, primus_actor_module, monkeypatch):
    calls = _run_init(megatron_actor_module, primus_actor_module, monkeypatch, with_primus=True)
    calls.clear()

    megatron_actor_module.init(Namespace())

    # A wrapper left behind on the module would re-run the phases for anything that
    # initialises Megatron later in this process.
    assert calls == ["megatron_init"]


def test_init_is_untouched_when_patches_are_off(megatron_actor_module, primus_actor_module, monkeypatch):
    calls = _run_init(megatron_actor_module, primus_actor_module, monkeypatch, with_primus=False)

    assert calls == ["megatron_init", "past_primus_phases"]


def test_the_megatron_actor_carries_no_primus_logic(megatron_actor_module):
    # The Megatron backend stays free of Primus: everything Primus needs lives in the
    # subclass, so a plain --train-backend megatron run cannot be affected by it.
    assert "primus" not in inspect.getsource(megatron_actor_module).lower()

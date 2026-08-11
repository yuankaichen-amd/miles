import sys
import textwrap

import pytest

PRIMUS_YAML = """
work_group: test
user_name: test
exp_name: primus_bridge_test
workspace: {workspace}

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml
    model: llama3.1_8B.yaml
    overrides:
      train_iters: 50
      tensor_model_parallel_size: 1
      enable_primus_turbo: true
"""


def _write_config(tmp_path):
    config = tmp_path / "primus_exp.yaml"
    config.write_text(textwrap.dedent(PRIMUS_YAML).format(workspace=tmp_path / "output"))
    return config


def _parse(monkeypatch, tmp_path, extra_argv):
    import miles.backends.megatron_utils.arguments as megatron_arguments
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(miles_arguments, "miles_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: None)
    monkeypatch.setattr(miles_arguments, "sglang_validate_args", lambda args: None)

    argv = [
        "pytest",
        "--train-backend",
        "primus",
        "--rollout-batch-size",
        "1",
        "--primus-config",
        str(_write_config(tmp_path)),
    ] + extra_argv
    monkeypatch.setattr(sys, "argv", argv)

    return miles_arguments.parse_args()


@pytest.fixture(autouse=True)
def _require_primus():
    pytest.importorskip("primus.core.config.primus_config")
    pytest.importorskip("megatron.training.arguments")


def test_primus_yaml_supplies_megatron_args(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, [])

    # Values come from the llama3.1_8B model preset that the YAML selects.
    assert args.num_layers == 32
    assert args.hidden_size == 4096
    assert args.train_iters == 50
    assert args.tensor_model_parallel_size == 1


def test_explicit_cli_flag_overrides_primus_yaml(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, ["--tensor-model-parallel-size", "4", "--train-iters", "7"])

    assert args.tensor_model_parallel_size == 4
    assert args.train_iters == 7
    # Untouched keys still come from the YAML.
    assert args.num_layers == 32


def test_primus_override_applies_before_megatron(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, ["--primus-override", "train_iters=123", "num_layers=4"])

    assert args.train_iters == 123
    assert args.num_layers == 4


def test_primus_only_params_are_carried_not_dropped(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, [])

    # Primus feature flags have no Megatron argument, so they must survive on the side.
    assert args.primus_params["enable_primus_turbo"] is True
    assert "use_turbo_attention" in args.primus_params


def test_primus_only_params_stay_out_of_the_args_namespace(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, [])

    # The Primus config is the only entry point; nothing is promoted to a top-level arg.
    assert not hasattr(args, "enable_primus_turbo")
    assert not hasattr(args, "use_turbo_attention")
    # Miles' own flags keep their Miles values even where the Primus config names them.
    assert args.offload is False
    assert args.mlflow_run_name is None


def test_strict_config_rejects_unmapped_params(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="not defined by Miles' Megatron"):
        _parse(monkeypatch, tmp_path, ["--primus-strict-config"])


def test_primus_config_requires_the_primus_backend(monkeypatch, tmp_path):
    # A Primus config under another backend would silently skip every Primus patch.
    with pytest.raises(AssertionError, match="requires --train-backend primus"):
        _parse(monkeypatch, tmp_path, ["--train-backend", "megatron"])


def test_primus_backend_requires_a_config(monkeypatch):
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(sys, "argv", ["pytest", "--train-backend", "primus", "--rollout-batch-size", "1"])

    with pytest.raises(AssertionError, match="requires --primus-config"):
        miles_arguments.parse_args()


def test_primus_backend_parses_megatron_arguments(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, [])

    # Primus trains with Megatron, so the Megatron-only arguments must be present.
    assert args.train_backend == "primus"
    assert args.pipeline_model_parallel_size == 1


def test_unknown_primus_module_lists_alternatives(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="no module 'trainer'"):
        _parse(monkeypatch, tmp_path, ["--primus-module", "trainer"])

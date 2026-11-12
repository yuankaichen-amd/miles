"""Turn one PyTorch-to-HIP task (dataset row or KernelBench file) into AKA v3 files.

``analyze`` runs inside a throwaway child process: it executes dataset code. It
returns either ``{"drop": reason}`` or the rendered ``task.yaml``,
``workloads.json``, ``reference.py`` and ``baseline/kernel.py`` texts plus
metadata. Shapes come from the meta device, so huge workloads cost nothing;
value statistics and the determinism/in-place probes run on CPU only when the
inputs are small.
"""

from __future__ import annotations

import inspect
import json
import keyword
import math
import re
import textwrap
from typing import Any

import yaml

SMALL_NUMEL = 16 * 1024 * 1024
STATS_NUMEL = 128 * 1024 * 1024
TARGET_W1_BYTES = 64 << 20
MAX_W1_BYTES = 4 << 30
MAX_SCALE = 1024
RANDOM_TOKENS = re.compile(
    r"\b(?:torch\.)?(?:rand|randn|randint|randperm|bernoulli|multinomial|normal|rand_like|randn_like|"
    r"randint_like|poisson)\s*\(|\.(?:normal_|uniform_|bernoulli_|exponential_|geometric_|cauchy_|"
    r"log_normal_|random_)\s*\("
)
DYNAMIC_SHAPE_TOKENS = re.compile(r"\b(?:nonzero|unique|unique_consecutive|masked_select|bincount)\s*\(")
CATEGORY_RULES = [
    ("attention", r"attention|attn|sdpa|flash"),
    ("convolution", r"conv"),
    ("normalization", r"norm|layernorm|batchnorm|groupnorm|instancenorm|rmsnorm"),
    ("embedding", r"embed"),
    ("sorting", r"sort|topk|argsort|kthvalue"),
    ("gemm", r"matmul|gemm|linear|bmm|mm_|einsum|addmm|baddbmm|dense|mlp|ffn|feedforward"),
    ("activation", r"relu|gelu|silu|swish|mish|sigmoid|tanh|elu|softplus|hardtanh|hardswish|"
                   r"hardsigmoid|leaky|selu|softsign|activation|glu"),
    ("reduction", r"softmax|sum|mean|max|min|reduce|prod|argmax|argmin|cumsum|cumprod|logsumexp|"
                  r"norm_|var|std|loss|entropy"),
    ("elementwise", r"add|mul|div|sub|scale|clamp|abs|exp|log|pow|sqrt|elementwise|residual|bias"),
]
RESERVED_NAMES = {"reference", "run", "module_fn", "torch", "nn", "F", "math", "np", "self", "Model"}
FLOAT_DTYPES = {"bfloat16", "float16", "float32", "float64"}
AKA_DTYPES = FLOAT_DTYPES | {"int8", "uint8", "int16", "int32", "int64", "bool"}


class Drop(Exception):
    pass


def _dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _literal(value: Any) -> str:
    import torch

    if value is None or isinstance(value, (bool, str)):
        return repr(value)
    if isinstance(value, int):
        return repr(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if isinstance(value, torch.dtype):
        return f"torch.{_dtype_name(value)}"
    if isinstance(value, torch.Size):
        return f"torch.Size({list(value)!r})"
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, torch.Tensor):
        raise Drop("nested_tensor_arg")
    raise Drop(f"unsupported_constant_arg:{type(value).__name__}")


def _contains_tensor(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_tensor(v) for v in value.values())
    return False


def _identifier(raw: str, taken: set[str]) -> str:
    name = re.sub(r"\W", "_", raw).strip("_") or "t"
    if name[0].isdigit():
        name = "t" + name
    if keyword.iskeyword(name) or name in RESERVED_NAMES:
        name += "_in"
    base, index = name, 1
    while name in taken:
        index += 1
        name = f"{base}{index}"
    taken.add(name)
    return name


def operator_name(raw: str) -> str:
    name = re.sub(r"\W+", "_", raw or "op").strip("_").lower() or "op"
    if name[0].isdigit():
        name = "op_" + name
    if keyword.iskeyword(name):
        name += "_op"
    return name[:60].rstrip("_")


def category(name: str, code: str) -> str:
    text = f"{name} {code[:4000]}".lower()
    for cat, pattern in CATEGORY_RULES:
        if re.search(pattern, name.lower()):
            return cat
    for cat, pattern in CATEGORY_RULES[:6]:
        if re.search(pattern, text):
            return cat
    return "other"


def _init_args(spec: dict, init):
    if spec["source"] == "gpumode" and isinstance(init, (list, tuple)) and len(init) == 2 \
            and isinstance(init[0], (list, tuple)) and isinstance(init[1], dict):
        return list(init[0]), dict(init[1])
    return list(init), {}


def _find_model_class(namespace: dict, need_fn: bool):
    import torch

    candidates = []
    for value in namespace.values():
        if isinstance(value, type) and issubclass(value, torch.nn.Module) and value.__module__ == "__t2h__":
            params = inspect.signature(value.forward).parameters
            if not need_fn or "fn" in params:
                candidates.append(value)
    if not candidates:
        raise Drop("no_model_class_with_fn" if need_fn else "no_model_class")
    for value in candidates:
        if value.__name__ == "Model":
            return value
    return candidates[-1]


def _distribution(tensor, source_code: str) -> dict:
    """An AKA distribution matching the observed values (or the factory used when values are unknown)."""
    import torch

    dtype = _dtype_name(tensor.dtype)
    if tensor.device.type == "meta":
        if dtype in FLOAT_DTYPES:
            if re.search(r"torch\.rand\s*\(", source_code) and not re.search(r"torch\.randn\s*\(", source_code):
                return {"kind": "uniform", "low": 0.0, "high": 1.0}
            return {"kind": "normal", "mean": 0.0, "std": 1.0}
        raise Drop("large_non_float_input_unknown_range")
    if tensor.numel() == 0:
        raise Drop("empty_input")
    if dtype == "bool":
        return {"kind": "randint", "low": 0, "high": 2}
    if dtype not in FLOAT_DTYPES:
        low, high = int(tensor.min()), int(tensor.max())
        return {"kind": "randint", "low": low, "high": high + 1}
    values = tensor.detach().double()
    if not bool(torch.isfinite(values).all()):
        raise Drop("non_finite_input_values")
    low, high = float(values.min()), float(values.max())
    if low == high:
        if low == 0.0:
            return {"kind": "uniform", "low": -0.1, "high": 0.1}
        a, b = sorted((low * 0.5, low * 1.5))
        return {"kind": "uniform", "low": round(a, 6), "high": round(b, 6)}
    mean, std = float(values.mean()), float(values.std()) if values.numel() > 1 else 1.0
    span = high - low
    # Uniform draws fill [low, high] flatly; normal draws put ~0.3% beyond 3 sigma.
    if values.numel() >= 64 and std > 0 and abs(span / std - math.sqrt(12)) < 0.35:
        return {"kind": "uniform", "low": _snap(low, span), "high": _snap(high, span)}
    if values.numel() >= 64 and std > 0:
        mean = 0.0 if abs(mean) < 0.05 * std else _round(mean)
        std = 1.0 if abs(std - 1.0) < 0.05 else _round(std)
        return {"kind": "normal", "mean": mean, "std": std}
    return {"kind": "uniform", "low": _round(low), "high": _round(high) if high > low else _round(low + 1)}


def _snap(bound: float, span: float) -> float:
    """Sampled extremes sit just inside the factory's range (torch.rand -> [0, 1)); snap to it."""
    nice = round(bound * 2) / 2
    return float(nice) if abs(bound - nice) < 0.05 * span else _round(bound)


def _round(x: float) -> float:
    if x == 0:
        return 0.0
    digits = 3 - int(math.floor(math.log10(abs(x))))
    return float(round(x, max(digits, 0)))


def _flatten_outputs(result) -> list:
    import torch

    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)) and result and all(isinstance(r, torch.Tensor) for r in result):
        return list(result)
    raise Drop("non_tensor_output")


def _to_meta(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to("meta")
    return value


def _same(a, b) -> bool:
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(torch.equal(torch.nan_to_num(a, nan=1.2345), torch.nan_to_num(b, nan=1.2345))) and \
            bool(torch.equal(torch.isnan(a), torch.isnan(b)))
    return bool(torch.equal(a, b))


class Analyzer:
    def __init__(self, spec: dict):
        self.spec = spec
        self.kind = spec["kind"]  # "row" or "kernelbench"
        self.code = spec["functional"] if self.kind == "row" else spec["module"]
        self.flags: list[str] = []

    def _exec(self):
        import torch

        namespace: dict[str, Any] = {"__name__": "__t2h__"}
        try:
            exec(compile(self.code, f"<{self.spec['task_id']}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - dataset code
            raise Drop(f"exec_failed:{type(exc).__name__}") from exc
        for required in ("get_inputs", "get_init_inputs"):
            if not callable(namespace.get(required)):
                raise Drop(f"missing_{required}")
        clash = {"reference", "run"} & set(namespace)
        if clash:
            raise Drop(f"name_clash:{','.join(sorted(clash))}")
        self.ns = namespace
        torch.manual_seed(0)
        init_args, init_kwargs = _init_args(self.spec, namespace["get_init_inputs"]())
        if _contains_tensor(init_args) or _contains_tensor(init_kwargs):
            if self.kind == "kernelbench":
                raise Drop("tensor_init_args")
        self.init_args, self.init_kwargs = init_args, init_kwargs
        cls = _find_model_class(namespace, need_fn=self.kind == "row")
        if self.kind == "row":
            # The functional form is whatever forward's `fn` defaults to (module_fn, frm_fn, ...).
            fn = inspect.signature(cls.forward).parameters["fn"].default
            name = getattr(fn, "__name__", "")
            if not callable(fn) or namespace.get(name) is not fn:
                raise Drop("missing_module_fn")
            self.fn, self.fn_name = fn, name
        try:
            with torch.device("cpu"):
                self.model = cls(*init_args, **init_kwargs).eval()
        except Exception as exc:  # noqa: BLE001
            raise Drop(f"model_init_failed:{type(exc).__name__}") from exc
        self.model_cls = cls

    def _inputs(self, device: str):
        import torch

        torch.manual_seed(1)
        with torch.device(device):
            try:
                inputs = self.ns["get_inputs"]()
            except Exception as exc:  # noqa: BLE001
                raise Drop(f"get_inputs_failed:{type(exc).__name__}") from exc
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return list(inputs)

    def _record(self, inputs):
        """The exact (args, kwargs) the model passes to module_fn."""
        captured = {}

        def recorder(*args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            raise StopIteration

        try:
            self.model(*inputs, fn=recorder)
        except StopIteration:
            pass
        except Exception as exc:  # noqa: BLE001
            raise Drop(f"forward_failed:{type(exc).__name__}") from exc
        if "args" not in captured:
            raise Drop("module_fn_not_called")
        return list(captured["args"]), dict(captured["kwargs"])

    def _call_spec(self, args, kwargs, activation_ids):
        """Name every module_fn argument; tensors become inputs, the rest literals."""
        fn = self.fn
        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
        except TypeError as exc:
            raise Drop("module_fn_bind_failed") from exc
        params = inspect.signature(fn).parameters
        taken: set[str] = {self.fn_name}
        positional, keywords, tensors = [], [], []
        seen: dict[int, str] = {}
        for name, value in bound.arguments.items():
            kind = params[name].kind
            values = list(value) if kind == inspect.Parameter.VAR_POSITIONAL else \
                list(value.items()) if kind == inspect.Parameter.VAR_KEYWORD else [value]
            for index, item in enumerate(values):
                key = None
                if kind == inspect.Parameter.VAR_KEYWORD:
                    key, item = item
                raw = key or (f"{name}{index}" if kind == inspect.Parameter.VAR_POSITIONAL else name)
                expr = self._arg_expr(raw, item, taken, tensors, seen, activation_ids)
                if kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
                    keywords.append((key or name, expr))
                else:
                    positional.append(expr)
        return positional, keywords, tensors

    def _arg_expr(self, raw, item, taken, tensors, seen, activation_ids):
        import torch

        if isinstance(item, torch.Tensor):
            if id(item) in seen:
                return seen[id(item)]
            name = _identifier(raw, taken)
            taken.add(name)
            seen[id(item)] = name
            role = "activation" if id(item) in activation_ids else "weight"
            tensors.append((name, item, role))
            return name
        if _contains_tensor(item):
            raise Drop("nested_tensor_arg")
        return _literal(item)

    def _kb_call_spec(self, inputs):
        import torch

        params = [p for p in inspect.signature(self.model.forward).parameters.values()]
        taken: set[str] = set()
        positional, tensors = [], []
        seen: dict[int, str] = {}
        names = [p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        for index, item in enumerate(inputs):
            raw = names[index] if index < len(names) else f"arg{index}"
            positional.append(self._arg_expr(raw, item, taken, tensors, seen, {id(t) for t in inputs}))
        state = {}
        for pname, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            name = _identifier(pname.replace(".", "_"), taken)
            state[pname] = name
            tensors.append((name, tensor.detach(), "weight"))
        if not isinstance(self.model, torch.nn.Module):
            raise Drop("no_model_class")
        return positional, state, tensors

    def run(self) -> dict:
        import torch

        self._exec()
        meta_inputs = self._inputs("meta")
        numel = sum(t.numel() for t in meta_inputs if isinstance(t, torch.Tensor))
        small = numel <= SMALL_NUMEL
        # Real values give the input distributions; only small tasks also run the CPU probes.
        inputs = self._inputs("cpu") if numel <= STATS_NUMEL else meta_inputs
        activation_ids = {id(t) for t in inputs if isinstance(t, torch.Tensor)}
        if self.kind == "row":
            args, kwargs = self._record(inputs)
            positional, keywords, tensors = self._call_spec(args, kwargs, activation_ids)
            state = None
        else:
            positional, state, tensors = self._kb_call_spec(inputs)
            keywords = []
        if not tensors:
            raise Drop("no_tensor_inputs")
        for name, tensor, _ in tensors:
            if _dtype_name(tensor.dtype) not in AKA_DTYPES:
                raise Drop(f"unsupported_input_dtype:{_dtype_name(tensor.dtype)}")
        self.positional, self.keywords, self.state = positional, keywords, state
        self.tensors = tensors
        self.input_names = [n for n, _, _ in tensors]

        outputs = self._call({n: _to_meta(t) for n, t, _ in tensors}, meta=True)
        if outputs is None:
            if not small:
                if DYNAMIC_SHAPE_TOKENS.search(self.code):
                    raise Drop("data_dependent_shape")
                raise Drop("meta_inference_failed_large")
            outputs = self._call({n: t for n, t, _ in tensors}, meta=False, strict=True)
            self.flags.append("meta_unsupported")
        for out in outputs:
            if _dtype_name(out.dtype) not in AKA_DTYPES:
                raise Drop(f"unsupported_output_dtype:{_dtype_name(out.dtype)}")
            if out.numel() == 0:
                raise Drop("empty_output")
        self.outputs = [(tuple(o.shape), _dtype_name(o.dtype)) for o in outputs]

        if small:
            self._probe_cpu()
        else:
            if RANDOM_TOKENS.search(self._fn_source()):
                raise Drop("random_op_static")
            if re.search(r"inplace\s*=\s*True|\.\w+_\(", self._fn_source()):
                self.flags.append("possible_inplace_static")
        workloads, axes, batch = self._axes()
        return self._render(workloads, axes, batch)

    def _fn_source(self) -> str:
        """The compute code: every top-level def/class except the input factories."""
        import ast

        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return self.code
        # Module __init__ randomness only seeds weights, which become inputs; scan what runs per call.
        parts = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name not in ("get_inputs", "get_init_inputs"):
                parts.append(ast.get_source_segment(self.code, node) or "")
            elif isinstance(node, ast.ClassDef):
                parts += [ast.get_source_segment(self.code, item) or "" for item in node.body
                          if isinstance(item, ast.FunctionDef) and item.name != "__init__"]
        return "\n".join(parts)

    def _call(self, values: dict, meta: bool, strict: bool = False):
        import torch
        from torch.func import functional_call

        def resolve(expr):
            return values[expr] if expr in values else eval(expr, self.ns)  # noqa: S307 - our literal

        try:
            with torch.no_grad(), torch.device("meta" if meta else "cpu"):
                if self.kind == "row":
                    args = [resolve(e) for e in self.positional]
                    kwargs = {k: resolve(e) for k, e in self.keywords}
                    result = self.fn(*args, **kwargs)
                else:
                    args = [resolve(e) for e in self.positional]
                    result = functional_call(self.model, {p: values[n] for p, n in self.state.items()},
                                             tuple(args))
        except Drop:
            raise
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise Drop(f"reference_failed:{type(exc).__name__}") from exc
            return None
        return _flatten_outputs(result)

    def _probe_cpu(self):
        import torch

        base = {n: t for n, t, _ in self.tensors}
        clones = {n: t.clone() for n, t in base.items()}
        torch.manual_seed(11)
        first = self._call(clones, meta=False, strict=True)
        changed = [n for n in base if not _same(base[n], clones[n])]
        if changed:
            raise Drop("inplace_input_mutation")
        torch.manual_seed(12)
        second = self._call({n: t.clone() for n, t in base.items()}, meta=False, strict=True)
        if len(first) != len(second) or any(a.shape != b.shape for a, b in zip(first, second)):
            raise Drop("data_dependent_shape")
        if not all(_same(a, b) for a, b in zip(first, second)):
            raise Drop("nondeterministic")
        for out in first:
            if out.is_floating_point() and not bool(torch.isfinite(out).all()):
                self.flags.append("non_finite_reference_output")
                break

    def _axes(self):
        """Var axis `batch` on dim 0 of the activations, verified by a scaled meta run."""
        import torch

        acts = [(n, t) for n, t, r in self.tensors if r == "activation" and t.dim() >= 1]
        if not acts:
            return [{"B": None}], None, None
        b = acts[0][1].shape[0]
        scaled_names = {n for n, t in acts if t.shape[0] == b}
        total_bytes = sum(t.numel() * t.element_size() for _, t, _ in self.tensors)
        total_bytes += sum(math.prod(s) * torch.empty((), dtype=getattr(torch, d)).element_size()
                           for s, d in self.outputs)
        if total_bytes < TARGET_W1_BYTES:
            factor = min(MAX_SCALE, max(4, 2 ** math.ceil(math.log2(TARGET_W1_BYTES / max(total_bytes, 1)))))
            while factor > 4 and total_bytes * factor > MAX_W1_BYTES:
                factor //= 2
            b1 = b * factor
        elif total_bytes * 4 <= MAX_W1_BYTES:
            b1 = b * 4
        elif b >= 4:
            b1 = b // 4
        else:
            b1 = None
        if b1 is not None:
            values = {}
            for n, t, _ in self.tensors:
                m = _to_meta(t)
                if n in scaled_names:
                    m = torch.empty((b1, *m.shape[1:]), dtype=m.dtype, device="meta")
                values[n] = m
            outs = self._call(values, meta=True)
            predicted = [((b1, *s[1:]) if s and s[0] == b else s) for s, _ in self.outputs]
            if outs is None or [tuple(o.shape) for o in outs] != predicted:
                b1 = None
        if b1 is None:
            return [{"B": b}], None, b
        return [{"B": b}, {"B": b1}], scaled_names, b

    def _shape(self, shape, scaled: bool, b):
        if scaled and shape and shape[0] == b:
            return ["batch", *[int(x) for x in shape[1:]]]
        return [int(x) for x in shape]

    def _render(self, workloads, scaled_names, b) -> dict:
        spec = self.spec
        scaled_names = scaled_names or set()
        var_axis = bool(scaled_names)
        bs = [w["B"] for w in workloads]
        if var_axis:
            axes = {"batch": {"kind": "var", "min": 1, "max": max(bs),
                              "description": "Leading (batch) dimension of the activations."}}
        else:
            axes = {"batch": {"kind": "var", "min": 1, "max": 1,
                              "description": "Placeholder; this task has one fixed shape."}}
        inputs = {}
        for name, tensor, role in self.tensors:
            decl = {"kind": "tensor",
                    "shape": self._shape(tensor.shape, var_axis and name in scaled_names, b),
                    "dtype": _dtype_name(tensor.dtype),
                    "distribution": _distribution(tensor, self.code)}
            if role != "activation":
                decl["role"] = "weight"
            inputs[name] = decl
        out_names = ["out"] if len(self.outputs) == 1 else [f"out{i}" for i in range(len(self.outputs))]
        out_names = [n if n not in inputs else n + "_result" for n in out_names]
        outputs = {n: {"shape": self._shape(s, var_axis, b), "dtype": d}
                   for n, (s, d) in zip(out_names, self.outputs)}
        compare = {n: {"bound": "baseline_relative" if d in FLOAT_DTYPES else "exact"}
                   for n, (_, d) in zip(out_names, self.outputs)}
        cases = []
        for index, w in enumerate(workloads):
            cases.append({"id": f"w{index}", "axes": {"batch": w["B"] if var_axis else 1},
                          "checks": ["correctness", "performance"],
                          "source": "get_inputs()" if index == 0 else f"batch {bs[0]} -> {w['B']}"})
        name = operator_name(spec["kernel_name"])
        description = self._description()
        task = {
            "schema_version": 3,
            "description": description,
            "operator": {"name": name, "category": category(spec["kernel_name"], self.code)},
            "interface": {"axes": axes, "inputs": inputs, "outputs": outputs},
            "reference": {"file": "reference.py", "symbol": "reference"},
            "baseline": {"language": "python", "entry": "baseline/kernel.py",
                         "provenance": spec["provenance"]},
            "compare": compare,
            "workloads": "workloads.json",
            "platform_support": {"required_arch": ["gfx950"]},
        }
        return {
            "task_yaml": yaml.safe_dump(task, sort_keys=False, width=110),
            "workloads_json": json.dumps({"cases": cases}, indent=2) + "\n",
            "reference_py": self._reference_py(out_names),
            "baseline_py": self._baseline_py(out_names),
            "meta": {"flags": self.flags, "workloads": len(cases), "batch_scaled": var_axis,
                     "inputs": len(inputs), "outputs": len(outputs),
                     "input_bytes_w0": int(sum(t.numel() * t.element_size() for _, t, _ in self.tensors))},
        }

    def _description(self) -> str:
        doc = ""
        if self.kind == "row":
            doc = inspect.getdoc(self.fn) or ""
        else:
            doc = inspect.getdoc(self.model_cls) or ""
        doc = doc.split("\n\n")[0].strip().replace("\n", " ")
        src = {"cudaagent": "CUDA-Agent-Ops", "gpumode": "GPUMode/KernelBook", "kernelbench": "KernelBench"}[
            self.spec["source"]]
        what = f"`reference()` computes {self.fn_name if self.kind == 'row' else self.model_cls.__name__ + '.forward'}" \
               f" from the {src} task `{self.spec['kernel_name']}` (see reference.py)."
        literals = [e for e in self.positional if e not in self.input_names] + \
                   [f"{k}={e}" for k, e in self.keywords if e not in self.input_names]
        tail = f" Fixed non-tensor arguments: {', '.join(literals)}." if literals else ""
        return textwrap.shorten(f"{doc} {what}{tail}".strip(), width=1200, placeholder=" ...")

    def _call_text(self, wrap: str) -> str:
        def arg(expr):
            return f"{wrap}({expr})" if expr in self.input_names else expr

        if self.kind == "row":
            parts = [arg(e) for e in self.positional] + [f"{k}={arg(e)}" for k, e in self.keywords]
            return f"{self.fn_name}({', '.join(parts)})"
        state = ", ".join(f"{p!r}: {wrap}({n})" for p, n in self.state.items())
        args = ", ".join(arg(e) for e in self.positional)
        return f"_t2h_functional_call(_t2h_model(), {{{state}}}, ({args}{',' if len(self.positional) == 1 else ''}))"

    def _model_helper(self) -> str:
        if self.kind == "row":
            return ""
        init = _literal(self.init_args)
        kwargs = _literal(self.init_kwargs)
        return textwrap.dedent(f"""
            from torch.func import functional_call as _t2h_functional_call

            _T2H_MODEL = []


            def _t2h_model():
                # Parameters and buffers are task inputs; the instance only supplies structure.
                if not _T2H_MODEL:
                    import torch as _torch
                    with _torch.device("meta"):
                        _T2H_MODEL.append({self.model_cls.__name__}(*{init}, **{kwargs}).eval())
                return _T2H_MODEL[0]
            """)

    def _reference_py(self, out_names) -> str:
        sig = ", ".join(self.input_names)
        return (f'"""float64 reference for {self.spec["task_id"]}; the task code below is verbatim."""\n'
                f"{self.code.rstrip()}\n\n# --- t2h harness ---\n{self._model_helper()}\n\n"
                "def _t2h_hp(t):\n"
                "    return t.double() if t.is_floating_point() else t\n\n\n"
                f"def reference(*, {sig}):\n"
                f"    return {self._call_text('_t2h_hp')}\n")

    def _baseline_py(self, out_names) -> str:
        sig = ", ".join(self.input_names + out_names)
        lines = [f'"""Trusted PyTorch baseline for {self.spec["task_id"]} at native dtype."""',
                 self.code.rstrip(), "", "# --- t2h harness ---", self._model_helper(), "",
                 "def _t2h_id(t):", "    return t", "", "",
                 f"def run(*, {sig}):", f"    result = {self._call_text('_t2h_id')}"]
        if len(out_names) == 1:
            lines.append(f"    {out_names[0]}.copy_(result)")
        else:
            for index, name in enumerate(out_names):
                lines.append(f"    {name}.copy_(result[{index}])")
        return "\n".join(lines) + "\n"


def analyze(spec: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    torch.Tensor.cuda = lambda self, *a, **k: self  # dataset code sometimes calls .cuda()
    torch.nn.Module.cuda = lambda self, *a, **k: self
    try:
        return Analyzer(spec).run()
    except Drop as exc:
        return {"drop": str(exc)}
    except RecursionError:
        return {"drop": "recursion_error"}

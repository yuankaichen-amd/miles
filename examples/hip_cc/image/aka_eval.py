#!/usr/bin/env python3
"""aka-eval: the agent's check commands inside a hip-cc-rollout container (plan §7.1, §7.3).

    aka-eval compile       local gfx950 build (no GPU): AKA source audit, compiler errors, the
                           op-schema check and a per-kernel resource report
    aka-eval correctness   MI355X correctness on Crusoe, through the node's eval-proxy
    aka-eval performance   MI355X timing against the PyTorch baseline, through the eval-proxy
    aka-eval counters      MI355X hardware counters (observation only), through the eval-proxy
    aka-eval status        remaining check budgets

Only /workspace/solution/kernel.hip is built or sent. The eval-proxy owns every budget, including
the compile budget, so deleting files in the container doesn't reset anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(os.environ.get("CC_WORKSPACE", "/workspace"))
SOURCE = WORKSPACE / "solution" / "kernel.hip"
CONTRACT = WORKSPACE / "_aka_eval" / "contract.json"
EVAL_URL = os.environ.get("CC_EVAL_URL", "http://127.0.0.1:8081")
TOKEN_FILE = Path(os.environ.get("CC_EVAL_TOKEN_FILE", "/home/agent/.cc/eval_token"))
BUILD_ROOT = Path(os.environ.get("CC_BUILD_ROOT", "/tmp/aka-build"))
ARCH = "gfx950"
FLAGS = ("-O3",)
REMARK_FLAG = "-Rpass-analysis=kernel-resource-usage"
LDS_BYTES = 163840
LDS_GRANULE = 1280
COMPILE_TIMEOUT_S = 300
ERROR_LINES = 20
REMOTE_ACTIONS = ("correctness", "performance", "counters")
USAGE = "usage: aka-eval compile|correctness|performance|counters|status"


def proxy(path: str, payload: dict, timeout: float) -> dict:
    try:
        token = TOKEN_FILE.read_text().strip()
    except OSError:
        return {"ok": False, "text": "eval-proxy is not configured in this container; check not run"}
    request = urllib.request.Request(EVAL_URL.rstrip("/") + path, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json", "authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except ValueError:
            return {"ok": False, "text": f"eval-proxy error {exc.code}"}
    except (OSError, ValueError) as exc:
        return {"ok": False, "text": f"eval-proxy unreachable ({type(exc).__name__}); check not run, not counted"}


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _round_up(value: int, granule: int) -> int:
    return -(-value // granule) * granule


REMARK = re.compile(r"remark:[ \t]+([A-Za-z][A-Za-z ]*(?:\[[^\]\n]*\])?):[ \t]+([^\n\[]+?)[ \t]*"
                    r"\[-Rpass-analysis=kernel-resource-usage\]")
FIELDS = {"VGPRs": "vgpr", "AGPRs": "agpr", "SGPRs": "sgpr", "TotalSGPRs": "sgpr", "ScratchSize [bytes/lane]": "scratch",
          "VGPRs Spill": "vgpr_spill", "SGPRs Spill": "sgpr_spill", "LDS Size [bytes/block]": "lds",
          "Occupancy [waves/SIMD]": "occupancy", "Dynamic Stack": "dynamic_stack"}


def parse_resource_remarks(text: str) -> list[dict]:
    """Per-kernel fields from clang's ``-Rpass-analysis=kernel-resource-usage`` remarks."""
    kernels: list[dict] = []
    for key, value in REMARK.findall(text):
        key, value = key.strip(), value.strip()
        if key == "Function Name":
            kernels.append({"name": value})
        elif kernels and key in FIELDS:
            field = FIELDS[key]
            kernels[-1][field] = value if field == "dynamic_stack" else int(value) if value.isdigit() else value
    unique, seen = [], set()
    for kernel in kernels:
        key = json.dumps(kernel, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(kernel)
    return unique


def sanitize_name(name: str, demangled: dict[str, str]) -> str:
    text = demangled.get(name, name)
    text = re.sub(r"[^A-Za-z0-9_:<>,() *&]", "", text)
    return text if len(text) <= 80 else text[:77] + "..."


def demangle(names: list[str]) -> dict[str, str]:
    tool = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "lib" / "llvm" / "bin" / "llvm-cxxfilt"
    if not names or not tool.is_file():
        return {}
    try:
        out = subprocess.run([str(tool)], input="\n".join(names), capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(zip(names, out.splitlines()))


def resource_report(kernels: list[dict]) -> str:
    if not kernels:
        return "resources: no device kernels reported by the compiler"
    names = demangle([k["name"] for k in kernels])
    lines = ["resources (gfx950, per kernel):"]
    for kernel in kernels:
        lds = int(kernel.get("lds", 0) or 0)
        occ_lds = LDS_BYTES // _round_up(lds, LDS_GRANULE) if lds else None
        lines.append(
            f"- {sanitize_name(kernel['name'], names)}: VGPR {kernel.get('vgpr', '?')}, AGPR {kernel.get('agpr', '?')}, "
            f"SGPR {kernel.get('sgpr', '?')}, scratch {kernel.get('scratch', '?')} B/lane, spills "
            f"{kernel.get('vgpr_spill', 0)} VGPR / {kernel.get('sgpr_spill', 0)} SGPR, LDS {lds} B, "
            f"occupancy {kernel.get('occupancy', '?')} waves/SIMD"
            + (f", occ_lds {occ_lds} workgroups/CU" if occ_lds is not None else ""))
    return "\n".join(lines)


def first_errors(log: str) -> str:
    keep = [line for line in log.splitlines()
            if ("error" in line or "fatal" in line or line.startswith((" ", "\t")) and "|" in line)
            and "-Rpass-analysis" not in line and not line.startswith(("[", "FAILED:", "ninja:"))
            and "clang++" not in line and "hipcc" not in line]
    keep = keep or log.strip().splitlines()[-ERROR_LINES:]
    return "\n".join(line[:300] for line in keep[:ERROR_LINES])


BUILD_SCRIPT = r"""
import os, sys, torch
from torch.utils import cpp_extension
os.environ["PYTORCH_ROCM_ARCH"] = sys.argv[3]
cpp_extension.load(name=sys.argv[1], sources=[sys.argv[2]], extra_include_paths=[os.path.dirname(sys.argv[2])],
                   extra_cuda_cflags=sys.argv[4:], build_directory=os.path.dirname(os.path.dirname(sys.argv[2])),
                   is_python_module=False, verbose=True)
op = getattr(getattr(torch.ops, sys.argv[1]), "run").default
print("ARENA_SCHEMA " + str(op._schema))
"""


def check_schema(schema_text: str) -> str | None:
    """AKA loader._check_op_schema against the workspace contract summary."""
    import torch  # noqa: PLC0415

    contract = json.loads(CONTRACT.read_text())
    call_keys = list(contract["inputs"]) + list(contract["outputs"])
    checked = set(contract["outputs"]) | set(contract.get("mutable", ()))
    schema = torch._C.parse_schema(schema_text)
    names = [argument.name for argument in schema.arguments]
    if sorted(names) != sorted(call_keys):
        return f"run takes {names}, not the call keys {call_keys}"
    writes = {a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write}
    if writes != checked or schema.returns:
        return f"{schema} must write exactly {sorted(checked)}, each a Tensor(a!), and return ()"
    return None


def compile_candidate(code: str) -> tuple[bool, str]:
    sys.path.insert(0, "/opt/cc")
    from aka_audit import PolicyError, audit_hip_sources  # noqa: PLC0415

    lines = []
    try:
        audit_hip_sources({"kernel.hip": code}, "kernel.hip")
        lines.append("audit: OK")
        audit_ok = True
    except PolicyError as exc:
        lines.append(f"audit: FAIL: {exc}")
        audit_ok = False
    digest = code_sha(code)[:16]
    package = f"aka_hip_candidate_{digest}"
    build = BUILD_ROOT / package
    (build / "src").mkdir(parents=True, exist_ok=True)
    source = build / "src" / "kernel.hip"
    source.write_text(code)
    env = {**os.environ, "MAX_JOBS": "2", "OMP_NUM_THREADS": "1"}
    try:
        proc = subprocess.run([sys.executable, "-c", BUILD_SCRIPT, package, str(source), ARCH, *FLAGS, REMARK_FLAG],
                              capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S, env=env, cwd=str(build))
        log = proc.stdout + proc.stderr
        built = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        log = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        built = False
        lines.append(f"compile: TIMEOUT after {COMPILE_TIMEOUT_S}s")
    schema = next((line[len("ARENA_SCHEMA "):] for line in log.splitlines() if line.startswith("ARENA_SCHEMA ")), None)
    if built and schema:
        problem = check_schema(schema)
        lines.insert(0, "compile: OK (gfx950)" if problem is None else f"compile: FAIL: {problem}")
        built = problem is None
    elif built:
        lines.insert(0, "compile: FAIL: solution/kernel.hip registers no op run in "
                        "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m)")
        built = False
    elif not any(line.startswith("compile: TIMEOUT") for line in lines):
        lines.insert(0, "compile: FAIL\n" + first_errors(log))
    kernels = parse_resource_remarks(log)
    saved = build / "resources.json"
    if kernels:
        saved.write_text(json.dumps(kernels))
    elif built and saved.is_file():
        # ninja skips an unchanged build, so the remarks come from the first build of this code
        kernels = json.loads(saved.read_text())
    lines.append(resource_report(kernels))
    return built and audit_ok, "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("compile", "status", *REMOTE_ACTIONS):
        print(USAGE, file=sys.stderr)
        return 2
    action = argv[0]
    if action == "status":
        print(proxy("/status", {}, 30).get("text", ""))
        return 0
    code = SOURCE.read_text(encoding="utf-8", errors="replace")
    if action == "compile":
        grant = proxy("/compile", {"sha256": code_sha(code)}, 30)
        if not grant.get("ok"):
            print(grant.get("text", "compile not allowed"))
            return 3
        ok, report = compile_candidate(code)
        proxy("/compile/result", {"sha256": code_sha(code), "ok": ok}, 30)
        print(report)
        print(grant.get("text", ""))
        return 0 if ok else 1
    reply = proxy("/check", {"action": action, "code": code}, 1800)
    print(reply.get("text", json.dumps(reply)[:2500]))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

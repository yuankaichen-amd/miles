"""gfx950 validation of hip-cc-rollout against the Crusoe HIP grader (plan §10 P4 gate).

For each test kernel (relu6 good / wrong / syntax / audit) this compiles the code inside the
rollout image exactly as ``aka-eval compile`` does, then sends the same code through the
eval-proxy code path (``EvalProxy`` with the real ``GraderClient``) and ``/grade`` on Crusoe, and
checks that the image and the grader agree: the image compiles and audits what the grader
compiles and audits, the good kernel is correct and scores on MI355X, counters come back as
"unavailable; not counted" while the pool returns 501.

Runs on a compute node; the grader token is read in-process from ``HIP_SANDBOX_TOKEN_FILE``.

    python3 validate_image_grader.py --image hip-cc-rollout:<tag> --workspace <relu6 ws> --out <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cc_runner import EpisodeSpec, EvalProxy, JsonlLog  # noqa: E402
from cc_workspace import container_args, workspace_tar  # noqa: E402
from hip_grader_client import GraderClient  # noqa: E402

TASK = "t2h/cudaagent/fused_torch_1_relu6_72ecee24"
VERSION = "6e0402dbba69efa9"
KERNELS = ("relu6_good", "relu6_wrong", "relu6_syntax", "relu6_audit")
COMPILE_IN_IMAGE = r"""
import importlib.machinery, json, sys, time
ae = importlib.machinery.SourceFileLoader("ae", "/opt/cc/bin/aka-eval").load_module()
for name in sys.argv[1:]:
    code = open(f"/tmp/k/{name}.hip").read()
    t = time.time()
    ok, report = ae.compile_candidate(code)
    print("RESULT " + json.dumps({"kernel": name, "ok": ok, "seconds": round(time.time() - t, 1), "report": report}))
"""


def compile_in_image(image: str, workspace: Path) -> dict[str, dict]:
    name = f"cc-validate-{int(time.time())}"
    args = container_args(name, image, 10001, 10001)
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    try:
        subprocess.run(["docker", "exec", "-i", name, "tar", "-x", "-C", "/workspace"], input=workspace_tar(workspace),
                       check=True)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for kernel in KERNELS:
                data = (HERE / "kernels" / f"{kernel}.hip").read_bytes()
                info = tarfile.TarInfo(f"k/{kernel}.hip")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        subprocess.run(["docker", "exec", "-i", name, "tar", "-x", "-C", "/tmp"], input=buffer.getvalue(), check=True)
        proc = subprocess.run(["docker", "exec", name, "python3", "-c", COMPILE_IN_IMAGE, *KERNELS],
                              capture_output=True, text=True, timeout=1800)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            row = json.loads(line[7:])
            results[row["kernel"]] = row
    if len(results) != len(KERNELS):
        print(proc.stdout[-3000:], proc.stderr[-3000:])
    return results


async def grader_side(out: Path) -> dict[str, dict]:
    client = GraderClient()
    pool = concurrent.futures.ThreadPoolExecutor(4)
    episode = f"p4-validate-{int(time.time())}"
    spec = EpisodeSpec(episode=episode, task_id=TASK, task_version=VERSION, workspace_dir="/nonexistent",
                       gateway_url="http://127.0.0.1:1", gateway_token="unused", model="none",
                       budgets={"correctness": 10, "performance": 4, "counters": 1})
    log = JsonlLog(out / "eval-proxy.jsonl")
    proxy = EvalProxy(spec, client, log, pool)
    results = {}
    loop = asyncio.get_running_loop()
    for kernel in KERNELS:
        code = (HERE / "kernels" / f"{kernel}.hip").read_text()
        row = {}
        started = time.time()
        row["correctness_ok"], row["correctness_text"] = await proxy.check("correctness", code)
        row["correctness_s"] = round(time.time() - started, 1)
        if kernel == "relu6_good":
            row["correctness_again"] = (await proxy.check("correctness", code))[1][-60:]
            started = time.time()
            row["performance_ok"], row["performance_text"] = await proxy.check("performance", code)
            row["performance_s"] = round(time.time() - started, 1)
            row["counters_ok"], row["counters_text"] = await proxy.check("counters", code)
        started = time.time()
        grade = await loop.run_in_executor(pool, lambda: client.call_with_retry(
            "POST", "/grade", {"episode": f"{episode}-{kernel}", "task_id": TASK, "task_version": VERSION,
                               "code": code}))
        row["grade_kind"], row["grade_status"], row["grade"] = grade.kind, grade.status, grade.body
        row["grade_s"] = round(time.time() - started, 1)
        results[kernel] = row
        print(json.dumps({"kernel": kernel, **row})[:3000], flush=True)
    row = {"proxy_metrics": proxy.metrics(), "grader_stats": client.stats.snapshot()}
    log.close()
    results["_proxy"] = row
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image = compile_in_image(args.image, Path(args.workspace))
    for kernel, row in image.items():
        print(f"image compile {kernel}: ok={row['ok']} {row['seconds']}s\n{row['report']}\n", flush=True)
    grader = asyncio.run(grader_side(out))

    def g(kernel: str, key: str):
        return (grader.get(kernel, {}).get("grade") or {}).get(key)

    verdicts = {
        "good_compiles_in_image": image.get("relu6_good", {}).get("ok") is True,
        "good_correct_on_mi355x": grader["relu6_good"]["correctness_ok"] and g("relu6_good", "correct") is True,
        "good_graded": grader["relu6_good"]["grade_kind"] == "ok" and g("relu6_good", "compiled") is True,
        "good_dedupe": "not charged" in grader["relu6_good"].get("correctness_again", ""),
        "wrong_compiles_both_fails_grader": image.get("relu6_wrong", {}).get("ok") is True
        and g("relu6_wrong", "correct") is False,
        "syntax_fails_both": image.get("relu6_syntax", {}).get("ok") is False and g("relu6_syntax", "compiled") is False,
        "audit_fails_both": image.get("relu6_audit", {}).get("ok") is False
        and "audit: FAIL" in image.get("relu6_audit", {}).get("report", "")
        and (g("relu6_audit", "audit") is False or g("relu6_audit", "correct") is False),
        "counters_charged_only_on_a_result": (
            grader["_proxy"]["proxy_metrics"]["used"].get("counters") == 0
            and "not counted" in grader["relu6_good"].get("counters_text", ""))
        or ("counters check 1/1" in grader["relu6_good"].get("counters_text", "")
            and '"status": "error"' not in grader["relu6_good"].get("counters_text", "")),
        "resource_report_present": "VGPR" in image.get("relu6_good", {}).get("report", ""),
    }
    report = {"image": args.image, "verdicts": verdicts, "image_compile": image, "grader": grader}
    (out / "validate-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2))
    ok = all(verdicts.values())
    print("IMAGE VALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

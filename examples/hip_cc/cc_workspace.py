"""Agent workspaces for the HIP Claude Code rollouts (plan §1 "Workspace and prompt", §4.2, §6.4).

The task store tarball holds more than an agent may see (``t2h.json`` provenance, the
PyTorch baseline entry and, in task_version 6e0402dbba69efa9, a top-level ``hip_ref/``
directory with the dataset's reference HIP). This module is the only way rollout code turns
a task into a workspace, and it reads the tarball through an allowlist:

* only ``tasks/<split>/<source>/<name>/{task.yaml, workloads.json, reference.py}`` of the one
  requested task is extracted; ``hip_ref/``, ``t2h.json``, ``baseline/`` and the
  feasibility tarball are never opened;
* the rendered workspace is AKA's ``render_workspace`` output reduced to ``config.yaml``,
  ``README.md``, ``reference/reference.py``, ``solution/kernel.hip`` and an ``_aka_eval/``
  holding the shim and a contract summary without seeds;
* containers get the workspace through a tar stream into a tmpfs ``/workspace``; the only host
  mount is the episode's read-only socket directory at ``/fwd``, and ``check_docker_args``
  refuses any other bind, ``/shared_inference``, ``/home`` or a path naming
  ``hip_ref``/``feasibility``.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AKA_ROOT = Path("/home/yuankach/hip-cc/AgentKernelArena")
PACKAGE_FILES = ("task.yaml", "workloads.json", "reference.py")
FORBIDDEN_PATH = re.compile(r"hip_ref|feasibility", re.IGNORECASE)
WORKSPACE_FILES = ("config.yaml", "README.md", "reference/reference.py", "solution/kernel.hip",
                   "_aka_eval/__init__.py", "_aka_eval/evaluate.py", "_aka_eval/contract.json",
                   "_aka_eval/CANDIDATE.md")


class IsolationError(RuntimeError):
    pass


KB_POINTER = "Reference notes on HIP and MI355X are in `/kb` (read-only).\n\n"

def check_path_allowed(path: str | Path) -> None:
    """Every file rollout-side code opens from the task store goes through this."""
    if FORBIDDEN_PATH.search(str(path)):
        raise IsolationError(f"refusing to read {path}: hip_ref material is not for rollouts")


def check_docker_args(args: list[str], sock_root: str | None = None) -> None:
    """Containers may not bind host paths; tmpfs and named volumes only.

    The one exception is the episode's socket directory: a direct child of ``sock_root`` bound
    read-only at ``/fwd`` (it holds only the gateway and eval-proxy unix sockets).
    """
    flat = " ".join(args)
    for index, arg in enumerate(args):
        value = None
        if arg in ("-v", "--volume", "--mount"):
            value = args[index + 1] if index + 1 < len(args) else ""
        elif arg.startswith(("--volume=", "--mount=", "-v=")):
            value = arg.split("=", 1)[1]
        if value is None:
            continue
        source = value.split(":", 1)[0] if arg in ("-v", "--volume") or arg.startswith(("--volume=", "-v=")) else \
            next((p.split("=", 1)[1] for p in value.split(",") if p.startswith(("source=", "src="))), "")
        if sock_root and _is_socket_bind(value, sock_root):
            continue
        if source.startswith("/") or FORBIDDEN_PATH.search(source):
            raise IsolationError(f"host mount {source!r} is not allowed in rollout containers")
    if "--privileged" in args or "--volumes-from" in flat:
        raise IsolationError("privileged containers and --volumes-from are not allowed")
    if FORBIDDEN_PATH.search(flat) or any(root in flat for root in ("/shared_inference", "/shared_nfs")):
        raise IsolationError("docker arguments name a forbidden host path")


def _is_socket_bind(value: str, sock_root: str) -> bool:
    parts = value.split(":")
    if len(parts) != 3 or parts[1] != "/fwd" or parts[2] != "ro":
        return False
    source, root = Path(parts[0]), Path(sock_root)
    return source.is_absolute() and source.parent == root and ".." not in source.parts and \
        not FORBIDDEN_PATH.search(parts[0]) and not any(r in parts[0] for r in ("/shared_inference", "/shared_nfs",
                                                                                 "/home"))


def _task_member_prefix(names: list[str], task_id: str) -> str:
    _, source, name = task_id.split("/")
    suffix = f"/{source}/{name}/task.yaml"
    hits = [n for n in names if n.endswith(suffix) and "/tasks/" in n]
    if len(hits) != 1:
        raise IsolationError(f"{task_id}: expected one task.yaml in the package, found {len(hits)}")
    return hits[0][: -len("task.yaml")]


def extract_task(package: Path, task_id: str, dest: Path) -> Path:
    """Copy the allowlisted package files of one task into ``dest``."""
    check_path_allowed(package)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package) as tar:
        prefix = _task_member_prefix(tar.getnames(), task_id)
        for name in PACKAGE_FILES:
            member = tar.getmember(prefix + name)
            check_path_allowed(member.name)
            if not member.isfile():
                raise IsolationError(f"{member.name} is not a regular file")
            (dest / name).write_bytes(tar.extractfile(member).read())
    return dest


def _aka():
    sys.path.insert(0, str(HERE))
    from build_t2h_tasks import aka_validator  # noqa: PLC0415

    return aka_validator()


SHIM = '''"""python3 -m _aka_eval.evaluate candidate compile|correctness|performance|counters

The same as `aka-eval <action>`: compile builds solution/kernel.hip here for gfx950; the other
actions go to MI355X through the node's eval-proxy.
"""
import os
import sys

if len(sys.argv) != 3 or sys.argv[1] != "candidate":
    sys.exit("usage: python3 -m _aka_eval.evaluate candidate compile|correctness|performance|counters")
os.execv("/opt/cc/bin/aka-eval", ["aka-eval", sys.argv[2]])
'''


def render_workspace(package: Path, task_id: str, workspace: Path, *, task_version: str, split: str) -> list[str]:
    """Render the agent-visible workspace of one task; returns its files."""
    tc = _aka()
    if workspace.exists():
        raise IsolationError(f"{workspace} already exists")
    with tempfile.TemporaryDirectory(prefix="ccws-") as tmp:
        task_root = extract_task(package, task_id, Path(tmp) / "task")
        staged = Path(tmp) / "staged"
        shutil.copytree(task_root, staged)
        tc.render_workspace(task_root, staged, task_id=task_id, target_language="hip")
        contract = json.loads((staged / "_aka_eval/contract.json").read_text())
        for case in contract["cases"]:
            case.pop("seeds", None)
        contract.pop("baseline", None)
        config = (staged / "config.yaml").read_text()
        candidate_doc = (staged / "_aka_eval/CANDIDATE.md").read_text()
        workspace.mkdir(parents=True)
        files = {
            "config.yaml": config + f"t2h:\n  task_id: {task_id}\n  task_version: {task_version}\n  split: {split}\n"
                           "  editable: [solution/kernel.hip]\n  target_arch: gfx950\n",
            "README.md": _readme(task_id, (task_root / "task.yaml").read_text(), candidate_doc),
            "reference/reference.py": (task_root / "reference.py").read_text(),
            "solution/kernel.hip": (staged / "solution/kernel.hip").read_text(),
            "_aka_eval/__init__.py": "",
            "_aka_eval/evaluate.py": SHIM,
            "_aka_eval/contract.json": json.dumps(contract, indent=2) + "\n",
            "_aka_eval/CANDIDATE.md": candidate_doc,
        }
        for relative, text in files.items():
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return sorted(files)


def _readme(task_id: str, task_yaml: str, candidate_doc: str) -> str:
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(task_yaml)
    return (f"# {task_id}\n\n"
            "Implement `reference` in `reference/reference.py` as a HIP kernel in `solution/kernel.hip`, registered as "
            "`run`, matching it within the bound and faster than the PyTorch baseline on MI355X (gfx950). "
            "Only `solution/kernel.hip` counts.\n\n"
            f"## Task\n\n{raw.get('description', '').strip()}\n\n"
            "## Rules\n\nNo PyTorch, rocBLAS or hipBLAS calls in the kernel, no memoizing outputs, no special-casing "
            "inputs.\n\n"
            "## Checks\n\n```\naka-eval compile       # local gfx950 build, audit and register/LDS report\n"
            "aka-eval correctness   # MI355X, against the PyTorch reference\n"
            "aka-eval performance   # MI355X, timing against the PyTorch baseline\n"
            "aka-eval counters      # MI355X hardware counters, after correctness passes\n"
            "aka-eval status        # remaining check budgets\n```\n\n"
            f"{KB_POINTER}"
            f"## Interface\n\n{candidate_doc}")


def workspace_tar(workspace: Path, *, with_kb: bool = True) -> bytes:
    """The workspace as a tar stream for ``docker exec -i <c> tar -x -C /workspace``.

    ``with_kb=False`` (the /kb-off ablation arm) drops the README's /kb pointer.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(workspace.rglob("*")):
            check_path_allowed(path.relative_to(workspace))
            info = tar.gettarinfo(str(path), arcname=path.relative_to(workspace).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if not with_kb and info.name == "README.md":
                data = path.read_bytes().replace(KB_POINTER.encode(), b"")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    return buffer.getvalue()


def container_args(name: str, image: str, uid: int, gid: int, network: str = "none", *,
                   fwd_dir: str | None = None, sock_root: str | None = None,
                   labels: dict[str, str] | None = None) -> list[str]:
    """``docker run`` arguments for a rollout container (plan §4.2), checked by ``check_docker_args``."""
    args = ["docker", "run", "-d", "--name", name, "--user", f"{uid}:{gid}", "--network", network,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--cpus", "2", "--memory", "6g", "--read-only",
            "--tmpfs", f"/workspace:rw,exec,size=512m,uid={uid},gid={gid}",
            "--tmpfs", f"/tmp:rw,exec,size=2g,uid={uid},gid={gid}",
            "--tmpfs", f"/home/agent:rw,size=256m,uid={uid},gid={gid}",
            "-e", "HOME=/home/agent", "-w", "/workspace"]
    for key, value in (labels or {}).items():
        args += ["--label", f"{key}={value}"]
    if fwd_dir:
        args += ["-v", f"{fwd_dir}:/fwd:ro"]
    args += [image, "sleep", "infinity"]
    check_docker_args(args, sock_root)
    return args

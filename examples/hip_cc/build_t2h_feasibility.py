"""Feasibility tarball for the hip_ref flag (format: t2h-task-package-format.md, "Feasibility tarball").

For every task in a built t2h package that has a dataset ``hip_ref_code``, ships the row's
``pytorch_functional_code`` and ``hip_ref_code`` verbatim plus its tolerances, so Crusoe can
check whether the reference HIP builds and passes on gfx950. Source rows are re-read by
``row_file#row_index`` and must match the ``row_sha256`` recorded in the package.

    python3 build_t2h_feasibility.py --package $DATA/t2h/t2h-<version>.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

DATA = Path("/shared_inference/yuankach/glm53/datasets/hip")
# build_hip_data.py writes these constants into every rft label; manifest rows carry none.
DATASET_ATOL, DATASET_RTOL = 1e-4, 1e-3


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_tasks(tarball: Path) -> tuple[str, list[dict]]:
    tasks = []
    with tarfile.open(tarball) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith("/t2h.json"):
                tasks.append(json.load(tar.extractfile(member)))
    version = tarball.name.removeprefix("t2h-").removesuffix(".tar.gz")
    return version, sorted(tasks, key=lambda t: t["task_id"])


def source_rows(tasks: list[dict]) -> dict[tuple[str, int], tuple[str, dict]]:
    wanted: dict[str, set[int]] = {}
    for task in tasks:
        wanted.setdefault(task["row_file"], set()).add(task["row_index"])
    rows = {}
    for row_file, indices in wanted.items():
        with (DATA / row_file).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in indices:
                    line = line.rstrip("\n")
                    rows[(row_file, index)] = (line, json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    package = Path(args.package)
    version, tasks = package_tasks(package)
    tasks = [t for t in tasks if t.get("has_hip_ref")]
    rows = source_rows(tasks)
    staging = Path(tempfile.mkdtemp(prefix="t2h-feas-")) / version
    counts, tags = Counter(), set()
    for task in tasks:
        line, row = rows[(task["row_file"], task["row_index"])]
        if sha256_bytes(line.encode()) != task["row_sha256"]:
            raise SystemExit(f"row_sha256 mismatch for {task['task_id']}")
        label = row.get("label", row)
        hip_ref = label.get("hip_ref_code") or label.get("hip_reference_code") or ""
        functional = label.get("pytorch_functional_code") or ""
        if sha256_bytes(hip_ref.encode()) != task["hip_ref_code_sha256"] or not functional:
            raise SystemExit(f"hip_ref/functional mismatch for {task['task_id']}")
        tag = "t" + sha256_bytes(task["task_id"].encode())[:12]
        assert tag not in tags
        tags.add(tag)
        meta = {"atol": label.get("atol", DATASET_ATOL), "rtol": label.get("rtol", DATASET_RTOL), "tag": tag}
        root = staging / task["task_id"].replace("/", "__")
        root.mkdir(parents=True)
        (root / "functional.py").write_text(functional, encoding="utf-8")
        (root / "hip_ref.hip").write_text(hip_ref, encoding="utf-8")
        (root / "meta.json").write_text(json.dumps(meta) + "\n")
        counts[task["split"]] += 1
    lines = [f"{sha256_bytes(p.read_bytes())}  {p.relative_to(staging).as_posix()}"
             for p in sorted(p for p in staging.rglob("*") if p.is_file())]
    (staging / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    tarball = package.parent / f"t2h-{version}-feasibility.tar.gz"
    with tarfile.open(tarball, "w:gz", compresslevel=6) as tar:
        tar.add(staging, arcname=version)
    shutil.rmtree(staging.parent)
    digest = sha256_bytes(tarball.read_bytes())
    Path(f"{tarball}.sha256").write_text(f"{digest}  {tarball.name}\n")
    print(json.dumps({"tarball": str(tarball), "sha256": digest, "tasks": len(tasks), "files": len(lines),
                      "per_split": dict(counts)}))


if __name__ == "__main__":
    main()

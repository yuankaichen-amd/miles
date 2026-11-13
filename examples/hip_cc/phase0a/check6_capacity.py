"""Phase 0a check 6: do 384 rollout containers fit at 2 vCPU and 6 GB each?

The planned layout is 32 containers per rollout node and 8 per trainer node. Reads
``scontrol show node`` for every node in CHECK6_PARTITION (default amd-arad-burst) and this node's
live state (cores, memory, the per-uid process limit, what this job's stack already uses), and
reports how many containers fit per node next to SGLang/Megatron, and the concurrency that fits.

    python3 check6_capacity.py   (on the job's node, through phase0a/driver.sh or srun --overlap)
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
from collections import Counter
from pathlib import Path

VCPU, MEM_GB = 2, 6
PLAN = {"rollout": 32, "trainer": 8}
# Host reservation per node besides the containers: SGLang TP8 or Megatron, Ray, gateway, runner.
RESERVE = {"rollout": {"cpus": 32, "mem_gb": 256}, "trainer": {"cpus": 48, "mem_gb": 768}}


def scontrol_nodes(partition: str) -> list[dict]:
    text = subprocess.run(["scontrol", "-o", "show", "node"], capture_output=True, text=True, check=True).stdout
    nodes = []
    for line in text.splitlines():
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        if partition in fields.get("Partitions", "").split(","):
            nodes.append({"name": fields.get("NodeName"), "cpus": int(fields.get("CPUTot", 0)),
                          "mem_gb": int(fields.get("RealMemory", 0)) // 1024, "gres": fields.get("Gres"),
                          "state": fields.get("State")})
    return nodes


def fit(cpus: int, mem_gb: int, role: str) -> int:
    reserve = RESERVE[role]
    return max(0, min((cpus - reserve["cpus"]) // VCPU, (mem_gb - reserve["mem_gb"]) // MEM_GB))


def main() -> int:
    partition = os.environ.get("CHECK6_PARTITION", "amd-arad-burst")
    nodes = scontrol_nodes(partition)
    shapes = Counter((n["cpus"], n["mem_gb"]) for n in nodes)
    per_shape = [{"cpus": c, "mem_gb": m, "nodes": k, "fit_rollout": fit(c, m, "rollout"),
                  "fit_trainer": fit(c, m, "trainer")} for (c, m), k in shapes.most_common()]
    common = per_shape[0] if per_shape else {"fit_rollout": 0, "fit_trainer": 0}
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    live = {"node": os.uname().nodename, "nproc": os.cpu_count(),
            "mem_total_gb": int(Path("/proc/meminfo").read_text().split()[1]) // (1024 * 1024),
            "uid_nproc_limit": [soft, hard],
            "containers_running": len(subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.split())}
    rollout_ok = common["fit_rollout"] >= PLAN["rollout"]
    trainer_ok = common["fit_trainer"] >= PLAN["trainer"]
    concurrency = {"rollout_per_node": min(PLAN["rollout"], common["fit_rollout"]),
                   "trainer_per_node": min(PLAN["trainer"], common["fit_trainer"])}
    report = {"partition": partition, "per_shape": per_shape, "live": live, "plan": PLAN,
              "vcpu": VCPU, "mem_gb": MEM_GB, "reserve": RESERVE, "fits_plan": rollout_ok and trainer_ok,
              "concurrency_that_fits": concurrency,
              "note": "containers run as uid 10001 with a pids limit each, so the per-uid limit above "
                      "bounds only the host-side runner, gateway and docker clients"}
    out = Path(os.environ["CC_OUT"])
    (out / "check6.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("CHECK 6: recorded;", "384 containers fit" if report["fits_plan"] else f"fits {concurrency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

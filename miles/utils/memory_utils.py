import gc
import json
import logging
import os
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "peak_allocated_GB": _byte_to_gb(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_GB": _byte_to_gb(torch.cuda.max_memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False, reset_peak: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    # Ray's log dedup folds these lines across ranks; MILES_MEMORY_LOG_DIR keeps one file per rank.
    if log_dir := os.environ.get("MILES_MEMORY_LOG_DIR"):
        try:
            with open(os.path.join(log_dir, f"rank{dist.get_rank():05d}.jsonl"), "a") as f:
                f.write(json.dumps({"time": round(time.time(), 1), "msg": msg, **memory_info}) + "\n")
        except OSError:
            pass
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return memory_info

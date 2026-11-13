"""Phase 0a check 1: config.json allows CC_CONTEXT, and a 120K-token prompt returns from SGLang."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    model, context, url = os.environ["MODEL"], int(os.environ["CC_CONTEXT"]), os.environ["SGLANG_URL"]
    config = json.loads(Path(model, "config.json").read_text())
    config = config.get("text_config", config)
    limit = config.get("max_position_embeddings", 0)
    rng = random.Random(0)
    prompt_tokens = int(os.environ.get("CHECK1_TOKENS", "120000"))
    input_ids = [rng.randrange(1000, min(config.get("vocab_size", 150000), 150000) - 1000) for _ in range(prompt_tokens)]
    started = time.time()
    response = httpx.post(f"{url}/generate", json={"input_ids": input_ids, "sampling_params": {
        "max_new_tokens": 16, "temperature": 0.0}}, timeout=1800)
    seconds = round(time.time() - started, 1)
    body = response.json() if response.status_code == 200 else {}
    meta = body.get("meta_info", {})
    result = {"max_position_embeddings": limit, "rope_scaling": config.get("rope_scaling"), "cc_context": context,
              "prompt_tokens": prompt_tokens, "status": response.status_code, "seconds": seconds,
              "engine_prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": meta.get("completion_tokens"),
              "finish_reason": meta.get("finish_reason"), "error": None if body else response.text[:500]}
    result["pass"] = (limit >= context and response.status_code == 200
                      and meta.get("prompt_tokens") == prompt_tokens and (meta.get("completion_tokens") or 0) > 0)
    out = Path(os.environ["CC_OUT"])
    (out / "check1.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print("CHECK 1:", "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

import build_t2h_tasks as b  # noqa: E402
import t2h_contract as c  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for index, line, row in b.read_jsonl(b.DATA / "rft/train.jsonl"):
    if row["label"].get("task_type") != "pytorch_to_hip" or row["label"]["source"] != "gpumode":
        continue
    spec = b.row_spec(row, split="train", row_file="rft/train.jsonl", row_index=index, line=line)
    a = c.Analyzer(spec)
    try:
        a._exec()
        inputs = a._inputs("cpu")
        args, kwargs = a._record(inputs)
        pos, kw, tensors = a._call_spec(args, kwargs, {id(t) for t in inputs})
        a.positional, a.keywords, a.state, a.tensors = pos, kw, None, tensors
        a.input_names = [t[0] for t in tensors]
        values = {nm: c._to_meta(t) for nm, t, _ in tensors}
        with torch.no_grad(), torch.device("meta"):
            out = a.ns["module_fn"](*[values[e] if e in values else eval(e, a.ns) for e in pos],
                                    **{k: values[e] if e in values else eval(e, a.ns) for k, e in kw})
        print("OK", spec["task_id"], [o.shape for o in c._flatten_outputs(out)])
    except Exception:
        print("FAIL", spec["task_id"])
        traceback.print_exc(limit=4)
    n -= 1
    if n == 0:
        break

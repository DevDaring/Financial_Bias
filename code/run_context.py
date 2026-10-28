"""
Run frozen verifiers on a passage pack (context experiment, or any jsonl of {item_id, evidence, text, truth}).
Passage-level contract for every arm (claim_level=True): no probability/coverage obligation.

    python run_context.py --pack Submission/mechanism/context_pack_v1.jsonl --judge qwen3_next \
        --methods generic_rubric forced_tool_judge exec_P exec_D exec_R --live --workers 8
Output: Submission/mechanism/context_runs/<method>__<judge>.jsonl (append-only, resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators as K  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from generate import Candidate  # noqa: E402
from run_comparators import with_deadline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--judge", default="qwen3_next")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset", default=None, help="file with item_ids to run (one per line)")
    ap.add_argument("--call-timeout", type=float, default=None)
    ap.add_argument("--item-timeout", type=float, default=600)
    ap.add_argument("--dispute-family", default=None)
    ap.add_argument("--reassess", default=None, help="reassessment family for exec_P (ReCheck 1.x)")
    ap.add_argument("--full-text", action="store_true", help="whole-explanation contract (fresh natural narratives)")
    a = ap.parse_args()
    if not a.live:
        print("dry by default: add --live")
        return 2
    if a.call_timeout:
        import gateway as G
        G.GATEWAY.timeout = a.call_timeout
    pack = [json.loads(l) for l in open(a.pack)]
    if a.subset:
        keep = {l.strip() for l in open(a.subset) if l.strip()}
        pack = [r for r in pack if r["item_id"] in keep]
    if a.limit:
        pack = pack[: a.limit]
    out = Path(a.out) if a.out else Path(a.pack).parent / (Path(a.pack).stem.replace("_pack", "") + "_runs")
    out.mkdir(parents=True, exist_ok=True)
    claim_level = not a.full_text
    for m in a.methods:
        path = out / f"{m}__{a.judge}.jsonl"
        done = {json.loads(l)["item_id"] for l in open(path)} if path.exists() else set()
        todo = [r for r in pack if r["item_id"] not in done]
        lock = threading.Lock()
        fn = K.METHODS[m]

        def work(r, m=m, fn=fn, path=path):
            eo = eo_from(r["evidence"])
            cand = Candidate(borrower_id=r.get("borrower_id", r["item_id"]), arm=r.get("arm", "passage"),
                             text=r["text"], family=r.get("family") or r.get("writer") or "llama4_maverick")
            t0 = time.time()

            def call():
                kw = dict(experiment=f"{Path(a.pack).stem}", claim_level=claim_level)
                if m.startswith("exec_"):
                    kw["dispute_family"] = a.dispute_family
                    kw["reassess_family"] = a.reassess
                v = fn(cand, eo, family=a.judge, **kw)
                return {"accepted": v.accepted, "score": v.score, "detail": v.detail, "error": v.error,
                        "calls": v.calls, "n_calls": len(v.calls or [])}
            res = with_deadline(call, a.item_timeout)
            row = {"item_id": r["item_id"], "method": m, "judge": a.judge, "truth": r.get("truth"),
                   **{k: r.get(k) for k in ("base_id", "context", "relation", "dataset", "group", "borrower_id",
                                            "family", "arm", "stratum")},
                   "seconds": round(time.time() - t0, 2), **res}
            with lock, open(path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, todo))
        print(m, a.judge, "done", len(todo), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)

"""
Same-transcript policy comparison on the LOCKED v1 set (Improvement_Plan WS2.4 and WS2.7). The saved first-stage
judge response of every live ReCheck 1.4 (P) run is processed by each policy, so differences come from the policy,
not from judge sampling:
  none       any structurally valid objection blocks (A1 rule)
  execute    objections recomputed; no reassessment
  P          execute + the saved reassessment (as run live)
  D          every unconfirmed objection or unsupported claim is a dispute -> defer
  R / R11    disputes resolved by the independent reader (GPT-OSS-120B; dispute/1.0 or 1.1), live reader calls
Also writes the reader ledger: what the reader did with each dispute, and its token cost.

    python replay_locked.py --judge qwen3_next [--live]
Output: Submission/locked/replay_locked_<judge>.jsonl and replay_locked_<judge>_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispute as DP  # noqa: E402
import executable_verifier as EV  # noqa: E402
import json_parse as JP  # noqa: E402
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from score_m1 import SUB  # noqa: E402

LK = SUB / "locked"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3_next")
    ap.add_argument("--reader", default="gptoss120b")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    recs = {json.loads(l)["item_id"]: json.loads(l) for l in open(LK / "fresh_narratives_v1.jsonl")}
    P = [json.loads(l) for l in open(LK / "fresh_narratives_v1_runs" / f"exec_P__{a.judge}.jsonl")]
    dst = LK / f"replay_locked_{a.judge}.jsonl"
    done = {json.loads(l)["item_id"] for l in open(dst)} if dst.exists() else set()
    lock = threading.Lock()

    def work(r):
        i = r["item_id"]
        d0 = r.get("detail") or {}
        row = {"item_id": i, "first_stage": "code_only"}
        if not d0.get("llm_called") or not d0.get("raw"):
            for p in ("none", "execute", "P", "D", "R", "R11"):
                row[p] = r["accepted"]
        else:
            rec = recs[i]
            eo = eo_from(rec["evidence"])
            text = W.clean(rec["text"])
            tx = W.execute_text(text, eo)
            d, _ = JP.parse_local(d0["raw"][0], validate=EV.valid_ev)
            if d is None:
                for p in ("none", "execute", "D", "R", "R11"):
                    row[p] = None
                row["P"] = r["accepted"]
                row["first_stage"] = "unparsed"
            else:
                row["first_stage"] = "llm"
                for p, how in (("none", "none"), ("execute", "execute")):
                    ws = EV.check_witnesses(W.witnesses_from_json(d), text, eo, tx, how)
                    dec, _w, _s = EV._resolve(d, tx, ws, True)
                    row[p] = {"accept": True, "reject": False}.get(dec)
                row["P"] = r["accepted"]
                ws = EV.check_witnesses(W.witnesses_from_json(d), text, eo, tx, "execute")
                dec, why, status = EV._resolve(d, tx, ws, True)
                contested, omis = EV.disputes(d, tx, ws, status, eo)
                if dec == "reject":
                    for p in ("D", "R", "R11"):
                        row[p] = False
                elif dec == "accept" and not contested and not omis:
                    for p in ("D", "R", "R11"):
                        row[p] = True
                else:
                    row["D"] = None
                    row["disputed"] = True
                    for p, ver in (("R", "1.0"), ("R11", "1.1")):
                        if not a.live:
                            row[p] = "dry"
                            continue
                        dec2, why2, rec2, calls = DP.resolve(contested, omis, tx, text, eo, a.reader,
                                                             experiment="locked-replay", item=i, version=ver)
                        row[p] = {"accept": True, "reject": False}.get(dec2)
                        row[p + "_why"] = why2
                        row[p + "_tokens"] = sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)
                                                 for c in calls)
        with lock, open(dst, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    todo = [r for r in P if r["item_id"] not in done]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    rows = [json.loads(l) for l in open(dst)]
    print(a.judge, len(rows), Counter(r["first_stage"] for r in rows))
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)

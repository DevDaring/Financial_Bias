"""
ReCheck 2-R on the SAME saved first-stage judge responses as ReCheck 1.4 (Future_Plan review §3.1-3.2).
Only the independent reinterpretation call is new, so R vs D vs P-1.4 isolates the dispute policy.

    python replay_dispute.py --judge qwen3_next --file exec_P__qwen3_next__v1.4.jsonl --reader gptoss120b --live
Output (append-only, resumable): Submission/m1/comparators/replayR__<judge>__<reader>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispute as DP  # noqa: E402
import executable_verifier as EV  # noqa: E402
import json_parse as JP  # noqa: E402
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from replay_same_transcript import records  # noqa: E402
from score_m1 import M1, final_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3_next")
    ap.add_argument("--file", default=None)
    ap.add_argument("--reader", default="gptoss120b")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--version", default="1.0", help="dispute version: 1.0 (frozen R) or 1.1 (R11)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--call-timeout", type=float, default=None)
    a = ap.parse_args()
    if a.call_timeout:
        import gateway as G
        G.GATEWAY.timeout = a.call_timeout
    src = M1 / "comparators" / (a.file or f"exec_P__{a.judge}.jsonl")
    dst = M1 / "comparators" / (f"replayR__{a.judge}__{a.reader}.jsonl" if a.version == "1.0"
                                else f"replayR11__{a.judge}__{a.reader}.jsonl")
    done = {json.loads(l)["item_id"] for l in open(dst)} if dst.exists() else set()
    fl, recs = final_labels(), records()
    todo = []
    for line in open(src):
        r = json.loads(line)
        if r["item_id"] in fl and r["item_id"] not in done:
            todo.append(r)
    lock = threading.Lock()
    if not a.live:
        print(f"{len(todo)} items; dry (no reinterpretation calls). Add --live.")

    def work(r):
        k, truth, _b = fl[r["item_id"]]
        d0 = r.get("detail") or {}
        row = {"item_id": r["item_id"], "truth": truth, "dataset": r["dataset"], "group": r.get("group"),
               "method": "exec_R_replay", "judge": a.judge, "reader": a.reader, "calls": list(d0.get("calls") or [])[:1]}
        if not d0.get("llm_called") or not d0.get("raw"):
            row.update(accepted=r["accepted"], reason=d0.get("reason"), path="code_only")
        else:
            rec = recs[(k["borrower_id"], k["arm"], k["family"])]
            eo = eo_from(rec["evidence"])
            text = W.clean(rec["text"])
            tx = W.execute_text(text, eo)
            d, _how = JP.parse_local(d0["raw"][0], validate=EV.valid_ev)
            if d is None:
                row.update(accepted=None, reason="first_stage_unparsed", path="unparsed")
            else:
                ws = EV.check_witnesses(W.witnesses_from_json(d), text, eo, tx, "execute")
                dec, why, status = EV._resolve(d, tx, ws, True)
                contested, omis = EV.disputes(d, tx, ws, status, eo)
                row["disputes"] = {"claims": contested, "omissions": omis}
                if dec == "reject":
                    row.update(accepted=False, reason=why, path="first_stage")
                elif dec == "accept" and not contested and not omis:
                    row.update(accepted=True, reason=why, path="first_stage")
                elif not a.live:
                    row.update(accepted=None, reason="dry", path="dispute")
                else:
                    dec2, why2, rec2, calls = DP.resolve(contested, omis, tx, text, eo, a.reader,
                                                         experiment="m1-replayR", item=r["item_id"],
                                                         version=a.version)
                    row.update(accepted={"accept": True, "reject": False}.get(dec2), reason=why2, path="dispute",
                               reinterpretation=rec2)
                    row["calls"] += calls
        if a.live:
            with lock, open(dst, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, todo))
    from collections import Counter
    print(Counter((r["truth"], r["accepted"], r["path"]) for r in rows))
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)

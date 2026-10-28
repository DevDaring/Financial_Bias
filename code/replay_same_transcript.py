"""
Same-transcript objection ablation (Future_Plan review §3.2; CPU only, no API calls).

Every live ReCheck (exec_P) row stores its first-stage judge response. Replaying that identical
response through different deterministic postprocessors isolates the objection handling from
model variability and from the different live calls that A1/A2 made:

  none         any structurally present objection blocks (A1 postprocessor)
  plausibility contract plausibility check only (A2 postprocessor)
  execute      objections recomputed; a dismissed objection against a code-supported claim is
               ignored and one against a residual claim leaves it unresolved (ReCheck 1.x first round)
  execute+R    as run live: execute, then the saved reassessment response (ReCheck 1.x)
  dispute      execute, but ANY judge objection or 'contradicted' status that code does not confirm
               is a code-judge dispute and defers (the conservative disputes-to-defer gate)

Items decided by code alone (no LLM call) are identical under every policy and are reported
separately. Output: Submission/m1/replay_same_transcript_<judge>.json

    python replay_same_transcript.py --judge qwen3_next --file exec_P__qwen3_next__v1.4.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import executable_verifier as EV  # noqa: E402
import json_parse as JP  # noqa: E402
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from score_m1 import M1, final_labels  # noqa: E402

POLICIES = ("none", "plausibility", "execute", "execute+R", "dispute")


def records():
    latest = {}
    for line in open(M1 / "records.jsonl"):
        r = json.loads(line)
        latest[(r["borrower_id"], r["arm"], r["family"])] = r
    return latest


def decide(d, text, eo, tx, how, dispute=False):
    ws = EV.check_witnesses(W.witnesses_from_json(d), text, eo, tx, how)
    dec, why, status = EV._resolve(d, tx, ws, True)
    if dispute and dec == "accept":
        # a judge objection or 'contradicted' status that code did not confirm is a dispute
        contested = [w for w in ws if not w.blocks] + \
                    [c for c, s in status.items() if s in ("contradicted", "insufficient")]
        if contested:
            return None, "code_judge_dispute", ws
    return dec, why, ws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3_next")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    path = M1 / "comparators" / (a.file or f"exec_P__{a.judge}.jsonl")
    fl = final_labels()
    recs = records()
    out_rows, unparsable = [], 0
    for line in open(path):
        r = json.loads(line)
        if r["item_id"] not in fl:
            continue
        k, truth, _b = fl[r["item_id"]]
        d0 = r.get("detail") or {}
        row = {"item_id": r["item_id"], "truth": truth, "dataset": r["dataset"], "group": r.get("group"),
               "borrower_id": k["borrower_id"], "llm": bool(d0.get("llm_called"))}
        if not d0.get("llm_called") or not d0.get("raw"):
            for p in POLICIES:
                row[p] = r["accepted"]
            row["path"] = "code_only" if not d0.get("llm_called") else "no_response"
            out_rows.append(row)
            continue
        rec = recs[(k["borrower_id"], k["arm"], k["family"])]
        eo = eo_from(rec["evidence"])
        text = W.clean(rec["text"])
        tx = W.execute_text(text, eo)
        d, how_parsed = JP.parse_local(d0["raw"][0], validate=EV.valid_ev)
        if d is None:
            unparsable += 1
            for p in POLICIES:
                row[p] = None
            row["path"] = "first_stage_unparsed"
            out_rows.append(row)
            continue
        row["path"] = "llm"
        for p in ("none", "plausibility", "execute"):
            dec, why, ws = decide(d, text, eo, tx, p)
            row[p] = {"accept": True, "reject": False}.get(dec)
            row[p + "_why"] = why
        row["n_objections"] = len(W.witnesses_from_json(d))
        dec, why, ws = decide(d, text, eo, tx, "execute", dispute=True)
        row["dispute"] = {"accept": True, "reject": False}.get(dec)
        row["dispute_why"] = why
        # the live run's own outcome (execute + saved reassessment)
        row["execute+R"] = r["accepted"]
        row["live_reason"] = d0.get("reason")
        out_rows.append(row)

    def cells(p, rows):
        c = Counter()
        for x in rows:
            c[("wrong" if x["truth"] else "correct", {True: "release", False: "reject", None: "defer"}[x[p]])] += 1
        return {f"{a}_{b}": c[(a, b)] for a in ("correct", "wrong") for b in ("release", "reject", "defer")}

    llm_rows = [x for x in out_rows if x["path"] == "llm"]
    res = {"note": "DEVELOPMENT (299 adjudicated M1 items). Same saved first-stage judge response replayed through "
                   "each postprocessor; code-only items are identical across policies.",
           "source": path.name, "n": len(out_rows), "n_llm_replayed": len(llm_rows),
           "n_first_stage_unparsed": unparsable,
           "all_items": {p: cells(p, out_rows) for p in POLICIES},
           "llm_items_only": {p: cells(p, llm_rows) for p in POLICIES},
           "transitions_execute_vs_none": Counter(
               f"{'wrong' if x['truth'] else 'correct'}:{x['none']}->{x['execute']}" for x in llm_rows
               if x["none"] != x["execute"]),
           "transitions_dispute_vs_execute": Counter(
               f"{'wrong' if x['truth'] else 'correct'}:{x['execute']}->{x['dispute']}" for x in llm_rows
               if x["execute"] != x["dispute"]),
           "rows": out_rows}
    dst = M1 / f"replay_same_transcript_{a.judge}.json"
    dst.write_text(json.dumps(res, indent=1, default=str))
    print(path.name, "n", len(out_rows), "llm", len(llm_rows), "unparsed", unparsable)
    for p in POLICIES:
        print(f"  {p:13}", res["all_items"][p])
    print("  execute vs none:", dict(res["transitions_execute_vs_none"]))
    print("  dispute vs execute:", dict(res["transitions_dispute_vs_execute"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

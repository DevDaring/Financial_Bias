"""
Analysis of the controlled context experiment (Future_Plan review §6). Prespecified before any run:

  outcomes     error escape  = wrong passages released / wrong passages
               correct retention = correct passages released / correct passages
               unresolved rate; LLM calls and tokens
  primary test method x context INTERACTION on error escape:
               DiD = [escape(B1R) - escape(ReCheck-R)]_claims - [escape(B1R) - escape(ReCheck-R)]_alone
               (and the same for filler vs alone, and for correct retention), with a base-case cluster bootstrap
               (80 base cases resampled; 4,000 draws). A significant effect in one context and not another is
               NOT an interaction test; only the DiD interval is.
  secondary    every method against B1R; per relation

    python analyze_context.py   -> Submission/mechanism/context_analysis.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack_m1 import SUB  # noqa: E402

RUNS = SUB / "mechanism" / "context_v1_runs"
B = 4000
REF, BASE = "exec_R", "generic_rubric"
CONTEXTS = ("alone", "claims", "filler")


def load():
    data = defaultdict(dict)          # (method, judge) -> item -> row
    for p in sorted(RUNS.glob("*.jsonl")):
        m, j = p.stem.split("__")
        for l in open(p):
            r = json.loads(l)
            data[(m, j)][r["item_id"]] = r
    return data


def rates(rows, items):
    out = {}
    for ctx in CONTEXTS:
        w = [rows[i] for i in items if i in rows and rows[i]["context"] == ctx and rows[i]["truth"]]
        c = [rows[i] for i in items if i in rows and rows[i]["context"] == ctx and not rows[i]["truth"]]
        out[ctx] = {"escape": sum(r["accepted"] is True for r in w) / max(1, len(w)),
                    "retention": sum(r["accepted"] is True for r in c) / max(1, len(c)),
                    "unresolved": sum(r["accepted"] is None for r in w + c) / max(1, len(w + c)),
                    "n_wrong": len(w), "n_correct": len(c)}
    return out


def main() -> int:
    import argparse
    global RUNS, REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--ref", default="exec_R")
    args = ap.parse_args()
    RUNS, REF = SUB / "mechanism" / f"context_{args.tag}_runs", args.ref
    pack = [json.loads(l) for l in open(SUB / "mechanism" / f"context_pack_{args.tag}.jsonl")]
    base_of = {r["item_id"]: r["base_id"] for r in pack}
    rel_of = {r["item_id"]: r["relation"] for r in pack}
    bases = sorted({r["base_id"] for r in pack})
    by_base = defaultdict(list)
    for r in pack:
        by_base[r["base_id"]].append(r["item_id"])
    data = load()
    rng = random.Random(20261008)
    res = {"n_passages": len(pack), "n_base": len(bases), "reference": REF, "judges": {}}
    for j in sorted({j for _m, j in data}):
        meth = {m: rows for (m, jj), rows in data.items() if jj == j and len(rows) >= 0.95 * len(pack)}
        jr = {"methods": {}, "interaction": {}, "by_relation": {}}
        allit = [i for b in bases for i in by_base[b]]
        for m, rows in meth.items():
            rt = rates(rows, allit)
            calls = [c for r in rows.values() for c in (r.get("calls") or []) if isinstance(c, dict)]
            jr["methods"][m] = {"by_context": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                   for kk, vv in v.items()} for k, v in rt.items()},
                                "calls_per_item": round(len(calls) / max(1, len(rows)), 2),
                                "tokens_per_item": round(sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)
                                                             for c in calls) / max(1, len(rows)), 1)}
            rel = defaultdict(lambda: [0, 0, 0, 0])
            for i, r in rows.items():
                k = rel_of[i]
                if r["truth"]:
                    rel[k][0] += r["accepted"] is True
                    rel[k][1] += 1
                else:
                    rel[k][2] += r["accepted"] is True
                    rel[k][3] += 1
            jr["by_relation"][m] = {k: {"escape": f"{v[0]}/{v[1]}", "retention": f"{v[2]}/{v[3]}"} for k, v in rel.items()}
        comps = [(REF, x) for x in meth if x not in (REF,)] if REF in meth else []
        import numpy as np
        bidx = {b: n for n, b in enumerate(bases)}
        cidx = {c: n for n, c in enumerate(CONTEXTS)}

        def counts(rows):
            # [base, context, truth(0 correct, 1 wrong), (released, total)]
            arr = np.zeros((len(bases), 3, 2, 2))
            for i, r in rows.items():
                if i not in base_of:
                    continue
                arr[bidx[base_of[i]], cidx[r["context"]], int(bool(r["truth"])), 0] += r.get("accepted") is True
                arr[bidx[base_of[i]], cidx[r["context"]], int(bool(r["truth"])), 1] += 1
            return arr
        cnt = {m: counts(rows) for m, rows in meth.items()}
        W = np.random.default_rng(20261008).multinomial(len(bases), [1 / len(bases)] * len(bases), size=B)  # draws x bases

        def rate_w(arr, w, ctx, wrong):
            num = w @ arr[:, cidx[ctx], int(wrong), 0]
            den = w @ arr[:, cidx[ctx], int(wrong), 1]
            return num / np.maximum(den, 1)
        one = np.ones((1, len(bases)))
        for a, b in comps:
            out = {}
            for metric, wrong in (("escape", True), ("retention", False)):
                for ctx in ("claims", "filler"):
                    def did(w):
                        ra = lambda c: rate_w(cnt[a], w, c, wrong)   # noqa: E731
                        rb = lambda c: rate_w(cnt[b], w, c, wrong)   # noqa: E731
                        return (rb(ctx) - ra(ctx)) - (rb("alone") - ra("alone"))
                    pt = float(did(one)[0])
                    bs = np.sort(did(W))
                    out[f"{metric}|{ctx}_vs_alone"] = {
                        "DiD": round(pt, 4), "ci95": [round(float(bs[int(.025 * B)]), 4), round(float(bs[int(.975 * B)]), 4)],
                        "X_minus_R_alone": round(float(rate_w(cnt[b], one, "alone", wrong)[0] - rate_w(cnt[a], one, "alone", wrong)[0]), 4),
                        f"X_minus_R_{ctx}": round(float(rate_w(cnt[b], one, ctx, wrong)[0] - rate_w(cnt[a], one, ctx, wrong)[0]), 4)}
            jr["interaction"][f"{b}_vs_{a}"] = out
        res["judges"][j] = jr
    out = "context_analysis.json" if (args.tag == "v1" and REF == "exec_R") else f"context_analysis_{args.tag}_{REF}.json"
    (SUB / "mechanism" / out).write_text(json.dumps(res, indent=1))
    for j, jr in res["judges"].items():
        print("=====", j)
        for m, v in jr["methods"].items():
            print(f"  {m:20}", " | ".join(f"{c}: esc {v['by_context'][c]['escape']:.2f} ret {v['by_context'][c]['retention']:.2f} "
                                          f"unr {v['by_context'][c]['unresolved']:.2f}" for c in CONTEXTS), f"calls {v['calls_per_item']}")
        for k, v in jr["interaction"].items():
            for kk, vv in v.items():
                print(f"  {k:36} {kk:24} DiD {vv['DiD']:+.3f} {vv['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

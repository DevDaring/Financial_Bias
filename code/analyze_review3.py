"""
Second review (13 Oct 2026), analyses from the frozen fresh-v2 records; no model call (all post hoc).

  operations   per executable operation of ReCheck-P (GPT-OSS-120B): how often it ran, supported, contradicted, and
               how often a narrative it contradicted was labelled wrong (precision) -- Q2
  tokens       calls, input and output tokens per narrative, truncations and USD per 1000 narratives -- Q7
  escalation   a router replayed from saved outputs: ReCheck-P with GPT-OSS-120B, and only the narratives it defers
               go to Grok 4.5 given the precomputed facts (B1F) -- Q8
  residual     raters' error types among the wrong narratives ReCheck-P released -- Q4

    python analyze_review3.py
Output: Submission/locked/review3_analysis.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import analyze_review2 as R2
from analyze_utility import PRICE

RUNS = R2.AL.LK / "fresh_v2_runs"
J = R2.JUDGE


def load(m, j=J):
    return R2.jl(RUNS / f"{m}__{j}.jsonl")


def operations(nar, P):
    st = defaultdict(Counter)
    caught = defaultdict(set)
    for i, r in P.items():
        ex = (r.get("detail") or {}).get("execution") or {}
        ops = [o for c in ex.get("claims", []) for o in c.get("ops", [])] + list(ex.get("text_ops") or [])
        for o in ops:
            st[o["op"]]["n"] += 1
            st[o["op"]][o["status"]] += 1
            if o["status"] == "contradiction":
                caught[o["op"]].add(i)
    nw = sum(1 for v in nar.values() if v["truth"])
    out = {}
    for op, c in st.items():
        ids = [i for i in caught[op] if i in nar]
        w = sum(1 for i in ids if nar[i]["truth"])
        out[op] = {"executed": c["n"], "support": c.get("support", 0), "contradiction": c.get("contradiction", 0),
                   "narratives_contradicted": len(ids), "of_which_wrong": w,
                   "precision": round(100 * w / len(ids), 1) if ids else None,
                   "share_of_wrong_narratives": round(100 * w / nw, 1)}
    return dict(sorted(out.items(), key=lambda x: -x[1]["executed"]))


def tokens(nar):
    out = {}
    for p in sorted(RUNS.glob("*.jsonl")):
        m, j = p.stem.split("__")
        rows = {k: v for k, v in R2.jl(p).items() if k in nar}
        calls = [c for r in rows.values() for c in (r.get("calls") or [])]
        n = len(rows)
        if not calls:
            continue
        pt = sum(c.get("prompt_tokens", 0) for c in calls)
        ct = sum(c.get("completion_tokens", 0) for c in calls)
        o = {"calls_per_narrative": round(len(calls) / n, 2), "input_tokens_per_call": round(pt / len(calls)),
             "output_tokens_per_call": round(ct / len(calls)), "tokens_per_narrative": round((pt + ct) / n),
             "truncated_calls_pct": round(100 * sum(c.get("finish_reason") == "length" for c in calls) / len(calls), 1)}
        if j in PRICE:
            o["usd_per_1000"] = round(1000 * (pt * PRICE[j][0] + ct * PRICE[j][1]) / 1e6 / n, 3)
        out[f"{m}|{j}"] = o
    return out


def escalation(nar, boot, P, B1F, G):
    """Release decisions of the router, and its cost per 1000 narratives (saved token counts, list prices)."""
    deferred = [i for i, r in P.items() if (r.get("detail") or {}).get("decision") == "unresolved"]
    router = dict(P, **{i: {"accepted": G[i].get("accepted") is True} for i in deferred})
    cheap = dict(P, **{i: {"accepted": B1F[i].get("accepted") is True} for i in deferred})

    def usd(rows, j, ids):
        return sum(c.get("prompt_tokens", 0) * PRICE[j][0] + c.get("completion_tokens", 0) * PRICE[j][1]
                   for i in ids for c in (rows[i].get("calls") or [])) / 1e6

    ids = list(nar)
    cost = {"ReCheck-P": usd(P, J, ids), "Grok+B1F": usd(G, "grok45", ids), "B1F": usd(B1F, J, ids)}
    cost["router"] = cost["ReCheck-P"] + usd(G, "grok45", [i for i in deferred if i in nar])
    sec = {"ReCheck-P": sum(P[i].get("seconds") or 0 for i in ids), "Grok+B1F": sum(G[i].get("seconds") or 0 for i in ids)}
    sec["router"] = sec["ReCheck-P"] + sum(G[i].get("seconds") or 0 for i in deferred if i in nar)
    n = len(ids)
    out = {"deferred": sum(1 for i in deferred if i in nar),
           "usd_per_1000": {k: round(1000 * v / n, 3) for k, v in cost.items()},
           "seconds_per_narrative": {k: round(v / n, 1) for k, v in sec.items()},
           "counts": {k: dict(zip(("correct", "wrong"), boot.cw(v, ids)[:2]))
                      for k, v in (("router", router), ("router_same_judge", cheap), ("ReCheck-P", P),
                                   ("B1F", B1F), ("Grok+B1F", G))},
           "router_vs_B1F": boot.diff(router, B1F), "router_vs_Grok": boot.diff(router, G),
           "router_vs_ReCheck-P": boot.diff(router, P), "router_same_judge_vs_B1F": boot.diff(cheap, B1F)}
    return out


def residual(nar, P):
    c = Counter()
    for i, v in nar.items():
        if v["truth"] and P.get(i, {}).get("accepted") is True:
            for t in v.get("types") or ["unspecified"]:
                c[t] += 1
    return dict(c.most_common())


def main() -> int:
    nar, _ = R2.setup("provided")
    P, B1F, G = load("exec_P"), load("facts_judge"), load("facts_judge", "grok45")
    boot = R2.Boot(nar)
    res = {"operations": operations(nar, P), "tokens": tokens(nar), "escalation": escalation(nar, boot, P, B1F, G),
           "residual_types": residual(nar, P)}
    out = R2.AL.LK / "review3_analysis.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    e = res["escalation"]
    print(json.dumps({"operations": res["operations"], "tokens": res["tokens"], "residual": res["residual_types"],
                      "escalation": {k: e[k] for k in ("deferred", "usd_per_1000", "seconds_per_narrative", "counts")},
                      **{k: {m: e[k][m] for m in ("wrong_release", "correct_release", "U2")}
                         for k in ("router_vs_B1F", "router_vs_Grok", "router_vs_ReCheck-P", "router_same_judge_vs_B1F")}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

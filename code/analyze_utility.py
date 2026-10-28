"""
Decision analysis of the fresh v2 test (secondary outcome: utility for lambda in 1, 2, 5, 10, declared in
analyze_locked.py and reported for the locked test). Utility per 100 narratives = correct released - lambda x wrong
released. For each judge, ReCheck-P is compared with every other verifier run with the same judge, and with the
strongest system run (Grok 4.5 + B1F). Break-even lambda* = (correct-release gap) / (wrong-release gap): ReCheck-P
has the higher utility for every lambda above lambda* when it releases fewer wrong narratives.

Uses the same labels, strata and bootstrap as analyze_locked.py (borrower = unit, resampled within strata).
Reads only label columns (the annotator, checker and participant tabs are never read).

    python analyze_utility.py
Output: Submission/locked/fresh_v2_utility.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import analyze_locked as AL

LAMBDAS = (1, 2, 5, 10)
GRID = [0.25 * k for k in range(1, 41)]   # lambda 0.25..10, for the figure
B = 4000
# list prices (USD per million input/output tokens), as stated in the paper
PRICE = {"gptoss120b": (0.037, 0.17), "qwen3_next": (0.10, 1.10), "grok45": (0.50, 1.50)}
LABELS_DIR = AL.SUB / "packs" / "Annotated" / "IDBI_Annotator_Checks"


def main() -> int:
    AL.RUNS = AL.LK / "fresh_v2_runs"
    adj = AL.LK / "adjudication_fresh_v2_independent.csv"
    if not adj.exists():
        adj = AL.LK / "adjudication_fresh_v2.csv"
    AL.CFG.update(key=AL.SUB / "annotation_key_fresh_v2.jsonl", sheet="FRESH_annotation_sheet_",
                  adjudication=adj, records=AL.LK / "fresh_natural_v2_final.jsonl")
    data = AL.runs()
    lab, _dis, _dup = AL.labels(LABELS_DIR)
    nar = {i: v for i, v in lab.items() if v["arm"] != "template"}
    strata = defaultdict(list)
    for i, v in nar.items():
        strata[v["stratum"]].append(i)
    rng = random.Random(20261010)
    samples = [[i for v in strata.values() for i in (rng.choice(v) for _ in v)] for _ in range(B)]

    def stats(rows, items):
        w = sum(1 for i in items if nar[i]["truth"] and rows.get(i, {}).get("accepted") is True)
        c = sum(1 for i in items if not nar[i]["truth"] and rows.get(i, {}).get("accepted") is True)
        return c, w

    def compare(A, Bm):
        n = len(nar)
        ca, wa = stats(A, nar)
        cb, wb = stats(Bm, nar)
        boot = [(stats(A, s), stats(Bm, s)) for s in samples]
        out = {"dC": ca - cb, "dW": wa - wb}
        for lam in LAMBDAS:
            pt = 100 * ((ca - lam * wa) - (cb - lam * wb)) / n
            bs = sorted(100 * ((x[0] - lam * x[1]) - (y[0] - lam * y[1])) / n for x, y in boot)
            out[f"lambda_{lam}"] = {"diff_per_100": round(pt, 2), "ci95": [round(bs[int(.025 * B)], 2),
                                                                           round(bs[int(.975 * B)], 2)]}
        out["grid"] = []
        for lam in GRID:
            bs = sorted(100 * ((x[0] - lam * x[1]) - (y[0] - lam * y[1])) / n for x, y in boot)
            out["grid"].append([lam, round(100 * ((ca - lam * wa) - (cb - lam * wb)) / n, 2),
                                round(bs[int(.025 * B)], 2), round(bs[int(.975 * B)], 2)])
        # break-even: ReCheck-P better for lambda > lambda* when it releases fewer wrong narratives
        if wa < wb:
            lam_star = max(0.0, (cb - ca) / (wb - wa))
            bs = sorted(max(0.0, (y[0] - x[0]) / (y[1] - x[1])) if y[1] > x[1] else float("inf") for x, y in boot)
            out["break_even_lambda"] = {"point": round(lam_star, 2), "upper95": round(bs[int(.95 * B)], 2)}
        return out

    res = {"n_narratives": len(nar), "lambdas": LAMBDAS, "comparisons": {}, "calls_per_item": {}}
    for (m, j), rows in data.items():
        calls = [c for r in rows.values() for c in (r.get("calls") or []) if isinstance(c, dict)]
        res["calls_per_item"][f"{m}|{j}"] = round(len(calls) / max(1, len(rows)), 2)
        if j in PRICE:
            usd = sum(c.get("prompt_tokens", 0) * PRICE[j][0] + c.get("completion_tokens", 0) * PRICE[j][1]
                      for c in calls) / 1e6
            res.setdefault("usd_per_1000", {})[f"{m}|{j}"] = round(1000 * usd / max(1, len(rows)), 3)
        res.setdefault("seconds_per_item", {})[f"{m}|{j}"] = round(sum(r.get("seconds") or 0 for r in rows.values())
                                                                  / max(1, len(rows)), 1)
    for (m, j) in sorted(data):
        if m != "exec_P":
            continue
        for (m2, j2) in sorted(data):
            if (m2, j2) == (m, j) or not (j2 == j or (m2, j2) == ("facts_judge", "grok45")):
                continue
            res["comparisons"][f"exec_P|{j} vs {m2}|{j2}"] = compare(data[(m, j)], data[(m2, j2)])
    out = AL.LK / "fresh_v2_utility.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["comparisons"].items():
        print(f"{k:52}", {l: v[f'lambda_{l}']['diff_per_100'] for l in LAMBDAS}, "λ*", v.get("break_even_lambda"),
              "CI λ=2", v["lambda_2"]["ci95"])
    print("calls", res["calls_per_item"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

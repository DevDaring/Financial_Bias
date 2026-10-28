"""
Run-to-run variation and a same-judge ensemble baseline (fresh v2 test, GPT-OSS-120B judge; post hoc).

Hosted models are not deterministic at temperature 0, so the frozen ReCheck-P and B1F were each run three times on
the same 774 narratives. This script reports
  runs       wrong and correct releases per run, mean and spread, for both verifiers
  averaged   the ReCheck-P minus B1F difference averaged over the three run pairs, with a bootstrap interval that
             resamples borrowers within strata (same labels and strata as the primary analysis)
  sc3        B1F-SC3: the three B1F runs combined by majority vote (release if at least two of three release), i.e.
             the same judge with three times the spend, against a single ReCheck-P run and against ReCheck-P's
             own majority vote

    python analyze_runs3.py
Output: Submission/locked/runs3_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import analyze_review2 as R2

ABL = R2.ABL
RUNS = R2.AL.LK / "fresh_v2_runs"
J = R2.JUDGE


def main() -> int:
    nar, _ = R2.setup("provided")
    P = [R2.jl(RUNS / f"exec_P__{J}.jsonl"), R2.jl(ABL / "repeat" / f"exec_P__{J}.jsonl"), R2.jl(ABL / "run3" / f"exec_P__{J}.jsonl")]
    F = [R2.jl(RUNS / f"facts_judge__{J}.jsonl"), R2.jl(ABL / "b1f_run2" / f"facts_judge__{J}.jsonl"),
         R2.jl(ABL / "b1f_run3" / f"facts_judge__{J}.jsonl")]
    if any(len(x) < len(nar) for x in P + F):
        print("incomplete runs:", [len(x) for x in P + F])
        return 1
    boot = R2.Boot(nar)
    res = {"runs": {}}
    for name, runs in (("ReCheck-P", P), ("B1F", F)):
        cw = [boot.cw(r, list(nar))[:2] for r in runs]
        res["runs"][name] = {"correct": [c for c, _ in cw], "wrong": [w for _, w in cw],
                             "correct_mean": round(mean(c for c, _ in cw), 1), "wrong_mean": round(mean(w for _, w in cw), 1),
                             "correct_sd": round(pstdev(c for c, _ in cw), 1), "wrong_sd": round(pstdev(w for _, w in cw), 1)}
    # averaged over the three run pairs: bootstrap the mean of the per-pair differences
    def avg_stat(items):
        out = []
        for a, b in zip(P, F):
            ca, wa, nc, nw = boot.cw(a, items)
            cb, wb, _, _ = boot.cw(b, items)
            n = len(items)
            out.append([100 * (wa - wb) / max(1, nw), 100 * (ca - cb) / max(1, nc)] +
                       [100 * ((ca - l * wa) - (cb - l * wb)) / n for l in R2.LAMBDAS])
        return [mean(x[k] for x in out) for k in range(len(out[0]))], out
    pt, per_pair = avg_stat(list(nar))
    bs = [avg_stat(s)[0] for s in boot.samples]
    names = ["wrong_release", "correct_release"] + [f"U{l}" for l in R2.LAMBDAS]
    res["averaged"] = {}
    for k, nm in enumerate(names):
        col = sorted(b[k] for b in bs)
        res["averaged"][nm] = {"diff": round(pt[k], 2), "ci95": [round(col[int(.025 * R2.B)], 2), round(col[int(.975 * R2.B)], 2)],
                               "one_sided_95": [round(col[int(.05 * R2.B)], 2), round(col[int(.95 * R2.B)], 2)],
                               "per_pair": [round(x[k], 2) for x in per_pair]}

    def vote(runs):
        return {i: {"accepted": sum(r[i].get("accepted") is True for r in runs) >= 2} for i in runs[0]}
    SC3 = vote(F)
    P3 = vote(P)
    res["sc3"] = {"B1F_SC3_vs_ReCheck_P_single": boot.diff(P[0], SC3),
                  "ReCheck_P_SC3_vs_B1F_SC3": boot.diff(P3, SC3)}
    res["sc3"]["B1F_SC3_counts"] = dict(zip(("correct", "wrong"), boot.cw(SC3, list(nar))[:2]))
    out = R2.AL.LK / "runs3_analysis.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"runs": res["runs"], "averaged": {k: {kk: v[kk] for kk in ("diff", "ci95", "per_pair")}
                                                         for k, v in res["averaged"].items()},
                      "sc3": {k: (v if "counts" in k else {m: v[m] for m in ("wrong_release", "correct_release", "U2", "counts")})
                              for k, v in res["sc3"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

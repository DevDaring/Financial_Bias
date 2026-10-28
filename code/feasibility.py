"""
Statistical feasibility (Future_Plan §F): released, independently labelled borrowers
needed for a simultaneous one-sided Clopper-Pearson certificate, and a simulation of
how often a certificate succeeds when the true selective error is not zero.

  1. zero-error lower bounds n >= ceil(log(alpha)/log(1-eps)), alpha = delta/(M*G)
  2. simulation: per stratum, n_cal calibration borrowers, coverage c, true selective
     error r; P(all G strata certify at eps) over 4,000 draws
  3. development inputs: the M1 coverage/error of the code layer and of the replayed
     P (development, optimistic) are printed for orientation, not used as truth

    python feasibility.py   -> Submission/feasibility.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import clopper_pearson_upper, required_n  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def p_certify(n_cal: int, coverage: float, err: float, eps: float, alpha: float, G: int, draws: int, rng) -> float:
    ok = 0
    for _ in range(draws):
        good = True
        for _g in range(G):
            n_rel = sum(1 for _ in range(n_cal) if rng.random() < coverage)
            k = sum(1 for _ in range(n_rel) if rng.random() < err)
            if clopper_pearson_upper(k, n_rel, alpha) > eps:
                good = False
                break
        ok += good
    return ok / draws


def main() -> int:
    rng = random.Random(20261007)
    delta, G = 0.05, 4
    zero = {}
    for eps in (0.05, 0.02, 0.01):
        zero[str(eps)] = {f"M={M}": required_n(eps, delta / (M * G)) for M in (1, 8)}
    sim = []
    for eps in (0.05, 0.02):
        for M in (1, 8):
            alpha = delta / (M * G)
            for cov in (0.6, 0.8):
                for err in (0.0, 0.01, 0.02):
                    if err >= eps:
                        continue
                    for n_cal in (150, 250, 400, 700, 1000):
                        p = p_certify(n_cal, cov, err, eps, alpha, G, 600, rng)
                        sim.append({"eps": eps, "M": M, "coverage": cov, "true_selective_error": err,
                                    "n_cal_per_stratum": n_cal, "p_all_strata_certify": round(p, 3)})
    out = {"delta": delta, "G": G, "zero_error_lower_bounds_released_per_stratum": zero, "simulation": sim,
           "reading": "the smallest n_cal per stratum with p_all_strata_certify >= 0.8 is the planning size for "
                      "that (eps, M, coverage, error) scenario; with natural error ~1-2% after release, eps=0.05 "
                      "needs several hundred calibration borrowers per stratum, eps=0.02 is rarely feasible"}
    (ROOT / "Submission" / "feasibility.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(zero, indent=1))
    for s in sim:
        if s["p_all_strata_certify"] >= 0.8 and not any(t["eps"] == s["eps"] and t["M"] == s["M"]
                                                        and t["coverage"] == s["coverage"]
                                                        and t["true_selective_error"] == s["true_selective_error"]
                                                        and t["n_cal_per_stratum"] < s["n_cal_per_stratum"]
                                                        and t["p_all_strata_certify"] >= 0.8 for t in sim):
            print(f"eps={s['eps']} M={s['M']} cov={s['coverage']} err={s['true_selective_error']}: "
                  f"n_cal/stratum={s['n_cal_per_stratum']} (P={s['p_all_strata_certify']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

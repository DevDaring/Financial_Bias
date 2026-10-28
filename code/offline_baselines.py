"""
Two comparators computed from cached outputs (no new API calls), on the final M1 labels.

1. risk_gate  (Badshah et al., COLM 2026 style: risk-controlled judging with abstention)
   Score = the judge's self-reported probability that the explanation is correct (the Mermigkis
   verifier's confidence, same judge model). Two-fold cross-fitting by BORROWER: on the calibration
   fold, choose the lowest threshold whose one-sided Clopper-Pearson upper bound on selective error is
   <= eps (0.05, delta 0.05); apply it to the other fold. Below the threshold the item is deferred
   (unresolved), never rejected. Departure: the original routes to retrieval or a stronger judge;
   here abstention goes to the audited template, as for every other method.

2. ws_cascade (Kiyani et al., ICML 2026 weak/strong verification; comparators.WeakStrongPolicy)
   Weak score = the same confidence. Strong verifier = (a) human review (the original setting: the
   label is revealed when queried; reported as review rate) or (b) the Grok 4.5 generic judge
   (a fallible strong stage). Items are streamed in a fixed seeded order.

Writes Submission/m1/comparators/{risk_gate,ws_cascade_human,ws_cascade_grok}__<judge>.jsonl
in the schema analyze_phaseA.py reads.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators as K  # noqa: E402
from policy import clopper_pearson_upper  # noqa: E402
from run_comparators import OUT  # noqa: E402
from score_m1 import SUB, final_labels  # noqa: E402

EPS, DELTA, SEED = 0.05, 0.05, 20261007


def load(name):
    p = OUT / f"{name}.jsonl"
    return {json.loads(l)["item_id"]: json.loads(l) for l in open(p)} if p.exists() else {}


def conf(r):
    s = r.get("score")
    return None if s is None else float(s)


def gate(items, scores, truth):
    """lowest threshold t with CP upper bound(selective error) <= EPS on these items; None if none."""
    cands = sorted({scores[i] for i in items if scores.get(i) is not None})
    for t in cands:
        rel = [i for i in items if scores.get(i) is not None and scores[i] >= t]
        if not rel:
            continue
        k = sum(1 for i in rel if truth[i])
        if clopper_pearson_upper(k, len(rel), DELTA) <= EPS:
            return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["qwen3_next", "grok45"])
    a = ap.parse_args()
    key = {r["item_id"]: r for r in map(json.loads, open(SUB / "annotation_key_m1.jsonl")) if not r.get("_meta")}
    fl = final_labels()
    truth = {i: v for i, (_k, v, _b) in fl.items()}
    items = sorted(truth)
    grok_b1 = load("generic_judge2__grok45")
    for judge in a.judges:
        mv = load(f"mermigkis_verifier__{judge}")
        if not mv:
            print(f"no mermigkis_verifier__{judge}; skipped")
            continue
        scores = {i: conf(mv[i]) for i in items if i in mv}
        base = {i: {"item_id": i, "truth": truth[i], "dataset": key[i]["dataset"], "group": key[i].get("group"),
                    "family": key[i]["family"], "arm": key[i]["arm"], "live": True} for i in items if i in mv}
        # ---- risk gate, 2-fold cross-fitting by borrower
        bs = sorted({key[i]["borrower_id"] for i in base})
        rng = random.Random(SEED)
        rng.shuffle(bs)
        fold = {b: k % 2 for k, b in enumerate(bs)}
        rows = []
        for f in (0, 1):
            cal = [i for i in base if fold[key[i]["borrower_id"]] != f]
            tst = [i for i in base if fold[key[i]["borrower_id"]] == f]
            t = gate(cal, scores, truth)
            for i in tst:
                s = scores.get(i)
                acc = (True if (t is not None and s is not None and s >= t) else None)
                rows.append(dict(base[i], method="risk_gate", judge=judge, accepted=acc, score=s,
                                 threshold=t, fold=f, calls=mv[i].get("calls") or []))
        with open(OUT / f"risk_gate__{judge}.jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps(r, default=str) + "\n")
        # ---- weak/strong cascade
        order = list(base)
        random.Random(SEED).shuffle(order)
        for variant in ("human", "grok"):
            if variant == "grok" and (not grok_b1 or judge == "grok45"):
                continue
            pol = K.WeakStrongPolicy(alpha=0.10, beta=0.10, seed=SEED)
            out = []
            for i in order:
                s = scores.get(i)
                w = 0.5 if s is None else s
                correct = not truth[i]
                if variant == "human":
                    strong = (lambda c=correct: c)
                else:
                    g = grok_b1.get(i, {})
                    strong = (lambda g=g: bool(g.get("accepted")))
                st = pol.step(w, strong)
                calls = list(mv[i].get("calls") or [])
                if st["queried"] and variant == "grok":
                    calls += grok_b1.get(i, {}).get("calls") or []
                out.append(dict(base[i], method=f"ws_cascade_{variant}", judge=judge,
                                accepted=(st["decision"] == "accept"), region=st["region"],
                                strong_queried=st["queried"], calls=calls))
            with open(OUT / f"ws_cascade_{variant}__{judge}.jsonl", "w") as fo:
                for r in out:
                    fo.write(json.dumps(r, default=str) + "\n")
        print(judge, "risk_gate thresholds", sorted({r['threshold'] for r in rows}, key=str),
              "deferred", sum(1 for r in rows if r["accepted"] is None), "of", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

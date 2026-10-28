"""
Robustness of the binder to missing aliases (review 2, Q1), code layer only: no model call.

Each evidence item's plain-language aliases are dropped at random with probability p (field names are kept), the
frozen code layer (executable/1.6 with its gate) is re-run on the fresh v2 narratives, and the code decision
(release by code alone / reject on an executed contradiction / send to the judge) is compared with the frozen one.
The safety-relevant quantity is the number of wrong narratives released by code alone; a missing alias should send
a claim to the judge rather than release it.

    python analyze_alias.py
Output: Submission/locked/alias_perturbation.json
"""
from __future__ import annotations

import copy
import json
import random
from collections import Counter

import analyze_review2 as R2
import witness as W
from build_pack_m1 import eo_from

PACK = R2.AL.LK / "fresh_v2_narratives.jsonl"
PS = (0.25, 0.5, 1.0)
SEEDS = (1, 2, 3, 4, 5)


def decide(r, evidence):
    tx = W.execute_text(W.clean(r["text"]), eo_from(evidence))
    routed = sum(1 for c in tx.claims if c.status not in ("support", "contradiction"))
    if tx.contradictions:
        return "reject", routed, len(tx.claims)
    if not tx.residual_claims and all(o.status == W.SUPPORT for o in tx.text_ops):
        return "release", routed, len(tx.claims)
    return "judge", routed, len(tx.claims)


def drop(evidence, p, rng):
    ev = copy.deepcopy(evidence)
    for it in ev["items"]:
        it["aliases"] = [a for a in (it.get("aliases") or []) if rng.random() >= p]
    return ev


def summary(dec, nar):
    c = Counter()
    for i, (d, routed, n) in dec.items():
        c[d] += 1
        c[f"{d}|{'wrong' if nar[i]['truth'] else 'correct'}"] += 1
        c["claims"] += n
        c["claims_routed"] += routed
    return dict(c)


def main() -> int:
    nar, _ = R2.setup("provided")
    pack = [r for r in (json.loads(l) for l in open(PACK)) if r["item_id"] in nar]
    base = {r["item_id"]: decide(r, r["evidence"]) for r in pack}
    res = {"n": len(pack), "aliases": sum(len(i.get("aliases") or []) for r in pack for i in r["evidence"]["items"]),
           "frozen": summary(base, nar), "perturbed": {}}
    for p in PS:
        runs = []
        for s in SEEDS if p < 1 else (0,):
            rng = random.Random(1000 * s + int(100 * p))
            dec = {r["item_id"]: decide(r, drop(r["evidence"], p, rng)) for r in pack}
            sm = summary(dec, nar)
            moves = Counter(f"{base[i][0]}->{d[0]}" for i, d in dec.items() if d[0] != base[i][0])
            sm["moves"] = dict(moves)
            sm["moves_wrong"] = dict(Counter(f"{base[i][0]}->{d[0]}" for i, d in dec.items()
                                             if d[0] != base[i][0] and nar[i]["truth"]))
            runs.append(sm)
        res["perturbed"][str(p)] = runs
    out = R2.AL.LK / "alias_perturbation.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"frozen": res["frozen"], **{p: [{k: r.get(k) for k in ("release", "release|wrong", "reject",
                     "judge", "claims_routed", "moves", "moves_wrong")} for r in v] for p, v in res["perturbed"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

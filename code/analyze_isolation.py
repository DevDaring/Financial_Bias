"""
Contribution-isolating comparisons on the locked v1 set (Improvement_Plan WS2), now development data.

  1. same-transcript policies (replay_locked.py): none / execute / P / D / R / R11, six cells, paired differences
     against D (borrower bootstrap within strata), and the reader ledger (what R and R11 did with each dispute)
  2. blind two-judge baselines: release only if BOTH judges (Qwen and GPT-OSS, same rubric) accept
  3. matched coverage: error among released narratives when every method releases the same number k of narratives,
     ranked inside each method's released set by ONE shared score (the share of the narrative's material claims the
     code layer executed as supported; ties broken by item id). k = the release count of ReCheck-D.

    python analyze_isolation.py -> Submission/locked/isolation_analysis.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from dev_sets import locked_items  # noqa: E402
from score_m1 import SUB  # noqa: E402

LK = SUB / "locked"
B = 3000


def cells(dec, truth):
    c = Counter()
    for i, t in truth.items():
        a = dec.get(i, "missing")
        c[f"{'wrong' if t else 'correct'}_{ {True: 'release', False: 'reject', None: 'defer'}.get(a, 'missing')}"] += 1
    return dict(c)


def main() -> int:
    items = {x["item_id"]: x for x in locked_items()}
    truth = {i: x["truth"] for i, x in items.items()}
    strata = defaultdict(list)
    for i, x in items.items():
        strata[x["rec"]["stratum"]].append(i)
    rng = random.Random(20261015)
    draws = [[i for v in strata.values() for i in (rng.choice(v) for _ in v)] for _ in range(B)]

    def paired(a, b):
        out = {}
        for kind, w in (("wrong_release", True), ("correct_release", False)):
            def rate(its, d):
                x = [i for i in its if truth[i] == w]
                return sum(d.get(i) is True for i in x) / max(1, len(x))
            pt = rate(list(truth), a) - rate(list(truth), b)
            bs = sorted(rate(s, a) - rate(s, b) for s in draws)
            out[kind] = {"diff": round(pt, 4), "ci95": [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)]}
        return out

    res = {"n": len(truth), "n_wrong": sum(truth.values()), "replay": {}, "methods": {}, "matched": {}}
    # 1. same-transcript policies
    for j in ("qwen3_next", "gemma27_vllm"):
        p = LK / f"replay_locked_{j}.jsonl"
        if not p.exists():
            continue
        rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}
        if len(rows) < 0.95 * len(truth):
            res["replay"][j] = {"incomplete": len(rows)}
            continue
        pol = {k: {i: r.get(k) for i, r in rows.items()} for k in ("none", "execute", "P", "D", "R", "R11")}
        rj = {"cells": {k: cells(v, truth) for k, v in pol.items()},
              "vs_D": {k: paired(v, pol["D"]) for k, v in pol.items() if k != "D"}}
        for k in ("R", "R11"):
            led = Counter()
            tok = 0
            for i, r in rows.items():
                if r.get("disputed"):
                    led[f"{'wrong' if truth[i] else 'correct'}_{ {True: 'release', False: 'reject', None: 'defer'}[r.get(k)]}"] += 1
                    tok += r.get(k + "_tokens") or 0
            rj[f"ledger_{k}"] = dict(led, disputes=sum(led.values()), reader_tokens=tok)
        res["replay"][j] = rj
    # 2./3. live methods on the locked set
    runs = LK / "fresh_narratives_v1_runs"
    dec = {}
    for p in sorted(runs.glob("*.jsonl")):
        m, j = p.stem.split("__")
        rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}
        if len(rows) >= 0.95 * len(truth):
            dec[f"{m}|{j}"] = {i: r.get("accepted") for i, r in rows.items()}
    for a, b in (("generic_rubric|qwen3_next", "generic_rubric|gptoss120b"), ("facts_judge|qwen3_next", "facts_judge|gptoss120b")):
        if a in dec and b in dec:
            dec[f"both({a.split('|')[0]})|qwen+gptoss"] = {i: (True if dec[a].get(i) is True and dec[b].get(i) is True
                                                             else False) for i in truth}
    # shared score: share of material claims executed as supported by the code layer (executable/1.5 behaviour)
    gate = W.ZERO_CALL_GATE
    W.ZERO_CALL_GATE = False
    score = {}
    for i, x in items.items():
        tx = W.execute_text(W.clean(x["rec"]["text"]), eo_from(x["rec"]["evidence"]))
        mat = [c for c in tx.claims if c.status != "nonmaterial"]
        score[i] = sum(c.status == W.SUPPORT for c in mat) / max(1, len(mat))
    W.ZERO_CALL_GATE = gate
    for k_, d in dec.items():
        c = cells(d, truth)
        rel = c.get("correct_release", 0) + c.get("wrong_release", 0)
        res["methods"][k_] = dict(c, released=rel, selective_error=round(c.get("wrong_release", 0) / max(1, rel), 4))
    kref = res["methods"].get("exec_D|qwen3_next", {}).get("released")
    if kref:
        for k_, d in dec.items():
            relset = sorted((i for i in truth if d.get(i) is True), key=lambda i: (-score[i], i))
            top = relset[:kref]
            w = sum(truth[i] for i in top)
            res["matched"][k_] = {"k": len(top), "wrong": w, "error_rate": round(w / max(1, len(top)), 4),
                                  "below_k": len(relset) < kref}
    (LK / "isolation_analysis.json").write_text(json.dumps(res, indent=1))
    for j, rj in res["replay"].items():
        print("==", j, {k: (v.get("correct_release"), v.get("wrong_release")) for k, v in rj.get("cells", {}).items()})
        for k in ("ledger_R", "ledger_R11"):
            print("  ", k, rj.get(k))
        for k, v in rj.get("vs_D", {}).items():
            print("   vs D", k, v)
    print("== methods (correct/wrong released, selective error) and matched coverage at k =", kref)
    for k_, v in sorted(res["methods"].items()):
        mm = res["matched"].get(k_, {})
        print(f"  {k_:44} {v.get('correct_release', 0):>4}/{v.get('wrong_release', 0):<4} {v['selective_error']:.3f}  "
              f"matched k={mm.get('k')} wrong={mm.get('wrong')} err={mm.get('error_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Paraphrase and adversarial-wording stress test: for each verifier (GPT-OSS-120B judge), the error escape (wrong
sentences released) and retention (correct sentences released) on the original sentences and on each rewrite
style, and the share of decisions that flip between an original and its rewrite. Rewrites that the human meaning
check (round 4, task 2) found to change the meaning are excluded when that check is available.

Intervals: bootstrap over base cases (both truth versions of a base case and all their rewrites move together).

    python analyze_paraphrase.py
Output: Submission/mechanism/paraphrase_analysis.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SUB = Path(__file__).resolve().parents[2] / "Submission"
MECH = SUB / "mechanism"
RUNS = MECH / "paraphrase_runs"
METHODS = ("exec_P", "facts_judge", "generic_rubric")
JUDGE = "gptoss120b"
B = 2000


def main() -> int:
    pack = {json.loads(l)["item_id"]: json.loads(l) for l in open(MECH / "paraphrase_pack.jsonl")}
    chk = MECH / "paraphrase_check.json"
    excluded = {c["paraphrase_item_id"] for c in json.load(open(chk))["changed"]} if chk.exists() else set()
    res = {"excluded_by_human_check": len(excluded), "methods": {}}
    bases = sorted({r["base_id"] for r in pack.values()})
    rng = random.Random(20261013)
    samples = [[rng.choice(bases) for _ in bases] for _ in range(B)]
    for m in METHODS:
        p = RUNS / f"{m}__{JUDGE}.jsonl"
        if not p.exists():
            continue
        rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}
        acc = {i: r.get("accepted") is True for i, r in rows.items() if i not in excluded}
        by = defaultdict(lambda: defaultdict(list))      # style -> base -> [(truth, accepted, flipped)]
        for i, a in acc.items():
            r = pack[i]
            orig = acc.get(r["source_id"] + "|orig")
            st = r["style"] if r["style"] == "original" else r["style"]
            flip = None if r["style"] == "original" or orig is None else (a != orig)
            by[st][r["base_id"]].append((r["truth"], a, flip))
        out = {}
        for st, d in by.items():
            def stat(bs):
                w = [a for b in bs for t, a, _ in d.get(b, []) if t]
                c = [a for b in bs for t, a, _ in d.get(b, []) if not t]
                f = [x for b in bs for _, _, x in d.get(b, []) if x is not None]
                return (100 * sum(w) / max(1, len(w)), 100 * sum(c) / max(1, len(c)),
                        100 * sum(f) / max(1, len(f)) if f else None, len(w), len(c))
            pt = stat(bases)
            bs = [stat(s) for s in samples]
            ci = lambda k: [round(sorted(x[k] for x in bs)[int(.025 * B)], 1), round(sorted(x[k] for x in bs)[int(.975 * B)], 1)]
            out[st] = {"escape": round(pt[0], 1), "escape_ci": ci(0), "retention": round(pt[1], 1),
                       "retention_ci": ci(1), "n_wrong": pt[3], "n_correct": pt[4]}
            if pt[2] is not None:
                out[st]["flip"] = round(pt[2], 1)
                out[st]["flip_ci"] = ci(2)
        res["methods"][m] = out
    (MECH / "paraphrase_analysis.json").write_text(json.dumps(res, indent=1))
    for m, v in res["methods"].items():
        for st, x in v.items():
            print(f"{m:15} {st:12} escape {x['escape']:5} {x['escape_ci']} retention {x['retention']:5} "
                  f"flip {x.get('flip')} n {x['n_wrong']}/{x['n_correct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

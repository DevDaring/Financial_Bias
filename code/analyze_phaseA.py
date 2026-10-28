"""
Phase A analysis (development): every method on the 245 agreed M1 items with one judge model.

Per method: false acceptance a, unnecessary withholding b (rejected correct), deferred (unresolved),
released-correct coverage, LLM calls, tokens and list-price cost. Paired differences P - X in
released-correct coverage and in errors accepted, with borrower-cluster bootstrap intervals
(borrowers resampled within dataset x group). Per-dataset group gaps in withholding (reject OR defer
of a correct narrative) for every method.

    python analyze_phaseA.py --judge qwen3_next  -> Submission/m1/phaseA_<judge>.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_comparators import OUT, PRICE  # noqa: E402

B = 2000
LABEL = {"always_accept": "control: always accept", "always_reject": "control: always reject",
         "pt_audit": "Pratama & Tseng audit", "generic_judge2": "B1 generic judge",
         "generic_tool_judge": "B2 generic + same tools", "generic_second_look": "B2' generic + second look",
         "contract": "B3 contract verifier (v1.0)", "zandi_judge": "Zandi TFC/TDC", "mermigkis_verifier": "Mermigkis verifier",
         "pbp_judge": "Tapwal PbP", "exec_A0": "A0 code only", "exec_A1": "A1 P without objection checks",
         "exec_A2": "A2 P with plausibility checks", "exec_PX": "PX P without claim execution",
         "exec_P": "P executable verification", "risk_gate": "Risk-controlled gate (Badshah-style)",
         "ws_cascade_human": "Weak/strong cascade, human strong (Kiyani)",
         "ws_cascade_grok": "Weak/strong cascade, Grok strong (Kiyani)"}


def norm_group(g):
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(g or "").lower(), "other")


TAG = ""


def load(method, judge):
    p = OUT / f"{method}__{judge}__{TAG}.jsonl"
    if TAG and not p.exists():
        p = OUT / f"{method}__none__{TAG}.jsonl"
    if not TAG or not p.exists():
        p = OUT / f"{method}__{judge}.jsonl"
    if not p.exists():
        p = OUT / f"{method}__none.jsonl"
    if not p.exists():
        return None
    rows = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("live") is False and "none" not in p.name:
            continue
        rows[r["item_id"]] = r
    return rows


def borrower_of(item_id, key):
    return key[item_id]["borrower_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3_next")
    ap.add_argument("--tag", default="", help="use <method>__<judge>__<tag>.jsonl where present (e.g. v1.4)")
    ap.add_argument("--labels", choices=["agreed", "final"], default="agreed",
                    help="final: re-score every row against the adjudicated labels (score_m1.final_labels)")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    root = Path(__file__).resolve().parents[2] / "Submission"
    key = {r["item_id"]: r for r in map(json.loads, open(root / "annotation_key_m1.jsonl")) if not r.get("_meta")}
    pin, pout = PRICE.get(a.judge, (0, 0))
    relabel = None
    if a.labels == "final":
        from score_m1 import final_labels
        relabel = {i: v for i, (_k, v, _b) in final_labels().items()}
    data, table = {}, {}
    for m in LABEL:
        rows = load(m, a.judge)
        if not rows:
            continue
        if relabel is not None:
            rows = {i: dict(r, truth=relabel[i]) for i, r in rows.items() if i in relabel}
        data[m] = rows
        err = [r for r in rows.values() if r["truth"]]
        ok = [r for r in rows.values() if r["truth"] is False]
        calls = sum(len(r.get("calls") or []) for r in rows.values())
        tin = sum(c.get("prompt_tokens") or 0 for r in rows.values() for c in (r.get("calls") or []))
        tout = sum(c.get("completion_tokens") or 0 for r in rows.values() for c in (r.get("calls") or []))
        rel_ok = sum(1 for r in ok if r["accepted"] is True)
        table[m] = {"method": LABEL[m], "n": len(rows),
                    "errors_accepted": f"{sum(1 for r in err if r['accepted'] is True)}/{len(err)}",
                    "correct_rejected": f"{sum(1 for r in ok if r['accepted'] is False)}/{len(ok)}",
                    "deferred": sum(1 for r in rows.values() if r["accepted"] is None),
                    "deferred_correct": sum(1 for r in ok if r["accepted"] is None),
                    "correct_released": f"{rel_ok}/{len(ok)}",
                    "selective_error": (round(sum(1 for r in err if r['accepted'] is True) /
                                              max(1, sum(1 for r in rows.values() if r['accepted'] is True)), 4)),
                    "human_reviews": sum(1 for r in rows.values() if r.get("method") == "ws_cascade_human"
                                         and r.get("strong_queried")),
                    "llm_calls": calls, "calls_per_item": round(calls / len(rows), 2),
                    "input_tokens": tin, "output_tokens": tout,
                    "usd_list": round(tin / 1e6 * pin + tout / 1e6 * pout, 4),
                    "usd_per_correct_released": round((tin / 1e6 * pin + tout / 1e6 * pout) / max(1, rel_ok), 5)}

    rng = random.Random(20261007)
    items = sorted(set.intersection(*[set(v) for v in data.values()])) if data else []
    strata = defaultdict(lambda: defaultdict(list))
    for i in items:
        r = next(iter(data.values()))[i]
        strata[(r["dataset"], norm_group(r.get("group")))][borrower_of(i, key)].append(i)
    base = {k: list(v.values()) for k, v in strata.items()}

    def stat(sample_items, m, kind):
        rs = [data[m][i] for i in sample_items]
        if kind == "released_correct":
            ok = [r for r in rs if r["truth"] is False]
            return sum(1 for r in ok if r["accepted"] is True) / max(1, len(ok))
        er = [r for r in rs if r["truth"]]
        return sum(1 for r in er if r["accepted"] is True) / max(1, len(er))

    diffs = {}
    if "exec_P" in data:
        for m in data:
            if m == "exec_P":
                continue
            out = {}
            for kind in ("released_correct", "errors_accepted"):
                allit = [i for v in base.values() for b in v for i in b]
                point = stat(allit, "exec_P", kind) - stat(allit, m, kind)
                bs = []
                for _ in range(B):
                    smp = [i for v in base.values() for b in (rng.choice(v) for _ in v) for i in b]
                    bs.append(stat(smp, "exec_P", kind) - stat(smp, m, kind))
                bs.sort()
                out[kind] = {"P_minus_X": round(point, 4), "ci95": [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)]}
            diffs[m] = out

    gaps = {}
    for m in data:
        g = {}
        for ds in sorted({d for d, _ in base}):
            vals = {}
            for grp in ("female", "male"):
                its = [i for b in base.get((ds, grp), []) for i in b if data[m][i]["truth"] is False]
                vals[grp] = (sum(1 for i in its if data[m][i]["accepted"] is not True), len(its))
            g[ds] = {k: f"{v[0]}/{v[1]}" for k, v in vals.items()}
            g[ds]["male_minus_female_withheld"] = round(vals["male"][0] / max(1, vals["male"][1])
                                                        - vals["female"][0] / max(1, vals["female"][1]), 4)
        gaps[m] = g

    out = {"note": "DEVELOPMENT: 245 agreed M1 items (36 borrowers, 9 per dataset x group), labels before adjudication; "
                   "checker executable/1.3 and contract rules developed on these items. Withheld = rejected or deferred.",
           "judge": a.judge, "price_usd_per_Mtok": [pin, pout], "methods": table,
           "paired_P_minus_X_cluster_bootstrap": diffs, "group_withholding_by_dataset": gaps,
           "n_common_items": len(items)}
    out["tag"] = a.tag or "v1.3 (paid run as first executed)"
    out["labels"] = a.labels
    (root / "m1" / f"phaseA_{a.judge}{('_' + a.tag) if a.tag else ''}{'_final' if a.labels == 'final' else ''}.json").write_text(json.dumps(out, indent=1))
    print(f"{'method':34} {'err acc':>8} {'corr rej':>9} {'defer':>5} {'corr rel':>9} {'calls':>6} {'$':>7}")
    for m, v in table.items():
        print(f"{v['method']:34} {v['errors_accepted']:>8} {v['correct_rejected']:>9} {v['deferred']:>5} "
              f"{v['correct_released']:>9} {v['llm_calls']:>6} {v['usd_list']:>7}")
    for m, d in diffs.items():
        print(f"P - {LABEL[m]:30} released_correct {d['released_correct']}  errors_accepted {d['errors_accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

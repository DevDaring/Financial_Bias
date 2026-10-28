"""
Zero-cost mechanism evidence (development): replay the objections that three
generic judges ALREADY raised on the M1 candidates (judge/2.0, cached in
Submission/m1/records.jsonl) through executable witness checking, and score the
resulting decisions against the annotators' agreed labels.

What this is: an offline approximation of the proposed method P built from cached
calls. The judges saw the whole text with the generic prompt, not P's residual-claim
prompt or witness schema, so P itself must still be run live. What it can show:

  1. objection outcomes: how many generic-judge objections to CORRECT explanations
     the recomputation disproves, and how many objections to WRONG explanations it
     verifies (objection-level confusion, by judge);
  2. decision-level effect of each layer on the same judge outputs:
        judge alone | code + judge (A1: no objection checks) |
        code + judge + plausibility checks (A2) | code + judge + execution (P-replay)
     A dismissed objection cannot be reassessed offline: those items are reported as
     unresolved (withheld) - a conservative choice for coverage;
  3. per-group false rejection with borrower-cluster bootstrap intervals for the
     male - female difference (items from the same borrower are resampled together).

Output: Submission/m1/replay_objections_dev.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as C  # noqa: E402
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from score_m1 import M1, agreed_natural  # noqa: E402

JUDGES = ["gpt56_luna", "grok420", "sonnet5"]
TYPE_RULE = {"sign_reversal": "R_DIRECTION", "invented_factor": "R_INVENTED", "unsupported_trend": "R_TEMPORAL",
             "unsupported_causal_or_action": "R_CAUSAL", "omission_of_essential_driver": "R_OMISSION"}
SEED = 20261007
B = 2000


def norm_group(g) -> str:
    """Taiwan codes sex as male/female, Home Credit CODE_GENDER as M/F: one audit label."""
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(g or "").strip().lower(), "other")


def to_witness(v: dict) -> W.Witness:
    t = str(v.get("type", "")).lower().replace(" ", "_")
    det = str(v.get("detail") or "")
    rule = TYPE_RULE.get(t)
    if t == "numeric_or_unit_error":
        rule = "R_SCALE" if any(k in det.lower() for k in ("log-odds", "log odds", "percentage point", "probability change",
                                                          "as a percentage", "scale")) else "R_VALUE"
    return W.Witness(rule=rule or "R_INVENTED", quote=str(v.get("quote") or ""),
                     evidence_ids=[str(x) for x in (v.get("evidence_ids") or [])], reason=det,
                     operation=W.RULE_OP.get(rule or "", "SEMANTIC"))


def plausibility(w: W.Witness, text, eo) -> bool:
    o = C.check_objection(C.Objection(rule=w.rule, quote=w.quote, evidence_ids=list(w.evidence_ids),
                                      reason=w.reason), text, eo)
    return bool(o.valid)


def decide(judge_acc, ws, tx, layer: str):
    """accept | reject | unresolved for one layer on one item."""
    if layer == "judge":
        return "accept" if judge_acc else "reject"
    if tx.contradictions:
        return "reject"
    if not tx.residual_claims and all(o.status == W.SUPPORT for o in tx.text_ops):
        return "accept"
    if judge_acc:
        return "accept"
    if layer == "A1":
        return "reject"
    if layer == "A2":
        return "reject" if any(x["plausible"] for x in ws) else "unresolved"
    return "reject" if any(x["outcome"] in ("verified", "semantic_plausible") for x in ws) else "unresolved"


def rate(rows, truth, dec):
    xs = [r for r in rows if r["truth"] == truth]
    k = sum(1 for r in xs if r[dec] == ("accept" if truth else "reject"))
    return k, len(xs)


def cluster_boot_diff(rows, key, rng):
    """male - female false-reject rate on correct items. Borrowers are resampled with
    replacement WITHIN each dataset x group stratum (items of one borrower move together)."""
    strata = defaultdict(lambda: defaultdict(list))        # (dataset, group) -> borrower -> [rejected?]
    for r in rows:
        if not r["truth"] and r["group"] in ("male", "female"):
            strata[(r["dataset"], r["group"])][r["borrower"]].append(r[key] == "reject")

    def est(sample, g):
        xs = [x for (ds, gg), bs in sample.items() if gg == g for v in bs for x in v]
        return sum(xs) / len(xs) if xs else float("nan")
    base = {k2: list(v.values()) for k2, v in strata.items()}
    point = est(base, "male") - est(base, "female")
    diffs = []
    for _ in range(B):
        s = {k2: [rng.choice(v) for _ in v] for k2, v in base.items()}
        diffs.append(est(s, "male") - est(s, "female"))
    diffs.sort()
    by_ds = {}
    for ds in sorted({d for d, _ in base}):
        sub = {k2: v for k2, v in base.items() if k2[0] == ds}
        by_ds[ds] = round(est(sub, "male") - est(sub, "female"), 4)
    return {"diff": round(point, 4), "ci95": [round(diffs[int(0.025 * B)], 4), round(diffs[int(0.975 * B)], 4)],
            "by_dataset": by_ds,
            "n_borrowers": {f"{d}|{g}": len(v) for (d, g), v in sorted(base.items())},
            "n_items": {f"{d}|{g}": sum(len(x) for x in v) for (d, g), v in sorted(base.items())}}


def main() -> int:
    latest = {}
    for line in open(M1 / "records.jsonl"):
        r = json.loads(line)
        latest[(r["borrower_id"], r["arm"], r["family"])] = r
    agreed = agreed_natural()
    rows, obj = [], Counter()
    examples = defaultdict(list)
    for i, (k, truth) in sorted(agreed.items()):
        rec = latest.get((k["borrower_id"], k["arm"], k["family"]))
        if not rec:
            continue
        eo = eo_from(rec["evidence"])
        text = W.clean(rec["text"])
        tx = W.execute_text(text, eo)
        row = {"item_id": i, "truth": bool(truth), "group": norm_group(k.get("group") or rec.get("group")),
               "borrower": rec["borrower_id"], "dataset": rec["dataset"], "family": rec["family"],
               "code_reject": bool(tx.contradictions),
               "code_detail": [o.detail for o in tx.contradictions][:2]}
        for j in JUDGES:
            jv = (rec.get("judges") or {}).get(j) or {}
            if jv.get("error") or not isinstance(jv.get("accepted"), bool):
                for layer in ("judge", "A1", "A2", "P"):
                    row[f"{j}:{layer}"] = "unresolved"
                continue
            ws = []
            for v in jv.get("violations") or []:
                w = W.execute_witness(to_witness(v), text, eo, tx)
                pl = plausibility(w, text, eo)
                ws.append({"rule": w.rule, "outcome": w.outcome, "plausible": pl, "quote": w.quote[:120],
                           "detail": (w.execution or {}).get("detail", "")[:120], "problems": w.problems[:3]})
                obj[(j, "error" if truth else "correct", w.outcome)] += 1
                obj[(j, "error" if truth else "correct", "plausible" if pl else "implausible")] += 1
                if len(examples[(truth, w.outcome)]) < 4:
                    examples[(truth, w.outcome)].append({"item": i, "judge": j, "rule": w.rule, "quote": w.quote[:140],
                                                         "execution": (w.execution or {}).get("detail", ""),
                                                         "problems": w.problems[:2]})
            for layer in ("judge", "A1", "A2", "P"):
                row[f"{j}:{layer}"] = decide(jv["accepted"], ws, tx, layer)
            row[f"{j}:witnesses"] = ws
        rows.append(row)

    rng = random.Random(SEED)
    table = {}
    for j in JUDGES:
        for layer in ("judge", "A1", "A2", "P"):
            key = f"{j}:{layer}"
            fa, ne = rate(rows, True, key)
            fr, nc = rate(rows, False, key)
            un = sum(1 for r in rows if r[key] == "unresolved")
            unc = sum(1 for r in rows if r[key] == "unresolved" and not r["truth"])
            cov = sum(1 for r in rows if not r["truth"] and r[key] == "accept")
            g = {}
            for grp in ("female", "male"):
                fr_g, nc_g = rate([r for r in rows if r["group"] == grp], False, key)
                fa_g, ne_g = rate([r for r in rows if r["group"] == grp], True, key)
                cv_g = sum(1 for r in rows if r["group"] == grp and not r["truth"] and r[key] == "accept")
                g[grp] = {"false_accept": f"{fa_g}/{ne_g}", "false_reject": f"{fr_g}/{nc_g}",
                          "coverage_on_correct": f"{cv_g}/{nc_g}"}
            table[key] = {"false_accept_a": f"{fa}/{ne}", "false_reject_b": f"{fr}/{nc}",
                          "unresolved": un, "unresolved_on_correct": unc,
                          "coverage_on_correct": f"{cov}/{nc}", "by_group": g,
                          "male_minus_female_b": cluster_boot_diff(rows, key, rng)}
    objections = defaultdict(dict)
    for (j, t, o), n in sorted(obj.items()):
        objections[j].setdefault(t, {})[o] = n
    out = {"note": "DEVELOPMENT replay of cached judge/2.0 objections through executable witnesses "
                   "(executable/1.0); agreed natural M1 items before adjudication; extraction rules were "
                   "developed on these items. Not a run of the proposed prompt.",
           "n_items": len(rows), "n_borrowers": len({r['borrower'] for r in rows}),
           "objections": objections, "decisions": table,
           "examples": {f"{'error' if t else 'correct'}|{o}": v for (t, o), v in examples.items()}}
    # agreed-CORRECT items that the contract says are wrong: code rejects, or >= 2 judges raise an objection
    # that survives execution. Not counted as verifier errors until the author adjudicates them.
    conflicts = []
    for r in rows:
        if r["truth"]:
            continue
        code_rej = r["code_reject"]
        surv = [j for j in JUDGES if any(x["outcome"] in ("verified", "semantic_plausible")
                                         for x in r.get(f"{j}:witnesses", []))]
        if code_rej or len(surv) >= 2:
            conflicts.append({"item_id": r["item_id"], "dataset": r["dataset"], "family": r["family"],
                              "code_rejects": code_rej, "judges_with_surviving_objection": surv,
                              "objections": {j: [x for x in r.get(f"{j}:witnesses", [])
                                                 if x["outcome"] in ("verified", "semantic_plausible")][:2]
                                             for j in surv}})
    out["contract_vs_label_conflicts"] = conflicts
    import csv
    with open(M1 / "contract_label_conflicts_for_adjudication.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["item_id", "dataset", "producer", "code_rejects", "judges", "objection_rules", "quotes",
                     "AUTHOR_DECISION (error / correct / unresolved)", "AUTHOR_NOTE"])
        for c in conflicts:
            obs = [x for v in c["objections"].values() for x in v]
            wr.writerow([c["item_id"], c["dataset"], c["family"], c["code_rejects"],
                         ";".join(c["judges_with_surviving_objection"]), ";".join(sorted({x["rule"] for x in obs})),
                         " | ".join(x["quote"] for x in obs[:3]), "", ""])
    print(f"contract-vs-label conflicts for adjudication: {len(conflicts)}")
    (M1 / "replay_objections_dev.json").write_text(json.dumps(out, indent=1, default=str))
    for key, v in table.items():
        print(f"{key:20} a={v['false_accept_a']:7} b={v['false_reject_b']:8} unres={v['unresolved']:3} "
              f"cov={v['coverage_on_correct']:8} m-f b={v['male_minus_female_b']['diff']:+.3f} "
              f"{v['male_minus_female_b']['ci95']} by ds {v['male_minus_female_b']['by_dataset']}")
    print(json.dumps(objections, indent=0)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

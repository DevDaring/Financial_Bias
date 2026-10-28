"""
Analyses requested in the review (all post hoc, on the fresh v2 test; the prespecified claims are unchanged).

  attribution   each wrong narrative released by ReCheck-P: path (code alone / judge accepted / accepted on
                reassessment) x raters' error type x whether a claim was outside the executable language
  strict        a stricter policy for causal and invented-factor claims: withhold a narrative whose release
                rests on judge support for such a claim (replayed exactly from the frozen records)
  fairness      by dataset x sex: wrong release, correct withheld, deferral and dispute rates, differences with CIs
  coverage      grammar coverage: claims settled by code, per operation and per routing reason
  ablation      leave-one-gate-rule-out (exact counterfactual: frozen outcomes except the narratives whose code
                decision changes), a repeat of the frozen system, a second reassessment, objections unchecked
                (A1) and the deterministic Pratama-Tseng audit
  labels        label rules: provided, both raters must flag an error, and leave-one-rater-pair-out (pairs are
                reported as anonymous indices; rater identities are never read or written)

Reads only label columns (never the annotator, checker or participant tabs).

    python analyze_review2.py
Output: Submission/locked/review2_analysis.json
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import analyze_locked as AL

B = 4000
LAMBDAS = (1, 2, 5)
LABELS_DIR = AL.SUB / "packs" / "Annotated" / "IDBI_Annotator_Checks"
ABL = AL.LK / "fresh_v2_ablation"
JUDGE = "gptoss120b"
SEMANTIC = {"R_INVENTED", "R_CAUSAL"}
GATE = {"R_FIELD_IDENTITY", "R_NEGATION_SCOPE", "R_CODE_SCOPE", "R_TARGET"}


def jl(p):
    return {json.loads(l)["item_id"]: json.loads(l) for l in open(p)} if Path(p).exists() else {}


INDEPENDENT = AL.LK / "adjudication_fresh_v2_independent.csv"   # protocol adjudicator (primary)
FIRST_AUTHOR = AL.LK / "adjudication_fresh_v2.csv"


def setup(mode="provided", adjudication=None):
    AL.ADJ_MODE = mode
    AL.RUNS = AL.LK / "fresh_v2_runs"
    adj = adjudication or (INDEPENDENT if INDEPENDENT.exists() else FIRST_AUTHOR)
    AL.CFG.update(key=AL.SUB / "annotation_key_fresh_v2.jsonl", sheet="FRESH_annotation_sheet_",
                  adjudication=adj, records=AL.LK / "fresh_natural_v2_final.jsonl")
    lab, dis, _ = AL.labels(LABELS_DIR)
    return {i: v for i, v in lab.items() if v["arm"] != "template"}, dis


class Boot:
    def __init__(self, nar, seed=20261013):
        strata = defaultdict(list)
        for i, v in nar.items():
            strata[v["stratum"]].append(i)
        rng = random.Random(seed)
        self.nar = nar
        self.samples = [[i for v in strata.values() for i in (rng.choice(v) for _ in v)] for _ in range(B)]

    def cw(self, rows, items):
        c = sum(1 for i in items if not self.nar[i]["truth"] and rows.get(i, {}).get("accepted") is True)
        w = sum(1 for i in items if self.nar[i]["truth"] and rows.get(i, {}).get("accepted") is True)
        nc = sum(1 for i in items if not self.nar[i]["truth"])
        nw = sum(1 for i in items if self.nar[i]["truth"])
        return c, w, nc, nw

    def diff(self, A, Bm):
        """A minus B: wrong-release and correct-release rate differences (points) and net benefit per 100."""
        def stat(items):
            ca, wa, nc, nw = self.cw(A, items)
            cb, wb, _, _ = self.cw(Bm, items)
            n = len(items)
            return ([100 * (wa - wb) / max(1, nw), 100 * (ca - cb) / max(1, nc)] +
                    [100 * ((ca - l * wa) - (cb - l * wb)) / n for l in LAMBDAS])
        pt = stat(list(self.nar))
        bs = [stat(s) for s in self.samples]
        names = ["wrong_release", "correct_release"] + [f"U{l}" for l in LAMBDAS]
        out = {}
        for k, nm in enumerate(names):
            col = sorted(b[k] for b in bs)
            out[nm] = {"diff": round(pt[k], 2), "ci95": [round(col[int(.025 * B)], 2), round(col[int(.975 * B)], 2)]}
        ca, wa, nc, nw = self.cw(A, list(self.nar))
        out["counts"] = {"correct_released": ca, "wrong_released": wa, "n_correct": nc, "n_wrong": nw}
        return out


def residual_tags(row):
    return {t for c in (row.get("detail") or {}).get("execution", {}).get("claims", []) for t in (c.get("residual") or [])}


def attribution(nar, P):
    out = Counter()
    types = Counter()
    for i, v in nar.items():
        r = P.get(i)
        if not (r and v["truth"] and r.get("accepted") is True):
            continue
        why = (r["detail"] or {}).get("reason", "")
        path = ("code alone" if why == "all_claims_executed_supported" else
                "judge accepted after reassessment" if "reassessment" in why else "judge accepted")
        tags = residual_tags(r)
        sem = "semantic claim (causal/invented) judged" if tags & SEMANTIC else (
            "gate-routed claim judged" if tags & GATE else "other residual" if tags else "no residual claim")
        out[(path, sem if path != "code alone" else "-")] += 1
        for t in v.get("types") or ["(no type recorded)"]:
            types[(path, t)] += 1
    return {"by_path": {f"{a} | {b}": n for (a, b), n in out.most_common()},
            "by_path_and_type": {f"{a} | {b}": n for (a, b), n in types.most_common()}}


def strict(P):
    """Withhold a release that rests on judge support for a causal or invented-factor claim."""
    S = {}
    for i, r in P.items():
        r2 = dict(r)
        why = (r.get("detail") or {}).get("reason", "")
        if r.get("accepted") is True and why != "all_claims_executed_supported" and residual_tags(r) & SEMANTIC:
            r2["accepted"] = None
        S[i] = r2
    return S


def fairness(nar, runs):
    out = {}
    for name, rows in runs.items():
        cell = {}
        for ds in ("taiwan", "home_credit"):
            g = {}
            for grp in sorted({v["group"] for v in nar.values() if v["dataset"] == ds}):
                its = [i for i, v in nar.items() if v["dataset"] == ds and v["group"] == grp and i in rows]
                w = [i for i in its if nar[i]["truth"]]
                c = [i for i in its if not nar[i]["truth"]]
                g[grp] = {"n": len(its),
                          "wrong_released": round(100 * sum(rows[i].get("accepted") is True for i in w) / max(1, len(w)), 1),
                          "correct_withheld": round(100 * sum(rows[i].get("accepted") is not True for i in c) / max(1, len(c)), 1),
                          "deferred": round(100 * sum(rows[i].get("accepted") is None for i in its) / max(1, len(its)), 1)}
            cell[ds] = g
        out[name] = cell
    # differences between groups with bootstrap CIs, ReCheck-P
    rng = random.Random(7)
    P = runs["ReCheck-P"]
    gaps = {}
    for ds in ("taiwan", "home_credit"):
        grs = sorted({v["group"] for v in nar.values() if v["dataset"] == ds})
        by = {g: [i for i, v in nar.items() if v["dataset"] == ds and v["group"] == g and i in P] for g in grs}

        def rates(sel):
            res = {}
            for g, its in sel.items():
                w = [i for i in its if nar[i]["truth"]]
                c = [i for i in its if not nar[i]["truth"]]
                res[g] = (100 * sum(P[i].get("accepted") is True for i in w) / max(1, len(w)),
                          100 * sum(P[i].get("accepted") is not True for i in c) / max(1, len(c)))
            return res
        pt = rates(by)
        bs = [rates({g: [rng.choice(its) for _ in its] for g, its in by.items()}) for _ in range(B)]
        a, b = grs
        for k, nm in enumerate(("wrong_released", "correct_withheld")):
            col = sorted(x[a][k] - x[b][k] for x in bs)
            gaps[f"{ds}|{nm}|{a}-{b}"] = {"diff": round(pt[a][k] - pt[b][k], 1),
                                          "ci95": [round(col[int(.025 * B)], 1), round(col[int(.975 * B)], 1)]}
    out["gaps_ReCheck-P"] = gaps
    return out


def coverage(P):
    claims = Counter()
    ops = Counter()
    res = Counter()
    narr = Counter()
    for r in P.values():
        d = r.get("detail") or {}
        narr[d.get("reason", "")] += 1
        for c in d.get("execution", {}).get("claims", []):
            claims[c["status"]] += 1
            for t in c.get("residual") or []:
                res[t] += 1
            for o in c.get("ops", []):
                ops[(o["op"], o["status"])] += 1
    n = sum(claims.values())
    return {"claims": dict(claims), "n_claims": n,
            "settled_by_code_pct": round(100 * (claims["support"] + claims["contradiction"]) / max(1, n), 1),
            "residual_reasons": dict(res.most_common()), "ops": {f"{a}|{b}": k for (a, b), k in sorted(ops.items())},
            "narrative_reasons": dict(narr.most_common())}


def ablation(boot, P, B1F):
    out = {}
    variants = {}
    for d in sorted(ABL.glob("gate_off_*")):
        alt = jl(d / f"exec_P__{JUDGE}.jsonl")
        changed = [l.strip() for l in open(d / "changed_items.txt") if l.strip()] if (d / "changed_items.txt").exists() else []
        if changed and len(alt) < len(changed):
            continue                       # still running
        variants[d.name] = dict(P, **{i: alt[i] for i in changed if i in alt})
        out.setdefault("changed_items", {})[d.name] = len(changed)
    for name, sub, meth in (("repeat", "repeat", "exec_P"), ("A1_objections_unchecked", "A1", "exec_A1"),
                            ("pt_audit", "pt_audit", "pt_audit")):
        rows = jl(ABL / sub / f"{meth}__{JUDGE}.jsonl")
        if len(rows) >= len(boot.nar):
            variants[name] = rows
    r2 = jl(ABL / "reassess2" / f"exec_P2__{JUDGE}.jsonl")
    todo = [i for i, r in P.items() if str((r.get("detail") or {}).get("reason", "")).endswith("unresolved after reassessment")]
    if r2 and len(r2) >= len(todo):
        variants["reassess2"] = dict(P, **r2)
    variants["strict_semantic"] = strict(P)
    for name, rows in variants.items():
        out[name] = {"vs_frozen_P": boot.diff(rows, P), "vs_B1F": boot.diff(rows, B1F)}
    return out


def label_rules(P, B1F):
    out = {}
    for mode in ("provided", "first_author", "all_yes"):
        nar, _ = (setup_all_yes() if mode == "all_yes" else
                  setup("provided", FIRST_AUTHOR) if mode == "first_author" else setup("provided"))
        out[mode] = Boot(nar).diff(P, B1F)
        out[mode]["n"] = len(nar)
    nar, _ = setup("provided")
    pair = {i: tuple(sorted(v["raters"])) for i, v in nar.items()}
    order = {p: f"pair {k + 1}" for k, p in enumerate(sorted(set(pair.values())))}   # anonymous indices
    loo = {}
    for p, lab in order.items():
        sub = {i: v for i, v in nar.items() if pair[i] != p}
        loo[lab] = Boot(sub).diff(P, B1F)
        loo[lab]["n"] = len(sub)
    out["leave_one_pair_out"] = loo
    return out


def setup_all_yes():
    """A disagreement counts as an error only if both raters flagged one (i.e. disagreements are correct)."""
    nar, dis = setup("any_yes")
    for i, v in nar.items():
        if v.get("basis") == "any_rater_yes":
            v["truth"] = False
            v["basis"] = "both_raters_yes"
    return nar, dis


def main() -> int:
    nar, _ = setup("provided")
    P = jl(AL.LK / "fresh_v2_runs" / f"exec_P__{JUDGE}.jsonl")
    B1F = jl(AL.LK / "fresh_v2_runs" / f"facts_judge__{JUDGE}.jsonl")
    boot = Boot(nar)
    res = {"attribution": attribution(nar, P),
           "fairness": fairness(nar, {"ReCheck-P": P, "B1F": B1F}),
           "coverage": coverage(P),
           "ablation": ablation(boot, P, B1F),
           "labels": label_rules(P, B1F)}
    out = AL.LK / "review2_analysis.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: (v if k != "ablation" else {a: b["vs_frozen_P"] for a, b in v.items() if isinstance(b, dict)
                                                   and "vs_frozen_P" in b}) for k, v in res.items()}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

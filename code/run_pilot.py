"""
The bounded pilot from plan §9.

  "Use about 60 development borrowers balanced across supported groups/risk
   bands, two producers, and two judges. Build the evidence template and
   independent rubric; compare grounded and typed-evidence prose, inspect
   natural failures, and test seeded violations. Determine whether verification
   can be independently evaluated, useful content survives, and there is a
   plausible benefit over templates."

This runs end to end on a real corpus and writes three artifacts:

  pilot_records.jsonl   one row per (borrower, arm, producer) with verdicts
  pilot_report.json     group metrics, policy sweep, decomposition, cost
  llm_trace.jsonl       every attempt, with the immutable model id used

What it does NOT do: supply ground-truth labels. Structural violations are a
proxy so the plumbing can be exercised; E1 requires independent annotation,
and the report marks the labels as PROXY so no number here is mistaken for a
finished result.

Usage
    python run_pilot.py --dataset sba --n 60
    python run_pilot.py --dataset sba --n 12 --dry-run     # no LLM calls
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "saarthi" / "training"))

import registry
from evidence import (EvidenceObject, EvidenceType, map_feature_to_concept,
                      new_item)
from gateway import TRACE_PATH, trace_report
from generate import ARMS, Candidate, generate
from policy import (Record, decompose, evaluate, policy_family, report,
                    select_policy, summarise)
from substrate import Substrate, build
from verify import combined_check, semantic_check, structural_check

OUT = Path(__file__).resolve().parents[2] / "Submission"
OUT.mkdir(exist_ok=True)

# documented feature -> concept maps (deterministic, per dataset)
CONCEPTS: Dict[str, Dict[str, str]] = {
    "sba": {"grappv": "LEVERAGE_LEVEL_HIGH", "sba_appv": "LEVERAGE_LEVEL_HIGH",
            "disbursementgross": "LEVERAGE_LEVEL_HIGH",
            "term": "TENURE_ASSOCIATION", "noemp": "REVENUE_LEVEL_LOW",
            "naics": "SECTOR_ASSOCIATION", "newexist": "TENURE_ASSOCIATION",
            "urbanrural": "SECTOR_ASSOCIATION"},
    "home_credit": {"amt_credit": "LEVERAGE_LEVEL_HIGH",
                    "amt_income_total": "REVENUE_LEVEL_LOW",
                    "amt_annuity": "LIQUIDITY_LEVEL_LOW",
                    "ext_source": "BUREAU_SCORE_LEVEL",
                    "days_birth": "UNMAPPED_FEATURE"},
    "taiwan": {"limit_bal": "LEVERAGE_LEVEL_HIGH", "pay_": "REPAYMENT_HISTORY_ADVERSE",
               "bill_amt": "LEVERAGE_LEVEL_HIGH", "pay_amt": "LIQUIDITY_LEVEL_LOW"},
}


# ---------------------------------------------------------------------------
def load_corpus(dataset: str, cap: int = 60_000):
    """Raw frame, target, audit groups. Uses the existing adapters."""
    import adapters
    c = adapters.load(dataset)
    X, y = c.native.reset_index(drop=True), c.y.reset_index(drop=True)
    prot = c.protected.reset_index(drop=True) if c.protected is not None else pd.DataFrame()
    if len(X) > cap:
        rng = np.random.RandomState(20260502)
        keep = rng.choice(len(X), cap, replace=False)
        X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
        # v1 indexed prot by X.index AFTER reset_index (0..cap-1), which paired
        # each sampled borrower with the wrong protected attribute. Use `keep`.
        if len(prot):
            prot = prot.iloc[keep].reset_index(drop=True)
    if prot.shape[1]:
        col = prot.columns[0]
        groups = prot[col].astype(str)
        top = groups.value_counts().head(2).index.tolist()
        groups = groups.where(groups.isin(top), "other")
        gname = col
    else:
        groups, gname = pd.Series(["all"] * len(X)), "none"
    return X, y, groups, gname


def shap_drivers(sub: Substrate, X_row: pd.DataFrame, k: int = 4):
    """Top-k attributions on the RAW MARGIN - the scale SHAP explains (P1)."""
    try:
        import shap
        Z = sub.prep.transform(X_row)
        ex = shap.TreeExplainer(sub.model)
        vals = ex.shap_values(Z)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        v = np.asarray(vals)[0]
        order = np.argsort(-np.abs(v))[:k]
        return [(Z.columns[i], float(v[i])) for i in order if abs(v[i]) > 1e-9]
    except Exception:
        return []


def build_evidence(dataset: str, bid: str, X_row: pd.DataFrame,
                   sub: Substrate, group: str) -> EvidenceObject:
    eo = EvidenceObject(borrower_id=bid, dataset=dataset, schema_version="evidence/1.1",
                        prediction_time="documented decision time",
                        group_attributes={"audit_group": group},
                        provenance={"model_id": sub.model_id,
                                    "calibrator_id": sub.calibrator_id,
                                    "transformer_id": sub.prep.transformer_id,
                                    "split": sub.split_sizes})
    doc = CONCEPTS.get(dataset, {})
    row = X_row.iloc[0]
    n = 0
    for col in X_row.columns[:8]:
        v = row[col]
        if pd.isna(v):
            continue
        n += 1
        eo.add(new_item(f"f{n}", EvidenceType.OBSERVED_FACT, source_field=str(col),
                        value=(float(v) if isinstance(v, (int, float, np.number))
                               else str(v)),
                        observed_at="documented decision time",
                        concept=map_feature_to_concept(str(col), doc)))
    for i, (feat, val) in enumerate(shap_drivers(sub, X_row), start=1):
        eo.add(new_item(f"a{i}", EvidenceType.MODEL_ATTRIBUTION,
                        model_id=sub.model_id,
                        transformer_id=sub.prep.transformer_id,
                        feature_group=str(feat), contribution=round(val, 5),
                        direction="increases_risk" if val > 0 else "decreases_risk",
                        explained_scale=sub.attribution_scale,
                        reference_background=sub.reference_background,
                        concept=map_feature_to_concept(str(feat), doc)))
    pd_hat = float(sub.predict_pd(X_row)[0])
    eo.add(new_item("p1", EvidenceType.MODEL_PREDICTION, calibrated_pd=round(pd_hat, 5),
                    target_definition=f"{dataset} documented default label",
                    horizon="documented outcome horizon (NOT assumed 12 months)",
                    model_id=sub.model_id, calibrator_id=sub.calibrator_id))
    if not eo.by_type(EvidenceType.MODEL_ATTRIBUTION):
        eo.add(new_item("u1", EvidenceType.UNSUPPORTED,
                        reason="attribution unavailable for this record",
                        concept="NO_SUPPORTED_ESTIMATE"))
    return eo


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sba")
    ap.add_argument("--n", type=int, default=60, help="development borrowers")
    ap.add_argument("--producers", nargs="*", default=registry.PILOT_PRODUCERS)
    ap.add_argument("--judges", nargs="*", default=registry.PILOT_JUDGES)
    ap.add_argument("--arms", nargs="*", default=["template", "grounded", "typed"])
    ap.add_argument("--model-kind", default="gbdt")
    ap.add_argument("--dry-run", action="store_true", help="template arm only")
    ap.add_argument("--seed", type=int, default=20260502)
    a = ap.parse_args()

    t0 = time.time()

    # Hard gate BEFORE any spend: an experiment whose judges are not
    # independent of its producers produces nothing worth paying for.
    if not a.dry_run:
        print("[0/5] probing API keys", flush=True)
        from providers import prune_dead_keys
        prune_dead_keys()
        panel = registry.assert_independent_panel(
            a.producers, a.judges, require_full_matrix=True)
        print(f"[0/5] judge panel independent: "
              f"producers={panel['producers']} judges={panel['judges']} "
              f"({panel['cross_publisher_pairs']} cross-publisher cells)", flush=True)

    print(f"[1/5] corpus {a.dataset}", flush=True)
    X, y, groups, gname = load_corpus(a.dataset)
    print(f"      n={len(X):,}  positives={int(y.sum()):,}  audit group='{gname}' "
          f"levels={sorted(groups.unique())}")

    print(f"[2/5] substrate ({a.model_kind}, 3-way split, P0 fixes on)", flush=True)
    sub = build(X, y, name=a.dataset, model_kind=a.model_kind,
                calibrator_kind="isotonic", groups=groups, threshold=0.5,
                seed=a.seed)
    print(f"      {sub.model_id}  test AUC={sub.metrics['auc']:.4f} "
          f"ECE={sub.metrics['ece']:.4f} n_test={sub.metrics['n']:,}")
    for g, d in sub.subgroup.items():
        print(f"        group {g:<10} n={d['n']:<6} base={d['base_rate']:<7} "
              f"TPR={d['tpr']} FPR={d['fpr']}")

    # sample development borrowers from the TEST fold, balanced by group+band
    test_idx = np.array(sub.metrics["_test_index"])
    pd_test = sub.predict_pd(X.iloc[test_idx])
    band = pd.cut(pd_test, [-.01, .2, .5, 1.01], labels=["low", "med", "high"])
    frame = pd.DataFrame({"idx": test_idx, "group": groups.iloc[test_idx].values,
                          "band": band})
    per = max(1, a.n // max(1, frame.groupby(["group", "band"], observed=True).ngroups))
    picked = (frame.groupby(["group", "band"], observed=True, group_keys=False)
              .apply(lambda d: d.sample(min(len(d), per), random_state=a.seed)))
    sel = picked["idx"].tolist()[:a.n]
    print(f"[3/5] {len(sel)} development borrowers "
          f"(balanced over {frame.groupby(['group','band'], observed=True).ngroups} group x band cells)")

    arms = ["template"] if a.dry_run else a.arms
    producers = [] if a.dry_run else a.producers
    print(f"[4/5] generate+verify  arms={arms} producers={producers} judges={a.judges}",
          flush=True)

    records: List[Record] = []
    rows: List[dict] = []
    for n_done, i in enumerate(sel, 1):
        Xi = X.iloc[[i]]
        g = str(groups.iloc[i])
        eo = build_evidence(a.dataset, f"{a.dataset}-{i}", Xi, sub, g)

        for arm in arms:
            fams = [""] if arm == "template" else producers
            for fam in fams:
                cand = generate(eo, arm, fam, temperature=0.2, seed=a.seed,
                                experiment=f"pilot:{a.dataset}")
                st = structural_check(cand, eo)
                jverdicts = {}
                if not a.dry_run and cand.ok:
                    for j in a.judges:
                        if arm != "template" and registry.get(j).publisher == \
                           registry.get(fam).publisher:
                            continue          # gateway would refuse; skip cleanly
                        v = semantic_check(cand, eo, j,
                                           experiment=f"pilot:{a.dataset}")
                        jverdicts[j] = v
                acc = [v.accepted for v in jverdicts.values()]
                judge_accept = (all(acc) if acc else None)
                records.append(Record(
                    borrower_id=eo.borrower_id, group=g, arm=arm,
                    family=fam or "template", released=True,
                    # PROXY label - E1 replaces this with independent annotation
                    loss=(not st.accepted) if cand.ok else True,
                    informative=len(cand.text.split()) >= 25,
                    completeness=eo.completeness(),
                    structural_ok=st.accepted,
                    judge_accept=judge_accept,
                    judge_violations=sum(v.n for v in jverdicts.values()),
                    latency_ms=cand.latency_ms,
                    template_correct=True, template_informative=True))
                rows.append({
                    "borrower_id": eo.borrower_id, "group": g, "arm": arm,
                    "family": cand.family, "model_id": cand.model_id,
                    "provider": cand.provider, "route_index": cand.route_index,
                    "error": cand.error, "text": cand.text,
                    "claims": cand.claims,
                    "evidence_fingerprint": eo.fingerprint(),
                    "evidence": eo.to_dict(include_groups=True),
                    "completeness": eo.completeness(),
                    "structural_accepted": st.accepted,
                    "structural_violations": st.violations,
                    "judges": {k: v.to_dict() for k, v in jverdicts.items()},
                    "prompt_tokens": cand.prompt_tokens,
                    "completion_tokens": cand.completion_tokens})
        if n_done % 10 == 0 or n_done == len(sel):
            print(f"      {n_done}/{len(sel)} borrowers  ({time.time()-t0:.0f}s)",
                  flush=True)

    print("[5/5] policy sweep + report", flush=True)
    pols = policy_family()
    chosen, sweep = select_policy(records, pols, eps=0.10, min_coverage=0.50)
    rep = {
        "LABELS": "PROXY (structural check). E1 must replace with independent "
                  "annotation before any reported error rate is credible.",
        "dataset": a.dataset, "n_borrowers": len(sel),
        "audit_group_field": gname,
        "arms": arms, "producers": producers, "judges": a.judges,
        "substrate": {"model_id": sub.model_id, "calibrator_id": sub.calibrator_id,
                      "transformer_id": sub.prep.transformer_id,
                      "metrics": {k: v for k, v in sub.metrics.items()
                                  if k != "_test_index"},
                      "subgroup": sub.subgroup, "splits": sub.split_sizes,
                      "attribution_scale": sub.attribution_scale},
        "model_manifest": registry.manifest(sorted(set(producers + a.judges))),
        "policy_sweep": sweep,
        "selected_policy": chosen.name if chosen else None,
        "per_arm": {},
        "decomposition": decompose(records),
        "llm_trace": trace_report(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    for arm in arms:
        sub_r = [r for r in records if r.arm == arm]
        if sub_r:
            m = evaluate(sub_r, pols[0])
            rep["per_arm"][arm] = {"n": len(sub_r), **summarise(m),
                                   "per_group": {g: v.to_dict() for g, v in m.items()}}

    (OUT / "pilot_records.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in rows))
    (OUT / "pilot_report.json").write_text(json.dumps(rep, indent=2, default=str))

    print(f"\n── pilot summary ({rep['wall_seconds']}s) ──")
    for arm, d in rep["per_arm"].items():
        print(f"  {arm:<10} n={d['n']:<4} minCov={d['min_coverage']:<7} "
              f"worstErr={d['worst_group_error']} minUseful={d['min_useful_rate']}")
    print(f"  selected policy: {rep['selected_policy'] or 'INFEASIBLE'}")
    t = rep["llm_trace"]
    print(f"  llm: {t['successful']}/{t['attempts']} attempts ok, "
          f"{t['off_primary_route']} served off primary route")
    print(f"\n  -> {OUT/'pilot_report.json'}")
    print(f"  -> {OUT/'pilot_records.jsonl'}")
    print(f"  -> {TRACE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

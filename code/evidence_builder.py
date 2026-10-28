"""
Evidence v2 builder (plan §2 P0: evidence construction and feature meanings).

Replaces run_pilot.build_evidence, which showed the first eight raw columns
regardless of what drove the prediction, used adapter-internal names
(PAY_1 does not exist in the source), labelled one-hot dummies as if they
were the borrower's category, and assigned evaluative concepts by field name.

v2 rules
  1. Attributions are TreeSHAP values on the model's raw log-odds margin,
     SUMMED over all transformed columns of one source field (a categorical
     field's one-hot dummies become one attribution for that field). The
     additivity check (base + sum = raw margin) is asserted per borrower.
  2. The facts shown are exactly those needed to interpret the displayed
     drivers: each driver's recorded value, plus the inputs of a derived
     driver. Nothing else.
  3. Names are source names; each fact carries its documented meaning, unit,
     observation period, documented code label, and plain-language aliases.
     A missing value is shown as missing, never as an imputed number.
  4. Concepts are neutral unless a documented value predicate licenses an
     evaluative one (feature_semantics.concept_for).
  5. The prediction states the dataset's documented target and horizon.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import feature_semantics as FS
from evidence import EvidenceItem, EvidenceObject, EvidenceType, new_item

SCHEMA = "evidence/2.0"
TOP_K = 4


class GroupedShap:
    """TreeSHAP on the raw margin, grouped back to source fields."""

    def __init__(self, sub):
        import shap
        self.sub = sub
        self.explainer = shap.TreeExplainer(sub.model)
        ev = self.explainer.expected_value
        self.base = float(ev[1] if np.ndim(ev) and len(np.atleast_1d(ev)) > 1 else np.atleast_1d(ev)[0])
        self.groups = sub.prep.column_groups()

    def explain(self, X_row: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
        Z = self.sub.prep.transform(X_row)
        vals = self.explainer.shap_values(Z)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        v = np.asarray(vals)[0]
        col = {c: float(v[i]) for i, c in enumerate(Z.columns)}
        grouped = {src: sum(col.get(c, 0.0) for c in cols) for src, cols in self.groups.items()}
        margin = float(self.sub.model.predict(Z, raw_score=True)[0])
        return grouped, self.base, margin


def _fmt_value(v) -> str:
    if v is None:
        return "not recorded"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    if isinstance(v, float):
        if abs(v) < 1:     # never scientific notation (4e-06 would read as 4 and 6)
            return np.format_float_positional(v, precision=4, fractional=False, trim="-")
        return f"{v:,.2f}"
    return str(v)


def _raw(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return float(x)
    if isinstance(x, (np.floating,)):
        f = float(x)
        return None if np.isnan(f) else f
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    return None if s in ("nan", "None", "") else s


def build_evidence_v2(dataset: str, bid: str, X_row: pd.DataFrame, sub,
                      shap_g: GroupedShap, group: str, *, k: int = TOP_K) -> EvidenceObject:
    target, horizon, decision_time = FS.TARGETS[dataset]
    eo = EvidenceObject(
        borrower_id=bid, dataset=dataset, prediction_time=decision_time,
        group_attributes={"audit_group": group}, schema_version=SCHEMA,
        provenance={"model_id": sub.model_id, "calibrator_id": sub.calibrator_id,
                    "transformer_id": sub.prep.transformer_id,
                    "split": sub.split_sizes, "attribution": "TreeSHAP, grouped by source field"})
    row = X_row.iloc[0]
    grouped, base, margin = shap_g.explain(X_row)
    total = base + sum(grouped.values())
    if abs(total - margin) > 1e-3 * max(1.0, abs(margin)):
        raise AssertionError(f"{bid}: SHAP additivity failed ({total:.5f} vs margin {margin:.5f})")
    eo.provenance.update({"shap_base_value": round(base, 5), "raw_margin": round(margin, 5)})

    drivers = sorted(((f, v) for f, v in grouped.items() if abs(v) > 1e-9),
                     key=lambda t: -abs(t[1]))[:k]

    fact_id: Dict[str, str] = {}

    def add_fact(internal: str) -> str:
        if internal in fact_id:
            return fact_id[internal]
        spec = FS.spec_for(dataset, internal)
        raw = _raw(row[internal]) if internal in row.index else None
        label = FS.decode_value(spec, raw) if raw is not None else None
        missing_label = "not recorded"
        if raw is None and spec.missing_means is not None:
            raw, missing_label = spec.missing_means      # determinate missingness
            label = missing_label
        eq = {}
        if spec.equivalents is not None and isinstance(raw, float):
            eq = {kk: float(vv) for kk, vv in (spec.equivalents(raw) or {}).items()}
        eid = f"f{len(fact_id) + 1}"
        fact_id[internal] = eid          # reserve before recursing into inputs
        comps = [add_fact(c) for c in spec.components]
        eo.add(new_item(
            eid, EvidenceType.OBSERVED_FACT, source_field=spec.display, value=raw,
            unit=spec.unit, observed_at=spec.observed_at,
            missing=raw is None and missing_label == "not recorded",
            concept=FS.concept_for(spec, raw), meaning=spec.meaning + (
                f". Note: {spec.special}" if spec.special and spec.special != "normalized" else ""),
            value_label=(missing_label if raw is None else label), series=spec.series,
            period_order=spec.order, aliases=list(spec.aliases), equivalents=eq,
            components=comps, internal_name=internal))
        fact_id[internal] = eid
        return eid

    for i, (feat, val) in enumerate(drivers, start=1):
        fid = add_fact(feat)
        spec = FS.spec_for(dataset, feat)
        raw = _raw(row[feat]) if feat in row.index else None
        eo.add(new_item(
            f"a{i}", EvidenceType.MODEL_ATTRIBUTION, model_id=sub.model_id,
            transformer_id=sub.prep.transformer_id, feature_group=spec.display,
            contribution=round(val, 4),
            direction="increases_risk" if val > 0 else "decreases_risk",
            explained_scale="raw log-odds margin of the uncalibrated model (not the calibrated PD)",
            reference_background="TreeSHAP expected value over the training fold",
            concept=FS.concept_for(spec, raw), aliases=list(spec.aliases),
            components=[fid], internal_name=feat))

    pd_hat = float(sub.predict_pd(X_row)[0])
    eo.add(new_item("p1", EvidenceType.MODEL_PREDICTION, calibrated_pd=round(pd_hat, 4),
                    target_definition=target, horizon=horizon, model_id=sub.model_id,
                    calibrator_id=sub.calibrator_id,
                    aliases=["probability of default", "default probability", "PD",
                             "estimated probability", "risk estimate"]))
    if not drivers:
        eo.add(new_item("u1", EvidenceType.UNSUPPORTED,
                        reason="attribution unavailable for this record",
                        concept="NO_SUPPORTED_ESTIMATE"))
    # facts first, then attributions, then prediction (stable reading order)
    order = {EvidenceType.OBSERVED_FACT: 0, EvidenceType.MODEL_ATTRIBUTION: 1,
             EvidenceType.MODEL_PREDICTION: 2, EvidenceType.HYPOTHETICAL_ACTION: 3,
             EvidenceType.UNSUPPORTED: 4}
    eo.items.sort(key=lambda it: (order[it.etype], int(it.eid[1:]) if it.eid[1:].isdigit() else 0))
    return eo


TYPE_LABEL = {"observed_fact": "Observed fact", "model_attribution": "Model attribution",
              "model_prediction": "Model prediction", "hypothetical_action": "Hypothetical action",
              "unsupported": "Not available"}


def render_evidence_v2(eo: EvidenceObject) -> str:
    """Human-readable evidence for annotators: no borrower id, no group."""
    lines = []
    for i in eo.items:
        t = TYPE_LABEL[i.etype.value]
        if i.etype is EvidenceType.OBSERVED_FACT:
            v = _fmt_value(i.value) if (i.value is not None or not i.value_label) else i.value_label
            lab = f" ({i.value_label})" if i.value_label and i.value is not None else ""
            unit = f" {i.unit}" if i.unit and i.value is not None and not i.value_label else ""
            eq = ""
            if i.equivalents.get("years") is not None:
                eq = f" [= {i.equivalents['years']} years]"
            when = f"; recorded {i.observed_at}" if i.observed_at else ""
            comp = (f"; computed from {', '.join('[' + c + ']' for c in i.components)}"
                    if i.components else "")
            lines.append(f"[{i.eid}] {t}: {i.source_field} = {v}{unit}{lab}{eq}"
                         f"  — {i.meaning}{when}{comp}")
        elif i.etype is EvidenceType.MODEL_ATTRIBUTION:
            arrow = "RAISES" if i.direction == "increases_risk" else "LOWERS"
            lines.append(f"[{i.eid}] {t}: {i.feature_group} {arrow} the model's risk score "
                         f"(contribution {i.contribution:+.4f} on the {i.explained_scale}; "
                         f"value in [{i.components[0]}])" if i.components else
                         f"[{i.eid}] {t}: {i.feature_group} {arrow} the model's risk score "
                         f"(contribution {i.contribution:+.4f} on the {i.explained_scale})")
        elif i.etype is EvidenceType.MODEL_PREDICTION:
            lines.append(f"[{i.eid}] {t}: estimated probability of default = "
                         f"{i.calibrated_pd:.1%} (outcome: {i.target_definition}; "
                         f"horizon: {i.horizon})")
        elif i.etype is EvidenceType.HYPOTHETICAL_ACTION:
            lines.append(f"[{i.eid}] {t}: {i.raw_edits} -> modelled PD {i.rescored_pd}")
        else:
            lines.append(f"[{i.eid}] {t}: {i.reason}")
    return "\n".join(lines)

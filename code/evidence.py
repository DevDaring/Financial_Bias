"""
Typed evidence contract (plan §4.1).

One versioned evidence object per borrower, built BEFORE any generation. The
same object feeds the deterministic template, the constrained generator, and
the independent evaluator, so all three argue from identical facts.

Every item carries an id. Every released claim must cite the ids it rests on;
a claim citing nothing is unsupported by construction.

Boundary the plan insists on: attribution fidelity is fidelity to the
*specified model explanation*, not truth about why a borrower will default. A
faithful explanation can describe a biased predictor.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "evidence/2.0"   # 2.0: source names, meanings, periods; 1.1 kept zeros


class EvidenceType(str, Enum):
    OBSERVED_FACT = "observed_fact"
    MODEL_ATTRIBUTION = "model_attribution"
    MODEL_PREDICTION = "model_prediction"
    HYPOTHETICAL_ACTION = "hypothetical_action"
    UNSUPPORTED = "unsupported"


# What a generator is ALLOWED to say about each type. Enforced in verify.py.
PERMITTED_STATEMENT = {
    EvidenceType.OBSERVED_FACT:
        "State the recorded value, with its unit and observation time.",
    EvidenceType.MODEL_ATTRIBUTION:
        "State that, relative to the model's reference, this feature moves the "
        "model's risk score up or down. Never state a magnitude in PD terms.",
    EvidenceType.MODEL_PREDICTION:
        "State the model's estimate for the documented outcome and horizon.",
    EvidenceType.HYPOTHETICAL_ACTION:
        "State that under this modelled scenario the estimated score changes. "
        "Never assert a real-world causal effect.",
    EvidenceType.UNSUPPORTED:
        "State that no supported estimate is available. Never fill the gap.",
}


@dataclass
class EvidenceItem:
    eid: str
    etype: EvidenceType
    # observed fact
    source_field: Optional[str] = None
    value: Any = None
    unit: Optional[str] = None
    observed_at: Optional[str] = None
    missing: bool = False
    # model attribution
    model_id: Optional[str] = None
    transformer_id: Optional[str] = None
    feature_group: Optional[str] = None
    contribution: Optional[float] = None
    direction: Optional[str] = None          # increases_risk | decreases_risk
    explained_scale: Optional[str] = None    # e.g. "raw log-odds margin"
    reference_background: Optional[str] = None
    # model prediction
    calibrated_pd: Optional[float] = None
    target_definition: Optional[str] = None
    horizon: Optional[str] = None
    calibrator_id: Optional[str] = None
    # hypothetical action
    action_id: Optional[str] = None
    raw_edits: Dict[str, Any] = field(default_factory=dict)
    feasibility_rules: List[str] = field(default_factory=list)
    rescored_pd: Optional[float] = None
    # unsupported
    reason: Optional[str] = None
    # free-form concept label used by the taxonomy mapping
    concept: Optional[str] = None
    # --- evidence/2.0 additions (all optional; v1 objects load unchanged) ---
    meaning: Optional[str] = None            # documented meaning of the field
    value_label: Optional[str] = None        # documented label of an encoded value
    series: Optional[str] = None             # time-indexed series id
    period_order: Optional[int] = None       # 0 = most recent period in series
    aliases: List[str] = field(default_factory=list)       # names text may use
    equivalents: Dict[str, float] = field(default_factory=dict)  # e.g. years
    components: List[str] = field(default_factory=list)    # eids of inputs
    internal_name: Optional[str] = None      # model column (audit only)

    def permitted(self) -> str:
        return PERMITTED_STATEMENT[self.etype]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["etype"] = self.etype.value
        # Drop only genuinely empty fields. The v1 filter was
        # `v not in (None, {}, [], False)`, and since 0 == False in Python it
        # silently deleted every recorded zero (e.g. PAY_0 = 0, "paid duly").
        # Evidence must keep zeros; only the default `missing=False` flag and
        # empty containers are elided.
        def keep(k, v):
            if v is None:
                return False
            if isinstance(v, (dict, list)) and not v:
                return False
            if k == "missing" and v is False:
                return False
            return True
        return {k: v for k, v in d.items() if keep(k, v)}


@dataclass
class EvidenceObject:
    """Everything a generator, template or evaluator may use for one borrower."""
    borrower_id: str
    dataset: str
    prediction_time: Optional[str] = None
    items: List[EvidenceItem] = field(default_factory=list)
    # audit-only; NEVER passed to the generator in the neutral condition
    group_attributes: Dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    provenance: Dict[str, Any] = field(default_factory=dict)

    # -- construction helpers ---------------------------------------------
    def add(self, item: EvidenceItem) -> EvidenceItem:
        self.items.append(item)
        return item

    def by_type(self, t: EvidenceType) -> List[EvidenceItem]:
        return [i for i in self.items if i.etype is t]

    def ids(self) -> List[str]:
        return [i.eid for i in self.items]

    def get(self, eid: str) -> Optional[EvidenceItem]:
        return next((i for i in self.items if i.eid == eid), None)

    @property
    def prediction(self) -> Optional[EvidenceItem]:
        p = self.by_type(EvidenceType.MODEL_PREDICTION)
        return p[0] if p else None

    @property
    def has_unsupported(self) -> bool:
        return bool(self.by_type(EvidenceType.UNSUPPORTED))

    def completeness(self) -> float:
        """Share of items that are usable (not missing, not unsupported).

        Used by the release policy as a runtime, group-unaware signal.
        """
        if not self.items:
            return 0.0
        bad = sum(1 for i in self.items
                  if i.missing or i.etype is EvidenceType.UNSUPPORTED)
        return round(1.0 - bad / len(self.items), 4)

    # -- serialisation -----------------------------------------------------
    def to_dict(self, include_groups: bool = False) -> dict:
        d = {"schema_version": self.schema_version,
             "borrower_id": self.borrower_id,
             "dataset": self.dataset,
             "prediction_time": self.prediction_time,
             "provenance": self.provenance,
             "items": [i.to_dict() for i in self.items]}
        if include_groups:
            d["group_attributes"] = self.group_attributes
        return d

    def for_generator(self) -> dict:
        """Exactly what a generator sees. Group attributes are excluded here;
        the cue study (E2) injects a cue explicitly and labels it as a cue.
        Checker-only fields (aliases, internal model column names) are
        removed so the producer and judge see the same contract annotators do."""
        d = self.to_dict(include_groups=False)
        for it in d["items"]:
            # only v2-only fields: v1 payloads (and fingerprints) stay identical
            for k in ("aliases", "internal_name"):
                it.pop(k, None)
        return d

    def fingerprint(self) -> str:
        """Stable hash of the generator-visible evidence.

        Two conditions that must differ only by a demographic cue can be
        checked by asserting identical fingerprints.
        """
        blob = json.dumps(self.for_generator(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Reason concepts: deterministic, dataset-documented, no trend words
# ---------------------------------------------------------------------------
#: The production app shipped a 10-code taxonomy containing trend concepts
#: (REVENUE_DECLINE) that a single-snapshot model cannot support. Those are
#: renamed to levels here, per the plan: "Revise the ten-code taxonomy wherever
#: it encodes a trend without longitudinal evidence."
REASON_CONCEPTS: Dict[str, str] = {
    "LIQUIDITY_LEVEL_LOW":      "Debt-service coverage is low at the observation time.",
    "LEVERAGE_LEVEL_HIGH":      "Borrowing is large relative to capacity at the observation time.",
    "REVENUE_LEVEL_LOW":        "Turnover is low relative to obligations at the observation time.",
    "REPAYMENT_HISTORY_ADVERSE": "Recorded repayment history contains adverse events.",
    "SECTOR_ASSOCIATION":       "The recorded sector is associated with the model's risk score.",
    "COLLATERAL_LEVEL_LOW":     "Recorded security is low relative to exposure.",
    "TENURE_ASSOCIATION":       "Loan tenure is associated with the model's risk score.",
    "BUREAU_SCORE_LEVEL":       "The recorded bureau score is at this level.",
    "TEXT_SIGNAL":              "The recorded free-text field contains a flagged term.",
    "UNMAPPED_FEATURE":         "The feature has no documented business meaning.",
    "NO_SUPPORTED_ESTIMATE":    "No supported estimate is available.",
}

#: Trend words a snapshot model cannot license. verify.py flags these.
FORBIDDEN_TREND_TERMS = [
    "declined", "declining", "decline", "fell", "falling", "dropped", "dropping",
    "rose", "rising", "increased", "increasing", "worsened", "worsening",
    "improved", "improving", "trend", "trending", "deteriorated", "deteriorating",
    "over time", "recently", "has been", "continues to",
]

#: Causal language a scored counterfactual cannot license.
FORBIDDEN_CAUSAL_TERMS = [
    "will cause", "causes", "caused by", "because of this", "as a result",
    "leads to", "will lead to", "guarantees", "ensures", "will reduce",
    "will improve", "will prevent",
]


def map_feature_to_concept(feature_group: str,
                           documented: Dict[str, str]) -> str:
    """Deterministic feature -> concept. Unknown features stay unmapped.

    `documented` is the dataset's own documented mapping. Nothing is guessed:
    an opaque feature must not become a confident business explanation.
    """
    key = (feature_group or "").strip().lower()
    for pat, concept in documented.items():
        if pat.lower() in key:
            return concept
    return "UNMAPPED_FEATURE"


# ---------------------------------------------------------------------------
def new_item(eid: str, etype: EvidenceType, **kw) -> EvidenceItem:
    return EvidenceItem(eid=eid, etype=etype, **kw)


if __name__ == "__main__":
    eo = EvidenceObject(borrower_id="SBA-000123", dataset="sba",
                        prediction_time="approval_date",
                        group_attributes={"region": "CA", "urban_rural": "urban"})
    eo.add(new_item("f1", EvidenceType.OBSERVED_FACT, source_field="GrAppv",
                    value=455000.0, unit="USD", observed_at="approval_date",
                    concept="LEVERAGE_LEVEL_HIGH"))
    eo.add(new_item("f2", EvidenceType.OBSERVED_FACT, source_field="Term",
                    value=84, unit="months", observed_at="approval_date",
                    concept="TENURE_ASSOCIATION"))
    eo.add(new_item("a1", EvidenceType.MODEL_ATTRIBUTION, model_id="lgbm-sba-s0",
                    transformer_id="prep-sba-s0", feature_group="GrAppv",
                    contribution=0.41, direction="increases_risk",
                    explained_scale="raw log-odds margin",
                    reference_background="training-fold mean",
                    concept="LEVERAGE_LEVEL_HIGH"))
    eo.add(new_item("p1", EvidenceType.MODEL_PREDICTION, calibrated_pd=0.31,
                    target_definition="MIS_Status == CHGOFF",
                    horizon="loan lifetime (not 12 months)",
                    model_id="lgbm-sba-s0", calibrator_id="isotonic-sba-s0"))
    eo.add(new_item("u1", EvidenceType.UNSUPPORTED,
                    reason="No longitudinal turnover series; trend not computable",
                    concept="NO_SUPPORTED_ESTIMATE"))
    print(json.dumps(eo.to_dict(include_groups=True), indent=2)[:900])
    print("\ncompleteness:", eo.completeness())
    print("fingerprint :", eo.fingerprint())
    print("groups hidden from generator:", "group_attributes" not in eo.for_generator())

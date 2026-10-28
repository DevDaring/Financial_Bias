"""
Credit-evidence contract (plan §4.1) and deterministic objection checks (§4.2 step 4).

The contract names the finite set of things a credit explanation may get
wrong. A verifier's objection must name one of these rules, quote the exact
words it objects to and cite the evidence it relies on. This module checks
those objections WITHOUT an LLM, so an objection that is malformed, quotes
text that is not there, cites irrelevant evidence, or calls a disclaimer a
causal claim can be told apart from a valid one.

An invalid objection does not make the candidate correct (plan §4.2 step 5):
it makes the verdict unresolved.

Version: contract/1.1 (6 Oct 2026).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from evidence import EvidenceObject, EvidenceType
import verify as V

CONTRACT_VERSION = "contract/1.1"

# ---------------------------------------------------------------------------
# 1. the rule set
# ---------------------------------------------------------------------------
#: rule -> (what it forbids, matching ErrorType value used by the annotation rubric)
RULES: Dict[str, Tuple[str, str]] = {
    "R_VALUE": ("A number attached to a field differs from that field's evidence value "
                "(beyond sensible rounding or a documented unit conversion).",
                V.ErrorType.NUMERIC_ERROR.value),
    "R_DIRECTION": ("An attribution is described as moving the model's risk score in the "
                    "direction opposite to its stated direction.",
                    V.ErrorType.SIGN_REVERSAL.value),
    "R_SCALE": ("An attribution (raw log-odds contribution) is presented as a percentage or as "
                "a change in the probability of default.",
                V.ErrorType.NUMERIC_ERROR.value),
    "R_TEMPORAL": ("A change over time is asserted that does not compare values for periods "
                   "that are all shown in the evidence.",
                   V.ErrorType.UNSUPPORTED_TREND.value),
    "R_CAUSAL": ("A real-world causal effect, or the effect of an action on the outcome, is "
                 "ASSERTED. Denying causation or describing the model is not a violation.",
                 V.ErrorType.UNSUPPORTED_CAUSAL.value),
    "R_TARGET": ("The predicted outcome, its definition or its horizon is stated wrongly.",
                 "other_material_error"),
    "R_INVENTED": ("A factor, fact or reason is named that is absent from the evidence.",
                   V.ErrorType.INVENTED_FACTOR.value),
    "R_OMISSION": ("The probability estimate or the strongest attribution is not conveyed.",
                   V.ErrorType.OMISSION.value),
    # contract/1.1 (6 Oct 2026): added from the M1 development labels, where these
    # four credit-evidence errors were the most frequent agreed material errors.
    "R_RANKING": ("A feature is called the strongest / main / largest driver when another "
                  "attribution has a larger absolute contribution.",
                  "other_material_error"),
    "R_FACT_AS_DRIVER": ("A field that appears only as a recorded fact (it has no attribution) is "
                         "said to raise or lower the model's risk score.",
                         V.ErrorType.INVENTED_FACTOR.value),
    "R_CODE_MEANING": ("A meaning is stated for a coded value whose meaning the source does not "
                       "document.", "other_material_error"),
    "R_COMPLETENESS": ("The text says no other factors matter, or that one factor is the only one, "
                       "when the evidence lists other attributions.", V.ErrorType.INVENTED_FACTOR.value),
}
#: rules that must quote the offending words (omission has nothing to quote)
NEEDS_QUOTE = set(RULES) - {"R_OMISSION"}
#: rules that must cite at least one existing evidence item
NEEDS_EVIDENCE = set(RULES) - {"R_INVENTED", "R_TEMPORAL", "R_CAUSAL"}


def rule_text() -> str:
    return "\n".join(f"{k}: {v[0]}" for k, v in RULES.items())


# ---------------------------------------------------------------------------
# 2. evidence validation (step 1)
# ---------------------------------------------------------------------------
def validate_evidence(eo: EvidenceObject) -> List[str]:
    """Problems that make the evidence insufficient to verify against.

    Evidence insufficiency is a separate failure mode from a false narrative
    (plan §4.2 step 1): such items are unresolved, never accepted or rejected.
    """
    problems: List[str] = []
    pred = eo.prediction
    if pred is None:
        problems.append("no model_prediction item")
    elif pred.calibrated_pd is None:
        problems.append(f"prediction {pred.eid} has no calibrated_pd")
    attrs = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
    if not attrs:
        problems.append("no model_attribution items")
    for a in attrs:
        if a.direction not in ("increases_risk", "decreases_risk"):
            problems.append(f"attribution {a.eid} has no direction")
        if a.contribution is None:
            problems.append(f"attribution {a.eid} has no contribution")
    ids = eo.ids()
    if len(ids) != len(set(ids)):
        problems.append("duplicate evidence ids")
    for it in eo.items:
        for c in it.components or []:
            if c not in ids:
                problems.append(f"{it.eid} lists unknown component {c}")
    return problems


# ---------------------------------------------------------------------------
# 3. objection checks (step 4)
# ---------------------------------------------------------------------------
@dataclass
class Objection:
    rule: str
    quote: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    reason: str = ""
    claim_id: Optional[str] = None
    # filled by check_objection
    valid: Optional[bool] = None
    problems: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                         "—": "-", " ": " "})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(_QUOTES)).strip().lower()


def quote_position(quote: str, text: str) -> int:
    """Start of the quote in the text (whitespace/quote-style/case-insensitive), or -1."""
    q, t = _norm(quote).strip(" .\"'"), _norm(text)
    if not q:
        return -1
    return t.find(q)


# a rationale that grants the quoted words are right while objecting to them
_SELF_CONTRADICTION = re.compile(
    r"\b(?<!not )(?:is|are|seems|looks)\s+(?:entirely\s+|fully\s+|actually\s+)?"
    r"(?:correct|accurate|appropriate|valid|fine|acceptable|supported|consistent with the evidence)\b"
    r"|\bcorrectly (?:states|notes|describes|says)\b|\bno (?:error|problem|issue)\b", re.I)
_CAUSAL_ASSERT = re.compile(
    r"\b(will (?:cause|lead|reduce|improve|prevent|lower|raise|increase|decrease|help)|causes?|caused by|"
    r"leads? to|results? in|because of|due to|drives?|driven by|ensures?|guarantees?|"
    r"(?:would|could|should) (?:reduce|lower|improve|prevent|cut|raise|increase|decrease))\b", re.I)
_DISCLAIMER = re.compile(
    r"\b(not (?:a |an )?(?:verified |demonstrated |proven |established |real-world )?(?:cause|causal)|"
    r"(?:does|do|did|should) not (?:mean|imply|show|prove|establish|indicate)|"
    r"describes? (?:the )?model|model'?s? (?:view|association|estimate)|associat(?:ed|ion)|"
    r"no (?:causal|real-world) (?:claim|effect))", re.I)
_SCALE_TERMS = re.compile(
    r"(percentage points?|pp\b|%|per ?cent|probability|chance|likelihood|\bpd\b)", re.I)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _item_names(eo: EvidenceObject, eid: str) -> List[str]:
    it = eo.get(eid)
    if it is None:
        return []
    names = V._names(it, eo.dataset)
    for c in it.components or []:
        comp = eo.get(c)
        if comp is not None:
            names += V._names(comp, eo.dataset)
    return names


def _mentions_item(span: str, eo: EvidenceObject, eid: str) -> bool:
    it = eo.get(eid)
    if it is None:
        return False
    if it.etype is EvidenceType.MODEL_PREDICTION:
        return bool(re.search(r"\b(probability|chance|likelihood|pd|default|risk|estimate)\b", span, re.I))
    names = _item_names(eo, eid)
    return bool(names) and bool(V._mentions(V._canon(span.lower()), names))


def check_objection(obj: Objection, text: str, eo: EvidenceObject) -> Objection:
    """Deterministic validity of ONE objection against the candidate text and evidence.

    Valid means: the rule exists, the quote is really in the text, the cited
    evidence exists and is about the quoted words, and the named rule fits
    what the quoted words do. It does NOT mean the objection is correct: that
    needs reference labels (the objection audit, plan §5.1).
    """
    p: List[str] = []
    if obj.rule not in RULES:
        p.append(f"unknown rule {obj.rule!r}")
        obj.valid, obj.problems = False, p
        return obj
    ids = [str(i) for i in obj.evidence_ids or []]
    unknown = [i for i in ids if eo.get(i) is None]
    if unknown:
        p.append(f"cites unknown evidence ids {unknown}")
    if obj.rule in NEEDS_EVIDENCE and not [i for i in ids if eo.get(i) is not None]:
        p.append("cites no existing evidence")

    span = obj.quote or ""
    if obj.rule in NEEDS_QUOTE:
        if not span.strip():
            p.append("no quote")
        elif quote_position(span, text) < 0:
            p.append("quote is not in the explanation")

    if _SELF_CONTRADICTION.search(obj.reason or ""):
        p.append("the rationale says the quoted words are correct")

    if not p:
        p += _rule_fits(obj, span, text, eo, ids)
    obj.valid, obj.problems = (not p), p
    return obj


def _rule_fits(obj: Objection, span: str, text: str, eo: EvidenceObject, ids: List[str]) -> List[str]:
    """Does the named rule describe what the quoted words actually do?"""
    p: List[str] = []
    r = obj.rule
    if r == "R_CAUSAL":
        if not _CAUSAL_ASSERT.search(span):
            p.append("quoted words contain no causal or action-effect assertion")
        elif _DISCLAIMER.search(span) and not re.search(r"\bwill\b", span, re.I):
            p.append("quoted words are a disclaimer or a model-association statement, not a causal claim")
    elif r == "R_TEMPORAL":
        low = span.lower()
        hits = [t for t in V._TREND_TERMS if re.search(r"\b" + re.escape(t) + r"\b", low)]
        if not hits:
            p.append("quoted words assert no change over time")
        else:
            pos = quote_position(span, text)
            before = text[max(0, pos - 80):pos] if pos >= 0 else ""
            k = min(low.find(h) for h in hits)
            if V._disclaimed(before + " " + span[:max(0, k)]):
                p.append("the change-over-time words are negated or disclaimed")
    elif r == "R_VALUE":
        if not _NUM.search(span):
            p.append("quoted words contain no number")
        elif ids and not any(_mentions_item(span, eo, i) for i in ids if eo.get(i) is not None):
            p.append("the cited evidence is not about the quoted words")
    elif r == "R_DIRECTION":
        if not re.search(V._UP_INFL + "|" + V._DN_INFL + "|" + V._UP_BARE + "|" + V._DN_BARE
                         + r"|higher|lower|more|less|adds?|pushes|pulls", span, re.I):
            p.append("quoted words state no direction")
        attrs = [i for i in ids if (eo.get(i) is not None and eo.get(i).etype is EvidenceType.MODEL_ATTRIBUTION)]
        if not attrs:
            p.append("direction objection cites no attribution")
        elif not any(_mentions_item(span, eo, i) for i in attrs):
            p.append("the cited attribution is not about the quoted words")
    elif r == "R_SCALE":
        if not (_NUM.search(span) and _SCALE_TERMS.search(span)):
            p.append("quoted words present no percentage or probability change")
        if not any(eo.get(i) is not None and eo.get(i).etype is EvidenceType.MODEL_ATTRIBUTION for i in ids):
            p.append("scale objection cites no attribution")
    elif r == "R_TARGET":
        if not any(eo.get(i) is not None and eo.get(i).etype is EvidenceType.MODEL_PREDICTION for i in ids):
            p.append("target objection cites no prediction item")
    elif r == "R_INVENTED":
        # the quoted factor must not be one the evidence names
        named = [i.eid for i in eo.items if i.etype is not EvidenceType.MODEL_PREDICTION
                 and _mentions_item(span, eo, i.eid)]
        if named and not _NUM.search(span):
            p.append(f"the quoted factor is in the evidence ({', '.join(named[:3])})")
    elif r == "R_OMISSION":
        low = text.lower()
        for i in ids:
            it = eo.get(i)
            if it is None:
                continue
            if it.etype is EvidenceType.MODEL_PREDICTION:
                pdv = it.calibrated_pd
                if pdv is not None and any(V._num_matches(n, pdv * 100) or V._num_matches(n, pdv)
                                           for n in V._signed_numbers(text)):
                    p.append(f"the estimate in {i} is stated in the explanation")
            elif V._mentions(V._canon(low), _item_names(eo, i)):
                p.append(f"{i} is mentioned in the explanation")
    return p


def objections_from_json(d: dict) -> List[Objection]:
    """Objections from a contract-verifier or judge response (tolerant of key names)."""
    out: List[Objection] = []
    for c in (d.get("claims") or []):
        if not isinstance(c, dict) or str(c.get("status", "")).lower() != "contradicted":
            continue
        out.append(Objection(rule=str(c.get("rule", "")).upper(), quote=str(c.get("quote") or ""),
                             evidence_ids=[str(x) for x in (c.get("evidence_ids") or [])],
                             reason=str(c.get("reason") or ""), claim_id=str(c.get("claim_id") or "") or None))
    for o in (d.get("omissions") or []):
        if isinstance(o, dict):
            out.append(Objection(rule="R_OMISSION", evidence_ids=[str(x) for x in (o.get("evidence_ids") or [])],
                                 reason=str(o.get("reason") or "")))
    return out


# ---------------------------------------------------------------------------
# 4. claims to assess
# ---------------------------------------------------------------------------
def claims_of(text: str, typed_claims: Optional[List[dict]] = None) -> List[dict]:
    """Numbered claims for the semantic step: typed claims as given, else clauses
    grouped into sentences (the same splitter the structural checker uses)."""
    if typed_claims:
        return [{"claim_id": f"c{k + 1}", "text": c.get("text", ""),
                 "evidence_ids": list(c.get("evidence_ids") or [])} for k, c in enumerate(typed_claims)]
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", V._protect(text))
    out, pos = [], 0
    for s in sents:
        start = text.find(s.strip()[:20].replace("\x00", "."), pos) if s.strip() else -1
        seg = text[pos: pos + len(s)] if start < 0 else text[start: start + len(s.strip())]
        pos = (start if start >= 0 else pos) + len(seg)
        seg = seg.strip()
        if seg:
            out.append({"claim_id": f"c{len(out) + 1}", "text": seg, "evidence_ids": []})
    return out


# ---------------------------------------------------------------------------
# 5. deterministic contract detectors (contract/1.1; run alongside structural/2.6)
# ---------------------------------------------------------------------------
_RANK_CLAIM = re.compile(
    r"\b(?:strongest|largest|biggest|greatest|main|primary|principal|most important|most influential|"
    r"leading|dominant|top|single (?:largest|biggest|most important))\s+(?:(?!and\b|or\b)\w+\s+){0,3}?"
    r"(?:driver|factor|contributor|reason|influence|feature|signal|risk factor|determinant)\b(?!s)"
    r"|\b(?:mainly|primarily|chiefly|largely|mostly|predominantly)\s+(?:driven|explained|determined)\s+by\b"
    r"|\bthe (?:biggest|largest|strongest|main) (?:push|pull|effect|contribution|impact)\b"
    r"|\b(?:drives|dominates|matters most|contributes most|has the (?:largest|biggest|strongest) (?:effect|impact|contribution))\b",
    re.I)
_DIRECTION_ON_SCORE = re.compile(
    r"\b(?:" + V._UP_INFL + "|" + V._DN_INFL + "|" + V._UP_BARE + "|" + V._DN_BARE +
    r"|adds? to|contributes? to|pushes? (?:up|down)|pulls? down|offsets?|mitigates?)\b"
    r"(?:\s+\w+){0,4}?\s+(?:risk|score|probability|pd|likelihood|default|estimate)", re.I)
_MEANING_WORDS = re.compile(
    r"\b(?:paid duly|pays? duly|duly|on[- ]time|timely|no (?:payment )?delays?|without (?:any )?delay|not delayed|"
    r"no (?:late|missed) payments?|paid in full|revolving credit|no consumption|no (?:card )?use|in good standing)\b", re.I)
_HEDGE = re.compile(r"\b(?:not defined|undefined|undocumented|not documented|unclear|unknown|not specified|"
                    r"no documented meaning|meaning is not)\b", re.I)
_COMPLETE_CLAIM = re.compile(
    r"\b(?:no other (?:\w+\s+)?(?:factors?|drivers?|features?|variables?|influences?)|"
    r"(?:the )?only (?:\w+\s+)?(?:factor|driver|reason|feature)|sole (?:factor|driver|reason)|nothing else|"
    r"no additional (?:factors?|drivers?)|all other factors? (?:are|were) (?:neutral|irrelevant))\b", re.I)


def _attr_names(eo: EvidenceObject) -> Dict[str, List[str]]:
    return {a.eid: _item_names(eo, a.eid) for a in eo.by_type(EvidenceType.MODEL_ATTRIBUTION)}


def _fact_only(eo: EvidenceObject) -> List:
    linked = set()
    groups = set()
    for a in eo.by_type(EvidenceType.MODEL_ATTRIBUTION):
        linked.update(a.components or [])
        groups.add(V._canon(str(a.feature_group or "").lower()))
    return [f for f in eo.by_type(EvidenceType.OBSERVED_FACT)
            if f.eid not in linked and V._canon(str(f.source_field or "").lower()) not in groups]


def contract_violations(text: str, eo: EvidenceObject) -> List[dict]:
    """Ranking, fact-as-driver, undocumented-code-meaning and completeness errors."""
    out: List[dict] = []
    attrs = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
    names = _attr_names(eo)
    sents = [s.strip() for s in re.split(r"(?<=[.;!?])\s+", text) if s.strip()]

    def mentioned(sent: str) -> List[str]:
        low = V._canon(sent.lower())
        return [eid for eid, ns in names.items() if ns and V._mentions(low, ns)]

    if attrs:
        top = max(attrs, key=lambda a: abs(a.contribution or 0.0))
        topv = abs(top.contribution or 0.0)
        for s in sents:
            for m in _RANK_CLAIM.finditer(s):
                if V._governed_neg(s[:m.start()]) or re.search(r"\bone of the\s*$", s[:m.start()], re.I):
                    continue
                ms = mentioned(s)
                if not ms:
                    continue
                if top.eid in ms or any(abs(abs(eo.get(e).contribution or 0.0) - topv) <= 0.005 for e in ms):
                    continue
                out.append({"type": "R_RANKING", "rule": "R_RANKING", "span": s[:200],
                            "detail": f"named {ms} but the strongest attribution is {top.eid} "
                                      f"({top.feature_group}, {top.contribution:+.4f})",
                            "evidence_ids": [top.eid] + ms})
                break

    fo = _fact_only(eo)
    if fo:
        for s in sents:
            if not _DIRECTION_ON_SCORE.search(s):
                continue
            low = V._canon(s.lower())
            hits = [f for f in fo if V._mentions(low, V._names(f, eo.dataset))]
            if hits and not mentioned(s):
                out.append({"type": "R_FACT_AS_DRIVER", "rule": "R_FACT_AS_DRIVER", "span": s[:200],
                            "detail": f"direction given for fact-only field(s) "
                                      f"{[f.source_field for f in hits]}",
                            "evidence_ids": [f.eid for f in hits]})

    undoc = [i for i in eo.items if _HEDGE.search(str(i.value_label or ""))]
    for s in sents:
        if not _MEANING_WORDS.search(s) or _HEDGE.search(s):
            continue
        low = V._canon(s.lower())
        hits = [i for i in undoc if V._mentions(low, V._names(i, eo.dataset)) or V._period_mentions(s, i)]
        if hits:
            out.append({"type": "R_CODE_MEANING", "rule": "R_CODE_MEANING", "span": s[:200],
                        "detail": f"meaning stated for undocumented code(s): "
                                  f"{[(i.source_field, i.value_label) for i in hits][:3]}",
                        "evidence_ids": [i.eid for i in hits]})

    if len(attrs) >= 2:
        all_named = set(mentioned(text))
        for s in sents:
            m = _COMPLETE_CLAIM.search(s)
            if m and not V._governed_neg(s[:m.start()]) and not re.search(r"\bnot\s+(?:the\s+)?$", s[:m.start()], re.I) \
                    and len(all_named) < len(attrs):
                out.append({"type": "R_COMPLETENESS", "rule": "R_COMPLETENESS", "span": s[:200],
                            "detail": f"completeness claim, but {len(attrs) - len(all_named)} attribution(s) "
                                      f"are not mentioned",
                            "evidence_ids": [a.eid for a in attrs if a.eid not in all_named]})
                break
    return out

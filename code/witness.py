"""
Executable evidence verification (Future_Plan §D1-D3, 7 Oct 2026).

A claim and an objection to it are both turned into TYPED OPERATIONS over the
evidence object. Code retrieves the operands and recomputes the relation; an
LLM may propose an operation (a *witness*) but never supplies its result.

    claim    -> (span, operation, evidence ids, asserted value/direction/rank/...)
    witness  -> (claim, rule, operation, evidence ids, quote, claimed result)

Every executed operation ends in one of three states:

    support        the evidence licenses what the span asserts
    contradiction  the evidence contradicts it, with an unambiguous binding
    unresolved     the binding is ambiguous (two candidate fields, a negated or
                   mixed direction, no dated comparison computable, a referent
                   the evidence does not name) - never treated as proved

Operations (the finite language; not a logic for unrestricted prose):

    VALUE         field-bound number vs recorded value (units, %, rounding)
    DIRECTION     stated direction vs attribution sign (near-zero = unresolved)
    RANK          "strongest / main driver" vs argmax |contribution| (ties ok)
    SCALE         raw log-odds contribution presented as a % / PD change
    TEMPORAL      asserted change vs field-matched dated observations
    CODE          meaning stated for a code vs the source dictionary
    FACT_DRIVER   direction given to a field that has no attribution
    COVERAGE      PD and strongest driver conveyed (text level)
    COMPLETENESS  "no other factors" vs the attributions listed (text level)

Causal assertions, invented factors and target/horizon statements need
semantic interpretation; they are RESIDUAL content routed to the semantic
step, with their unresolved status explicit.

Proposition (Future_Plan §D3). Conditional on correct evidence, a correct
parse into this operation language and correct operation code, an executed
contradiction establishes a violation of the declared contract. Extraction
errors, semantic blind spots and dictionary errors remain empirical and are
measured separately; nothing here is a formal verification of free text.

Declared attribution set: the displayed attributions. The evidence builder
displays the top-k source fields by |TreeSHAP|, so the displayed argmax is the
model-wide argmax; a rank claim beyond first place ("second largest of all
features") is out of scope and stays unresolved.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import contract as C
import verify as V
from evidence import EvidenceItem, EvidenceObject, EvidenceType

EXEC_VERSION = "executable/1.6"   # 1.6 (9 Oct): zero-call gate (field identity, negation, codes, horizon); 1.5:   # 1.5 (7 Oct, before any locked evaluation): zero-rounding guard in verify._num_matches
# 1.1 (7 Oct): TARGET operation, two-noun directions, context-checked semantic witnesses
# 1.2 (7 Oct, after the single scoring of mechanism pack v2): negation scope up to three words
#     before a horizon ('not to a one-year horizon'); action-effect wording ('increasing X would make
#     default less likely') counts as a causal assertion. Pack v2 is development data from 1.2 on.
# 1.3 (7 Oct, after the 6-item live smoke test): an R_INVENTED witness whose quoted factor shares a
#     distinctive word with an evidence field name or meaning ('application weekday' vs
#     WEEKDAY_APPR_PROCESS_START) is not plausible: the factor may be named by an unrecognised alias.
# 1.4 (7 Oct, after Phase A on M1 showed deferral driven by unrecognised driver names, unevenly by
#     group): secondary binding - (a) a FAMILY stem ('external score' for EXT_SOURCE_1/2/3) is bound
#     to the member whose value is written next to it, or to the only member present; otherwise it is
#     a known-but-ambiguous mention (not an invented factor); (b) a DESCRIPTION phrase taken from one
#     item's documented meaning ('identity document', 'current employment') binds that item when it
#     identifies exactly one item. Development change; reported separately from the 1.3 paid run.
#     Measured on M1: (a) alone raised code-layer acceptance of correct narratives 103 -> 120/209 and
#     error rejection 18 -> 20/36 with no new false rejection; (b) added 2 false rejections and 2
#     mechanism-pack misses (a description phrase bound the wrong period's field), so (b) is OFF.

SUPPORT, CONTRA, UNRES = "support", "contradiction", "unresolved"
OPS = ("VALUE", "DIRECTION", "RANK", "SCALE", "TEMPORAL", "CODE", "FACT_DRIVER", "TARGET", "COVERAGE", "COMPLETENESS")
RULE_OP = {"R_VALUE": "VALUE", "R_DIRECTION": "DIRECTION", "R_RANKING": "RANK", "R_SCALE": "SCALE",
           "R_TEMPORAL": "TEMPORAL", "R_CODE_MEANING": "CODE", "R_FACT_AS_DRIVER": "FACT_DRIVER",
           "R_OMISSION": "COVERAGE", "R_COMPLETENESS": "COMPLETENESS", "R_TARGET": "TARGET"}
OP_RULE = {v: k for k, v in RULE_OP.items()}
SEMANTIC_RULES = {"R_CAUSAL", "R_INVENTED"}

STRICT_WORDING = True     # 1.5: an 'evidence wording' number must come from the bound item's own wording
ZERO_TOL = 0.005          # |contribution| below this has no reliable direction
TIE_TOL = 0.005           # attributions within this of the maximum are tied for first


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class OpResult:
    op: str
    status: str                                  # support | contradiction | unresolved
    evidence_ids: List[str] = field(default_factory=list)
    span: str = ""
    asserted: Any = None                         # what the text says (read by code)
    recomputed: Any = None                       # what the evidence gives (computed by code)
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaimExec:
    claim_id: str
    text: str
    start: int
    ops: List[OpResult] = field(default_factory=list)
    residual: List[str] = field(default_factory=list)   # semantic content: R_CAUSAL, R_INVENTED, R_TARGET
    material: bool = True

    @property
    def status(self) -> str:
        """contradiction | support | unresolved | nonmaterial."""
        if any(o.status == CONTRA for o in self.ops):
            return CONTRA
        if not self.material and not self.ops and not self.residual:
            return "nonmaterial"
        if self.ops and all(o.status == SUPPORT for o in self.ops) and not self.residual:
            return SUPPORT
        return UNRES

    def to_dict(self) -> dict:
        return {"claim_id": self.claim_id, "text": self.text, "status": self.status,
                "residual": self.residual, "ops": [o.to_dict() for o in self.ops]}


# ---------------------------------------------------------------------------
# text and binding
# ---------------------------------------------------------------------------
def clean(text: str) -> str:
    """Markdown emphasis and code marks removed; nothing else changes."""
    t = re.sub(r"(\*\*|__|`)", "", text or "")
    return re.sub(r"(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)", "", t)


_PD_WORDS = re.compile(r"\b(probabilit(?:y|ies)|chance|likelihood|\bpd\b|default rate|risk estimate|"
                       r"estimated (?:risk|probability)|predict(?:s|ed)?|estimat(?:e|es|ed))\b", re.I)
_TIME_UNIT = re.compile(r"^\s*(?:-|\s)?(?:calendar\s+)?(?:months?|years?|weeks?|days?)\b", re.I)


@dataclass
class Mention:
    eid: str
    kind: str          # attr | fact | pred
    start: int
    end: int


def _item_kind(it: EvidenceItem) -> str:
    return {EvidenceType.MODEL_ATTRIBUTION: "attr", EvidenceType.OBSERVED_FACT: "fact",
            EvidenceType.MODEL_PREDICTION: "pred"}.get(it.etype, "other")


def _name_spans(span: str, it: EvidenceItem, dataset: str) -> List[Tuple[int, int]]:
    low = V._canon(span.lower())
    out = []
    for n in V._names(it, dataset):
        n = V._canon(n)
        for m in re.finditer(rf"(?<![\w]){re.escape(n)}(?![\w])", low):
            out.append((m.start(), m.end()))
    if it.etype is EvidenceType.OBSERVED_FACT:
        for p in V._period_mentions(span, it):
            out.append((p, p + 1))
    return out


def bind(span: str, eo: EvidenceObject) -> List[Mention]:
    """Evidence items the span names. A name match contained in a LONGER match
    of another item is dropped ('repayment status' inside 'largest repayment
    status code'); identical spans for an attribution and its own fact are kept."""
    raw: List[Mention] = []
    for it in eo.items:
        k = _item_kind(it)
        if k in ("attr", "fact"):
            for s, e in _name_spans(span, it, eo.dataset):
                raw.append(Mention(it.eid, k, s, e))
    keep = []
    for m in raw:
        if any((o.start <= m.start and m.end <= o.end and (o.end - o.start) > (m.end - m.start)
                and o.eid != m.eid) for o in raw):
            continue
        keep.append(m)
    keep += _secondary_mentions(span, eo, keep)
    if _PD_WORDS.search(span) and eo.prediction is not None:
        mm = _PD_WORDS.search(span)
        keep.append(Mention(eo.prediction.eid, "pred", mm.start(), mm.end()))
    # one mention per eid (first position)
    seen, out = set(), []
    for m in sorted(keep, key=lambda x: (x.start, x.eid)):
        if m.eid not in seen:
            seen.add(m.eid)
            out.append(m)
    return out


_STOP = {"the", "of", "a", "an", "to", "for", "in", "on", "at", "by", "with", "which", "did", "does", "how", "many",
         "much", "what", "was", "were", "is", "are", "and", "or", "he", "she", "his", "her", "client", "clients",
         "before", "after", "application", "applied", "loan", "this", "that", "when", "who", "from", "as", "it",
         "its", "not", "be", "has", "have", "had"}


def _content(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOP]


def _family_stems(eo: EvidenceObject) -> Dict[str, List[str]]:
    """'external score 2' -> stem 'external score' : items of the numbered family."""
    fam: Dict[str, set] = {}
    for it in eo.items:
        if it.etype not in (EvidenceType.MODEL_ATTRIBUTION, EvidenceType.OBSERVED_FACT):
            continue
        for n in list(it.aliases or []) + [str(it.source_field or it.feature_group or "").replace("_", " ")]:
            m = re.fullmatch(r"(.+?)\s+(?:from\s+)?(?:source\s+)?\d+", n.lower().strip())
            if m and len(m.group(1)) >= 5:
                fam.setdefault(m.group(1).strip(), set()).add(it.eid)
    return {k: sorted(v) for k, v in fam.items()}


def _meaning_phrases(eo: EvidenceObject) -> Dict[str, str]:
    """Distinctive description phrases (two content words, or one long word) unique to ONE field."""
    owner: Dict[str, set] = {}
    for it in eo.items:
        if it.etype not in (EvidenceType.MODEL_ATTRIBUTION, EvidenceType.OBSERVED_FACT) or not it.meaning:
            continue
        field = _field_key(eo, it)
        ws = _content(it.meaning)
        cands = {f"{a} {b}" for a, b in zip(ws, ws[1:])} | {w for w in ws if len(w) >= 9}
        for c in cands:
            if all(w in _GENERIC_WORDS or len(w) < 4 for w in c.split()):
                continue
            owner.setdefault(c, set()).add(field)
    return {c: next(iter(f)) for c, f in owner.items() if len(f) == 1}


def _field_key(eo: EvidenceObject, it: EvidenceItem) -> str:
    """An attribution and its own fact are one field (keyed by the attribution when there is one)."""
    if it.etype is EvidenceType.OBSERVED_FACT:
        for a in eo.by_type(EvidenceType.MODEL_ATTRIBUTION):
            if _same_field(eo, it.eid, a.eid):
                return a.eid
    return it.eid


_USE_MEANING = False   # description binding: measured, not adopted (see 1.4 note)


def _secondary_mentions(span: str, eo: EvidenceObject, primary: List["Mention"]) -> List["Mention"]:
    low = V._canon(span.lower())
    covered = [(m.start, m.end) for m in primary]
    out: List[Mention] = []
    free = lambda s, e: not any(a < e and s < b for a, b in covered)   # noqa: E731
    for stem, ids in _family_stems(eo).items():
        for m in re.finditer(rf"(?<![\w]){re.escape(V._canon(stem))}(?![\w])(?!\s*(?:from\s+)?(?:source\s+)?\d)", low):
            if not free(m.start(), m.end()):
                continue
            fields = sorted({_field_key(eo, eo.get(i)) for i in ids})
            pick = fields if len(fields) == 1 else []
            if not pick:
                nums = [n for n in V._signed_numbers(span[m.end():m.end() + 40])]
                hits = [f for f in fields if nums and any(_num_ok(n, _values(eo, eo.get(f)), _pct_values(eo.get(f)))
                                                         for n in nums[:1])]
                pick = hits if len(hits) == 1 else []
            if pick:
                it = eo.get(pick[0])
                out.append(Mention(pick[0], _item_kind(it), m.start(), m.end()))
            else:
                out.append(Mention("|".join(fields), "family", m.start(), m.end()))
            covered.append((m.start(), m.end()))
    for ph, fid in (_meaning_phrases(eo).items() if _USE_MEANING else []):
        for m in re.finditer(rf"(?<![\w]){re.escape(ph)}s?(?![\w])", low):
            if free(m.start(), m.end()):
                it = eo.get(fid)
                out.append(Mention(fid, _item_kind(it), m.start(), m.end()))
                covered.append((m.start(), m.end()))
    return out


def _attrs(eo: EvidenceObject) -> List[EvidenceItem]:
    return eo.by_type(EvidenceType.MODEL_ATTRIBUTION)


def _owner(eo: EvidenceObject, fact_eid: str) -> Optional[str]:
    """The attribution a fact belongs to: listed component, or the same field."""
    f = eo.get(fact_eid)
    for a in _attrs(eo):
        if fact_eid in (a.components or []):
            return a.eid
        if f is not None and V._canon(str(a.feature_group or "").lower()) == V._canon(str(f.source_field or "").lower()):
            return a.eid
    return None


def _same_field(eo: EvidenceObject, fact_eid: str, attr_eid: str) -> bool:
    """The fact IS the attribution's field (not merely one input of a derived field)."""
    f, a = eo.get(fact_eid), eo.get(attr_eid)
    if f is None or a is None:
        return False
    if V._canon(str(a.feature_group or "").lower()) == V._canon(str(f.source_field or "").lower()):
        return True
    comps = a.components or []
    return len(comps) == 1 and comps[0] == fact_eid


def fields_of(mentions: List[Mention], eo: EvidenceObject) -> Dict[str, List[str]]:
    """Group mentions into 'fields': attributions named directly, or through
    their own fact; facts that are only an INPUT of an attribution stay facts."""
    attrs = {m.eid for m in mentions if m.kind == "attr"}
    facts = [m.eid for m in mentions if m.kind == "fact"]
    via = set()
    for f in facts:
        o = _owner(eo, f)
        if o is not None and _same_field(eo, f, o):
            via.add(o)
    attrs |= via
    # a derived driver named by its definition: all of its direct inputs are named
    # ('the September bill as a share of the credit limit' = util_ratio)
    named_facts = set(facts)
    for a in _attrs(eo):
        own = [x for x in eo.by_type(EvidenceType.OBSERVED_FACT) if _same_field(eo, x.eid, a.eid)]
        for x in own:
            comps = list(x.components or [])
            if len(comps) >= 2 and set(comps) <= named_facts:
                attrs.add(a.eid)
                named_facts -= set(comps)
    facts = [x for x in facts if x in named_facts]
    loose = [f for f in facts if not any(_same_field(eo, f, a) for a in attrs)]
    return {"attrs": sorted(attrs), "facts": loose,
            "pred": [m.eid for m in mentions if m.kind == "pred"],
            "family": [m.eid for m in mentions if m.kind == "family"]}


def _inputs(eo: EvidenceObject, eid: str) -> set:
    """Every item reachable through `components` (inputs of inputs included)."""
    seen, stack = set(), list((eo.get(eid).components or []) if eo.get(eid) is not None else [])
    while stack:
        c = stack.pop()
        if c in seen or eo.get(c) is None:
            continue
        seen.add(c)
        stack += list(eo.get(c).components or [])
    return seen


def _fact_only_ids(eo: EvidenceObject) -> set:
    """Facts with no attribution of their own AND not an input (at any depth)
    of an attributed field: a direction for these cannot come from the model."""
    linked = set()
    for a in _attrs(eo):
        linked |= _inputs(eo, a.eid)
    return {f.eid for f in C._fact_only(eo) if f.eid not in linked}


# ---------------------------------------------------------------------------
# polarity reading
# ---------------------------------------------------------------------------
_CONTINUATION = re.compile(r"^\s*(?:,\s*)?(?:which|that|and|thereby|thus|so|as does|as do|also|both|"
                           r"(?:increas|rais|lower|reduc|decreas|push|pull|add|offset|mitigat)\w*ing)\b", re.I)


def subject_of(clause: str, f: Dict[str, List[str]], fs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """The clause's own named items; the sentence's only for a subjectless continuation
    ('..., which increases risk'). A clause whose subject is an unrecognised noun keeps
    no subject: it is unbound content for the semantic step, not someone else's claim."""
    if f["attrs"] or f["facts"]:
        return f
    if _CONTINUATION.search(clause):
        return fs
    return f


_SCORE_NP = r"(?:the\s+|its\s+|this\s+)?(?:(?:model|borrower|applicant)'?s?\s+)?(?:(?:estimated|predicted|raw|overall)\s+)?(?:risk\s+score|score|risk|estimate)"
#: executable/1.1: 'pushes the model's risk score up/down', 'moves the risk score higher' (two-noun objects)
_EXTRA_DIR = [
    (re.compile(r"\b(?:push|pushes|pushed|pushing|move|moves|moved|moving|drive|drives|drove|driving|nudges?|shifts?)\s+"
                + _SCORE_NP + r"\s+(?:up|upwards?|higher)\b", re.I), +1, False),
    (re.compile(r"\b(?:push|pushes|pushed|pushing|move|moves|moved|moving|drive|drives|drove|driving|pull|pulls|pulled|"
                r"pulling|nudges?|shifts?)\s+" + _SCORE_NP + r"\s+(?:down|downwards?|lower)\b", re.I), -1, False),
    (re.compile(r"\b(?:pulls?|pulled|pulling|drags?|dragged)\s+down\s+" + _SCORE_NP, re.I), -1, False),
]


def polarities(clause: str) -> List[Tuple[int, int, bool]]:
    """(polarity, position, negated) for each direction phrase about the score."""
    out = []
    for pat, pol, _passive in list(V._DIR_PATTERNS) + _EXTRA_DIR:
        for m in pat.finditer(clause):
            neg = V._governed_neg(clause[:m.start()]) or bool(re.search(r"\b(?:neither|nor)\s+$", clause[:m.start()], re.I))
            out.append((pol, m.start(), neg))
    # de-duplicate overlapping matches of the same polarity
    out.sort(key=lambda x: x[1])
    ded: List[Tuple[int, int, bool]] = []
    for p in out:
        if ded and abs(ded[-1][1] - p[1]) < 4 and ded[-1][0] == p[0]:
            continue
        ded.append(p)
    return ded


def _sign(a: EvidenceItem) -> int:
    c = a.contribution or 0.0
    if a.direction == "increases_risk":
        return +1
    if a.direction == "decreases_risk":
        return -1
    return (c > 0) - (c < 0)


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------
def _values(eo: EvidenceObject, it: EvidenceItem) -> List[float]:
    """Numbers an explanation may legitimately attach to an item."""
    vals = list(V._item_values(it))
    if it.etype is EvidenceType.MODEL_ATTRIBUTION:
        vals = [v for v in vals if v != it.contribution] + [it.contribution or 0.0, abs(it.contribution or 0.0)]
    for c in it.components or []:
        ci = eo.get(c)
        if ci is not None:
            vals += V._item_values(ci)
    if it.etype is EvidenceType.MODEL_ATTRIBUTION:      # the attribution's own fact(s)
        for f in eo.by_type(EvidenceType.OBSERVED_FACT):
            if _same_field(eo, f.eid, it.eid) and f.eid not in (it.components or []):
                vals += V._item_values(f)
    vals += V._text_numbers(it.value_label) if isinstance(it.value_label, str) else []
    if isinstance(it.value, str):
        vals += V._text_numbers(it.value)
    return vals


def _num_ok(n: dict, vals: List[float], pct_vals: List[float]) -> bool:
    if n["pct"] or n["pp"]:
        if any(V._num_matches(n, v) for v in pct_vals):
            return True
        m = dict(n, v=n["v"] / 100.0, lo=n["lo"] / 100.0, hi=n["hi"] / 100.0, tol=n["tol"] / 100.0)
        return any(V._num_matches(m, v, mag_ok=True) for v in vals)
    return any(V._num_matches(n, v, mag_ok=True) for v in vals)


def _pct_values(it: EvidenceItem) -> List[float]:
    out = [float(v) for k, v in (it.equivalents or {}).items() if "percent" in k.lower()]
    if it.calibrated_pd is not None:
        out.append(100.0 * it.calibrated_pd)
    return out


def _skip_number(clause: str, n: dict) -> bool:
    v = n["v"]
    if float(v).is_integer() and 1900 <= abs(v) <= 2100 and not (n["pct"] or n["pp"]):
        return True                                   # a year
    after = clause[n["end"]:n["end"] + 4]
    if re.match(r"\s*(?:st|nd|rd|th)\b", after):
        return True                                   # ordinal
    p = n["pos"]
    if p >= 2 and clause[p - 1] in "-_#" and clause[p - 2].isalnum():
        return True                                   # an identifier: 'taiwan-17424', 'doc#3'
    return False


def op_value(clause: str, eo: EvidenceObject, f: Dict[str, List[str]], restrict: Optional[List[str]] = None
             ) -> List[OpResult]:
    """VALUE for each number in the clause, bound to the nearest named item."""
    out: List[OpResult] = []
    nums = [n for n in V._signed_numbers(clause) if not _skip_number(clause, n)]
    if not nums:
        return out
    ments = [m for m in bind(clause, eo)]
    items = [m for m in ments if m.kind in ("attr", "fact", "pred")]
    if restrict is not None:
        items = [m for m in items if m.eid in restrict]
    all_vals = []
    for it in eo.items:
        all_vals += _values(eo, it)
    all_pct = [p for it in eo.items for p in _pct_values(it)]
    string_nums = V._string_numbers(eo)
    field_ids = set(f["attrs"]) | set(f["facts"])
    for n in nums:
        if not items:
            continue
        # nearest named item, preferring one written BEFORE the number
        def dist(m: Mention) -> float:
            d = n["pos"] - m.end if m.end <= n["pos"] else (m.start - n["end"]) + 25
            return abs(d)
        cand = sorted(items, key=dist)
        m0 = cand[0]
        it = eo.get(m0.eid)
        is_pred = it.etype is EvidenceType.MODEL_PREDICTION
        if is_pred and _TIME_UNIT.match(clause[n["end"]:n["end"] + 14]):
            continue                                  # horizon wording: residual R_TARGET
        own_vals = _values(eo, it)
        if is_pred:
            own_vals = [it.calibrated_pd] if it.calibrated_pd is not None else []
        ok = _num_ok(n, own_vals, _pct_values(it))
        shown = clause[max(0, n["pos"]):n["end"]].strip()
        if ok:
            out.append(OpResult("VALUE", SUPPORT, [it.eid], clause, shown, own_vals[:4], "number matches the bound item"))
            continue
        # does it match another item named in the same clause? then the binding, not the value, is uncertain
        others = [eo.get(m.eid) for m in cand[1:]]
        if any(o is not None and _num_ok(n, _values(eo, o) if o.etype is not EvidenceType.MODEL_PREDICTION
                                          else [o.calibrated_pd], _pct_values(o)) for o in others):
            # measured, not adopted: treating this as unresolved cut synthetic binding escapes 9 -> 3 of 55
            # but sent 20 more correct development narratives to the model (wording_intervention.py)
            out.append(OpResult("VALUE", SUPPORT, [m.eid for m in cand[1:]], clause, shown, None,
                                "number matches another item named in the clause"))
            continue
        # 1.5: only the wording of the BOUND item (its components and own fact) and of the outcome/horizon
        own_strs = [it] + [eo.get(c) for c in (it.components or [])] + \
            [x for x in eo.by_type(EvidenceType.OBSERVED_FACT) if _same_field(eo, x.eid, it.eid)] + \
            ([eo.prediction] if eo.prediction is not None else [])
        bound_nums = V._text_numbers(*[sv for x in own_strs if x is not None for sv in
                                       (x.value, x.unit, x.observed_at, x.horizon, x.target_definition,
                                        x.explained_scale, x.reason, x.meaning, x.value_label)])
        if any(V._num_matches(n, s) for s in (bound_nums if STRICT_WORDING else string_nums)):
            out.append(OpResult("VALUE", SUPPORT, [it.eid], clause, shown, None,
                                "number repeats the evidence wording (outcome, horizon, unit or label)"))
            continue
        small_count = (float(n["v"]).is_integer() and abs(n["v"]) <= 12 and not (n["pct"] or n["pp"]))
        if small_count:
            out.append(OpResult("VALUE", UNRES, [it.eid], clause, shown, own_vals[:4],
                                "small count or a number from evidence wording; binding not established"))
            continue
        named_ids = {m.eid for m in cand}
        derived = [d for d in eo.items if d.eid not in named_ids and (_inputs(eo, d.eid) & named_ids)
                   and _num_ok(n, _values(eo, d), _pct_values(d))]
        if derived:
            out.append(OpResult("VALUE", SUPPORT, [d.eid for d in derived[:2]], clause, shown, None,
                                "number matches a derived field whose inputs the clause names"))
            continue
        tight = _tight(clause, m0, n)
        other_field = _num_ok(n, all_vals, all_pct)
        n_fields = len(field_ids | ({it.eid} if not is_pred else set())) + (1 if f["pred"] else 0)
        if tight:
            out.append(OpResult("VALUE", CONTRA, [it.eid], clause, shown, own_vals[:4],
                                "number written next to the item's name does not match its value"
                                + (" (it matches a different field)" if other_field else "")))
        elif other_field:
            out.append(OpResult("VALUE", UNRES, [it.eid], clause, shown, own_vals[:4],
                                "number matches another field whose name may not be recognised; binding not established"))
        elif n_fields > 1 and not (is_pred and not field_ids):
            out.append(OpResult("VALUE", UNRES, [m.eid for m in cand[:2]], clause, shown, own_vals[:4],
                                "number does not match its nearest item, and the clause names several items"))
        else:
            out.append(OpResult("VALUE", CONTRA, [it.eid], clause, shown, own_vals[:4],
                                "number matches no evidence value and the clause names one item"))
    return out


_TIGHT_GAP = re.compile(r"[\s(:=,]*(?:(?:is|was|were|of|at|stood at|equal to|equals|amounted to|recorded(?: value)?|value|"
                        r"score|ratio|amount|totall?ing|reached|NT\$|\$|₹|currency units|about|approximately|around|"
                        r"roughly|=|:)[\s(:=,]*){0,4}", re.I)


def _tight(clause: str, m: "Mention", n: dict) -> bool:
    """The number is written directly after the item's name ('X (0.76)', 'X of 0.76', 'X was 0.76')."""
    if m.end > n["pos"]:
        return False
    gap = clause[m.end:n["pos"]]
    return len(gap) <= 30 and bool(_TIGHT_GAP.fullmatch(gap))


_LOGODDS_LABEL = re.compile(r"\b(log[- ]?odds|logit|raw (?:score|margin)|margin|shap|score units?|points? on the (?:raw )?score)\b", re.I)


def op_scale(clause: str, eo: EvidenceObject, f: Dict[str, List[str]]) -> List[OpResult]:
    """A contribution written as a percentage or as a change in probability."""
    out = []
    if not f["attrs"]:
        return out
    for n in V._signed_numbers(clause):
        for aid in f["attrs"]:
            a = eo.get(aid)
            c = abs(a.contribution or 0.0)
            if c == 0:
                continue
            as_pct = (n["pct"] or n["pp"]) and V._num_matches(n, 100 * c)
            as_prob = (n["pct"] or n["pp"]) and V._num_matches(n, c)
            raw = (not (n["pct"] or n["pp"])) and V._num_matches(n, c, mag_ok=True)
            shown = clause[n["pos"]:n["end"]].strip()
            if as_pct or as_prob:
                out.append(OpResult("SCALE", CONTRA, [aid], clause, shown, a.contribution,
                                    "a raw log-odds contribution is written as a percentage"))
            elif raw:
                after = clause[n["end"]:n["end"] + 60]
                before = clause[max(0, n["pos"] - 70):n["pos"]]
                if _LOGODDS_LABEL.search(after) or _LOGODDS_LABEL.search(before):
                    out.append(OpResult("SCALE", SUPPORT, [aid], clause, shown, a.contribution,
                                        "contribution labelled on the raw score scale"))
                elif re.search(r"\b(probability|chance|likelihood|pd)\b", before + " " + after, re.I) and \
                        re.search(r"\b(by|of)\s*$", before, re.I):
                    out.append(OpResult("SCALE", CONTRA, [aid], clause, shown, a.contribution,
                                        "a raw log-odds contribution is presented as a change in probability"))
    return out


def op_direction(clause: str, sentence: str, eo: EvidenceObject, f: Dict[str, List[str]],
                 fs: Dict[str, List[str]]) -> List[OpResult]:
    """Stated direction vs attribution sign; fact-only fields given a direction."""
    pols = polarities(clause)
    subj = subject_of(clause, f, fs)
    zero = _ZERO_EFFECT.search(clause)
    if zero and len(subj["attrs"]) == 1 and not V._governed_neg(clause[:zero.start()]):
        a = eo.get(subj["attrs"][0])
        st = SUPPORT if abs(a.contribution or 0.0) < ZERO_TOL else CONTRA
        return [OpResult("DIRECTION", st, [a.eid], clause, 0, a.contribution,
                         "the text says the driver has no effect; its contribution is "
                         + ("near zero" if st == SUPPORT else f"{a.contribution:+.4f}"))]
    if not pols and _ELLIPSIS.search(clause) and sentence != clause:
        rest = sentence.replace(clause, " ")
        ps = {p for p, _, neg in polarities(rest) if not neg}
        if len(ps) == 1:
            pols = [(ps.pop(), 0, False)]             # 'X raises risk, as does Y': Y inherits the direction
    if not pols:
        return []
    attrs, facts = subj["attrs"], subj["facts"]
    if not attrs and not facts:
        return []
    if any(neg for _, _, neg in pols):
        return [OpResult("DIRECTION", UNRES, attrs + facts, clause, None, None, "negated direction phrase")]
    ps = {p for p, _, _ in pols}
    if len(ps) > 1:
        if len(attrs) >= 2 and len(pols) == len(attrs):
            # 'X raises it while Y lowers it' - pair in order of mention
            ments = [m for m in bind(clause, eo) if m.eid in attrs]
            if len(ments) == len(pols):
                res = []
                for m, (p, _, _) in zip(sorted(ments, key=lambda x: x.start), sorted(pols, key=lambda x: x[1])):
                    res.append(_dir_one(eo.get(m.eid), p, clause))
                return res
        return [OpResult("DIRECTION", UNRES, attrs + facts, clause, None, None, "mixed directions in one clause")]
    p = ps.pop()
    out: List[OpResult] = []
    fo = _fact_only_ids(eo)
    if not attrs:
        if len(facts) == 1 and facts[0] in fo:
            return [OpResult("FACT_DRIVER", CONTRA, facts, clause, "+" if p > 0 else "-", "no attribution",
                             "a direction is given for a field that has no attribution")]
        return [OpResult("DIRECTION", UNRES, facts, clause, p, None,
                         "direction given for an input of a derived driver or several facts")]
    signs = {_sign(eo.get(a)) for a in attrs}
    if len(attrs) > 1 and len(signs) > 1:
        return [OpResult("DIRECTION", UNRES, attrs, clause, p, sorted(signs),
                         "one direction stated for several drivers with different signs")]
    for a in attrs:
        out.append(_dir_one(eo.get(a), p, clause))
    # other fields named as subjects in the same clause (outside parentheses) - 'X and Y
    # increase risk, as does Z' where Z has no attribution of its own: not proved either way
    prot = V._protect(clause)
    ments = {m.eid: m for m in bind(clause, eo)}
    extra = [x for x in facts if x in ments and prot[ments[x].start:ments[x].end].strip()
             and not any(x in _inputs(eo, a) for a in attrs)]
    if extra:
        out.append(OpResult("DIRECTION", UNRES, extra, clause, "+" if p > 0 else "-", None,
                            "the stated direction also covers a field without its own attribution"))
    return out


_ELLIPSIS = re.compile(r"^\s*,?\s*(?:as|so)\s+(?:does|do|did|is|are)\b|\b(?:as|so)\s+(?:does|do|did)\s+(?:the\s+)?\w", re.I)
_ZERO_EFFECT = re.compile(r"\b(?:has|have|had|with|shows?)\s+(?:no|zero|negligible|little or no)\s+(?:recorded\s+|measurable\s+|"
                          r"meaningful\s+|real\s+)?(?:effect|impact|influence|contribution)\b(?!\s+(?:on default|in the real world))|"
                          r"\b(?:does|do|did) not (?:affect|influence|change|move) (?:the\s+)?(?:model'?s?\s+)?(?:risk|score|estimate)\b|"
                          r"\bis neutral\b", re.I)


def _dir_one(a: EvidenceItem, p: int, clause: str) -> OpResult:
    if abs(a.contribution or 0.0) < ZERO_TOL:
        return OpResult("DIRECTION", UNRES, [a.eid], clause, p, a.contribution, "near-zero contribution")
    s = _sign(a)
    st = SUPPORT if s == p else CONTRA
    return OpResult("DIRECTION", st, [a.eid], clause, "+" if p > 0 else "-", a.direction,
                    "stated direction " + ("matches" if st == SUPPORT else "is opposite to") + " the attribution sign")


_RANK_SCOPE_OUT = re.compile(r"\b(second|third|fourth|least|smallest|weakest)\b", re.I)
_RANK_DOWN = re.compile(r"\b(?:strongest|largest|biggest|greatest|main|primary|principal|most important|leading|top)\s+"
                        r"(?:risk[- ]reducing|protective|negative|downward|mitigating|favourable|favorable)\b|"
                        r"\b(?:driver|factor|contributor)s?\s+(?:lowering|reducing|decreasing|pulling down)\b", re.I)
_RANK_UP = re.compile(r"\b(?:strongest|largest|biggest|greatest|main|primary|principal|most important|leading|top)\s+"
                      r"(?:risk[- ]increasing|positive|upward|adverse|aggravating)\b|"
                      r"\b(?:driver|factor|contributor)s?\s+(?:raising|increasing|pushing up)\b", re.I)
#: 'the strongest driver increasing risk' reads either as 'the strongest driver, which
#: increases risk' or as 'the strongest of the risk-increasing drivers': ambiguous scope
_RANK_PARTICIPLE = re.compile(r"\b(?:driver|factor|contributor)s?\s+(?:raising|increasing|pushing up|lowering|reducing|"
                              r"decreasing|pulling down)\b", re.I)


def op_rank(clause: str, sentence: str, eo: EvidenceObject, f: Dict[str, List[str]],
            fs: Dict[str, List[str]]) -> List[OpResult]:
    out: List[OpResult] = []
    attrs_all = _attrs(eo)
    if not attrs_all:
        return out
    for m in C._RANK_CLAIM.finditer(clause):
        pre = clause[:m.start()]
        if V._governed_neg(pre) or re.search(r"\bone of the\s*$", pre, re.I):
            continue
        if _RANK_SCOPE_OUT.search(clause[max(0, m.start() - 20):m.end() + 10]):
            out.append(OpResult("RANK", UNRES, [], clause, m.group(0), None, "rank beyond first place: out of scope"))
            continue
        subj = subject_of(clause, f, fs)
        if not (subj["attrs"] or subj["facts"]) and fs is not f and (fs["attrs"] or fs["facts"]):
            # 'The strongest driver is X': the subject may follow in a later clause of the sentence
            subj = fs if len(fs["attrs"]) == 1 else subj
        pool = attrs_all
        restr = None
        win = clause[m.start():m.end() + 30]
        if _RANK_DOWN.search(win):
            pool, restr = [a for a in attrs_all if _sign(a) < 0], "decreases_risk"
        elif _RANK_UP.search(win):
            pool, restr = [a for a in attrs_all if _sign(a) > 0], "increases_risk"
        if not pool:
            out.append(OpResult("RANK", CONTRA, subj["attrs"], clause, m.group(0), None,
                                f"no attribution has direction {restr}"))
            continue
        top = max(pool, key=lambda a: abs(a.contribution or 0.0))
        tv = abs(top.contribution or 0.0)
        tied = {a.eid for a in pool if abs(abs(a.contribution or 0.0) - tv) <= TIE_TOL}
        named = subj["attrs"]
        if restr and _RANK_PARTICIPLE.search(win) and len(named) == 1:
            gv = max(abs(a.contribution or 0.0) for a in attrs_all)
            g_tied = {a.eid for a in attrs_all if abs(abs(a.contribution or 0.0) - gv) <= TIE_TOL}
            if (named[0] in tied) != (named[0] in g_tied):
                out.append(OpResult("RANK", UNRES, [named[0]] + sorted(tied | g_tied), clause, named[0],
                                    {"restricted": sorted(tied), "overall": sorted(g_tied)},
                                    "rank scope is ambiguous (overall vs within one direction) and the readings disagree"))
                continue
        if len(named) == 1:
            st = SUPPORT if named[0] in tied else CONTRA
            out.append(OpResult("RANK", st, [named[0]] + sorted(tied), clause, named[0], sorted(tied),
                                f"argmax |contribution| over {'displayed' if not restr else restr} attributions"
                                + (" (tied)" if len(tied) > 1 else "")))
        elif not named and len(subj["facts"]) == 1 and subj["facts"][0] in _fact_only_ids(eo):
            out.append(OpResult("RANK", CONTRA, subj["facts"] + sorted(tied), clause, subj["facts"][0], sorted(tied),
                                "a field with no attribution is called the strongest driver"))
        else:
            out.append(OpResult("RANK", UNRES, named + subj["facts"], clause, named, sorted(tied),
                                "the rank claim's subject is not a single named driver"))
    return out


def op_temporal(clause: str, eo: EvidenceObject, f: Dict[str, List[str]]) -> List[OpResult]:
    out: List[OpResult] = []
    low = clause.lower()
    dir_spans = []
    for pat, _p, _ in V._DIR_PATTERNS:
        dir_spans += [(m.start(), m.end()) for m in pat.finditer(clause)]
    for t in V._TREND_TERMS:
        for m in re.finditer(r"\b" + re.escape(t) + r"\b", low):
            if any(s <= m.start() < e for s, e in dir_spans):
                continue                          # 'increases the risk score': a direction, not a trend
            pre = clause[:m.start()]
            if V._disclaimed(pre) or V._TREND_DISCLAIM.search(pre) or V._SCORE_SUBJ.search(pre):
                continue
            if re.match(r"\s+(?:the\s+)?(?:model'?s?\s+)?(?:risk|score|probability|estimate|odds)", clause[m.end():], re.I):
                continue
            items = f["attrs"] + f["facts"]
            if not items:
                out.append(OpResult("TEMPORAL", UNRES, [], clause, t, None, "change asserted about an unnamed quantity"))
                break
            out.append(_temporal_one(clause, eo, items, t))
            break
    return out[:1]


def _series_items(eo: EvidenceObject, eid: str) -> List[EvidenceItem]:
    """Dated facts of the item's series (through its own fact or one input)."""
    it = eo.get(eid)
    cand = [it] + [eo.get(c) for c in (it.components or [])]
    for f in eo.by_type(EvidenceType.OBSERVED_FACT):
        if it.etype is EvidenceType.MODEL_ATTRIBUTION and _same_field(eo, f.eid, it.eid):
            cand.append(f)
    sers = {c.series for c in cand if c is not None and c.series}
    if len(sers) != 1:
        return []
    s = sers.pop()
    return sorted([x for x in eo.by_type(EvidenceType.OBSERVED_FACT) if x.series == s and x.period_order is not None
                   and isinstance(x.value, (int, float))], key=lambda x: -x.period_order)


def _temporal_one(clause: str, eo: EvidenceObject, items: List[str], term: str) -> OpResult:
    up = term in V._TREND_UP
    dn = term in V._TREND_DN
    for eid in items:
        ser = _series_items(eo, eid)
        if len(ser) >= 2:
            named = [x for x in ser if V._period_mentions(clause, x)]
            pair = (named[0], named[-1]) if len(named) >= 2 else ((ser[0], ser[-1]) if len(ser) == 2 else None)
            if pair is None or not (up or dn):
                return OpResult("TEMPORAL", UNRES, [x.eid for x in ser], clause, term, None,
                                "several dated periods shown; the compared periods or direction are not determined")
            old, new = pair
            d = (new.value or 0) - (old.value or 0)
            if d == 0:
                return OpResult("TEMPORAL", CONTRA, [old.eid, new.eid], clause, term, 0,
                                "the dated values are equal: no change")
            st = SUPPORT if (d > 0) == up else CONTRA
            return OpResult("TEMPORAL", st, [old.eid, new.eid], clause, term, d,
                            f"{old.observed_at} {old.value} -> {new.observed_at} {new.value}")
    # no item named in the clause has two dated observations
    only = items[0]
    it = eo.get(only)
    if len(items) == 1 and (it.etype is EvidenceType.MODEL_ATTRIBUTION or it.series is None or
                            len(_series_items(eo, only)) < 2):
        return OpResult("TEMPORAL", CONTRA, items, clause, term, None,
                        "a change over time is asserted but the evidence shows one observation")
    return OpResult("TEMPORAL", UNRES, items, clause, term, None, "binding of the trend subject is ambiguous")


_MEANING = re.compile(C._MEANING_WORDS.pattern + r"|\bno (?:recorded |reported |documented )?(?:payment )?delays?\b|"
                      r"\bnot? (?:been )?(?:late|overdue|in arrears)\b|\bpaid (?:on time|as agreed)\b", re.I)


def op_code(clause: str, eo: EvidenceObject, f: Dict[str, List[str]]) -> List[OpResult]:
    out: List[OpResult] = []
    if not _MEANING.search(clause):
        return out
    hedged = bool(C._HEDGE.search(clause))
    cand = []
    for eid in f["facts"] + f["attrs"]:
        it = eo.get(eid)
        ids = [it] + [eo.get(c) for c in (it.components or [])]
        if it.etype is EvidenceType.MODEL_ATTRIBUTION:
            ids += [x for x in eo.by_type(EvidenceType.OBSERVED_FACT) if _same_field(eo, x.eid, it.eid)]
        cand += [x for x in ids if x is not None and x.value_label]
    seen = set()
    cand = [x for x in cand if not (x.eid in seen or seen.add(x.eid))]
    if not cand:
        return out
    undoc = [x for x in cand if C._HEDGE.search(str(x.value_label))]
    doc = [x for x in cand if x not in undoc]
    if undoc and not doc:
        st = SUPPORT if hedged else CONTRA
        out.append(OpResult("CODE", st, [x.eid for x in undoc], clause,
                            _MEANING.search(clause).group(0), [x.value_label for x in undoc][:3],
                            "the source does not document this code" + ("; the text says so" if hedged else
                                                                         "; the text states a meaning")))
    elif doc and not undoc:
        delay = [x for x in doc if isinstance(x.value, (int, float)) and x.value >= 1]
        if delay and re.search(r"duly|on[- ]time|no (?:\w+ )?delay|not (?:been )?late|timely", clause, re.I):
            out.append(OpResult("CODE", CONTRA, [x.eid for x in delay], clause, _MEANING.search(clause).group(0),
                                [x.value_label for x in delay][:3], "the documented code records a payment delay"))
        else:
            out.append(OpResult("CODE", SUPPORT, [x.eid for x in doc], clause, _MEANING.search(clause).group(0),
                                [x.value_label for x in doc][:3], "consistent with the documented code label"))
    else:
        out.append(OpResult("CODE", UNRES, [x.eid for x in cand], clause, None, None,
                            "documented and undocumented codes in one clause"))
    return out


_HORIZON = re.compile(r"\b(?:(\d+|one|two|three|six|twelve|twenty-four)[- ]?(month|year)s?|next (month|year)|"
                      r"(?:the )?coming (month|year)|annual(?:ly)?|lifetime|first (?:\w+ )?(?:loan )?instal?ments?)\b", re.I)
_NUMWORD = {"one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12, "twenty-four": 24}


def _horizon_key(m) -> str:
    t = m.group(0).lower()
    if m.group(1):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMWORD.get(m.group(1).lower(), 0)
        unit = m.group(2).lower()
        months = n * (12 if unit.startswith("year") else 1)
        return "next month" if months == 1 else f"{months} months"
    if "next month" in t or "coming month" in t:
        return "next month"
    if "next year" in t or "coming year" in t or "annual" in t:
        return "12 months"
    if "instal" in t:
        return "first instalments"
    return t


def op_target(clause: str, eo: EvidenceObject, f: Dict[str, List[str]]) -> List[OpResult]:
    """Stated horizon of the estimate vs the evidence horizon (documented, incl. 'not 12 months')."""
    pred = eo.prediction
    if pred is None or not f["pred"] or not pred.horizon:
        return []
    hz = pred.horizon.lower()
    has = {_horizon_key(m) for m in _HORIZON.finditer(hz)
           if not re.search(r"\bnot\s+(?:a\s+|an\s+|over\s+)?$", hz[:m.start()])}
    denied = {_horizon_key(m) for m in _HORIZON.finditer(hz)
              if re.search(r"\bnot\s+(?:a\s+|an\s+|over\s+)?$", hz[:m.start()])}
    out = []
    for m in _HORIZON.finditer(clause):
        k = _horizon_key(m)
        pre = clause[:m.start()]
        post = clause[m.end():m.end() + 12]
        # a duration is a HORIZON only when it frames the estimate ('within 12 months',
        # 'a 12-month estimate'), not when it measures something ('a delay of 2 months')
        if m.group(1) and not (re.search(r"\b(?:within|over|for|in|during|across|on)\s+(?:the\s+|a\s+|an\s+)?"
                                         r"(?:next\s+|coming\s+|following\s+|first\s+)?$", pre, re.I)
                               or re.match(r"\s*(?:horizon|window|period|estimate|outlook|forecast)\b", post, re.I)
                               or re.search(r"\bnot\s+(?:[\w-]+\s+){0,3}$", pre, re.I)):
            continue
        neg = bool(re.search(r"\b(?:not|never|no)\s+(?:[\w-]+\s+){0,3}$", pre, re.I)) or V._governed_neg(pre)
        if neg:
            st = SUPPORT if k in denied else UNRES
            out.append(OpResult("TARGET", st, [pred.eid], clause, f"not {k}", sorted(has),
                                "the text denies a horizon the evidence also excludes" if st == SUPPORT
                                else "a denied horizon the evidence does not mention"))
        elif k in has:
            out.append(OpResult("TARGET", SUPPORT, [pred.eid], clause, k, sorted(has), "horizon matches the evidence"))
        elif k in denied or has:
            out.append(OpResult("TARGET", CONTRA, [pred.eid], clause, k, sorted(has),
                                f"the evidence horizon is {sorted(has) or pred.horizon}, not {k}"))
    return out


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
#: an action said to change the outcome, beyond the structural checker's causal patterns
_ACTION_EFFECT = re.compile(
    r"(?:\b\w+ing\b|\bif\b)[^.;]{0,80}?\b(?:would|will|could|may|might|can)\s+(?:\w+\s+){0,2}?"
    r"(?:make|render)\s+(?:a\s+|the\s+)?default\s+(?:more|less)\s+likely\b", re.I)


def causal_assertion(text: str):
    for pat in list(V._CAUSAL_PATTERNS) + [_ACTION_EFFECT]:
        m = pat.search(text)
        if m:
            return m
    return None


def _residual(clause: str, eo: EvidenceObject, f: Dict[str, List[str]], has_ops: bool) -> List[str]:
    r = []
    m = causal_assertion(clause)
    if m and not V._disclaimed(clause[:m.start()]) and not C._DISCLAIMER.search(clause[:m.end() + 40]):
        r.append("R_CAUSAL")
    if f["pred"] and not has_ops:
        for m in re.finditer(r"\b(\d+|one|two|twelve|next|coming|following)\s*(?:-\s*)?(month|year|week)s?\b|"
                             r"\bannual|\bwithin\b", clause, re.I):
            if not V._governed_neg(clause[:m.start()]):
                r.append("R_TARGET")
                break
    if not (f["attrs"] or f["facts"] or f["pred"]) and not f.get("family") and not _CONTINUATION.search(clause) and \
            (polarities(clause) or C._RANK_CLAIM.search(clause)):
        r.append("R_INVENTED")
    return r


_ELLIPSIS_SPLIT = re.compile(r",\s*(?=(?:as|so)\s+(?:does|do|did)\b)", re.I)


def _subclauses(sent: str) -> List[Tuple[int, str]]:
    """Clauses, with an elliptical tail split off: 'X raises risk, as does Y' -> 'X raises risk' | 'as does Y'."""
    out = []
    for off, cl in V._clauses(sent):
        last = 0
        for m in _ELLIPSIS_SPLIT.finditer(cl):
            out.append((off + last, cl[last:m.start()]))
            last = m.end()
        out.append((off + last, cl[last:]))
    return [(o, c) for o, c in out if c.strip()]


def split_claims(text: str) -> List[Tuple[int, str]]:
    """Sentences, independently of any generator-supplied claim list (§D2 step 2).
    Decimals, abbreviations and parenthetical asides never split; a sentence may
    start in lower case (template sentences start with a field name)."""
    prot = V._protect(text)
    out, last = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=\S)", prot):
        seg = text[last:m.start()]
        if seg.strip():
            out.append((last + len(seg) - len(seg.lstrip()), seg.strip()))
        last = m.end()
    seg = text[last:]
    if seg.strip():
        out.append((last + len(seg) - len(seg.lstrip()), seg.strip()))
    return out


# ---------------------------------------------------------------------------
# executable/1.6 zero-call gate (Improvement_Plan WS1): code-only support is kept only for the validated grammar
# ---------------------------------------------------------------------------
ZERO_CALL_GATE = True
_NEG_CUE = re.compile(r"\b(?:not|no|never|neither|nor|without|cannot)\b|n't\b", re.I)
_HORIZON_CUE = re.compile(r"\b(?:term|instal?ments?|horizon|period|months?|years?|annual|next|within|over the)\b", re.I)
_UNDOC_SAID = re.compile(r"\bnot (?:documented|defined|described)|\bundocumented\b|\bno documented\b|\bunknown meaning", re.I)
_DICT_CACHE: Dict[str, List[Tuple[str, List[str]]]] = {}
_CODE_MEANING_CUE = re.compile(r"\b(?:paid|pay(?:s|ing)? (?:on time|duly|in full)|duly|on time|timely|punctual|no (?:payment )?"
                               r"delay|without delay|delay|late|behind|overdue|good|positive|clean|history|revolving|"
                               r"minimum|meaning|means|indicat|signal|suggest|reflect)\w*", re.I)


def _dictionary(dataset: str) -> List[Tuple[str, List[str]]]:
    if dataset not in _DICT_CACHE:
        try:
            import feature_semantics as FS
            sp = FS.specs(dataset)
        except Exception:  # noqa: BLE001
            sp = {}
        out = []
        for k, s_ in sp.items():
            names = {k.lower(), k.lower().replace("_", " ")} | {a.lower() for a in (getattr(s_, "aliases", None) or [])}
            out.append((k, sorted((n for n in names if len(n) >= 4), key=len, reverse=True)))
        _DICT_CACHE[dataset] = out
    return _DICT_CACHE[dataset]


def _evidence_fields(eo: EvidenceObject) -> set:
    out = set()
    for it in eo.items:
        for n in (it.source_field, it.feature_group):
            if n:
                out.add(str(n).replace(" (derived)", "").strip().lower())
        for c in it.components or []:
            ci = eo.get(c)
            if ci is not None and ci.source_field:
                out.add(str(ci.source_field).replace(" (derived)", "").strip().lower())
    return out


def absent_field_mentions(clause: str, eo: EvidenceObject) -> List[Tuple[str, int, int]]:
    """Documented dataset fields named in the clause that are NOT in this borrower's evidence (field identity)."""
    low = V._canon(clause.lower())
    ev = _evidence_fields(eo)
    present_spans = [(m.start, m.end) for m in bind(clause, eo)]
    hits = []
    for fld, names in _dictionary(eo.dataset):
        if fld.lower() in ev:
            continue
        for n in names:
            for m in re.finditer(rf"(?<![\w]){re.escape(V._canon(n))}(?![\w])", low):
                if any(a <= m.start() and m.end() <= b for a, b in present_spans):
                    continue                    # part of the name of a field that IS in the evidence
                hits.append((fld, m.start(), m.end()))
                break
    return hits


def gate(cl: str, eo: EvidenceObject, f: Dict[str, List[str]], ops: List[OpResult]) -> Tuple[List[OpResult], List[str]]:
    """Extra contradictions and residual tags for one clause (1.6). Residual tags send the claim to the judge."""
    extra, res = [], []
    present = [(m.start, m.end) for m in bind(cl, eo)]
    for fld, s0, e0 in absent_field_mentions(cl, eo):
        tail = cl[e0:e0 + 45]
        nums = V._signed_numbers(tail)
        conf = None
        # a name qualified in the same phrase by a field that IS in the evidence ('credit amount for the goods
        # (AMT_GOODS_PRICE)') is ambiguous, not a proved confusion: it goes to the judge
        qualified = any(abs(a - e0) <= 40 or abs(s0 - b) <= 40 for a, b in present)
        for n in ([] if qualified else nums[:1]):
            for it in eo.by_type(EvidenceType.OBSERVED_FACT):
                if _num_ok(n, V._item_values(it), _pct_values(it)) and n["v"] != 0:
                    conf = it
                    break
        if conf is not None:
            # the phrase also uses distinctive words of the quoted field's own names: ambiguous, not proved
            names = " ".join(str(x or "") for x in [conf.source_field, conf.meaning] + list(conf.aliases or []))
            for a_ in eo.by_type(EvidenceType.MODEL_ATTRIBUTION):
                if conf.eid in (a_.components or []) or _same_field(eo, conf.eid, a_.eid):
                    names += " " + " ".join(str(x or "") for x in [a_.feature_group] + list(a_.aliases or []))
            window = cl[max(0, s0 - 25):e0 + 25]
            if _words(window.replace("_", " ")) & _words(names.replace("_", " ")):
                conf = None
        if conf is not None:
            extra.append(OpResult("VALUE", CONTRA, [conf.eid], cl, cl[s0:e0 + len(tail)].strip()[:80], [conf.value],
                                  f"field identity: the text names {fld}, which is not in the evidence, for the value "
                                  f"of {conf.source_field}"))
        else:
            res.append("R_FIELD_IDENTITY")
    if _NEG_CUE.search(cl) and not _is_disclaimer(cl) and \
            any(o.op in ("DIRECTION", "FACT_DRIVER", "RANK") and o.status == SUPPORT for o in ops):
        res.append("R_NEGATION_SCOPE")
    coded = [eo.get(e) for o in ops for e in (o.evidence_ids or []) if eo.get(e) is not None]
    coded += [eo.get(e) for e in (f.get("facts") or []) if eo.get(e) is not None]
    if any(re.search(r"not (?:documented|defined)", f"{it.unit} {it.value_label}", re.I) and "code" in str(it.unit).lower()
           for it in coded) and _CODE_MEANING_CUE.search(cl) and not _UNDOC_SAID.search(cl):
        res.append("R_CODE_SCOPE")
    if f.get("pred") and _HORIZON_CUE.search(cl) and not any(o.op == "TARGET" and o.status == SUPPORT for o in ops):
        res.append("R_TARGET")
    return extra, res


def execute_claims(text: str, eo: EvidenceObject) -> List[ClaimExec]:
    text = clean(text)
    out: List[ClaimExec] = []
    for k, (start, sent) in enumerate(split_claims(text)):
        ce = ClaimExec(f"c{k + 1}", sent, start)
        fs = fields_of(bind(sent, eo), eo)
        for off, cl in _subclauses(sent):
            f = fields_of(bind(cl, eo), eo)
            ops = []
            ops += op_value(cl, eo, f)
            ops += op_scale(cl, eo, f)
            ops += op_direction(cl, sent, eo, f, fs)
            ops += op_rank(cl, sent, eo, f, fs)
            ops += op_temporal(cl, eo, f)
            ops += op_code(cl, eo, subject_of(cl, f, fs))
            ops += op_target(cl, eo, f)
            # a SCALE contradiction explains the same number's VALUE mismatch
            sc = {o.asserted for o in ops if o.op == "SCALE" and o.status == CONTRA}
            ops = [o for o in ops if not (o.op == "VALUE" and o.status == CONTRA and o.asserted in sc)]
            ce.ops += ops
            ce.residual += [r for r in _residual(cl, eo, f, any(o.op == "TARGET" for o in ops))
                            if r not in ce.residual]
            if ZERO_CALL_GATE:
                extra, gres = gate(cl, eo, f, ops)
                ce.ops += extra
                ce.residual += [r for r in gres if r not in ce.residual]
        ce.material = not (_is_disclaimer(sent) and not ce.ops and not ce.residual)
        out.append(ce)
    return out


_SCALE_NOTE = re.compile(r"\blog[- ]?odds\b|\braw (?:score|margin)\b|not (?:changes? in )?the (?:calibrated )?(?:probability|pd)\b|"
                         r"\bnot the calibrated\b|\brelative to (?:the|its) (?:model'?s )?reference\b", re.I)


def _is_disclaimer(s: str) -> bool:
    """A model/causation disclaimer or a scale note: no driver, no direction, no number."""
    return (bool(C._DISCLAIMER.search(s)) or bool(_SCALE_NOTE.search(s))) and not polarities(s) \
        and not C._RANK_CLAIM.search(s) and not re.search(r"\d", s)


# ---------------------------------------------------------------------------
# text-level operations
# ---------------------------------------------------------------------------
def op_coverage(text: str, eo: EvidenceObject, claims: List[ClaimExec]) -> List[OpResult]:
    text = clean(text)
    out: List[OpResult] = []
    pred = eo.prediction
    unbound = [c for c in claims if c.status == UNRES]
    if pred is not None and pred.calibrated_pd is not None:
        nums = V._signed_numbers(text)
        ok = any(_num_ok(n, [pred.calibrated_pd], [100 * pred.calibrated_pd]) for n in nums)
        out.append(OpResult("COVERAGE", SUPPORT if ok else CONTRA, [pred.eid], "", "probability of default",
                            pred.calibrated_pd, "the estimate is " + ("stated" if ok else "not stated")))
    attrs = _attrs(eo)
    if attrs:
        tv = max(abs(a.contribution or 0.0) for a in attrs)
        tied = [a for a in attrs if abs(abs(a.contribution or 0.0) - tv) <= TIE_TOL]
        named = set()
        for c in claims:
            named |= set(fields_of(bind(c.text, eo), eo)["attrs"])
        hit = [a.eid for a in tied if a.eid in named]
        if hit:
            out.append(OpResult("COVERAGE", SUPPORT, hit, "", "strongest driver", [a.eid for a in tied],
                                "the strongest attribution is mentioned"))
        elif unbound:
            out.append(OpResult("COVERAGE", UNRES, [a.eid for a in tied], "", None, [a.eid for a in tied],
                                "strongest attribution not named, but some claims could not be interpreted"))
        else:
            out.append(OpResult("COVERAGE", CONTRA, [a.eid for a in tied], "", None, [a.eid for a in tied],
                                "the strongest attribution is not mentioned"))
    return out


def op_completeness(text: str, eo: EvidenceObject, claims: List[ClaimExec]) -> List[OpResult]:
    text = clean(text)
    attrs = _attrs(eo)
    out: List[OpResult] = []
    if len(attrs) < 2:
        return out
    named = set()
    for c in claims:
        named |= set(fields_of(bind(c.text, eo), eo)["attrs"])
    for c in claims:
        m = C._COMPLETE_CLAIM.search(c.text)
        if not m or V._governed_neg(c.text[:m.start()]) or re.search(r"\bnot\s+(?:the\s+)?$", c.text[:m.start()], re.I):
            continue
        win = c.text[m.start():m.end() + 40]
        pool = attrs
        if re.search(r"\b(rais|increas|push(?:es|ing)? up|adverse)", win, re.I):
            pool = [a for a in attrs if _sign(a) > 0]
        elif re.search(r"\b(lower|reduc|decreas|protect|mitigat)", win, re.I):
            pool = [a for a in attrs if _sign(a) < 0]
        missing = [a.eid for a in pool if a.eid not in named]
        if not missing:
            out.append(OpResult("COMPLETENESS", SUPPORT, [a.eid for a in pool], c.text, m.group(0), None,
                                "every attribution in scope is mentioned"))
        elif any(x.status == UNRES for x in claims if x is not c):
            out.append(OpResult("COMPLETENESS", UNRES, missing, c.text, m.group(0), missing,
                                "some claims could not be interpreted"))
        else:
            out.append(OpResult("COMPLETENESS", CONTRA, missing, c.text, m.group(0), missing,
                                f"{len(missing)} attribution(s) in scope are not mentioned"))
        break
    return out


@dataclass
class TextExec:
    claims: List[ClaimExec]
    text_ops: List[OpResult]

    @property
    def contradictions(self) -> List[OpResult]:
        return [o for c in self.claims for o in c.ops if o.status == CONTRA] + \
               [o for o in self.text_ops if o.status == CONTRA]

    @property
    def residual_claims(self) -> List[ClaimExec]:
        return [c for c in self.claims if c.status == UNRES]

    def to_dict(self) -> dict:
        return {"version": EXEC_VERSION, "claims": [c.to_dict() for c in self.claims],
                "text_ops": [o.to_dict() for o in self.text_ops]}


def execute_text(text: str, eo: EvidenceObject) -> TextExec:
    claims = execute_claims(text, eo)
    return TextExec(claims, op_coverage(text, eo, claims) + op_completeness(text, eo, claims))


# ---------------------------------------------------------------------------
# witnesses: an objection as an executable account of a violation
# ---------------------------------------------------------------------------
WITNESS_OUTCOMES = ("verified", "disproved", "malformed", "unexecutable", "semantic_plausible", "semantic_invalid")


@dataclass
class Witness:
    rule: str
    quote: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    operation: Optional[str] = None
    claimed_result: Any = None
    alleged_conflict: str = ""
    claim_id: Optional[str] = None
    reason: str = ""
    # filled by execute_witness
    outcome: Optional[str] = None
    problems: List[str] = field(default_factory=list)
    execution: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def blocks(self) -> bool:
        """Does this witness establish a violation? (P: only executed or semantic-plausible.)"""
        return self.outcome in ("verified", "semantic_plausible")


def _sentence_at(text: str, pos: int) -> Tuple[int, str]:
    for start, s in split_claims(text):
        if start <= pos < start + len(s) + 1:
            return start, s
    return 0, text


def _clause_at(sent: str, rel: int) -> str:
    best = sent
    for off, cl in V._clauses(sent):
        if off <= rel < off + len(cl) + 1:
            best = cl
    return best


def execute_witness(w: Witness, text: str, eo: EvidenceObject, tx: Optional[TextExec] = None) -> Witness:
    """Check the witness's text binding and operands, then RECOMPUTE its relation.

    verified       the executed relation is a contradiction
    disproved      the executed relation is supported: the objection is false
    malformed      unknown rule/operation, quote not in text, unknown or irrelevant evidence ids
    unexecutable   the relation cannot be executed unambiguously (claim stays unresolved)
    semantic_*     causal / invented / target: contract plausibility checks only
    """
    text = clean(text)
    tx = tx or execute_text(text, eo)
    p: List[str] = []
    rule = (w.rule or "").upper()
    if rule not in C.RULES:
        w.outcome, w.problems = "malformed", [f"unknown rule {w.rule!r}"]
        return w
    op = RULE_OP.get(rule)
    if w.operation and op and str(w.operation).upper() not in (op, "SEMANTIC"):
        p.append(f"operation {w.operation} does not implement rule {rule}")
    ids = [str(i) for i in w.evidence_ids or []]
    bad = [i for i in ids if eo.get(i) is None]
    if bad:
        p.append(f"cites unknown evidence ids {bad}")
    if rule in SEMANTIC_RULES:
        o = C.check_objection(C.Objection(rule=rule, quote=w.quote, evidence_ids=ids, reason=w.reason), text, eo)
        probs = list(o.problems)
        pos = find_quote(text, w.quote) if w.quote else -1
        if rule == "R_CAUSAL" and pos >= 0:
            # the quote's CONTEXT decides: a causal phrase inside a disclaimer is not a causal claim,
            # and a causal construction the contract regex misses still counts ('...would fall')
            s0, sent = _sentence_at(text, pos)
            rel = pos - s0
            ctx = sent[:rel]
            if V._disclaimed(ctx) or C._DISCLAIMER.search(ctx) or re.search(r"\bnot (?:evidence|proof|a sign)\b", ctx, re.I):
                probs = [x for x in probs if "causal" not in x] + ["the quoted words sit inside a disclaimer"]
            elif causal_assertion(sent):
                probs = [x for x in probs if "no causal or action-effect assertion" not in x]
        if rule == "R_INVENTED" and w.quote:
            hit = _evidence_word_overlap(w.quote, eo)
            if hit:
                probs.append(f"the quoted factor shares the word(s) {hit[:3]} with evidence fields; it may be "
                             "an unrecognised name for an evidence item")
        w.outcome = "semantic_plausible" if (not probs and not p) else "semantic_invalid"
        w.problems = p + probs
        return w

    if op == "COVERAGE":
        res = [o for o in tx.text_ops if o.op == "COVERAGE" and (not ids or set(o.evidence_ids) & set(ids))]
        return _finish(w, res, p, "COVERAGE")
    if op == "COMPLETENESS":
        res = [o for o in tx.text_ops if o.op == "COMPLETENESS"]
        if w.quote and find_quote(text, w.quote) < 0:
            p.append("quote is not in the explanation")
        return _finish(w, res, p, "COMPLETENESS")

    if not (w.quote or "").strip():
        p.append("no quote")
    pos = find_quote(text, w.quote) if w.quote else -1
    if w.quote and pos < 0:
        p.append("quote is not in the explanation")
    if p:
        w.outcome, w.problems = "malformed", p
        return w
    # the clause around the quote, with the sentence for the subject: execution
    # sees the context (a negation outside the quote still counts)
    s_start, sent = _sentence_at(text, pos)
    clause = _clause_at(sent, max(0, pos - s_start))
    f = fields_of(bind(clause, eo), eo)
    fs = fields_of(bind(sent, eo), eo)
    named = set(f["attrs"] + f["facts"] + f["pred"]) | set(fs["attrs"] + fs["facts"] + fs["pred"])
    # cited evidence must be about the quoted words (a forged witness cites another field)
    if ids and op in ("VALUE", "DIRECTION", "SCALE", "CODE", "FACT_DRIVER", "TARGET") and \
            not any(i in named or _owner(eo, i) in named or _related(eo, i, named) for i in ids):
        w.outcome, w.problems = "malformed", ["the cited evidence is not about the quoted words"]
        return w
    if op == "VALUE":
        res = op_value(clause, eo, f, restrict=None)
        q = C._norm(w.quote)
        res = [o for o in res if o.asserted and C._norm(str(o.asserted)) in q] or res
    elif op == "SCALE":
        res = op_scale(clause, eo, f if f["attrs"] else fs)
    elif op in ("DIRECTION", "FACT_DRIVER"):
        res = [o for o in op_direction(clause, sent, eo, f, fs) if o.op in ("DIRECTION", "FACT_DRIVER")]
        if ids:
            res = [o for o in res if set(o.evidence_ids) & set(ids)] or res
    elif op == "RANK":
        res = op_rank(clause, sent, eo, f, fs) or op_rank(sent, sent, eo, fs, fs)
    elif op == "TEMPORAL":
        res = op_temporal(clause, eo, f if (f["attrs"] or f["facts"]) else fs)
    elif op == "CODE":
        res = op_code(clause, eo, f if (f["attrs"] or f["facts"]) else fs)
    elif op == "TARGET":
        res = op_target(clause, eo, f if f["pred"] else fs) or op_target(sent, eo, fs)
    else:
        res = []
    w = _finish(w, res, p, op)
    # never trust an LLM-supplied number: report when it disagrees with recomputation
    if w.claimed_result not in (None, "") and w.execution:
        rec = w.execution.get("recomputed")
        try:
            cr = float(str(w.claimed_result).replace(",", "").strip().rstrip("%"))
            recs = rec if isinstance(rec, list) else [rec]
            if recs and all(isinstance(r, (int, float)) for r in recs) and \
                    not any(abs(cr - float(r)) <= max(0.005 * abs(float(r)), 1e-6) or
                            abs(cr - 100 * float(r)) <= max(0.005 * abs(100 * float(r)), 1e-6) for r in recs):
                w.problems.append(f"claimed result {w.claimed_result} disagrees with the recomputed {rec}")
        except (TypeError, ValueError):
            pass
    return w


def _related(eo: EvidenceObject, eid: str, named: set) -> bool:
    it = eo.get(eid)
    if it is None:
        return False
    return any(c in named for c in (it.components or [])) or any(
        eid in ((eo.get(n).components or []) if eo.get(n) is not None else []) for n in named)


_GENERIC_WORDS = {"amount", "ratio", "score", "value", "recorded", "model", "risk", "rate", "days", "date", "total",
                  "level", "count", "number", "type", "status", "derived", "relative", "reference", "client",
                  "borrower", "applicant", "loan", "credit", "default", "driver", "factor", "feature", "raises",
                  "lowers", "increases", "decreases", "documented", "source", "evidence", "information"}


def _words(t: str) -> set:
    return {w for w in re.split(r"[^a-z]+", (t or "").lower()) if len(w) >= 4 and w not in _GENERIC_WORDS}


def _evidence_word_overlap(quote: str, eo: EvidenceObject) -> List[str]:
    """Distinctive words the quote shares with evidence field names, aliases or meanings."""
    q = _words(quote)
    vocab = set()
    for it in eo.items:
        if it.etype in (EvidenceType.OBSERVED_FACT, EvidenceType.MODEL_ATTRIBUTION):
            for t in [it.source_field, it.feature_group, it.meaning] + list(it.aliases or []):
                vocab |= _words(str(t or "").replace("_", " "))
    return sorted(q & vocab)


def find_quote(text: str, quote: str) -> int:
    """Start of the quote in the ORIGINAL text (case, whitespace and quote-style insensitive), or -1."""
    q = C._norm(quote).strip(" .\"'")
    if not q:
        return -1
    pat = r"\s+".join(re.escape(tok) for tok in q.split(" "))
    m = re.search(pat, text.translate(C._QUOTES), re.I)
    return m.start() if m else -1


def _finish(w: Witness, res: List[OpResult], p: List[str], op: str) -> Witness:
    w.problems = list(p)
    if p and any(x.startswith(("cites unknown", "quote is not", "operation ")) for x in p):
        w.outcome = "malformed"
        return w
    if not res:
        w.outcome = "unexecutable"
        w.problems.append(f"no {op} relation could be executed on the quoted words")
        return w
    con = [o for o in res if o.status == CONTRA]
    sup = [o for o in res if o.status == SUPPORT]
    pick = con[0] if con else (sup[0] if sup and len(sup) == len(res) else res[0])
    w.execution = pick.to_dict()
    w.outcome = "verified" if con else ("disproved" if sup and len(sup) == len(res) else "unexecutable")
    return w


def witnesses_from_json(d: dict) -> List[Witness]:
    out: List[Witness] = []
    for c in (d.get("claims") or []):
        if not isinstance(c, dict) or str(c.get("status", "")).lower() != "contradicted":
            continue
        wj = c.get("witness") if isinstance(c.get("witness"), dict) else c
        out.append(Witness(rule=str(wj.get("rule") or "").upper(), quote=str(wj.get("quote") or ""),
                           evidence_ids=[str(x) for x in (wj.get("evidence_ids") or [])],
                           operation=(str(wj.get("operation")).upper() if wj.get("operation") else None),
                           claimed_result=wj.get("claimed_result"),
                           alleged_conflict=str(wj.get("alleged_conflict") or ""),
                           claim_id=str(c.get("claim_id") or "") or None,
                           reason=str(wj.get("reason") or c.get("reason") or "")))
    for o in (d.get("omissions") or []):
        if isinstance(o, dict):
            out.append(Witness(rule="R_OMISSION", evidence_ids=[str(x) for x in (o.get("evidence_ids") or [])],
                               operation="COVERAGE", reason=str(o.get("reason") or "")))
    return out

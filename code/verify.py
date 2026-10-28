"""
Verification (plan §4.2 and §4.5).

Three separable checkers, because the paper must attribute errors to a stage:

  structural   deterministic. Cheap, high precision, no LLM. Catches uncited
               claims, forbidden trend/causal language, numeric values absent
               from evidence, and direction reversals.
  semantic     an LLM judge. Fallible. Its verdict is a SIGNAL, never a label.
  combined     structural gate then judge - the cheap realistic system.

The production judge must never supply the labels used to score it (plan:
"Labels must come from direct evidence checks and independent annotation, not
the production judge"). Independent labels live in labels.py.

Also here: the controlled corruption generator for E1. Corruptions are the
plan's six named error types plus omission.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from evidence import (FORBIDDEN_CAUSAL_TERMS, FORBIDDEN_TREND_TERMS,
                      EvidenceObject, EvidenceType)
from gateway import GATEWAY, JudgeIndependenceError
from generate import Candidate


class ErrorType(str, Enum):
    INVENTED_FACTOR = "invented_factor"
    SIGN_REVERSAL = "sign_reversal"
    NUMERIC_ERROR = "numeric_or_unit_error"
    UNSUPPORTED_TREND = "unsupported_trend"
    UNSUPPORTED_CAUSAL = "unsupported_causal_or_action"
    OMISSION = "omission_of_essential_driver"


@dataclass
class Verdict:
    accepted: bool
    checker: str                       # structural | semantic:<family> | combined
    violations: List[dict] = field(default_factory=list)
    judge_family: Optional[str] = None
    judge_model_id: Optional[str] = None
    latency_ms: int = 0
    error: Optional[str] = None
    parse_method: str = ""             # direct | repair | parser | unresolved | truncated
    raw: str = ""                      # judge response, kept for objection audit
    provider: str = ""
    route_index: int = 0
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_id: str = ""
    error_kind: str = ""               # transport (retry) | model (an observation)

    @property
    def n(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. structural checker - deterministic, no LLM
# ---------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)\s*(%|percent)?")


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUM.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append(v / 100.0 if m.group(2) else v)
    return out


def _evidence_numbers(eo: EvidenceObject) -> List[float]:
    vals: List[float] = []
    for i in eo.items:
        for v in (i.value, i.calibrated_pd, i.rescored_pd, i.contribution):
            if isinstance(v, (int, float)):
                vals.append(float(v))
        # Numbers embedded in evidence STRINGS are traceable too. Without this
        # a horizon field reading "NOT assumed 12 months" makes the digits 12
        # look like an invented figure - a false rejection produced by the
        # checker, not by the explanation. False rejections inflate b_g and
        # destroy coverage, so they matter as much as false accepts.
        for sv in (i.value, i.unit, i.observed_at, i.horizon,
                   i.target_definition, i.explained_scale, i.reason,
                   i.source_field, i.feature_group):
            if isinstance(sv, str):
                vals.extend(_numbers(sv))
    # allow rounded renderings of a PD, e.g. 0.3149 -> 31
    extra = []
    for v in vals:
        if 0.0 < v < 1.0:
            extra += [round(v * 100), round(v * 100, 1)]
        extra += [round(v), round(v, 2)]
    return vals + [float(x) for x in extra]


def structural_check_v1(cand: Candidate, eo: EvidenceObject,
                        *, require_citations: bool = True,
                        numeric_tolerance: float = 0.02) -> Verdict:
    """v1 checker, frozen for reproducing the 23 Sep 2026 pilot and §1.4."""
    v: List[dict] = []
    text = cand.text or ""
    low = text.lower()

    if not text.strip():
        return Verdict(False, "structural",
                       [{"type": "empty_output", "detail": "no text produced"}])

    # (a) forbidden language a snapshot model cannot license
    for term in FORBIDDEN_TREND_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", low):
            v.append({"type": ErrorType.UNSUPPORTED_TREND.value, "detail": term})
            break
    for term in FORBIDDEN_CAUSAL_TERMS:
        if term in low:
            v.append({"type": ErrorType.UNSUPPORTED_CAUSAL.value, "detail": term})
            break

    # (b) citations: every typed claim must cite ids that exist
    valid = set(eo.ids())
    if require_citations and cand.claims:
        for c in cand.claims:
            ids = c.get("evidence_ids") or []
            if not ids:
                v.append({"type": "uncited_claim", "detail": c.get("text", "")[:90]})
            else:
                bad = [i for i in ids if i not in valid]
                if bad:
                    v.append({"type": "unknown_evidence_id",
                              "detail": f"{bad} in: {c.get('text','')[:70]}"})

    # (c) numbers in prose must trace to evidence.
    # First strip text that is NOT a claim: the borrower id and any verbatim
    # evidence strings. Otherwise "borrower taiwan-1361" reads as a claim that
    # the number 1361 exists, and the checker rejects a correct explanation.
    # Such false rejections inflate b_g and suppress coverage, which is the
    # very failure mode this study measures - so the checker must not add to it.
    scan = text
    strip_tokens = [eo.borrower_id, eo.dataset]
    for i in eo.items:
        strip_tokens.append(i.eid)
        for sv in (i.source_field, i.feature_group, i.horizon,
                   i.target_definition, i.model_id, i.calibrator_id,
                   i.transformer_id, i.explained_scale, i.action_id,
                   i.observed_at, i.reason):
            if isinstance(sv, str) and len(sv) > 2:
                strip_tokens.append(sv)
    for tok in sorted({t for t in strip_tokens if t}, key=len, reverse=True):
        scan = re.sub(re.escape(tok), " ", scan, flags=re.I)

    ev = _evidence_numbers(eo)
    for n in _numbers(scan):
        if n in (0.0, 1.0) or abs(n) < 1e-9:
            continue
        if not any(abs(n - e) <= max(numeric_tolerance, abs(e) * numeric_tolerance)
                   for e in ev):
            v.append({"type": ErrorType.NUMERIC_ERROR.value, "detail": str(n)})

    # (d) direction reversal against attributions.
    # Scan EVERY mention, not just the first: a text can describe a driver
    # correctly once and contradict itself later, which is exactly what a
    # sign-reversal corruption does.
    UP = ("raise", "raises", "raising", "increase", "increases", "higher risk",
          "elevat", "adds to", "worse", "upward")
    DN = ("lower", "lowers", "lowering", "reduce", "reduces", "decrease",
          "decreases", "mitigat", "offset", "better", "downward")
    for a in eo.by_type(EvidenceType.MODEL_ATTRIBUTION):
        fg = (a.feature_group or "").lower()
        if not fg or fg not in low:
            continue
        contradicts = False
        for m in re.finditer(re.escape(fg), low):
            win = low[max(0, m.start() - 90): m.end() + 90]
            says_up = any(w in win for w in UP)
            says_dn = any(w in win for w in DN)
            if a.direction == "increases_risk" and says_dn and not says_up:
                contradicts = True
            if a.direction == "decreases_risk" and says_up and not says_dn:
                contradicts = True
            if contradicts:
                break
        if contradicts:
            v.append({"type": ErrorType.SIGN_REVERSAL.value,
                      "detail": a.feature_group})

    # (e) omission of the strongest driver
    attrs = sorted(eo.by_type(EvidenceType.MODEL_ATTRIBUTION),
                   key=lambda a: -abs(a.contribution or 0.0))
    if attrs:
        top = (attrs[0].feature_group or "").lower()
        if top and top not in low:
            v.append({"type": ErrorType.OMISSION.value, "detail": attrs[0].feature_group})

    return Verdict(accepted=not v, checker="structural_v1", violations=v)


# ---------------------------------------------------------------------------
# 1b. structural checker v2 - claim-scoped (plan §2 P0 "verify.py and rubric
#     disagree"; §5.2 rubric). Development: tuned on the v1 E1 development
#     items only; evaluated on the next, independently annotated round.
#
# v1 failure causes found against adjudicated labels (2 Oct 2026):
#   - the number reader dropped minus signs: "-0.33" read as 0.33, "-2" as 2;
#   - attribution language ("the risk score is increased by PAY_0") was read
#     as a trend over time;
#   - a driver named in plain words ("credit limit" for LIMIT_BAL) counted as
#     omitted; direction words were matched in a +-90 character window, so an
#     adjective ("a lower credit limit") could be read as a risk direction.
# ---------------------------------------------------------------------------
STRUCTURAL_VERSION = "structural/2.7"   # 2.7: M1 annotation misreads (field-name digits, 1.58M, HC aliases)
# 2.6: misreads seen in the first M1 outputs

# ---------------------------------------------------------------------------
# text utilities
# ---------------------------------------------------------------------------
_ABBR = re.compile(r"\b(e\.g|i\.e|approx|vs|etc|no|mr|mrs|dr|yrs|nt|incl|est|avg|max|min)\.", re.I)
# an aside's text may contain digit-grouping commas/decimal points ('50,000',
# '0.75'); only a comma NOT followed by a digit closes it
_AS_TXT = r"(?:[^,;.()]|[,.](?=\d))"
_ASIDE = re.compile(
    r"\([^()]{0,160}\)|,\s*(?:while|though|although|rather than|unlike|instead of|as opposed to|not)"
    r"\b" + _AS_TXT + r"{0,80},(?!\d)"
    r"|,\s*(?:defined as|which is (?:the )?(?:ratio|share|amount)|i\.e\.|that is|computed as|calculated as)"
    r"\b" + _AS_TXT + r"{0,90},(?!\d)"
    r"|(?:^|(?<=[.!?;]\s))\s*(?:unlike|compared (?:with|to)|in contrast to|relative to|besides|"
    r"apart from|rather than)\b" + _AS_TXT + r"{0,60},(?!\d)", re.I)
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?;])\s+|;\s*|\s+(?:while|whereas|but|although|though|however)\s+", re.I)
_GOV_NEG = re.compile(r"(?:\b(?:not|never|no longer|no|cannot|nor)|n't)\s+(?:\w+\s+)?$", re.I)
_NOT_ONLY = re.compile(r"\bnot\s+(?:only|just|merely)\s+$", re.I)


def _protect(text: str) -> str:
    """Same-length copy with abbreviation periods neutralised and asides blanked,
    used only to find clause boundaries and to read directions."""
    t = _ABBR.sub(lambda m: m.group(0)[:-1] + "\x00", text)
    return _ASIDE.sub(lambda m: " " * len(m.group(0)), t)


def _clauses(text: str) -> List[Tuple[int, str]]:
    """(start offset, clause) pieces; decimals, abbreviations and asides never split."""
    prot = _protect(text)
    out, last = [], 0
    for m in _CLAUSE_SPLIT.finditer(prot):
        if m.start() > last:
            out.append((last, text[last:m.start()]))
        last = m.end()
    if last < len(text):
        out.append((last, text[last:]))
    return out


_EPISTEMIC_NEG = re.compile(r"(?:\b(?:not|never|no|cannot)|n't)\s+(?:\w+\s+){0,2}(mean|means|imply|implies|suggest|"
                            r"suggests|indicate|indicates|show|shows|establish|establishes|prove|"
                            r"proves|demonstrate|demonstrates|evidence|say|says|tell|tells|read|"
                            r"conclude|claim|claims|state|states)\b", re.I)


def _governed_neg(before: str) -> bool:
    """A negator that governs the next word(s): 'does not raise', 'no change'."""
    tail = before[-40:]
    return bool(_GOV_NEG.search(tail)) and not _NOT_ONLY.search(tail)


_TREND_DISCLAIM = re.compile(
    r"\b(?:no information|no data|no evidence|no history|nothing can be said|cannot (?:tell|say|determine|"
    r"be said|be determined|show|infer|establish|reveal|identify|assess|see)|can't (?:tell|say|show|infer)|"
    r"(?:impossible|not possible) to (?:tell|say|infer|determine|show|establish|identify|assess)|"
    r"unable to (?:tell|say|determine|infer)|(?:does|do|did) not "
    r"(?:show|indicate|tell|reveal|say)|doesn't (?:show|indicate|tell)|single snapshots?|one snapshot)\b"
    r"[^.;]{0,90}$", re.I)


_NEAR_NEG = re.compile(r"(?:\b(?:not|no|never|nor)|n't)\s+(?:\w+\s+){0,3}$", re.I)


def _disclaimed(before: str) -> bool:
    """'This does not mean/imply (that) X causes ...', 'not a verified cause of':
    a disclaimer, not a claim."""
    return (_governed_neg(before) or bool(_EPISTEMIC_NEG.search(before[-80:]))
            or (bool(_NEAR_NEG.search(before[-40:])) and not _NOT_ONLY.search(before[-40:])))


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
_NUMTOK = re.compile(
    r"(?<![\w.])(?P<sign>[+\-−–]?)"
    r"(?P<num>\d{1,3}(?:[   ]\d{3})+(?!\d)|\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?P<mult>\s*(?:k|thousand|million|mn|bn|billion)\b|(?-i:M)\b)?"
    r"(?P<pp>\s*(?:percentage points?|pp\b|points?\b))?"
    r"(?P<pct>\s*(?:%|percent\b|per cent\b|in 100\b))?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "bn": 1e9, "billion": 1e9}
#: a number that is part of a field's NAME, not a value: "source 2", "document 9" (2.7)
_ID_BEFORE = re.compile(r"(?:\bsource|\bdocument|\bdoc|\bflag|\bnumber|\bno\.|#)\s*$", re.I)


_WORDNUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_WORDNUM.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
                 "eighty": 80, "ninety": 90, "": 0})
_SPELLED_PCT = re.compile(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|zero|one|two|three|four|"
                          r"five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                          r"seventeen|eighteen|nineteen)(?:[- ](one|two|three|four|five|six|seven|eight|nine))?"
                          r"\s+(?:percent|per cent)\b", re.I)


def _signed_numbers(text: str) -> List[dict]:
    """Each written number: value, tolerance, percent flag, explicit sign, range."""
    out: List[dict] = []
    for m in _NUMTOK.finditer(text):
        sign = m.group("sign").replace("−", "-").replace("–", "-")
        if sign == "-" and m.start() > 0 and text[m.start() - 1].isalnum():
            sign = ""                         # 'April-2005', 'PAY-2': a hyphen
        if _ID_BEFORE.search(text[max(0, m.start() - 12):m.start()]):
            continue                          # 'external source 2', 'document 9': a name
        raw = re.sub(r"[   ,]", "", m.group("num"))
        try:
            v = float(raw)
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        mult = _MULT.get((m.group("mult") or "").strip().lower(), 1.0)
        v *= mult
        if sign == "-":
            v = -v
        out.append({"v": v, "tol": 0.5 * mult * 10 ** (-dec), "pct": bool(m.group("pct")),
                    "pp": bool(m.group("pp")),
                    "signed": sign in ("+", "-"), "pos": m.start("num") - len(sign),
                    "end": m.end(), "lo": v, "hi": v})
    # '1 in 20' (a rate) and spelled-out percentages: read as percentages
    for m in re.finditer(r"\b1 in (\d{1,4})\b", text):
        nn = float(m.group(1))
        out = [o for o in out if not (m.start() <= o["pos"] < m.end())]
        out.append({"v": 100.0 / nn, "tol": 100.0 / (nn - 0.5) - 100.0 / nn + 1e-9, "pct": True, "pp": False,
                    "signed": False, "pos": m.start(), "end": m.end(), "lo": 100.0 / nn, "hi": 100.0 / nn})
    for m in _SPELLED_PCT.finditer(text):
        val = _WORDNUM.get(m.group(1).lower(), 0) + _WORDNUM.get((m.group(2) or "").lower(), 0)
        out.append({"v": float(val), "tol": 0.5, "pct": True, "pp": False, "signed": False,
                    "pos": m.start(), "end": m.end(), "lo": float(val), "hi": float(val)})
    out.sort(key=lambda o: o["pos"])
    # ranges 'a-b' / 'a to b': one claim covering an interval
    merged: List[dict] = []
    for n in out:
        if merged:
            p = merged[-1]
            gap = text[p["end"]:n["pos"]]
            if (re.fullmatch(r"\s*(?:-|–|to)\s*", gap) and not p.get("range")
                    and not (1900 <= abs(p["v"]) <= 2100 and float(p["v"]).is_integer())
                    and not (1900 <= abs(n["v"]) <= 2100 and float(n["v"]).is_integer())):
                lo, hi = sorted((p["v"], abs(n["v"]) if n["signed"] and n["v"] < 0 else n["v"]))
                p.update({"range": True, "lo": lo, "hi": hi, "end": n["end"],
                          "pct": p["pct"] or n["pct"], "tol": max(p["tol"], n["tol"])})
                continue
        merged.append(n)
    return merged


ZERO_ROUNDING_GUARD = True   # executable/1.5: a stated 0 is not a rounding of a value that is not about zero


def _num_matches(n: dict, e: float, mag_ok: bool = False) -> bool:
    tol = max(n["tol"], 0.005 * abs(e)) + 1e-9
    if ZERO_ROUNDING_GUARD and not n.get("range") and n["v"] == 0 and abs(e) > 0.005:
        return False
    if n.get("range"):
        return n["lo"] - tol <= e <= n["hi"] + tol
    if abs(n["v"] - e) <= tol:
        return True
    return mag_ok and not n["signed"] and abs(abs(n["v"]) - abs(e)) <= tol


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
def _spec_aliases(dataset: str, name: Optional[str]) -> List[str]:
    """Documented aliases for evidence that predates evidence/2.0 (no aliases
    stored). v1 used adapter-internal names and one-hot names like
    'EDUCATION_5.0', so a decimal one-hot suffix is stripped before lookup."""
    if not name or dataset not in ("taiwan", "home_credit"):
        return []
    try:
        import feature_semantics as FS
    except Exception:  # noqa: BLE001
        return []
    sp = FS.specs(dataset)
    base = re.sub(r"_-?\d+\.\d+$", "", name) if name not in sp else name
    s = sp.get(base) or sp.get(name)
    # The source display name is NOT added: v1 evidence showed internal names
    # (PAY_1 = source PAY_2), so adding display names would make PAY_2 in a v1
    # text refer to two different fields.
    return ([base] + list(s.aliases)) if s else []


_GENERIC_CONCEPTS = {"UNMAPPED_FEATURE", "NO_SUPPORTED_ESTIMATE"}


def _concept_aliases(concept: Optional[str]) -> List[str]:
    """A driver may be referred to by its concept label for NUMBER binding only."""
    if not concept or concept in _GENERIC_CONCEPTS:
        return []
    words = concept.lower().split("_")
    out = [" ".join(words)]
    while len(words) > 2 and words[-1] in ("high", "low", "adverse", "level", "association", "record"):
        words = words[:-1]
        out.append(" ".join(words))
    return out


def _names(item, dataset: str = "", concept: bool = False) -> List[str]:
    """Every name an explanation may use for an evidence item, lowercased.
    concept=True also accepts the concept label; used only to bind numbers.
    Naming a driver by its concept alone does not count as naming it (three
    unanimous v1 omission labels were concept-only mentions)."""
    extra = list(item.aliases or [])
    # internal model column names are NOT names an explanation uses (v2 PAY_2 is
    # the source PAY_3); only v1 evidence, which showed them, falls back to them
    base = [item.source_field, item.feature_group] + ([] if extra else [item.internal_name])
    if extra and item.internal_name:      # v2: add the CURRENT documented aliases
        extra += _spec_aliases(dataset, item.internal_name)[1:]
    if not extra:
        for b in base:
            extra += _spec_aliases(dataset, b)
    if concept:
        extra += _concept_aliases(item.concept)
    out: List[str] = []
    for b in base + extra:
        if not b:
            continue
        b = str(b).lower().replace(" (derived)", "").strip()
        for v in (b, b.replace("_", " ")):
            if len(v) >= 2 and v not in out:
                out.append(v)
    return out


_SERIES_NOUNS = {
    "repayment_status": [r"repayment[- ]status", r"repayment(?=[^.;]{0,25}\b(?:status|code|delay|late|duly)\b)",
                         r"status code", r"payment status", r"paid duly"],
    "bill_amount": [r"bills?(?: amounts?| statements?)?", r"billed amounts?", r"statement amounts?", r"balances?"],
    "payment_amount": [r"amounts? paid", r"payment amounts?", r"payments? of", r"payments? made", r"payments",
                       r"repayments? of", r"repayment amounts?", r"repaid",
                       r"paid(?!\s+(?:duly|on time|in full|late))", r"payment(?!\s+(?:status|delay|code))"],
}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]
# a capitalised month; 'May' only with a year or after in/of/for (else the modal 'may')
_MONTH_RE = re.compile(r"\b(?:(January|February|March|April|June|July|August|September|October|"
                       r"November|December)|(?:(?<=in )|(?<=of )|(?<=for ))(May)|(May)(?=\s+\d{4}))\b")


def _period_mentions(text: str, item) -> List[int]:
    """A series field named by its noun plus its own period in one clause,
    e.g. 'the September 2005 bill amount' for BILL_AMT1. Case-sensitive months."""
    if not item.series or item.series not in _SERIES_NOUNS:
        return []
    month = next((m for m in _MONTHS if m.lower() in (item.observed_at or "").lower()), None)
    if not month:
        return []
    out = []
    for off, cl in _clauses(text):
        months = [(mm.start(), next(g for g in mm.groups() if g)) for mm in _MONTH_RE.finditer(cl)]
        if not any(g == month for _, g in months):
            continue
        for pat in _SERIES_NOUNS[item.series]:
            for m in re.finditer(rf"\b{pat}\b", cl, flags=re.I):
                nearest = min(months, key=lambda t: abs(t[0] - m.start()))[1]
                if nearest == month:
                    out.append(off + m.start())
    return sorted(set(out))


_HYPH = str.maketrans({"-": " ", "\u2011": " ", "\u2010": " ", "\u2013": " ", "\u2012": " "})


def _canon(t: str) -> str:
    """Same-length normalisation for name matching: hyphens -> spaces,
    curly quotes -> straight, British/American -ise/-ize spellings."""
    t = t.replace("\u2019", "'").replace("\u2018", "'").translate(_HYPH)
    return re.sub(r"(?<=[a-z])is(?=e[ds]?\b|ing\b|ation)", "iz", t)


def _mentions(text_low: str, names: List[str]) -> List[int]:
    t = _canon(text_low)
    pos = []
    for n in names:
        n = _canon(n)
        for m in re.finditer(rf"(?<![\w]){re.escape(n)}(?![\w])", t):
            pos.append(m.start())
    return sorted(set(pos))


def _item_values(item) -> List[float]:
    vals = []
    # a v1 one-hot name carries its level: EDUCATION_5.0 means EDUCATION = 5.
    # Only a DECIMAL suffix is a level; PAY_2 or BILL_AMT1 are field indices.
    for nm in (item.feature_group, item.source_field):
        m = re.search(r"_(-?\d+\.\d+)$", nm or "")
        if m:
            vals.append(float(m.group(1)))
    for v in (item.value, item.calibrated_pd, item.rescored_pd, item.contribution):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    for v in (item.equivalents or {}).values():
        vals.append(float(v))
    return vals


def _text_numbers(*strings) -> List[float]:
    vals: List[float] = []
    for sv in strings:
        if isinstance(sv, str):
            for n in _signed_numbers(sv):
                vals += [n["v"], abs(n["v"])]
    return vals


def _string_numbers(eo: EvidenceObject) -> List[float]:
    return _text_numbers(*[sv for i in eo.items for sv in
                           (i.value, i.unit, i.observed_at, i.horizon, i.target_definition,
                            i.explained_scale, i.reason, i.meaning, i.value_label)])


# ---------------------------------------------------------------------------
# direction, trend and causal language
# ---------------------------------------------------------------------------
_OBJ = (r"(?:the\s+|its\s+|this\s+|a\s+|their\s+)?(?:(?:model|borrower|applicant|client|customer)'?s?\s+)?"
        r"(?:(?:estimated|predicted|overall|default|modelled|modeled|calculated|raw|uncalibrated|log-odds)\s+){0,2}"
        r"(risk|score|probability|pd|estimate|odds|chances?|likelihood|margin|log-odds|logit)(?:\s+of\s+default)?")
_OBJ_IT = r"(?:it|this)\b"
_UP_INFL = (r"raises|raised|raising|increases|increased|increasing|elevates|elevated|elevating|"
            r"adds to|added to|adding to|worsens|worsened|worsening|boosts|boosted|lifts|lifted")
_DN_INFL = (r"lowers|lowered|lowering|reduces|reduced|reducing|decreases|decreased|decreasing|"
            r"mitigates|mitigated|mitigating|offsets|offsetting|improves|improved|improving|"
            r"cuts|eases|eased|dampens|dampened")
_UP_BARE = r"raise|increase|elevate|boost|lift|worsen"
_DN_BARE = r"lower|reduce|decrease|mitigate|improve|ease|dampen"
_SKIP = r"(?:(?!(?:raise|rais|increas|elevat|lower|reduc|decreas|mitigat|offset|improv|worsen|higher|greater|less)\w*)\w+\s+){0,3}?"
_DIR_PATTERNS = [   # (pattern, polarity, passive)
    (re.compile(rf"\b(?:{_UP_INFL})\s+{_SKIP}{_OBJ}\b", re.I), +1, False),
    (re.compile(rf"\b(?:{_DN_INFL})\s+{_SKIP}{_OBJ}\b", re.I), -1, False),
    # bare forms count only when the risk/score object follows directly
    (re.compile(rf"\b(?:{_UP_BARE})\s+{_OBJ}\b", re.I), +1, False),
    (re.compile(rf"\b(?:{_DN_BARE})\s+{_OBJ}\b", re.I), -1, False),
    # pronoun object: 'X raises it', 'Y lowers it' (the text is about the score)
    (re.compile(rf"\b(?:{_UP_INFL}|{_UP_BARE})\s+{_OBJ_IT}", re.I), +1, False),
    (re.compile(rf"\b(?:{_DN_INFL}|{_DN_BARE})\s+{_OBJ_IT}", re.I), -1, False),
    # a credit score rises as risk falls: inverse polarity
    (re.compile(rf"\b(?:{_DN_INFL}|{_DN_BARE})\s+(?:the\s+|its\s+)?(?:(?:borrower|applicant|client)'?s?\s+)?"
                r"(?:credit\s*worthiness|creditworthiness|credit score)\b", re.I), +1, False),
    (re.compile(rf"\b(?:{_UP_INFL}|{_UP_BARE})\s+(?:the\s+|its\s+)?(?:(?:borrower|applicant|client)'?s?\s+)?"
                r"(?:credit\s*worthiness|creditworthiness|credit score)\b", re.I), -1, False),
    (re.compile(r"\b(?:higher|better|stronger)\s+credit(?:\s+score|worthiness)\b", re.I), -1, False),
    (re.compile(r"\b(?:lower|worse|weaker)\s+credit(?:\s+score|worthiness)\b", re.I), +1, False),
    (re.compile(r"\bmakes?\s+(?:a\s+)?default\s+more\s+likely\b", re.I), +1, False),
    (re.compile(r"\bmakes?\s+(?:a\s+)?default\s+less\s+likely\b", re.I), -1, False),
    (re.compile(r"\b(?:pushes|pushed|pushing|push|moves?|moved|takes?|took|drives?|drove)\s+"
                + _SKIP + _OBJ + r"\s+higher\b", re.I), +1, False),
    (re.compile(r"\b(?:pushes|pushed|pushing|push|moves?|moved|takes?|took|drives?|drove)\s+"
                + _SKIP + _OBJ + r"\s+lower\b", re.I), -1, False),
    (re.compile(r"\b(?:drives?|driving|drove|pushes|pushed|pushing|push|moves?|moving|moved)\s+"
                + _SKIP + _OBJ + r"\s+up\b|\b(?:drives?|driving|drove|pushes|pushed|pushing|push)\s+up\b", re.I), +1, False),
    (re.compile(r"\b(?:drives?|driving|drove|pushes|pushed|pushing|push|moves?|moving|moved|pulls?|pulled|pulling)\s+"
                + _SKIP + _OBJ + r"\s+down\b|\b(?:drives?|driving|drove|pushes|pushed|pushing|push|pulls?|pulled)\s+down\b", re.I), -1, False),
    (re.compile(r"\bcounts?\s+against\b|\bworks?\s+against\b|\bweighs?\s+against\b", re.I), +1, False),
    (re.compile(r"\b(?:counts?|works?|weighs?)\s+in\s+(?:the\s+)?(?:borrower'?s\s+|applicant'?s\s+|client'?s\s+)?favou?r\b", re.I), -1, False),
    (re.compile(r"\b(?:risk|score|probability|estimate|odds|chances?|likelihood|margin)\s+(?:is\s+|was\s+|are\s+|gets\s+|got\s+)?(?:\w+ly\s+|also\s+|further\s+){0,2}(?:raised|increased|pushed up|elevated|boosted|driven up|moved up)\b", re.I), +1, True),
    (re.compile(r"\b(?:risk|score|probability|estimate|odds|chances?|likelihood|margin)\s+(?:is\s+|was\s+|are\s+|gets\s+|got\s+)?(?:\w+ly\s+|also\s+|further\s+){0,2}(?:lowered|reduced|decreased|pushed down|mitigated|eased|driven down|moved down)\b", re.I), -1, True),
    (re.compile(r"\b(?:higher|greater|increased|elevated|more)\s+(?:default\s+|credit\s+(?=risk))?(?:risk|(?<!credit )score|probability)\b", re.I), +1, False),
    (re.compile(r"\b(?:lower|reduced|less|decreased|smaller)\s+(?:default\s+|credit\s+(?=risk))?(?:risk|(?<!credit )score|probability)\b", re.I), -1, False),
    (re.compile(r"\b(?:upward|positive|risk-increasing|adverse)\s+(?:shap\s+)?(?:pressure|contribution|effect|push|driver|factor|value)\b", re.I), +1, False),
    (re.compile(r"\b(?:downward|negative|risk-reducing|protective|mitigating)\s+(?:shap\s+)?(?:pressure|contribution|effect|push|driver|factor|value)\b", re.I), -1, False),
    (re.compile(r"\bcontribut\w*\s+positively\b", re.I), +1, False),
    (re.compile(r"\bcontribut\w*\s+negatively\b", re.I), -1, False),
    (re.compile(r"\bcontribut\w*\s+to\s+(?:a\s+)?(?:higher|increased|greater|increasing)\b", re.I), +1, False),
    (re.compile(r"\bcontribut\w*\s+to\s+(?:a\s+)?(?:lower|reduced|smaller|decreasing|reducing)\b", re.I), -1, False),
]

_TREND_EXTRA = ["grew", "grown", "growing", "went up", "went down", "gone up", "gone down", "climbed",
                "climbing", "jumped", "surged", "spiked", "dipped", "shrank", "shrunk", "shrinking",
                "increase in", "decrease in", "rise in", "fall in", "drop in", "decreased", "decreasing",
                "risen", "fallen", "gotten worse", "got worse", "become worse", "became worse",
                "getting worse", "gotten better", "got better", "become better", "became better",
                "getting better"]
_TREND_TERMS = [t for t in FORBIDDEN_TREND_TERMS if t != "has been"] + _TREND_EXTRA
_TREND_UP = {"rose", "rising", "increased", "increasing", "worsened", "worsening",
             "deteriorated", "deteriorating", "grew", "grown", "growing", "went up", "climbed",
             "climbing", "jumped", "surged", "increase in", "rise in", "gone up", "spiked", "risen",
             "gotten worse", "got worse", "become worse", "became worse", "getting worse"}
_TREND_DN = {"declined", "declining", "decline", "fell", "falling", "dropped", "dropping",
             "improved", "improving", "went down", "shrank", "shrunk", "shrinking",
             "decrease in", "fall in", "drop in", "gone down", "dipped", "decreased", "decreasing",
             "fallen", "gotten better", "got better", "become better", "became better", "getting better"}
_SCORE_SUBJ = re.compile(r"\b(risk|score|probability|estimate|odds|margin|contribution|pd|prediction)"
                         r"\s*(?:'s)?\s*(?:(?:is|was|are|were|be|been|being|gets|got|has|have|had)\s+){0,2}$", re.I)
_SCORE_OBJ = re.compile(r"^\s+" + _OBJ + r"\b", re.I)
_TIME_SPAN = re.compile(r"\b(?:since|between (?:january|february|march|april|may|june|july|august|september|"
                        r"october|november|december|\d{4}|the (?:two|last|first)|last|earlier)|"
                        r"over (?:the )?(?:last|past|recent)|from (?:january|february|"
                        r"march|april|may|june|july|august|september|october|november|december)|"
                        r"compared (?:with|to) (?:last|the previous|previous|earlier)|last month|previous "
                        r"month|month (?:after|on|over) month|over time|recently)\b", re.I)

_CAUSAL_TERMS = [t for t in FORBIDDEN_CAUSAL_TERMS if t != "as a result"]   # a connector, not a claim
_CAUSAL_PATTERNS = [
    re.compile(r"\b(?:would|could|will|may|might|can|should|is likely to|would likely)\s+"
               r"(?:\w+\s+){0,2}?(?:reduce|lower|decrease|improve|raise|increase|prevent|cut|eliminate|avoid)\s+"
               r"(?:the\s+)?(?:borrower'?s\s+|applicant'?s\s+|client'?s\s+)?(?:default\s+)?"
               r"(?:risk|chance|probability|likelihood)(?!\s+score)(?:\s+of\s+default)?\b", re.I),
    # advice: the explanation tells the borrower what to do
    re.compile(r"\b(?:should|needs? to|ought to|must|is advised to|(?:is |are )?(?:recommended|encouraged|advised) to|"
               r"(?:we |i )?(?:recommend|advise|suggest)(?:s)?(?: that)?)\s+(?:\w+\s+){0,3}?"
               r"(?:pay|paying|repay|reduce|reducing|lower|lowering|increase|increasing|raise|raising|improve|"
               r"avoid|clear|settle|cut)\b", re.I),
    re.compile(r"\bconsider(?:s|ed)?\s+(?:paying|repaying|reducing|lowering|increasing|raising|clearing|settling)\b", re.I),
    # an ACTION (gerund or if-clause) said to change the model's output
    re.compile(r"(?:\bif\b[^.;]{0,80}?|\b\w+ing\b[^.;]{0,60}?)\b(?:would|could|will|might|may)\s+(?:have\s+)?"
               r"(?:\w+\s+){0,2}?(?:lower(?:ed)?|reduc(?:e|ed)|decreas(?:e|ed)|rais(?:e|ed)|increas(?:e|ed)|"
               r"improv(?:e|ed)|cut)\s+(?:the\s+)?(?:model'?s\s+)?(?:predicted\s+|estimated\s+)?"
               r"(?:risk|pd|probability|estimate|score|chances?)\b", re.I),
    re.compile(r"\b(?:would|could|will|may|might|can)\s+(?:\w+\s+){0,4}?(?:prevent|avoid|cause)\s+(?:a\s+)?default", re.I),
    re.compile(r"\b(?:leads?|leading|led|lead|results?|resulting|resulted)\s+(?:to|in)\s+(?:a\s+|the\s+)?default\b", re.I),
    re.compile(r"\b(?:will|would|could|may|might|should)\s+(?:\w+\s+){0,2}?(?:go down|go up|fall|drop|decline|decrease|rise)\b", re.I),
    re.compile(r"\b(?:a\s+|the\s+|major\s+|main\s+)*cause\s+of\s+default|\bcauses?\s+(?:a\s+)?default\b|\bdrives?\s+(?:a\s+)?default\b", re.I),
]


def structural_check(cand: Candidate, eo: EvidenceObject,
                     *, require_citations: bool = True) -> Verdict:
    """Claim-scoped deterministic checks. Each violation carries the clause
    (span) and the evidence ids it was checked against, so a later objection
    audit can test whether the stated reason is itself valid."""
    v: List[dict] = []
    text = (cand.text or "").replace("\u2019", "'").replace("\u2018", "'").replace("*", " ").replace("`", " ")
    if not text.strip():
        return Verdict(False, STRUCTURAL_VERSION,
                       [{"type": "empty_output", "detail": "no text produced"}])
    low = text.lower()

    facts = eo.by_type(EvidenceType.OBSERVED_FACT)
    attrs = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
    pred = eo.prediction
    by_id = {i.eid: i for i in eo.items}
    named = [(i, _names(i, eo.dataset, concept=True)) for i in facts + attrs]
    real_names = {i.eid: set(_names(i, eo.dataset)) for i in facts + attrs}
    fact_attr = {c: a for a in attrs for c in (a.components or [])}

    def period_spans(clause: str) -> List[Tuple[int, int, object]]:
        """(start, end, item) for monthly fields named by noun + month."""
        out = []
        for f in facts:
            for pos in _period_mentions(clause, f):
                m = next((re.match(rf"{pat}\b", clause[pos:], flags=re.I)
                          for pat in _SERIES_NOUNS.get(f.series, [])
                          if re.match(rf"{pat}\b", clause[pos:], flags=re.I)), None)
                end = pos + (len(m.group(0)) if m else 1)       # the noun only, never digits after it
                out.append((pos, end, f))
                if f.eid in fact_attr:
                    out.append((pos, end, fact_attr[f.eid]))
        return out
    plain = {i.eid: _names(i, eo.dataset) for i in facts + attrs}
    pd_val = float(pred.calibrated_pd) if pred is not None and isinstance(
        pred.calibrated_pd, (int, float)) else None
    str_nums = _string_numbers(eo)
    all_vals = [(x, False) for i in eo.items if i.etype is not EvidenceType.MODEL_ATTRIBUTION
                for x in _item_values(i)]
    all_contribs = [(float(a.contribution or 0.0), True) for a in attrs]

    # evidence text an explanation may QUOTE (units, labels, meanings, periods,
    # outcome/horizon): digits inside such a quotation are not claims
    quotable = set()
    for it in eo.items:
        for sv in (it.unit, it.value_label, it.meaning, it.observed_at, it.target_definition,
                   it.horizon, it.explained_scale, it.reason):
            if isinstance(sv, str) and len(sv) >= 6:
                for part in [sv] + re.split(r"\s*[(;]\s*|\)\s*", sv):
                    part = part.strip(" ).").lower()
                    if len(part) >= 6 and re.search(r"\d", part):
                        quotable.add(_canon(part))

    def quoted_spans(cl_low: str) -> List[Tuple[int, int]]:
        t = _canon(cl_low)
        return [(m.start(), m.end()) for q in quotable for m in re.finditer(re.escape(q), t)]

    def bound_values(items) -> List[Tuple[float, bool]]:
        """The VALUES of the named fields. Not their contributions (a number is
        a contribution only when the text says so; handled separately) and not
        the inputs of a derived field (pay_max = 1 is not 'pay_max is -1')."""
        out = []
        for it in items:
            if it.etype is EvidenceType.OBSERVED_FACT:
                out += [(x, False) for x in _item_values(it)]
            if it.etype is EvidenceType.MODEL_ATTRIBUTION:
                for f in facts:
                    if f.eid in (it.components or [])[:1] or (f.source_field and f.source_field == it.feature_group):
                        out += [(x, False) for x in _item_values(f)]
                # v1 one-hot names carry their level (EDUCATION_5.0)
                out += [(x, False) for x in _item_values(it) if x != it.contribution]
            # a string value's own digits ('Trade: type 7') and the year of its
            # period; never numbers from unit/meaning/label text, which would make
            # every documented code a valid value of every repayment field
            out += [(x, False) for x in _text_numbers(it.value if isinstance(it.value, str) else None)]
            out += [(float(y), False) for y in re.findall(r"\b(?:19|20)\d\d\b", it.observed_at or "")]
        return out

    def series_mentions(cl_orig: str, cl_low: str) -> Dict[str, List]:
        ser: Dict[str, List] = {}
        months = {next(g for g in mm.groups() if g) for mm in _MONTH_RE.finditer(cl_orig)}
        for f in facts:
            if not f.series:
                continue
            noun = any(re.search(rf"\b{pat}\b", _canon(cl_low)) for pat in _SERIES_NOUNS.get(f.series, []))
            fmonth = next((m for m in _MONTHS if m.lower() in (f.observed_at or "").lower()), None)
            if (_mentions(cl_low, plain[f.eid]) or _period_mentions(cl_orig, f)
                    or (noun and fmonth in months)):
                if f not in ser.setdefault(f.series, []):
                    ser[f.series].append(f)
        return ser

    for off, clause in _clauses(text):
        cl = clause.lower()
        dl = _protect(clause).lower()       # asides blanked: for directions and trends
        prob_ctx = bool(re.search(r"\b(probability|chance|likelihood|pd|estimate[sd]?|default risk|risk of default)\b", cl))

        # (a) trend claims: a recorded quantity said to change over time
        for term in _TREND_TERMS:
            for m in re.finditer(rf"\b{re.escape(term)}\b", dl):
                pre, post = dl[:m.start()], dl[m.end():]
                if _governed_neg(pre) or _disclaimed(pre) or _TREND_DISCLAIM.search(pre):
                    continue
                if pre.endswith("-") or re.match(r"^[\s-]*(factor|driver|effect|contribution|feature|influence)s?\b", post):
                    continue          # 'risk-increasing', 'an increasing factor'
                pm = re.match(r"^\s+by\s+(.{0,60})", post)
                if pm and any(_mentions(pm.group(1), plain[a.eid]) for a in attrs):
                    continue          # '(score) ... decreased by <driver>': attribution, not time

                if not _TIME_SPAN.search(dl) and (_SCORE_SUBJ.search(pre[-60:]) or _SCORE_OBJ.match(post) or (
                        re.match(r"^\s+it\b", post) and re.search(r"\b(score|risk|probability)\b", dl))):
                    continue          # model-score movement, not a change over time
                ser = series_mentions(clause, cl)
                two = {k: fs for k, fs in ser.items() if len({f.period_order for f in fs}) >= 2}
                if two:
                    # a comparison of shown periods: check its direction
                    bad = False
                    for k, fs in two.items():
                        new = min(fs, key=lambda f: f.period_order)
                        old = max(fs, key=lambda f: f.period_order)
                        if not all(isinstance(x.value, (int, float)) for x in (new, old)):
                            continue
                        delta = new.value - old.value
                        up = term in _TREND_UP
                        dn = term in _TREND_DN
                        if k == "repayment_status" and not all(
                                x.value == -1 or x.value >= 1 for x in (new, old)):
                            bad = True        # ranking of undocumented codes is unknown
                            continue
                        if term in ("worsened", "worsening", "deteriorated", "deteriorating",
                                    "improved", "improving") and k != "repayment_status":
                            continue          # 'worse' is undefined for amounts
                        if (up and delta <= 0) or (dn and delta >= 0):
                            bad = True
                    if bad:
                        v.append({"type": ErrorType.UNSUPPORTED_TREND.value,
                                  "detail": f"{term}: direction contradicts the shown periods",
                                  "span": clause.strip()[:200]})
                    continue
                v.append({"type": ErrorType.UNSUPPORTED_TREND.value, "detail": term,
                          "span": clause.strip()[:200]})
                break

        # (b) causal / action claims (negation must govern the claim)
        hit = None
        for term in _CAUSAL_TERMS:
            for m in re.finditer(rf"\b{re.escape(term)}\b", cl):
                if not _disclaimed(cl[:m.start()]):
                    hit = term
                    break
            if hit:
                break
        if not hit:
            for pat in _CAUSAL_PATTERNS:
                m = pat.search(cl)
                if m and not _disclaimed(cl[:m.start()]):
                    hit = m.group(0)
                    break
        if hit:
            v.append({"type": ErrorType.UNSUPPORTED_CAUSAL.value, "detail": hit[:60],
                      "span": clause.strip()[:200]})

        # (c) numbers: bound to the field they are attached to
        pspans = period_spans(clause)
        mentioned = [i for i, ns in named if _mentions(cl, ns)]
        mentioned += [it for _, _, it in pspans if it not in mentioned]
        allspans = [(pos, pos + len(nm), it, nm in real_names[it.eid]) for it, ns in named
                    for nm in ns for pos in _mentions(cl, [nm])] + [(a, b, it, True) for a, b, it in pspans]
        # a concept label only binds where no real field name covers the same text
        real = [(a, b) for a, b, _, r in allspans if r]
        spans = sorted(((a, b, it) for a, b, it, r in allspans
                        if r or not any(ra <= a and b <= rb or (rb == b) for ra, rb in real)),
                       key=lambda t: (t[0], t[1]))
        mentioned = [it for it in mentioned if any(it is x for _, _, x in spans)] or mentioned
        respectively = "respectively" in cl
        resp_pairs = None
        if respectively:
            # 'A and B are x and y respectively': pair fields and numbers in order
            first = {}
            for s0, e0, it in spans:
                first.setdefault(it.eid, (s0, it))
            # one entry per field: a fact and its attribution name the same field
            fields, seen_names = [], set()
            for _, it in sorted(first.values(), key=lambda t: t[0]):
                key = (it.source_field or it.feature_group or it.eid).lower().replace(" (derived)", "")
                if key not in seen_names:
                    seen_names.add(key)
                    fields.append(it)
            nums = [n["pos"] for n in _signed_numbers(clause)
                    if not any(s0 <= n["pos"] < e0 for s0, e0, _ in spans)
                    and not (n["pos"] > 0 and re.match(r"[a-z_]", cl[n["pos"] - 1]))
                    and not (pd_val is not None and (n["pct"] and _num_matches(n, pd_val * 100)))]
            if fields and len(fields) == len(nums):
                # each number checked against its paired field (and that field's
                # fact/attribution twin)
                twin = lambda f: [x for x in mentioned if (x.source_field or x.feature_group or "").lower()
                                  .replace(" (derived)", "") == (f.source_field or f.feature_group or "")
                                  .lower().replace(" (derived)", "")] or [f]
                resp_pairs = {pos: twin(f) for pos, f in zip(nums, fields)}
            else:
                resp_pairs = {n["pos"]: list(mentioned) for n in _signed_numbers(clause)}   # pooled
        if resp_pairs is None:
            m2 = re.search(r"\band\b[^.;]{0,60}?\b(?:are|were|of|at|stand at)\s+[^.;]{0,40}?\band\b", cl)
            if m2:
                firsts = {}
                for s0, e0, it in spans:
                    if s0 < m2.end():
                        firsts.setdefault((it.source_field or it.feature_group or it.eid).lower()
                                          .replace(" (derived)", ""), (s0, it))
                flds = [it for _, it in sorted(firsts.values(), key=lambda t: t[0])]
                nums2 = [n["pos"] for n in _signed_numbers(clause) if n["pos"] > m2.start()
                         and not any(s0 <= n["pos"] < e0 for s0, e0, _ in spans)
                         and not (n["pos"] > 0 and re.match(r"[a-z_]", cl[n["pos"] - 1]))]
                if len(flds) == 2 and len(nums2) == 2:
                    tw = lambda f: [x for x in mentioned if (x.source_field or x.feature_group or "").lower()
                                    == (f.source_field or f.feature_group or "").lower()] or [f]
                    resp_pairs = {nums2[0]: tw(flds[0]), nums2[1]: tw(flds[1])}
                    respectively = True
        qspans = quoted_spans(cl)
        for n in _signed_numbers(clause):
            npos = n["pos"]
            if any(s0 <= npos < e0 for s0, e0, _ in spans):
                continue                  # digits inside a field name
            if any(s0 <= npos < e0 for s0, e0 in qspans):
                continue                  # digits inside quoted evidence text
            if (re.search(r"between\s+\S*$", cl[max(0, npos - 12):npos]) or re.search(r"between\s+0\s+and\s+$", cl[max(0, npos - 20):npos])) \
                    and any("between 0 and 1" in (f.unit or "") for f in facts) and n["v"] in (0.0, 1.0):
                continue                  # 'normalised between 0 and 1' quotes the unit
            if npos > 0 and re.match(r"[a-z_]", cl[npos - 1]):
                continue                  # 'f3', 'PAY_AMT1', 'a2'
            if eo.borrower_id and any(m.start() <= npos < m.end()
                                      for m in re.finditer(re.escape(eo.borrower_id.lower()), cl)):
                continue
            after_txt = cl[n["end"]:n["end"] + 24]
            if re.match(r"^(?:st|nd|rd|th)\b|^-(?:ranked|largest|strongest|biggest|most)", after_txt):
                continue                  # ordinals: '2nd strongest driver'
            if re.match(r"^\s*(?:of\s+the\s+)?(?:top\s+)?(?:main\s+)?(drivers?|factors?|features?|attributions?|items?|fields?|reasons?|sentences?)\b", after_txt):
                continue                  # 'the 4 drivers': a count of evidence items
            if 1900 <= n["v"] <= 2100 and n["v"] in str_nums:
                continue                  # a year that the evidence states
            # gaps are read with asides blanked: 'The credit term, defined as the
            # credit divided by the annuity, is 8.38' binds 8.38 to the credit term
            prot_cl = _protect(clause).lower()
            in_aside = lambda a, b: prot_cl[a:b].strip() == "" and cl[a:b].strip() != ""
            gap = lambda a, b: re.sub(r"\s+", " ", prot_cl[a:b])
            before = [(e0, it) for s0, e0, it in spans if not in_aside(s0, e0) and e0 <= npos
                      and len(gap(e0, npos)) <= 28
                      and (not re.search(r"[,;]|\band\b", gap(e0, npos))
                           # apposition: 'EXT_SOURCE_1, at 0.83,'
                           or re.fullmatch(r"\s*,\s*(?:at|of|is|was|=|with|:)?\s*(?:nt\$|\$)?\s*", gap(e0, npos)))]
            after = [(s0, it) for s0, e0, it in spans if 0 <= s0 - n["end"] <= 14
                     and (re.match(r"^[\s%]*(for|in|on|of)\s", cl[n["end"]:s0] + " ")
                          # 'the NT$200,000 credit limit': a CURRENCY amount written
                          # directly before an amount field
                          or (re.search(r"(?:nt\$|\$)\s*$", cl[max(0, npos - 4):npos])
                              and re.fullmatch(r"\s*", cl[n["end"]:s0])
                              and "dollar" in (it.unit or "").lower() + " " + "".join(
                                  (by_id[c].unit or "") for c in (it.components or [])[:1] if c in by_id).lower()))]
            if before:
                e_max = max(e for e, _ in before)
                before = [it for e, it in before if e == e_max]
            if after:
                s_min = min(s0 for s0, _ in after)
                after = [it for s0, it in after if s0 == s_min]
            # 'N in/for/of FIELD' names its field explicitly: it wins over a field
            # named earlier ('from 2 in PAY_2 to -2 in PAY_0')
            bound = after or before
            mm = re.match(r"^\s*(?:nt\$|\$)?\s*(?:nt dollars\s*)?(?:in|for|during)\s+(january|february|march|april|"
                          r"may|june|july|august|september|october|november|december)\b", after_txt)
            if mm:
                ser_here = [k for k, pats in _SERIES_NOUNS.items()
                            if any(re.search(rf"\b{pt}\b", _canon(cl)) for pt in pats)]
                month_f = [f for f in facts if f.series in ser_here
                           and mm.group(1) in (f.observed_at or "").lower()]
                if month_f:
                    bound = month_f
            if respectively and resp_pairs is not None:
                bound = resp_pairs.get(npos, bound)
            scope = bound or mentioned       # unbound: the fields this clause talks about
            if not scope and n["v"] in (0.0, 1.0) and not n["pct"]:
                continue
            cands: List[Tuple[float, bool]] = []
            prob_change = bool(re.search(r"\b(?:probability|chance|likelihood|pd|risk of default|default risk)\b"
                                         r"[^.;]{0,50}\bby\s*(?:about\s+|around\s+|roughly\s+)?$", cl[:npos]))
            if n.get("pp") or prob_change:
                # 'raises the probability of default by 0.61' / '1.6 percentage points':
                # the evidence states no probability change; only the PD itself is allowed
                if pd_val is not None and not n.get("pp"):
                    cands += [(pd_val, False), (pd_val * 100, False)]
            elif n["pct"]:
                def ratio_like(it):
                    fs = [it] + [by_id[c] for c in (it.components or []) if c in by_id]
                    return any("percent" in (f.equivalents or {}) or
                               (isinstance(f.value, (int, float)) and abs(f.value) <= 1.5)
                               for f in fs if f.etype is EvidenceType.OBSERVED_FACT)
                def ratio_facts():
                    # real ratios only: percent equivalents or fractions; never
                    # encoded codes (a status code 2 is not 200%)
                    return [f for f in facts if not f.value_label and (
                        "percent" in (f.equivalents or {}) or
                        (isinstance(f.value, (int, float)) and abs(f.value) <= 1.5))]
                share = bool(re.match(r"^\s*(?:of|of the)\b", after_txt)            # '22.4% of the limit'
                             or re.search(r"\b(?:ratio|share|proportion|utili[sz]ation|as a percent(?:age)?)\b", cl))
                rb = [it for it in (before or []) if ratio_like(it)]
                if before and not rb and not share:
                    # 'age is 200%': a percentage attached to a non-ratio field
                    if pd_val is not None and prob_ctx:
                        cands.append((pd_val * 100, False))
                elif rb:
                    # 'EXT_SOURCE_2 is 34%': only that field's own ratio
                    for it in rb:
                        for f in [it] + [by_id[c] for c in (it.components or []) if c in by_id]:
                            cands += [(x * 100, False) for x in _item_values(f) if abs(x) <= 10]
                            cands += [(x, False) for k, x in (f.equivalents or {}).items() if k == "percent"]
                elif prob_ctx and not share:
                    # 'the chance of default is 62%': only the PD
                    if pd_val is not None:
                        cands.append((pd_val * 100, False))
                else:
                    # a share 'of the credit limit': only ratios whose inputs include
                    # the named field; otherwise an unbound percentage
                    if pd_val is not None and not mentioned:
                        cands.append((pd_val * 100, False))
                    of_f = [it for it in (after or []) + [m for m in mentioned if m not in (after or [])]
                            if it.etype is EvidenceType.OBSERVED_FACT]
                    rf = ratio_facts()
                    if share and of_f:
                        rf = [f for f in rf if any(o.eid in (f.components or []) for o in of_f)] or rf
                    for it in rf:
                        cands += [(x * 100, False) for x in _item_values(it) if abs(x) <= 10]
                        cands += [(x, False) for k, x in (it.equivalents or {}).items() if k == "percent"]
            elif (re.search(r"(contribut\w*|log-odds|attribution|shap|weight|margin)\W+(?:\w+\W+){0,4}$",
                            cl[max(0, npos - 45):npos])
                  or re.search(r"\b(?:risk|score|margin)\b[^.;]{0,20}\bby\s*(?:about\s+|around\s+)?$", cl[max(0, npos - 45):npos])
                  or re.match(r"^\s*(?:log-odds|logit|on the (?:raw )?log-odds|on the (?:raw )?margin|in shap|shap)\b",
                              cl[n["end"]:n["end"] + 30])):
                # stated as a contribution: must be one of the scope's contributions
                if not scope:
                    cands += all_contribs
                for it in scope:
                    if it.etype is EvidenceType.MODEL_ATTRIBUTION:
                        cands.append((float(it.contribution or 0.0), True))
                    for a in attrs:
                        if it.eid in (a.components or [])[:1] or a.feature_group == it.source_field:
                            cands.append((float(a.contribution or 0.0), True))
            else:
                if pd_val is not None and (not scope or prob_ctx):
                    cands += [(pd_val, False), (pd_val * 100, False)]
                cands += bound_values(scope) if scope else all_vals
                if re.search(r"credit balance|in credit|overpaid", cl):
                    cands += [(abs(x), False) for x, _ in (bound_values(scope) if scope else all_vals) if x < 0]
            if not any(_num_matches(n, e, mag) for e, mag in cands):
                v.append({"type": ErrorType.NUMERIC_ERROR.value,
                          "detail": f"{n['v']:g}{'%' if n['pct'] else ''} not in "
                                    + (",".join(i.eid for i in scope) or "evidence"),
                          "span": clause.strip()[:200],
                          "evidence_ids": [i.eid for i in scope]})

        # (d) direction claims, bound to their driver (asides removed first)
        ment = sorted([(pos, a) for a in attrs for pos in _mentions(dl, plain[a.eid])]
                      + [(s0, it) for s0, _, it in period_spans(_protect(clause))
                         if it.etype is EvidenceType.MODEL_ATTRIBUTION],
                      key=lambda t: t[0])
        if ment:
            claimed: Dict[str, set] = {}
            taken = set()
            phr: List[Tuple[int, int]] = []
            for pat, pol, passive in _DIR_PATTERNS:
                for m in pat.finditer(dl):
                    if any(s <= m.start() < e for s, e in taken):
                        continue              # overlapping phrase already read
                    taken.add((m.start(), m.end()))
                    p = -pol if _governed_neg(dl[:m.start()]) else pol
                    if re.search(r"\bneither\b[^.;]*\bnor\b", dl[:m.start()]):
                        p = -pol          # 'neither X nor Y raises ...'
                    phr.append((m.start(), p))
                    before = [(pos, a) for pos, a in ment if pos < m.start()]
                    after = [(pos, a) for pos, a in ment if pos >= m.end()
                             and re.match(r"^\s*(?:\w+ly\s+)?(by|from|due to|because of|through)\b",
                                          dl[m.end():pos] + " ")]
                    if passive and after:
                        tgts = [after[0][1]]
                    elif before:
                        # coordinated subjects: 'X and Y both raise ...'
                        last = before[-1][0]
                        tgts = [before[-1][1]]
                        for pos, a in reversed(before[:-1]):
                            seg = dl[pos:last]
                            if (re.fullmatch(r"[\w\s'-]*?\s*(?:,|and|nor|as well as|together with)\s*(?:the\s+)?[\w\s'-]*?", seg)
                                    and len(seg) < 60 and not re.search(
                                        r"\b(?:is|are|was|were|has|have|had|matters?|raises?|lowers?|reduces?|increases?|"
                                        r"drives?|shows?|remains?|stays?|contributes?|appears?|seems?|which|that)\b", seg)):
                                tgts.append(a)
                                last = pos
                            else:
                                break
                    elif after:
                        tgts = [after[0][1]]
                    else:
                        continue
                    for t in tgts:
                        claimed.setdefault(t.eid, set()).add(p)
            # ellipsis: 'X raises the score, and so does Y' / 'as does Y'
            for em in re.finditer(r"\b(?:and so does|so does|as does|and likewise|likewise|and also does)\s+", dl):
                prev = [pp for st, pp in phr if st < em.start()]
                tgt = [a for pos, a in ment if 0 <= pos - em.end() <= 4]
                if prev and tgt:
                    claimed.setdefault(tgt[0].eid, set()).add(prev[-1])
            for a in attrs:
                ps = claimed.get(a.eid)
                if not ps or len(ps) > 1:
                    continue
                want = +1 if a.direction == "increases_risk" else -1
                if next(iter(ps)) != want:
                    v.append({"type": ErrorType.SIGN_REVERSAL.value, "detail": a.feature_group,
                              "span": clause.strip()[:200], "evidence_ids": [a.eid]})

    # (e) omission: the strongest driver, and the probability estimate
    if attrs:
        top = max(attrs, key=lambda a: abs(a.contribution or 0.0))
        top_fact = next((by_id[c] for c in (top.components or []) if c in by_id), None)
        if not (_mentions(low, plain[top.eid])
                or (top_fact is not None and (_mentions(low, plain[top_fact.eid])
                                              or _period_mentions(text, top_fact)))):
            v.append({"type": ErrorType.OMISSION.value, "detail": top.feature_group,
                      "evidence_ids": [top.eid]})
    if pd_val is not None:
        pd_ok = False
        for off, clause in _clauses(text):
            ctx = bool(re.search(r"\b(probability|chance|likelihood|pd|default|estimate[sd]?|predicts?|predicted|"
                                 r"risk|rates?|puts?)\b", clause.lower()))
            for n in _signed_numbers(clause):
                if n["pct"] and _num_matches(n, pd_val * 100) and ctx:
                    pd_ok = True
                elif not n["pct"] and abs(n["v"]) < 1 and _num_matches(n, pd_val) and ctx:
                    pd_ok = True
            if pd_ok:
                break
        if not pd_ok:
            v.append({"type": ErrorType.OMISSION.value, "detail": "probability estimate",
                      "evidence_ids": [pred.eid]})

    # (f) typed claims must cite ids that exist
    valid = set(eo.ids())
    if require_citations and cand.claims:
        for c in cand.claims:
            ids = c.get("evidence_ids") or []
            if not ids:
                v.append({"type": "uncited_claim", "detail": c.get("text", "")[:90]})
            elif [i for i in ids if i not in valid]:
                v.append({"type": "unknown_evidence_id",
                          "detail": f"{[i for i in ids if i not in valid]} in: {c.get('text', '')[:70]}"})

    seen, uniq = set(), []
    for x in v:
        k = (x["type"], x.get("detail"))
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return Verdict(accepted=not uniq, checker=STRUCTURAL_VERSION, violations=uniq)


# ---------------------------------------------------------------------------
# 2. semantic checker - an LLM judge (a fallible signal)
# ---------------------------------------------------------------------------
_JUDGE_SYS = (
    "You audit a credit explanation against the model evidence it must rest on. "
    "Flag ONLY: (1) a factor named in the text that is absent from the evidence; "
    "(2) a direction opposite to an attribution's stated direction; (3) a number "
    "that does not appear in the evidence; (4) an asserted change over time when "
    "the evidence is a single snapshot; (5) an asserted real-world causal effect "
    "from a modelled scenario. Plain-language restatement of evidence is fine. "
    "Omitting minor items is fine. Be strict on invention, lenient on wording. "
    'Reply with JSON only: {"accepted":true|false,"violations":['
    '{"type":"invented_factor|sign_reversal|numeric_or_unit_error|'
    'unsupported_trend|unsupported_causal_or_action","detail":"..."}]}')


# judge/2.0 (2 Oct 2026): aligned with the repaired rubric (plan §5.2). The
# v1 prompt above is the one the 2 Oct qualification used; it stays the
# default until judge/2.0 is itself qualified (qualify_judges.py --prompt 2).
_JUDGE_SYS_V2 = (
    "You audit a credit explanation against the model evidence it must rest on. "
    "Flag ONLY: (1) a factor, fact or reason named in the text that is absent from the "
    "evidence; (2) a direction opposite to an attribution's stated direction; (3) a number "
    "that does not match the evidence value of the field it is attached to (sensible "
    "rounding is fine; attribution contributions are on the raw log-odds scale and must not "
    "be presented as percentages or probability changes); (4) an asserted change over time, "
    "unless it compares values for periods that are all shown in the evidence; (5) an "
    "asserted real-world causal effect, or a claim that an action will change the outcome. "
    "Plain-language restatement of evidence is fine. Stating that the explanation describes "
    "the model rather than a verified cause is correct, not a causal claim. Omitting minor "
    "items is fine. Be strict on invention, lenient on wording. "
    'Reply with JSON only: {"accepted":true|false,"violations":['
    '{"type":"invented_factor|sign_reversal|numeric_or_unit_error|'
    'unsupported_trend|unsupported_causal_or_action","detail":"...","quote":"<exact words>",'
    '"evidence_ids":["..."]}]}')
JUDGE_PROMPTS = {1: _JUDGE_SYS, 2: _JUDGE_SYS_V2}


def semantic_check(cand: Candidate, eo: EvidenceObject, judge_family: str,
                   *, enforce_independence: bool = True,
                   experiment: str = "", prompt_version: int = 1) -> Verdict:
    payload = {"evidence": eo.for_generator(), "explanation": cand.text}
    msgs = [{"role": "system", "content": JUDGE_PROMPTS[prompt_version]},
            {"role": "user", "content": json.dumps(payload, default=str)[:12000]}]
    # v2: a trace key per candidate (borrower|arm|family) and room for quotes
    item = (f"{cand.borrower_id}|{cand.arm}|{cand.family}" if prompt_version >= 2
            else cand.borrower_id)
    mt = 1500 if prompt_version >= 2 else 400
    try:
        if enforce_independence and cand.family not in ("template", ""):
            r = GATEWAY.judged_by(judge_family, cand.family, msgs,
                                  temperature=0.0, max_tokens=mt, want_json=True,
                                  experiment=experiment, item_id=item)
        else:
            r = GATEWAY.complete(judge_family, msgs, temperature=0.0,
                                 max_tokens=mt, want_json=True,
                                 role=f"judge[{judge_family}]",
                                 experiment=experiment, item_id=item)
    except JudgeIndependenceError as e:
        return Verdict(False, f"semantic:{judge_family}", error=str(e), error_kind="config")
    except Exception as e:  # noqa: BLE001 - judge outage is a system-level case
        return Verdict(False, f"semantic:{judge_family}",
                       error=f"{type(e).__name__}: {e}", error_kind="transport")

    if prompt_version >= 2:
        # judge/2.x: JSON contract with the parser judge as last resort; a
        # verdict the parser cannot copy verbatim from the response is UNRESOLVED
        import json_parse as JP
        d, info = JP.parse_with_parser_judge(
            r.text, schema_hint=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
            verdict_key="accepted", experiment=experiment, item_id=item,
            role=f"judge[{judge_family}]", kind="judge", finish_reason=r.finish_reason)
        common = dict(judge_family=judge_family, judge_model_id=r.model_id,
                      latency_ms=r.latency_ms, parse_method=info["method"], raw=r.text[:4000],
                      provider=r.provider, route_index=r.route_index,
                      finish_reason=r.finish_reason, prompt_tokens=r.prompt_tokens,
                      completion_tokens=r.completion_tokens, call_id=r.call_id)
        if d is None:
            return Verdict(False, f"semantic:{judge_family}/judge-{prompt_version}",
                           error=f"{info['method']}_judge_output: {info.get('error', info.get('local'))}"[:200],
                           error_kind=("transport" if info["method"] == "parser_unavailable" else "model"),
                           **common)
        known = set(JP.VIOLATION_TYPES)
        viol = [x for x in (d.get("violations") or []) if isinstance(x, dict)
                and str(x.get("type", "")).lower().replace(" ", "_") in known]
        # one consistency rule for every parse path: accept <=> no violation
        if d["accepted"] != (not viol):
            return Verdict(False, f"semantic:{judge_family}/judge-{prompt_version}",
                           violations=viol, error="inconsistent_judge_output: verdict and violations disagree",
                           error_kind="model", **common)
        return Verdict(accepted=d["accepted"],
                       checker=f"semantic:{judge_family}/judge-{prompt_version}",
                       violations=viol, **common)

    try:
        t = re.sub(r"^```(?:json)?\s*", "", r.text.strip(), flags=re.I)
        t = re.sub(r"\s*```$", "", t)
        d = json.loads(t)
    except Exception:
        try:
            from json_repair import repair_json
            d = json.loads(repair_json(r.text))
        except Exception:
            return Verdict(False, f"semantic:{judge_family}",
                           judge_family=judge_family, judge_model_id=r.model_id,
                           latency_ms=r.latency_ms, error="unparseable_judge_output")
    # Wrong-shaped or non-boolean verdicts are UNRESOLVED, never silently
    # mapped to accept or reject (plan §2 P0: "Validate accepted booleans
    # strictly; malformed/truncated responses are unresolved"). bool("false")
    # is True in Python, so a string verdict must not reach bool().
    if not isinstance(d, dict) or not isinstance(d.get("accepted"), bool):
        return Verdict(False, f"semantic:{judge_family}", judge_family=judge_family,
                       judge_model_id=r.model_id, latency_ms=r.latency_ms,
                       error=f"malformed_judge_output:{type(d).__name__}")
    viol = [x for x in (d.get("violations") or []) if isinstance(x, dict)]
    acc = d["accepted"]
    return Verdict(accepted=acc and not viol,
                   checker=f"semantic:{judge_family}/judge-{prompt_version}",
                   violations=viol, judge_family=judge_family,
                   judge_model_id=r.model_id, latency_ms=r.latency_ms)


# ---------------------------------------------------------------------------
# 3. combined
# ---------------------------------------------------------------------------
def combined_check(cand: Candidate, eo: EvidenceObject, judge_family: str,
                   **kw) -> Verdict:
    s = structural_check(cand, eo)
    if not s.accepted:
        return Verdict(False, f"combined[struct|{judge_family}]", s.violations)
    j = semantic_check(cand, eo, judge_family, **kw)
    return Verdict(j.accepted, f"combined[struct|{judge_family}]", j.violations,
                   judge_family=j.judge_family, judge_model_id=j.judge_model_id,
                   latency_ms=j.latency_ms, error=j.error)


# ---------------------------------------------------------------------------
# 4. controlled corruptions for the E1 benchmark
# ---------------------------------------------------------------------------
def corrupt(cand: Candidate, eo: EvidenceObject, etype: ErrorType,
            rng: Optional[random.Random] = None) -> Tuple[Candidate, bool]:
    """Return (corrupted candidate, applied?). `applied=False` when the text
    offers no site for that corruption - never fake it."""
    rng = rng or random.Random(0)
    t = cand.text
    if not t.strip():
        return cand, False
    out, applied = t, False

    if etype is ErrorType.INVENTED_FACTOR:
        seen = {(a.feature_group or "").lower()
                for a in eo.by_type(EvidenceType.MODEL_ATTRIBUTION)}
        for f in ("a recent tax dispute", "an unresolved legal claim",
                  "a declining order book", "a supplier concentration problem"):
            if f.split()[-1] not in seen:
                out = t + f" The assessment also reflects {f}."
                applied = True
                break

    elif etype is ErrorType.SIGN_REVERSAL:
        for a in eo.by_type(EvidenceType.MODEL_ATTRIBUTION):
            fg = a.feature_group or ""
            if fg and fg.lower() in t.lower():
                out = re.sub(rf"({re.escape(fg)})", r"\1", t, count=1)
                out += (f" On balance {fg} lowers the modelled risk."
                        if a.direction == "increases_risk"
                        else f" On balance {fg} raises the modelled risk.")
                applied = True
                break

    elif etype is ErrorType.NUMERIC_ERROR:
        m = _NUM.search(t)
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
                out = t[:m.start(1)] + f"{v * 1.9:,.2f}" + t[m.end(1):]
                applied = True
            except ValueError:
                pass

    elif etype is ErrorType.UNSUPPORTED_TREND:
        out = t + " Turnover has declined steadily over recent months."
        applied = True

    elif etype is ErrorType.UNSUPPORTED_CAUSAL:
        out = t + " Restructuring the facility will reduce the default risk."
        applied = True

    elif etype is ErrorType.OMISSION:
        attrs = sorted(eo.by_type(EvidenceType.MODEL_ATTRIBUTION),
                       key=lambda a: -abs(a.contribution or 0.0))
        if attrs:
            top = attrs[0].feature_group or ""
            if top and top.lower() in t.lower():
                out = re.sub(re.escape(top), "another factor", t, flags=re.I)
                applied = True

    c = Candidate(**{**cand.__dict__, "text": out})
    c.arm = f"{cand.arm}+{etype.value}"
    return c, applied


if __name__ == "__main__":
    from evidence import EvidenceType as ET, new_item
    from generate import render_template
    eo = EvidenceObject(borrower_id="V-1", dataset="sba",
                        prediction_time="approval_date")
    eo.add(new_item("f1", ET.OBSERVED_FACT, source_field="GrAppv", value=455000.0,
                    unit="USD", observed_at="approval_date"))
    eo.add(new_item("a1", ET.MODEL_ATTRIBUTION, model_id="m0",
                    feature_group="GrAppv", contribution=0.41,
                    direction="increases_risk",
                    explained_scale="raw log-odds margin",
                    reference_background="training-fold mean"))
    eo.add(new_item("p1", ET.MODEL_PREDICTION, calibrated_pd=0.31,
                    target_definition="CHGOFF", horizon="loan lifetime",
                    model_id="m0", calibrator_id="iso0"))
    base = render_template(eo)
    print("clean ->", structural_check(base, eo).accepted)
    for et in ErrorType:
        c, applied = corrupt(base, eo, et)
        if not applied:
            print(f"  {et.value:<32} (no site, skipped)")
            continue
        v = structural_check(c, eo)
        print(f"  {et.value:<32} caught={not v.accepted:<5} {[x['type'] for x in v.violations]}")

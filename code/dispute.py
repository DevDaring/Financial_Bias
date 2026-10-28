"""
ReCheck 2: disputed-interpretation handling (Future_Plan review §3.1).

ReCheck 1.x treated the code's own reading as unquestionable: a judge objection that code did not
confirm was dismissed, and against a claim code had already supported it changed nothing, so the
narrative could be released. ReCheck 2 records such a code-judge disagreement as a DISPUTE.

  D  (disputes-to-defer)    any dispute defers. The conservative baseline the review requires.
  R  (reinterpret)          one INDEPENDENT reading of each disputed claim by a model of another
                            family, which sees the claim and the evidence but neither the judge's
                            objection nor the code's verdict. It returns each proposition's binding
                            (evidence ids), operation, asserted value/direction, polarity and an
                            ambiguity flag. Code executes THAT binding (not its own parse):
                              - any executed contradiction under the independent binding   -> reject
                              - every proposition executed as supported, none residual or
                                ambiguous, and the disputed words covered                 -> resolved
                              - anything else                                             -> defer
A dismissed objection never becomes positive support by itself, and an unresolved dispute defers.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Tuple

import contract as C
import gateway as G
import json_parse as JP
import verify as V
import witness as W
from evidence import EvidenceObject, EvidenceType

DISPUTE_VERSION = "dispute/1.1"
#: 1.1 (7 Oct, after the context experiment v1 showed that neutral framing prose was deferred): the reader may mark
#: a proposition NONE - it states nothing about this borrower, their data or the model's output. A disputed claim
#: whose reading is ONLY such propositions is non-evidential and does not block release. 1.0 behaviour: NON_EVIDENTIAL = False.
NON_EVIDENTIAL = True
EXEC_OPS = {"VALUE", "DIRECTION", "RANK", "SCALE", "TEMPORAL", "FACT_DRIVER", "CODE"}

REPARSE_SYS = (
    "You read sentences from a credit-risk explanation and record exactly what each one ASSERTS about the "
    "evidence. You do not judge whether anything is true.\n"
    "The evidence lists recorded facts (with ids), the model's attributions (contributions on the raw log-odds "
    "scale with a direction) and the probability of default.\n"
    "For every passage, list each proposition it asserts. For each proposition give:\n"
    "- quote: the exact words of the passage that assert it;\n"
    "- operation: VALUE (a number is given for a field), DIRECTION (a field is said to raise or lower the "
    "model's risk), RANK (a field is called the strongest/main/largest driver, optionally among raising or "
    "lowering ones), SCALE (an attribution is expressed as a percentage or probability change), TEMPORAL (a "
    "change over time between periods), CODE (a meaning is given to a coded value), TARGET (what is predicted "
    "or its horizon), CAUSAL (a real-world cause or the effect of an action), NONE (the words state nothing about "
    "this borrower, the borrower's data or the model's output: general framing such as how to read a summary), "
    "OTHER (anything else about the borrower, the data or the model);\n"
    "- evidence_ids: the ids of the evidence items the words refer to (empty if none fits);\n"
    "- asserted: for VALUE the number as written; for DIRECTION 'increases' or 'decreases'; for RANK "
    "'strongest', 'strongest_increasing' or 'strongest_decreasing'; for TEMPORAL 'increase' or 'decrease' "
    "with evidence_ids ordered oldest first; otherwise a short paraphrase;\n"
    "- polarity: 'asserted', 'negated' (the passage denies it) or 'conditional';\n"
    "- ambiguous: true if the reference, scope or meaning is unclear.\n"
    "For each coverage question, quote the words (if any) that convey the named evidence item and the "
    "direction of its effect, or give an empty quote.\n"
    'Reply with JSON only: {"passages":[{"pid":"p1","propositions":[{"quote":"","operation":"VALUE",'
    '"evidence_ids":[""],"asserted":"","polarity":"asserted","ambiguous":false}]}],'
    '"coverage":[{"evidence_id":"","quote":"","direction":"increases|decreases|none"}]}')
REPARSE_SYS_10 = REPARSE_SYS.replace(
    "NONE (the words state nothing about this borrower, the borrower's data or the model's output: general framing "
    "such as how to read a summary), OTHER (anything else about the borrower, the data or the model);",
    "OTHER (anything else);")
REPARSE_SCHEMA = ('{"passages": [{"pid": "p1", "propositions": [{"quote": "", "operation": "VALUE", '
                  '"evidence_ids": [""], "asserted": "", "polarity": "asserted", "ambiguous": false}]}], '
                  '"coverage": []}')


def _valid(o) -> bool:
    return isinstance(o, dict) and isinstance(o.get("passages"), list) and all(
        isinstance(p, dict) and isinstance(p.get("propositions", []), list) for p in o["passages"])


# ---------------------------------------------------------------------------
# execute an independently supplied binding
# ---------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _number(s: str) -> Optional[dict]:
    m = _NUM.search(str(s or ""))
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    dec = len(raw.split(".")[1]) if "." in raw else 0
    pct = "%" in str(s) or "percent" in str(s).lower()
    return {"v": v, "tol": 0.5 * 10 ** -dec, "pct": pct, "pp": False, "signed": False, "lo": v, "hi": v}


def _dir_word(s: str) -> int:
    s = str(s or "").lower()
    if re.search(r"increas|rais|higher|up|add", s):
        return +1
    if re.search(r"decreas|lower|reduc|down|offset", s):
        return -1
    return 0


def _attr_of(eo: EvidenceObject, eid: str):
    it = eo.get(eid)
    if it is None:
        return None
    if it.etype is EvidenceType.MODEL_ATTRIBUTION:
        return it
    own = W._owner(eo, eid)
    return eo.get(own) if own else None


def execute_proposition(p: dict, eo: EvidenceObject, text: str) -> Tuple[str, str]:
    """(status, detail): status in support | contradict | residual | invalid | none (non-evidential, 1.1)."""
    op = str(p.get("operation") or "").upper()
    ids = [str(i) for i in (p.get("evidence_ids") or []) if str(i).strip()]
    pol = str(p.get("polarity") or "asserted").lower()
    quote = str(p.get("quote") or "")
    if quote and W.find_quote(text, quote) < 0:
        return "invalid", "quote not in the passage"
    if p.get("ambiguous") is True:
        return "residual", "reader marked the reference or scope ambiguous"
    if op == "NONE":
        if ids:
            return "residual", "NONE cites evidence: not non-evidential"
        return "none", "states nothing about the borrower, the data or the model"
    if op == "CAUSAL":
        return ("support", "causation denied") if pol == "negated" else ("residual", "causal content")
    if op not in EXEC_OPS:
        return "residual", f"{op or 'OTHER'} is not executable"
    if pol == "conditional":
        return "residual", "conditional proposition"
    bad = [i for i in ids if eo.get(i) is None]
    if bad or not ids:
        return "invalid", f"unknown or missing evidence ids {bad}"
    neg = pol == "negated"
    if op == "VALUE":
        n = _number(p.get("asserted"))
        if n is None:
            return "invalid", "no number asserted"
        ok = False
        for i in ids:
            it = eo.get(i)
            vals = W._values(eo, it) if it.etype is not EvidenceType.MODEL_PREDICTION else [it.calibrated_pd]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if W._num_ok(n, vals, W._pct_values(it)):
                ok = True
        return ("support" if ok != neg else "contradict", f"number {n['v']} vs {ids}")
    if op in ("DIRECTION", "FACT_DRIVER"):
        d = _dir_word(p.get("asserted"))
        if d == 0:
            return "residual", "no direction asserted"
        out = []
        for i in ids:
            a = _attr_of(eo, i)
            if a is None:
                if neg:
                    out.append("support")
                    continue
                return "contradict", f"{i} has no attribution: a recorded fact given a direction (R_FACT_AS_DRIVER)"
            if abs(a.contribution or 0.0) < W.ZERO_TOL:
                return "residual", f"{a.eid} contribution is about zero"
            out.append("support" if (W._sign(a) == d) != neg else "contradict")
        return ("contradict", "direction differs from the attribution sign") if "contradict" in out \
            else ("support", "direction matches")
    if op == "RANK":
        at = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
        s = str(p.get("asserted") or "").lower()
        if "increas" in s:
            at = [x for x in at if x.direction == "increases_risk"]
        elif "decreas" in s:
            at = [x for x in at if x.direction == "decreases_risk"]
        if not at:
            return "residual", "no attribution in the named direction"
        top = max(abs(x.contribution or 0.0) for x in at)
        tops = {x.eid for x in at if abs(abs(x.contribution or 0.0) - top) <= W.TIE_TOL}
        named = {(_attr_of(eo, i).eid if _attr_of(eo, i) is not None else i) for i in ids}
        ok = bool(named & tops)
        return ("support" if ok != neg else "contradict", f"top {sorted(tops)} vs named {sorted(named)}")
    if op == "SCALE":
        ats = [i for i in ids if eo.get(i).etype is EvidenceType.MODEL_ATTRIBUTION]
        if ats and ("%" in str(p.get("asserted")) or "percent" in str(p.get("asserted")).lower()
                    or "probab" in str(p.get("asserted")).lower()) and not neg:
            return "contradict", "attribution presented as a percentage or probability change (R_SCALE)"
        return "residual", "scale proposition not decidable from the binding"
    if op == "TEMPORAL":
        d = _dir_word(p.get("asserted"))
        vals = []
        for i in ids:
            it = eo.get(i)
            try:
                vals.append(float(it.value))
            except (TypeError, ValueError):
                return "residual", "non-numeric series value"
        if len(vals) < 2 or d == 0:
            return "residual", "a change needs two dated values and a direction"
        actual = (vals[-1] > vals[0]) - (vals[-1] < vals[0])
        return ("support" if (actual == d) != neg else "contradict", f"values {vals}")
    if op == "CODE":
        for i in ids:
            it = eo.get(i)
            if not (it.value_label or it.meaning):
                return "contradict", f"{i}: the source does not document the code's meaning (R_CODE_MEANING)"
        return "residual", "documented meaning: wording needs semantic assessment"
    return "residual", "not executed"


# ---------------------------------------------------------------------------
# the reinterpretation call
# ---------------------------------------------------------------------------
def reinterpret(passages: List[Tuple[str, str]], coverage_ids: List[str], eo: EvidenceObject, family: str, *,
                experiment: str, item: str, max_tokens: int = 4000, version: str = "1.1"):
    payload = {"evidence": eo.for_generator(), "passages": [{"pid": p, "text": t} for p, t in passages],
               "coverage_questions": [{"evidence_id": i} for i in coverage_ids]}
    msgs = [{"role": "system", "content": REPARSE_SYS if version == "1.1" else REPARSE_SYS_10},
            {"role": "user", "content": json.dumps(payload, default=str)[:16000]}]
    t0 = time.time()
    r = G.GATEWAY.complete(family, msgs, temperature=0.0, max_tokens=max_tokens, want_json=True,
                           experiment=experiment, item_id=item + "|reinterpret", role=f"reinterpret[{family}]")
    call = {"family": family, "model_id": r.model_id, "provider": r.provider, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms or int(1000 * (time.time() - t0)),
            "finish_reason": r.finish_reason, "call_id": r.call_id, "role": "reinterpret"}
    d, how = JP.parse_local(r.text, validate=_valid)
    return d, call, (r.text or "")[:4000], how


def _overlaps(a: str, b: str) -> bool:
    wa = set(re.findall(r"[a-z0-9.]+", C._norm(a or "")))
    wb = set(re.findall(r"[a-z0-9.]+", C._norm(b or "")))
    if not wa or not wb:
        return False
    return len(wa & wb) / max(1, min(len(wa), len(wb))) >= 0.5


def resolve(contested: Dict[str, List[str]], omissions: List[List[str]], tx, text: str, eo: EvidenceObject,
            family: str, *, experiment: str, item: str, version: str = "1.1"):
    """contested: claim_id -> disputed quotes (may be empty). omissions: evidence-id lists the judge says are
    not conveyed but code found conveyed. Returns (decision, reason, record, calls)."""
    claims = {c.claim_id: c for c in tx.claims}
    passages = [(cid, claims[cid].text) for cid in contested if cid in claims]
    cov_ids = sorted({i for ids in omissions for i in ids if eo.get(i) is not None})
    if not passages and not cov_ids:
        return None, "dispute_without_passage", {}, []
    try:
        d, call, raw, how = reinterpret(passages, cov_ids, eo, family, experiment=experiment, item=item,
                                        version=version)
    except Exception as e:  # noqa: BLE001 - an outage leaves the dispute unresolved
        return None, f"reinterpretation_failed: {type(e).__name__}", {}, []
    nonev = NON_EVIDENTIAL and version == "1.1"
    rec = {"version": "dispute/" + version, "family": family, "parse": how, "raw": raw, "claims": {}, "coverage": []}
    if d is None:
        return None, "reinterpretation_unparseable", rec, [call]
    by_pid = {str(p.get("pid")): p for p in d.get("passages", []) if isinstance(p, dict)}
    verdicts = []
    for cid, quotes in contested.items():
        if cid not in claims:
            verdicts.append("residual")
            continue
        props = [p for p in (by_pid.get(cid) or {}).get("propositions", []) if isinstance(p, dict)]
        ex = [(p, *execute_proposition(p, eo, claims[cid].text)) for p in props]
        st = [s for _p, s, _d in ex]
        covered = all(any(_overlaps(q, p.get("quote", "")) for p, s, _d in ex if s == "support")
                      for q in quotes if q) if quotes else True
        # code's own binding of the claim vs the independent one (diagnostic)
        code_ids = sorted({e for o in claims[cid].ops for e in (o.evidence_ids or [])})
        ind_ids = sorted({str(e) for p in props for e in (p.get("evidence_ids") or [])})
        st_ev = [x for x in st if x != "none"] if nonev else st
        if "contradict" in st:
            v = "contradict"
        elif nonev and props and not st_ev:
            v = "support"                    # only non-evidential framing: breaks no contract rule
        elif props and st_ev and all(x == "support" for x in st_ev) and covered:
            v = "support"
        else:
            v = "residual"
        verdicts.append(v)
        rec["claims"][cid] = {"verdict": v, "covered": covered, "code_ids": code_ids, "independent_ids": ind_ids,
                              "propositions": [dict(p, status=s, detail=dd) for p, s, dd in ex]}
    for ids in omissions:
        got = [c for c in (d.get("coverage") or []) if isinstance(c, dict) and str(c.get("evidence_id")) in ids]
        ok = False
        for c in got:
            q = str(c.get("quote") or "")
            if q and W.find_quote(text, q) >= 0:
                dirn = _dir_word(c.get("direction"))
                a = _attr_of(eo, str(c.get("evidence_id")))
                if a is None or a.etype is not EvidenceType.MODEL_ATTRIBUTION:
                    ok = True                       # the probability estimate or a fact: conveyed
                elif dirn != 0 and dirn == W._sign(a):
                    ok = True                       # an attribution must be conveyed with its direction
        rec["coverage"].append({"ids": ids, "conveyed": ok})
        verdicts.append("support" if ok else "residual")
    if "contradict" in verdicts:
        return "reject", "dispute_resolved_contradiction", rec, [call]
    if verdicts and all(v == "support" for v in verdicts):
        return "accept", "dispute_resolved_supported", rec, [call]
    return None, "dispute_unresolved", rec, [call]

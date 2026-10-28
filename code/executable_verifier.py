"""
Executable evidence verification: the proposed method P (Future_Plan §D2).

  1. validate the evidence                      contract.validate_evidence      (no LLM)
  2. split the text into claims independently   witness.split_claims            (no LLM)
  3. execute every checkable relation           witness.execute_text            (no LLM)
       executed contradiction  -> reject (final; the trace is the witness)
       every claim supported and required content present -> accept, no LLM call
  4. semantic assessment of the RESIDUAL claims only (one LLM call). Every
     proposed contradiction must carry a witness: rule, operation, exact quote,
     evidence ids, claimed result.
  5. execute every witness                      witness.execute_witness         (no LLM)
       verified / semantic-plausible -> reject; disproved / malformed /
       unexecutable -> the objection is dismissed, the claim stays unresolved
  6. at most one prespecified reassessment, else unresolved (template/escalation)

The method checks both sides: a claim is accepted only with support (executed, or
semantic support for residual content), and an objection blocks only when its
relation is recomputed as a contradiction (or, for causal/invented/target content,
passes the contract plausibility checks). Deleting bad objections alone would
lower false rejection while raising unsafe release; step 3 prevents that.

Ablations (Future_Plan §E1), all with the same prompt, schema and model:
  P   exec claims + execute witnesses
  A1  exec claims + NO objection checking (any contradicted claim with a rule and a quote blocks)
  A2  exec claims + plausibility-only objection checks (contract.check_objection; no recomputation)
  PX  no claim execution (the LLM assesses every claim) + execute witnesses
  A0  code only: step 3, residual claims stay unresolved (no LLM)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import contract as C
import gateway as G
import json_parse as JP
import witness as W
from evidence import EvidenceObject, EvidenceType
from generate import Candidate

VERIFIER_VERSION = "executable-verifier/1.0"
MODES = {"P": (True, "execute"), "A1": (True, "none"), "A2": (True, "plausibility"),
         "PX": (False, "execute"), "A0": (True, "code_only"),
         # ReCheck 2 (dispute.py): a code-judge disagreement is a dispute, not a dismissal
         "D": (True, "execute"), "R": (True, "execute"), "R11": (True, "execute")}
#: dispute-resolution version per mode: R is frozen at dispute/1.0 (context v1, locked test primary);
#: R11 adds non-evidential framing (dispute/1.1, declared 7 Oct before any locked label was seen)
DISPUTE_VERSIONS = {"R": "1.0", "R11": "1.1"}
DISPUTE_FAMILY = "gptoss120b"    # independent reader for mode R (publisher distinct from every judge)

EV_SYS = (
    "You verify a credit-risk explanation against the model evidence it must rest on. The evidence "
    "lists recorded facts, the model's attributions (contributions on the raw log-odds scale, with a "
    "direction) and the calibrated probability of default.\n"
    "Each numbered claim has a code_status. 'checked_by_code' claims had their numbers, directions and "
    "rankings recomputed from the evidence by code; 'needs_assessment' claims contain content code could "
    "not check (an unrecognised factor name, a causal or horizon statement, an ambiguous reference).\n"
    "For EVERY claim marked needs_assessment, decide:\n"
    "- supported: the evidence licenses it (plain-language restatement, sensible rounding, documented "
    "unit conversions and disclaimers that deny causation are fine);\n"
    "- contradicted: it breaks one contract rule; you MUST give a witness that code will recompute: the "
    "rule, the operation, the EXACT words, the evidence ids, the value the evidence actually gives "
    "(claimed_result) and the conflict;\n"
    "- insufficient: the evidence does not settle it.\n"
    "You may also object to a checked_by_code claim, but only with a witness. Witnesses that code "
    "recomputes as supported are discarded. List omissions of the probability estimate or the strongest "
    "attribution under omissions.\n\n"
    "Contract rules:\n" + C.rule_text() + "\n\n"
    "Operations: VALUE (number vs field value), DIRECTION (stated direction vs attribution sign), RANK "
    "(strongest driver vs largest |contribution|), SCALE (log-odds shown as % or PD change), TEMPORAL "
    "(change over time vs dated values), CODE (meaning of a coded value), FACT_DRIVER (direction given to a "
    "field with no attribution), COVERAGE (omission), COMPLETENESS (\"no other factors\"), SEMANTIC (causal, "
    "invented factor, target/horizon).\n\n"
    'Reply with JSON only: {"claims":[{"claim_id":"c1","status":"supported|contradicted|insufficient",'
    '"witness":{"rule":"R_...","operation":"VALUE|...|SEMANTIC","quote":"<exact words>","evidence_ids":["..."],'
    '"claimed_result":"<what the evidence gives>","alleged_conflict":"..."},"reason":"..."}],'
    '"omissions":[{"evidence_ids":["..."],"reason":"..."}]}')

EV_SCHEMA = ('{"claims": [{"claim_id": "c1", "status": "supported|contradicted|insufficient", "witness": '
             '{"rule": "R_...", "operation": "VALUE", "quote": "<exact words>", "evidence_ids": ["<id>"], '
             '"claimed_result": "", "alleged_conflict": ""}, "reason": ""}], "omissions": []}')
_STATUS = {"supported", "contradicted", "insufficient"}


def valid_ev(o) -> bool:
    if not isinstance(o, dict) or not isinstance(o.get("claims"), list):
        return False
    for c in o["claims"]:
        if not isinstance(c, dict) or str(c.get("status", "")).lower() not in _STATUS:
            return False
    return isinstance(o.get("omissions", []), list)


@dataclass
class ExecVerdict:
    decision: str                                   # accept | reject | unresolved
    reason: str = ""
    mode: str = "P"
    checker: str = VERIFIER_VERSION
    judge_family: Optional[str] = None
    execution: dict = field(default_factory=dict)   # witness.TextExec.to_dict()
    witnesses: List[dict] = field(default_factory=list)
    claim_status: dict = field(default_factory=dict)
    llm_called: bool = False
    reassessed: bool = False
    calls: List[dict] = field(default_factory=list)
    parse: List[dict] = field(default_factory=list)
    raw: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> Optional[bool]:
        return {"accept": True, "reject": False}.get(self.decision)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["accepted"] = self.accepted
        return d


GATE_NOTE = {  # executable/1.6: why code left a claim to the judge, in plain words
    "R_FIELD_IDENTITY": "names a documented field that is NOT in this borrower's evidence: check that the number and "
                        "statement belong to the field actually named (e.g. goods price vs credit amount)",
    "R_NEGATION_SCOPE": "contains a negation: check the stated direction word by word against the attribution sign",
    "R_CODE_SCOPE": "describes a coded value whose meaning the source does not document",
    "R_TARGET": "states an outcome or horizon: check it against the prediction's target and horizon",
    "R_CAUSAL": "may assert a real-world cause or the effect of an action",
    "R_INVENTED": "names a factor that code could not find in the evidence",
}


def _payload(text: str, eo: EvidenceObject, tx: W.TextExec, exec_claims: bool, note: str = "") -> str:
    claims = []
    for c in tx.claims:
        if c.status == "nonmaterial":
            continue
        checked = exec_claims and c.status == W.SUPPORT
        e = {"claim_id": c.claim_id, "text": c.text, "code_status": "checked_by_code" if checked else "needs_assessment"}
        if exec_claims and c.status == W.UNRES:
            e["why_not_checked"] = sorted({o.detail for o in c.ops if o.status == W.UNRES} |
                                          {GATE_NOTE.get(r, r) for r in c.residual})[:5]
        claims.append(e)
    d = {"evidence": eo.for_generator(), "explanation": text, "claims": claims}
    if note:
        d["note"] = note
    return json.dumps(d, default=str)[:16000]


def _call(cand: Candidate, family: str, messages, experiment: str, item: str, max_tokens: int):
    t0 = time.time()
    gw = G.GATEWAY
    kw = dict(temperature=0.0, max_tokens=max_tokens, want_json=True, experiment=experiment, item_id=item,
              role=f"executable[{family}]")
    r = (gw.judged_by(family, cand.family, messages, **kw) if cand.family not in ("template", "")
         else gw.complete(family, messages, **kw))
    call = {"family": family, "model_id": r.model_id, "provider": r.provider, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms or int(1000 * (time.time() - t0)),
            "finish_reason": r.finish_reason, "call_id": r.call_id}
    d, info = JP.parse_with_parser_judge(r.text, schema_hint=EV_SCHEMA, validate=valid_ev, verdict_key=None,
                                         experiment=experiment, item_id=item, role=f"executable[{family}]",
                                         kind="contract", finish_reason=r.finish_reason)
    return d, info, call, r.text


def check_witnesses(ws: List[W.Witness], text: str, eo: EvidenceObject, tx: W.TextExec, how: str) -> List[W.Witness]:
    """execute (P/PX) | plausibility (A2) | none (A1)."""
    out = []
    known = {c.claim_id: c for c in tx.claims}
    for w in ws:
        if w.claim_id not in known and w.quote:          # attach the objection to the claim it quotes
            pos = W.find_quote(text, w.quote)
            for c in tx.claims:
                if pos >= 0 and c.start <= pos < c.start + len(c.text) + 1:
                    w.claim_id = c.claim_id
        if how == "execute":
            W.execute_witness(w, text, eo, tx)
        elif how == "plausibility":
            o = C.check_objection(C.Objection(rule=(w.rule or "").upper(), quote=w.quote,
                                              evidence_ids=list(w.evidence_ids), reason=w.reason), text, eo)
            w.outcome = ("semantic_plausible" if o.valid else "semantic_invalid") if w.rule in W.SEMANTIC_RULES \
                else ("verified" if o.valid else "malformed")
            w.problems = list(o.problems)
        else:   # none: a structurally present objection blocks
            ok = (w.rule or "").upper() in C.RULES and (bool((w.quote or "").strip()) or w.rule == "R_OMISSION")
            w.outcome = "verified" if ok else "malformed"
            w.problems = [] if ok else ["unknown rule or no quote"]
        out.append(w)
    return out


def _resolve(d: dict, tx: W.TextExec, ws: List[W.Witness], exec_claims: bool) -> tuple:
    """(decision or None, reason, claim_status). None = unresolved after this round."""
    status = {str(c.get("claim_id")): str(c.get("status")).lower() for c in d.get("claims", []) if isinstance(c, dict)}
    if any(w.blocks for w in ws):
        return "reject", "valid_witness", status
    need = {c.claim_id for c in tx.claims if c.status != "nonmaterial" and (not exec_claims or c.status == W.UNRES)}
    known = {c.claim_id for c in tx.claims}
    missing = [k for k in sorted(need) if k not in status]
    bad = [k for k in sorted(need) if status.get(k) in ("contradicted", "insufficient")]
    # a dismissed objection leaves a RESIDUAL claim unproved; against a claim that code
    # already supports (or an omission that code disproves) it changes nothing
    open_obj = [w for w in ws if not w.blocks and w.rule != "R_OMISSION"
                and (w.claim_id in need or w.claim_id not in known)]
    if not missing and not bad and not open_obj and not any(o.status == W.CONTRA for o in tx.text_ops):
        return "accept", "supported", status
    why = ("dismissed_objection" if open_obj else
           "unassessed_claims" if missing else "insufficient_or_unproved_claims")
    return None, why, status


def disputes(d: dict, tx: W.TextExec, ws: List[W.Witness], status: dict, eo: EvidenceObject):
    """Every code-judge disagreement after the first round: (claim_id -> disputed quotes, omission id lists)."""
    known = {c.claim_id for c in tx.claims}
    contested, omis = {}, []
    for w in ws:
        if w.blocks:
            continue
        if w.rule == "R_OMISSION":
            ids = [i for i in w.evidence_ids if eo.get(i) is not None]
            if not ids:
                at = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
                ids = [max(at, key=lambda x: abs(x.contribution or 0.0)).eid] if at else []
            omis.append(ids)
        else:
            contested.setdefault(w.claim_id if w.claim_id in known else "_unlocated", []).append(w.quote or "")
    need = {c.claim_id for c in tx.claims if c.status == W.UNRES}
    for cid in need:
        if status.get(cid) != "supported":
            contested.setdefault(cid, [])
    for cid, s_ in status.items():
        if s_ == "contradicted" and cid in known:
            contested.setdefault(cid, [])
    return contested, omis


def verify_executable(cand: Candidate, eo: EvidenceObject, judge_family: Optional[str] = None, *,
                      mode: str = "P", reassess_family: Optional[str] = None, experiment: str = "",
                      max_tokens: int = 2500, claim_level: bool = False,
                      dispute_family: Optional[str] = None) -> ExecVerdict:
    """claim_level=True: the text is one claim, not a full explanation, so whole-text requirements
    (PD stated, strongest driver present, completeness) and omission objections do not apply."""
    exec_claims, how = MODES[mode]
    item = f"{cand.borrower_id}|{cand.arm}|{cand.family}"
    out = ExecVerdict("unresolved", mode=mode, judge_family=judge_family)
    problems = C.validate_evidence(eo)
    if problems:
        out.reason = "insufficient_evidence: " + "; ".join(problems[:4])
        return out
    text = W.clean(cand.text)
    tx = W.execute_text(text, eo)
    if claim_level:
        tx.text_ops = []
    out.execution = tx.to_dict()

    if exec_claims:
        con = tx.contradictions
        if con:
            out.decision, out.reason = "reject", "executed_contradiction"
            return out
        if not tx.residual_claims and all(o.status == W.SUPPORT for o in tx.text_ops):
            out.decision, out.reason = "accept", "all_claims_executed_supported"
            return out
    if how == "code_only" or judge_family is None:
        out.reason = "residual_claims_need_semantic_assessment"
        return out

    msgs = [{"role": "system", "content": EV_SYS},
            {"role": "user", "content": _payload(text, eo, tx, exec_claims)}]
    out.llm_called = True
    try:
        d, info, call, raw = _call(cand, judge_family, msgs, experiment, item, max_tokens)
    except G.JudgeIndependenceError as e:
        out.reason = f"config: {e}"
        return out
    except Exception as e:  # noqa: BLE001 - an outage is an unresolved verification
        out.reason = f"transport: {type(e).__name__}: {str(e)[:120]}"
        return out
    out.calls.append(call); out.parse.append(info); out.raw.append(raw[:4000])
    if d is None:
        out.reason = f"unparseable: {info.get('method')}"
        return out
    ws = check_witnesses([w for w in W.witnesses_from_json(d) if not (claim_level and w.rule == "R_OMISSION")],
                         text, eo, tx, how)
    out.witnesses = [w.to_dict() for w in ws]
    dec, why, out.claim_status = _resolve(d, tx, ws, exec_claims)
    if mode in ("D", "R", "R11"):
        if dec == "reject":
            out.decision, out.reason = dec, why
            return out
        contested, omis = disputes(d, tx, ws, out.claim_status, eo)
        if dec == "accept" and not contested and not omis:
            out.decision, out.reason = dec, why
            return out
        out.claim_status["_disputes"] = {"claims": contested, "omissions": omis}
        if mode == "D" or (not contested and not omis):
            out.reason = "code_judge_dispute" if (contested or omis) else why
            return out
        dec2, why2, rec, calls = __import__("dispute").resolve(
            contested, omis, tx, text, eo, dispute_family or DISPUTE_FAMILY, experiment=experiment, item=item,
            version=DISPUTE_VERSIONS[mode])
        out.calls += calls
        out.claim_status["_reinterpretation"] = rec
        out.reassessed = bool(calls)
        if dec2:
            out.decision = dec2
        out.reason = why2
        return out
    if dec:
        out.decision, out.reason = dec, why
        return out
    if reassess_family is None:
        out.reason = why
        return out

    # one prespecified reassessment
    dismissed = [w for w in ws if not w.blocks]
    note = ("A previous review raised objections that code recomputed and did not confirm: "
            + "; ".join(f"{w.rule} '{(w.quote or '')[:60]}': {w.outcome}"
                        + (f" ({(w.execution or {}).get('detail', '')[:80]})" if w.execution else "")
                        for w in dismissed)[:1500]
            + " Re-assess; object only with a witness whose recomputation shows the error.")
    msgs2 = [{"role": "system", "content": EV_SYS},
             {"role": "user", "content": _payload(text, eo, tx, exec_claims, note=note)}]
    out.reassessed = True
    try:
        d2, info2, call2, raw2 = _call(cand, reassess_family, msgs2, experiment, item + "|reassess", max_tokens)
    except Exception as e:  # noqa: BLE001
        out.reason = f"{why}; reassessment failed: {type(e).__name__}"
        return out
    out.calls.append(call2); out.parse.append(info2); out.raw.append(raw2[:4000])
    if d2 is None:
        out.reason = f"{why}; reassessment unparseable"
        return out
    ws2 = check_witnesses([w for w in W.witnesses_from_json(d2) if not (claim_level and w.rule == "R_OMISSION")],
                          text, eo, tx, how)
    out.witnesses += [dict(w.to_dict(), reassessment=True) for w in ws2]
    dec2, why2, st2 = _resolve(d2, tx, ws2, exec_claims)
    out.claim_status = st2
    if dec2:
        out.decision, out.reason = dec2, why2 + "_on_reassessment"
    else:
        out.reason = f"{why}; unresolved after reassessment"
    return out

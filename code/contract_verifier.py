"""
Contract-aware verification (plan §4.2) - the proposed method.

  1. validate the evidence             contract.validate_evidence   (no LLM)
  2. deterministic relations           contract detectors GATE; structural checks PROPOSE
  3. structured semantic assessment    one LLM call: every claim supported / contradicted /
                                       insufficient, with rule, exact quote and evidence ids;
                                       each structural proposal confirmed or dismissed
  4. check every objection             contract.check_objection     (no LLM)
  5. resolve once or fall back         an invalid objection -> unresolved; at most one
                                       reassessment, then unresolved (template / escalation)
  6. decision                          accept | reject | unresolved, with full audit trail

The structural checker's misreads were the main source of false rejection in the
development labels, so its findings are not final here: the LLM must confirm a
proposal with a quote and a rule, and code checks that confirmation like any
other objection. Contract detectors (ranking, fact-as-driver, undocumented code
meaning, completeness) are high-precision on development data and gate directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import contract as C
import gateway as G
import json_parse as JP
from evidence import EvidenceObject
from generate import Candidate
from verify import structural_check

VERIFIER_VERSION = "contract-verifier/1.0"

CV_SYS = (
    "You verify a credit-risk explanation against the model evidence it must rest on. "
    "The evidence object lists recorded facts, the model's attributions (contributions on the raw "
    "log-odds scale, with a direction) and the calibrated probability of default. "
    "Assess EVERY numbered claim of the explanation:\n"
    "- supported: the evidence licenses what the claim says (plain-language restatement, sensible "
    "rounding and documented unit conversions are fine; saying that the explanation describes the "
    "model and not a verified cause is correct);\n"
    "- contradicted: the claim breaks one contract rule below; give that rule, the EXACT words of "
    "the claim that break it, and the evidence ids that show it;\n"
    "- insufficient: the evidence does not settle the claim either way.\n"
    "Also list omissions (rule R_OMISSION) and rule on each automatic finding: confirm it only if "
    "the quoted words really break the named rule, otherwise dismiss it.\n\n"
    "Contract rules:\n" + C.rule_text() + "\n\n"
    'Reply with JSON only: {"claims":[{"claim_id":"c1","status":"supported|contradicted|insufficient",'
    '"rule":"R_...|null","quote":"<exact words>","evidence_ids":["..."],"reason":"..."}],'
    '"omissions":[{"evidence_ids":["..."],"reason":"..."}],'
    '"findings":[{"finding_id":"s1","confirmed":true|false,"rule":"R_...","quote":"<exact words>",'
    '"evidence_ids":["..."],"reason":"..."}]}')

CV_SCHEMA = ('{"claims": [{"claim_id": "c1", "status": "supported|contradicted|insufficient", "rule": "R_...", '
             '"quote": "<exact words>", "evidence_ids": ["<id>"], "reason": "<text>"}], "omissions": [], '
             '"findings": [{"finding_id": "s1", "confirmed": true, "rule": "R_...", "quote": "", '
             '"evidence_ids": [], "reason": ""}]}')

_STATUS = {"supported", "contradicted", "insufficient"}


def valid_cv(o) -> bool:
    if not isinstance(o, dict) or not isinstance(o.get("claims"), list):
        return False
    for c in o["claims"]:
        if not isinstance(c, dict) or str(c.get("status", "")).lower() not in _STATUS:
            return False
    for k in ("omissions", "findings"):
        if not isinstance(o.get(k, []), list):
            return False
    return True


@dataclass
class ContractVerdict:
    decision: str                                  # accept | reject | unresolved
    reason: str = ""
    checker: str = VERIFIER_VERSION
    judge_family: Optional[str] = None
    gate_violations: List[dict] = field(default_factory=list)     # contract detectors (final)
    proposals: List[dict] = field(default_factory=list)           # structural findings to rule on
    objections: List[dict] = field(default_factory=list)          # every objection + validity
    claim_status: dict = field(default_factory=dict)
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


def _payload(cand: Candidate, eo: EvidenceObject, claims: List[dict], proposals: List[dict],
             note: str = "") -> str:
    d = {"evidence": eo.for_generator(),
         "explanation": cand.text,
         "claims": [{"claim_id": c["claim_id"], "text": c["text"]} for c in claims],
         "automatic_findings": [{"finding_id": p["finding_id"], "rule": p["rule"], "detail": p["detail"]}
                                for p in proposals]}
    if note:
        d["note"] = note
    return json.dumps(d, default=str)[:16000]


_STRUCT_RULE = {"numeric_or_unit_error": "R_VALUE", "sign_reversal": "R_DIRECTION",
                "unsupported_trend": "R_TEMPORAL", "unsupported_causal_or_action": "R_CAUSAL",
                "omission_of_essential_driver": "R_OMISSION", "invented_factor": "R_INVENTED",
                "uncited_claim": "R_INVENTED", "unknown_evidence_id": "R_INVENTED"}


def _call(cand, eo, family, messages, experiment, item, max_tokens):
    t0 = time.time()
    gw = G.GATEWAY
    if cand.family not in ("template", ""):
        r = gw.judged_by(family, cand.family, messages, temperature=0.0, max_tokens=max_tokens,
                         want_json=True, experiment=experiment, item_id=item, role=f"contract[{family}]")
    else:
        r = gw.complete(family, messages, temperature=0.0, max_tokens=max_tokens, want_json=True,
                        experiment=experiment, item_id=item, role=f"contract[{family}]")
    call = {"family": family, "model_id": r.model_id, "provider": r.provider, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms or int(1000 * (time.time() - t0)),
            "finish_reason": r.finish_reason, "call_id": r.call_id}
    d, info = JP.parse_with_parser_judge(r.text, schema_hint=CV_SCHEMA, validate=valid_cv, verdict_key=None,
                                         experiment=experiment, item_id=item, role=f"contract[{family}]",
                                         kind="contract", finish_reason=r.finish_reason)
    return d, info, call, r.text


def _objections(d: dict, text: str, eo: EvidenceObject, proposals: List[dict]) -> List[C.Objection]:
    objs = C.objections_from_json(d)
    known = {p["finding_id"]: p for p in proposals}
    for f in d.get("findings") or []:
        if isinstance(f, dict) and f.get("confirmed") is True and str(f.get("finding_id")) in known:
            p = known[str(f.get("finding_id"))]
            objs.append(C.Objection(rule=str(f.get("rule") or p["rule"]).upper(), quote=str(f.get("quote") or ""),
                                    evidence_ids=[str(x) for x in (f.get("evidence_ids") or p.get("evidence_ids") or [])],
                                    reason=str(f.get("reason") or ""), claim_id=str(f.get("finding_id"))))
    return [C.check_objection(o, text, eo) for o in objs]


def verify_contract(cand: Candidate, eo: EvidenceObject, judge_family: str, *,
                    reassess_family: Optional[str] = None, experiment: str = "",
                    max_tokens: int = 2500) -> ContractVerdict:
    item = f"{cand.borrower_id}|{cand.arm}|{cand.family}"
    # 1. evidence
    problems = C.validate_evidence(eo)
    if problems:
        return ContractVerdict("unresolved", reason="insufficient_evidence: " + "; ".join(problems[:4]),
                               judge_family=judge_family)
    # 2. deterministic: contract detectors gate, structural findings become proposals
    gate = C.contract_violations(cand.text, eo)
    if gate:
        return ContractVerdict("reject", reason="contract_detector", judge_family=judge_family,
                               gate_violations=gate)
    s = structural_check(cand, eo)
    proposals = [{"finding_id": f"s{k + 1}", "rule": _STRUCT_RULE.get(v["type"], "R_VALUE"),
                  "detail": f"{v['type']}: {v.get('detail', '')}"[:200],
                  "evidence_ids": v.get("evidence_ids") or []} for k, v in enumerate(s.violations)]
    claims = C.claims_of(cand.text, cand.claims)

    out = ContractVerdict("unresolved", judge_family=judge_family, proposals=proposals)
    msgs = [{"role": "system", "content": CV_SYS},
            {"role": "user", "content": _payload(cand, eo, claims, proposals)}]
    try:
        d, info, call, raw = _call(cand, eo, judge_family, msgs, experiment, item, max_tokens)
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

    # 3-4. claims and objections
    out.claim_status = {str(c.get("claim_id")): str(c.get("status")).lower() for c in d["claims"]
                        if isinstance(c, dict)}
    missing = [c["claim_id"] for c in claims if c["claim_id"] not in out.claim_status]
    objs = _objections(d, cand.text, eo, proposals)
    out.objections = [o.to_dict() for o in objs]
    valid = [o for o in objs if o.valid]
    if valid:
        out.decision, out.reason = "reject", "valid_objection"
        return out
    insufficient = [k for k, v in out.claim_status.items() if v == "insufficient"]
    if not objs and not missing and not insufficient:
        out.decision, out.reason = "accept", "all_claims_supported"
        return out

    # 5. one prespecified reassessment, else unresolved
    why = ("invalid_objection" if objs else "unassessed_claims" if missing else "insufficient_evidence_claims")
    if reassess_family is None:
        out.reason = why
        return out
    note = ("A previous review raised objections that failed automatic checks: "
            + "; ".join(f"{o.rule} '{o.quote[:60]}': {', '.join(o.problems)}" for o in objs if not o.valid)[:1500]
            + (f" Unassessed claims: {missing}." if missing else "")
            + " Re-assess carefully; object only with an exact quote and the evidence that shows the error.")
    msgs2 = [{"role": "system", "content": CV_SYS},
             {"role": "user", "content": _payload(cand, eo, claims, proposals, note=note)}]
    out.reassessed = True
    try:
        d2, info2, call2, raw2 = _call(cand, eo, reassess_family, msgs2, experiment, item + "|reassess", max_tokens)
    except Exception as e:  # noqa: BLE001
        out.reason = f"{why}; reassessment failed: {type(e).__name__}"
        return out
    out.calls.append(call2); out.parse.append(info2); out.raw.append(raw2[:4000])
    if d2 is None:
        out.reason = f"{why}; reassessment unparseable"
        return out
    st2 = {str(c.get("claim_id")): str(c.get("status")).lower() for c in d2["claims"] if isinstance(c, dict)}
    objs2 = _objections(d2, cand.text, eo, proposals)
    out.objections += [dict(o.to_dict(), reassessment=True) for o in objs2]
    if any(o.valid for o in objs2):
        out.decision, out.reason = "reject", "valid_objection_on_reassessment"
    elif not objs2 and all(c["claim_id"] in st2 for c in claims) and "insufficient" not in st2.values():
        out.decision, out.reason = "accept", "supported_on_reassessment"
    else:
        out.reason = f"{why}; unresolved after reassessment"
    return out

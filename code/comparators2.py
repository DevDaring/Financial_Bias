"""
Competitive baselines added after the 7 Oct manuscript review (Future_Plan review §5). Every one
uses the SAME twelve contract rules that ReCheck's judge sees (aligned rubric), so the comparison
varies the algorithm, not the task specification.

  generic_rubric     B1R  generic judge, one call, aligned rubric
  facts_judge        B1F  B1R plus the code's precomputed per-claim results (values, directions,
                          rankings recomputed from the evidence) given up front, but no ReCheck
                          release logic: tests whether ReCheck's gain is just "supplying calculations"
  forced_tool_judge  B2F  B1R with a REQUIRED tool round (lookup / value / direction / strongest /
                          dated values) before any verdict; a verdict without tool use is sent back
                          once; tool calls, arguments and results are logged (Findeis et al. 2025)
  claimwise_judge    B1C  the same claim inventory ReCheck uses, generic per-claim assessment and
                          FacTool-style aggregation (any non-supported claim or omission -> reject):
                          separates decomposition from executable semantics
  factool_judge           FacTool (Chern et al. 2023), knowledge-based-QA pipeline adapted: LLM claim
                          extraction with evidence queries -> evidence retrieval tool -> per-claim
                          verification; the response is factual iff every claim is
"""
from __future__ import annotations

import json
from typing import Optional

import contract as C
import gateway as G
import json_parse as JP
import witness as W
from comparators import CVerdict, TOOL_SPEC, _llm, _safe, _tool_calls, run_tool
from evidence import EvidenceObject
from generate import Candidate

RUBRIC = (
    "You verify a credit-risk explanation against the model evidence it must rest on. The evidence lists "
    "recorded facts, the model's attributions (contributions on the raw log-odds scale, with a direction) and "
    "the calibrated probability of default. The explanation has a material error if it breaks any rule below.\n"
    "Contract rules:\n" + C.rule_text() + "\n"
    "Plain-language restatement, sensible rounding, documented unit conversions and disclaimers that deny "
    "causation are fine. Omitting minor attributions is fine.")
PASSAGE_NOTE = ("\nThis text is a short passage, not a full explanation: it need not state the probability estimate "
                "or the strongest attribution. Do not report omissions; judge only whether what it states is correct.")


def _sys(base: str, claim_level: bool) -> str:
    return base.replace("Contract rules:", "Contract rules (R_OMISSION does not apply):") + PASSAGE_NOTE \
        if claim_level else base


VERDICT = ('Reply with JSON only: {"accepted":true|false,"violations":[{"type":"<rule, e.g. R_VALUE>",'
           '"detail":"...","quote":"<exact words>","evidence_ids":["..."]}]}')


def _verdict(method, family, d, calls, extra=None) -> CVerdict:
    viol = [x for x in (d.get("violations") or []) if isinstance(x, dict)]
    if d["accepted"] != (not viol):
        return CVerdict(method, None, family, error="inconsistent_judge_output",
                        detail=dict(extra or {}, violations=viol), calls=calls)
    return CVerdict(method, d["accepted"], family, detail=dict(extra or {}, violations=viol), calls=calls)


def _payload(cand, eo, **more) -> str:
    return json.dumps(dict({"evidence": eo.for_generator(), "explanation": cand.text}, **more), default=str)[:16000]


# ---------------------------------------------------------------------------
def generic_rubric(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "",
                   claim_level: bool = False, **_) -> CVerdict:
    def run():
        msgs = [{"role": "system", "content": _sys(RUBRIC, claim_level) + "\n" + VERDICT}, {"role": "user", "content": _payload(cand, eo)}]
        d, call, _raw = _llm(cand, family, msgs, schema=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
                             verdict_key="accepted", max_tokens=2500, experiment=experiment, role=f"b1r[{family}]")
        if d is None:
            return CVerdict("generic_rubric", None, family, error=f"unparseable: {call['parse']}", calls=[call])
        return _verdict("generic_rubric", family, d, [call])
    return _safe("generic_rubric", family, run)


# ---------------------------------------------------------------------------
def computed_facts(text: str, eo: EvidenceObject) -> list:
    tx = W.execute_text(W.clean(text), eo)
    out = []
    for c in tx.claims:
        if c.status == "nonmaterial":
            continue
        out.append({"claim": c.text, "computed": [
            {"operation": o.op, "evidence_ids": o.evidence_ids, "asserted": o.asserted,
             "result_from_evidence": o.recomputed, "code_finding": o.status, "note": o.detail}
            for o in c.ops]})
    for o in tx.text_ops:
        out.append({"whole_text": True, "operation": o.op, "evidence_ids": o.evidence_ids,
                    "code_finding": o.status, "note": o.detail})
    return out


FACTS_NOTE = ("\nYou also receive 'computed_results': for each sentence, the values, directions and rankings "
              "that a program recomputed from the evidence, with its finding (support / contradict / unresolved). "
              "The program reads text with simple rules, so its binding of words to evidence can be wrong or "
              "incomplete; use the results as aids and decide yourself.")


def facts_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "",
                claim_level: bool = False, **_) -> CVerdict:
    def run():
        msgs = [{"role": "system", "content": _sys(RUBRIC, claim_level) + FACTS_NOTE + "\n" + VERDICT},
                {"role": "user", "content": _payload(cand, eo, computed_results=computed_facts(cand.text, eo))}]
        d, call, _raw = _llm(cand, family, msgs, schema=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
                             verdict_key="accepted", max_tokens=2500, experiment=experiment, role=f"b1f[{family}]")
        if d is None:
            return CVerdict("facts_judge", None, family, error=f"unparseable: {call['parse']}", calls=[call])
        return _verdict("facts_judge", family, d, [call])
    return _safe("facts_judge", family, run)


# ---------------------------------------------------------------------------
FORCED = TOOL_SPEC.replace("you MAY request tool results ONCE", "you MUST first request tool results (one round)") \
    .replace("You may also reply with the verdict JSON immediately.",
             "You must check every number with compare_value, every stated direction with direction and every "
             "'strongest/main driver' statement with strongest before deciding. A verdict without tool use is "
             "not accepted.")
INSIST = ("You replied without using the tools. Request the tool results first: reply with JSON only "
          '{"tool_calls":[...]} checking each number, direction and ranking claim in the explanation.')


def forced_tool_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "",
                      max_calls: int = 3, claim_level: bool = False, **_) -> CVerdict:
    msgs = [{"role": "system", "content": _sys(RUBRIC, claim_level) + "\n" + VERDICT + FORCED},
            {"role": "user", "content": _payload(cand, eo)}]
    item = f"{cand.borrower_id}|{cand.arm}|{cand.family}"

    def run():
        calls, used, insisted = [], [], False
        gw = G.GATEWAY
        for k in range(max_calls):
            kw = dict(temperature=0.0, max_tokens=2500, want_json=True, experiment=experiment,
                      item_id=item + (f"|t{k}" if k else ""), role=f"b2f[{family}]")
            r = (gw.judged_by(family, cand.family, msgs, **kw) if cand.family not in ("template", "")
                 else gw.complete(family, msgs, **kw))
            calls.append({"family": family, "model_id": r.model_id, "provider": r.provider,
                          "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                          "latency_ms": r.latency_ms, "finish_reason": r.finish_reason})
            tc = _tool_calls(r.text)
            if tc is not None and k < max_calls - 1:          # up to two tool rounds; the last call is the verdict
                res = [run_tool(eo, c) for c in tc]
                used += [{"call": c, "result": x} for c, x in zip(tc, res)]
                msgs.extend([{"role": "assistant", "content": r.text},
                             {"role": "user", "content": "Tool results: " + json.dumps(res, default=str)[:8000]
                              + (" You may request one more tool round, or reply with the verdict JSON."
                                 if k < max_calls - 2 else " Now reply with the verdict JSON only.")}])
                continue
            if not used and not insisted and k < max_calls - 2:
                insisted = True
                msgs.extend([{"role": "assistant", "content": r.text}, {"role": "user", "content": INSIST}])
                continue
            d, info = JP.parse_with_parser_judge(r.text, schema_hint=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
                                                 verdict_key="accepted", experiment=experiment, item_id=item,
                                                 role=f"b2f[{family}]", kind="comparator",
                                                 finish_reason=r.finish_reason)
            extra = {"tools": used, "n_tools": len(used), "insisted": insisted}
            if d is None:
                return CVerdict("forced_tool_judge", None, family, error=f"unparseable: {info.get('method')}",
                                detail=extra, calls=calls)
            return _verdict("forced_tool_judge", family, d, calls, extra)
        return CVerdict("forced_tool_judge", None, family, error="no verdict within the call budget",
                        detail={"tools": used}, calls=calls)
    return _safe("forced_tool_judge", family, run)


# ---------------------------------------------------------------------------
CLAIMWISE = (RUBRIC + "\nThe explanation has been split into numbered claims. For EVERY claim decide: supported "
             "(the evidence licenses it), contradicted (it breaks a rule) or insufficient (the evidence does not "
             "settle it). Also list omissions of the probability estimate or the strongest attribution.\n"
             'Reply with JSON only: {"claims":[{"claim_id":"c1","status":"supported|contradicted|insufficient",'
             '"rule":"R_...","quote":"<exact words>","reason":"..."}],"omissions":[{"evidence_ids":["..."],"reason":"..."}]}')
CW_SCHEMA = ('{"claims": [{"claim_id": "c1", "status": "supported|contradicted|insufficient", "rule": "", '
             '"quote": "", "reason": ""}], "omissions": []}')


def _valid_cw(o) -> bool:
    return isinstance(o, dict) and isinstance(o.get("claims"), list) and all(
        isinstance(c, dict) and str(c.get("status", "")).lower() in ("supported", "contradicted", "insufficient")
        for c in o["claims"])


def claimwise_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "",
                    claim_level: bool = False, **_) -> CVerdict:
    def run():
        claims = [{"claim_id": f"c{k + 1}", "text": t} for k, (_s, t) in enumerate(W.split_claims(W.clean(cand.text)))]
        msgs = [{"role": "system", "content": _sys(CLAIMWISE, claim_level)},
                {"role": "user", "content": _payload(cand, eo, claims=claims)}]
        d, call, _raw = _llm(cand, family, msgs, schema=CW_SCHEMA, validate=_valid_cw, verdict_key=None,
                             max_tokens=3000, experiment=experiment, role=f"b1c[{family}]")
        if d is None:
            return CVerdict("claimwise_judge", None, family, error=f"unparseable: {call['parse']}", calls=[call])
        st = {str(c.get("claim_id")): str(c.get("status")).lower() for c in d["claims"]}
        missing = [c["claim_id"] for c in claims if c["claim_id"] not in st]
        bad = [k for k, v in st.items() if v != "supported"]
        om = [] if claim_level else [o for o in (d.get("omissions") or []) if isinstance(o, dict)]
        acc = not bad and not om and not missing
        return CVerdict("claimwise_judge", acc, family, detail={"n_claims": len(claims), "not_supported": bad,
                                                                "omissions": om, "missing": missing}, calls=[call])
    return _safe("claimwise_judge", family, run)


# ---------------------------------------------------------------------------
FT_EXTRACT = (
    "You are given a credit-risk explanation and the ids/names of the available evidence items. Extract every "
    "verifiable claim the explanation makes (one atomic statement each) and, for each claim, the evidence ids "
    "needed to verify it (queries).\n"
    'Reply with JSON only: {"claims":[{"claim":"<atomic claim>","queries":["<evidence id>"]}]}')
FT_VERIFY = (RUBRIC + "\nYou receive extracted claims, each with the evidence retrieved for it. For EACH claim give "
             "reasoning, whether it contains an error, and factuality (true if the evidence supports it under the "
             "rules). Also report whether the probability estimate and the strongest attribution are conveyed.\n"
             'Reply with JSON only: {"claims":[{"claim":"...","reasoning":"...","error":"<rule or none>",'
             '"factuality":true|false}],"omissions":[{"evidence_ids":["..."],"reason":"..."}]}')


def _index(eo: EvidenceObject) -> list:
    out = []
    for it in eo.items:
        d = it.to_dict()
        out.append({"id": it.eid, "type": d.get("etype"), "name": d.get("source_field") or d.get("feature_group")
                    or ("probability of default" if it.calibrated_pd is not None else "")})
    return out


def factool_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "", **_) -> CVerdict:
    def run():
        m1 = [{"role": "system", "content": FT_EXTRACT},
              {"role": "user", "content": json.dumps({"evidence_index": _index(eo), "explanation": cand.text})}]
        d1, c1, _ = _llm(cand, family, m1, schema='{"claims": [{"claim": "", "queries": [""]}]}',
                         validate=lambda o: isinstance(o, dict) and isinstance(o.get("claims"), list),
                         verdict_key=None, max_tokens=2500, experiment=experiment, role=f"factool-extract[{family}]")
        if d1 is None:
            return CVerdict("factool_judge", None, family, error=f"extract unparseable: {c1['parse']}", calls=[c1])
        cl = []
        for c in d1["claims"]:
            if not isinstance(c, dict):
                continue
            ev = [run_tool(eo, {"tool": "lookup", "args": {"evidence_id": q}})["result"]
                  for q in (c.get("queries") or [])[:6]]
            cl.append({"claim": str(c.get("claim", ""))[:400], "retrieved_evidence": ev})
        pred = [it.to_dict() for it in eo.items if it.calibrated_pd is not None]
        m2 = [{"role": "system", "content": FT_VERIFY},
              {"role": "user", "content": json.dumps({"explanation": cand.text, "claims": cl,
                                                      "prediction": pred,
                                                      "attributions_ranked": run_tool(eo, {"tool": "strongest",
                                                                                         "args": {}})["result"]},
                                                     default=str)[:16000]}]
        d2, c2, _ = _llm(cand, family, m2, schema='{"claims": [{"claim": "", "reasoning": "", "error": "", '
                                                   '"factuality": true}], "omissions": []}',
                         validate=lambda o: isinstance(o, dict) and isinstance(o.get("claims"), list),
                         verdict_key=None, max_tokens=3000, experiment=experiment, role=f"factool-verify[{family}]")
        if d2 is None:
            return CVerdict("factool_judge", None, family, error=f"verify unparseable: {c2['parse']}", calls=[c1, c2])
        nonf = [c for c in d2["claims"] if isinstance(c, dict) and c.get("factuality") is False]
        om = [o for o in (d2.get("omissions") or []) if isinstance(o, dict)]
        # FacTool decides on claim factuality only; it has no coverage notion, so omissions are logged, not used
        return CVerdict("factool_judge", not nonf, family,
                        detail={"n_claims": len(cl), "non_factual": nonf[:6], "omissions": om}, calls=[c1, c2])
    return _safe("factool_judge", family, run)


METHODS2 = {"generic_rubric": generic_rubric, "facts_judge": facts_judge, "forced_tool_judge": forced_tool_judge,
            "claimwise_judge": claimwise_judge, "factool_judge": factool_judge}


# ---------------------------------------------------------------------------
def parse_exec(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "",
               claim_level: bool = False, **_) -> CVerdict:
    """Structured parser + the same executor, WITHOUT a judge or dispute machinery (Improvement_Plan WS2.3).
    The reader model (``family``) parses EVERY claim into propositions with its own binding; code executes them.
    Any executed contradiction -> reject; every proposition supported (non-evidential framing allowed) and the
    probability estimate and strongest driver conveyed (code coverage check) -> release; otherwise defer."""
    import dispute as DP

    def run():
        text = W.clean(cand.text)
        tx = W.execute_text(text, eo)
        claims = [c for c in tx.claims if c.status != "nonmaterial"]
        d, call, raw, how = DP.reinterpret([(c.claim_id, c.text) for c in claims], [], eo, family,
                                           experiment=experiment, item=f"{cand.borrower_id}|parse_exec", version="1.1")
        if d is None:
            return CVerdict("parse_exec", None, family, error=f"unparseable: {how}", calls=[call])
        by = {str(p.get("pid")): p for p in d.get("passages", []) if isinstance(p, dict)}
        st, detail = [], {}
        for c in claims:
            props = [p for p in (by.get(c.claim_id) or {}).get("propositions", []) if isinstance(p, dict)]
            ex = [DP.execute_proposition(p, eo, c.text)[0] for p in props]
            v = ("contradict" if "contradict" in ex else
                 "support" if props and all(x in ("support", "none") for x in ex) else "residual")
            st.append(v)
            detail[c.claim_id] = v
        cov_ok = claim_level or all(o.status == W.SUPPORT for o in tx.text_ops if o.op == "COVERAGE")
        acc = False if "contradict" in st else (True if st and all(v == "support" for v in st) and cov_ok else None)
        return CVerdict("parse_exec", acc, family, detail={"claims": detail, "coverage_ok": cov_ok}, calls=[call],
                        error=None if acc is not None else "unresolved")
    return _safe("parse_exec", family, run)


METHODS2["parse_exec"] = parse_exec

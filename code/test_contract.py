"""
Unit tests for contract.py, contract_verifier.py and comparators.py.
No API calls: every LLM response comes from fakes.FakeGateway.

    python test_contract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators as K  # noqa: E402
import contract as C  # noqa: E402
import fakes  # noqa: E402
from contract_verifier import verify_contract  # noqa: E402
from evidence import EvidenceObject, EvidenceType, new_item  # noqa: E402
from generate import Candidate  # noqa: E402

RESULTS = []


def check(name, cond, info=""):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name:58} {info}")


def make_eo() -> EvidenceObject:
    eo = EvidenceObject(borrower_id="t-1", dataset="synthetic")
    eo.add(new_item("f1", EvidenceType.OBSERVED_FACT, source_field="CreditScore", value=530, unit="points",
                    aliases=["credit score", "bureau score"]))
    eo.add(new_item("f2", EvidenceType.OBSERVED_FACT, source_field="DSCR", value=0.54, unit="ratio",
                    aliases=["debt service coverage", "dscr"]))
    eo.add(new_item("f3", EvidenceType.OBSERVED_FACT, source_field="LoanAmount", value=455000, unit="INR",
                    aliases=["loan amount"]))
    eo.add(new_item("f4", EvidenceType.OBSERVED_FACT, source_field="PastDelays", value=0,
                    value_label="code 0 (not defined by the source)", aliases=["past delays"]))
    eo.add(new_item("a1", EvidenceType.MODEL_ATTRIBUTION, feature_group="CreditScore", contribution=0.90,
                    direction="increases_risk", components=["f1"], explained_scale="raw log-odds margin"))
    eo.add(new_item("a2", EvidenceType.MODEL_ATTRIBUTION, feature_group="DSCR", contribution=0.55,
                    direction="increases_risk", components=["f2"], explained_scale="raw log-odds margin"))
    eo.add(new_item("a3", EvidenceType.MODEL_ATTRIBUTION, feature_group="PastDelays", contribution=-0.20,
                    direction="decreases_risk", components=["f4"], explained_scale="raw log-odds margin"))
    eo.add(new_item("p1", EvidenceType.MODEL_PREDICTION, calibrated_pd=0.62, target_definition="default",
                    horizon="12 months"))
    return eo


GOOD = ("The model estimates a 62% probability of default. A credit score of 530 raises the model's risk score, "
        "and a DSCR of 0.54 also raises it. Past delays lower the risk score. This describes the model, "
        "not a verified cause of default.")


def cand(text, fam="nova_lite"):
    return Candidate(borrower_id="t-1", arm="grounded", text=text, family=fam)


def main() -> int:
    eo = make_eo()
    # ---------------- evidence validation
    check("valid evidence has no problems", C.validate_evidence(eo) == [])
    bad = make_eo(); bad.items = [i for i in bad.items if i.eid != "p1"]
    check("missing prediction is flagged", any("prediction" in p for p in C.validate_evidence(bad)))

    # ---------------- objection checks
    o = C.check_objection(C.Objection("R_CAUSAL", "This describes the model, not a verified cause of default",
                                      [], "unsupported causal claim"), GOOD, eo)
    check("disclaimer flagged as causal -> invalid", o.valid is False, o.problems)
    o = C.check_objection(C.Objection("R_DIRECTION", "lowers default risk sharply", ["a1"], "x"), GOOD, eo)
    check("quote not in text -> invalid", o.valid is False and "quote is not in the explanation" in o.problems)
    o = C.check_objection(C.Objection("R_DIRECTION", "A credit score of 530 raises the model's risk score",
                                      ["a9"], "wrong"), GOOD, eo)
    check("unknown evidence id -> invalid", o.valid is False)
    t_rev = GOOD.replace("Past delays lower the risk score", "Past delays raise the risk score")
    o = C.check_objection(C.Objection("R_DIRECTION", "Past delays raise the risk score", ["a3"],
                                      "a3 decreases risk"), t_rev, eo)
    check("real sign reversal objection -> valid", o.valid is True, o.problems)
    o = C.check_objection(C.Objection("R_DIRECTION", "Past delays raise the risk score", ["a3"],
                                      "this is correct, but"), t_rev, eo)
    check("self-contradicting rationale -> invalid", o.valid is False)
    o = C.check_objection(C.Objection("R_OMISSION", "", ["a1"], "credit score omitted"), GOOD, eo)
    check("omission objection for a mentioned driver -> invalid", o.valid is False)
    t_scale = GOOD + " The credit score adds 90 percentage points to the probability of default."
    o = C.check_objection(C.Objection("R_SCALE", "adds 90 percentage points to the probability of default",
                                      ["a1"], "contribution is log-odds"), t_scale, eo)
    check("scale objection on a pp claim -> valid", o.valid is True, o.problems)
    o = C.check_objection(C.Objection("R_TEMPORAL", "lower the risk score", [], "trend"), GOOD, eo)
    check("temporal objection without a trend word -> invalid", o.valid is False)
    o = C.check_objection(C.Objection("R_NOT_A_RULE", "x", [], ""), GOOD, eo)
    check("unknown rule -> invalid", o.valid is False)

    # ---------------- contract detectors
    check("clean text: no contract violations", C.contract_violations(GOOD, eo) == [],
          C.contract_violations(GOOD, eo))
    t = GOOD + " The DSCR is the strongest driver of this estimate."
    v = C.contract_violations(t, eo)
    check("wrong strongest driver -> R_RANKING", [x["rule"] for x in v] == ["R_RANKING"], v)
    t = GOOD + " The credit score is the strongest driver."
    check("right strongest driver -> no violation", C.contract_violations(t, eo) == [])
    t = GOOD + " DSCR is one of the strongest factors."
    check("'one of the strongest' is not a ranking claim", C.contract_violations(t, eo) == [])
    t = GOOD + " The loan amount of 455,000 also increases the risk score."
    v = C.contract_violations(t, eo)
    check("fact-only field given a direction -> R_FACT_AS_DRIVER", any(x["rule"] == "R_FACT_AS_DRIVER" for x in v), v)
    t = GOOD + " Past delays of code 0 mean the borrower paid on time."
    v = C.contract_violations(t, eo)
    check("meaning for undocumented code -> R_CODE_MEANING", any(x["rule"] == "R_CODE_MEANING" for x in v), v)
    t = GOOD + " Past delays of code 0 have no documented meaning, so nothing is said about paying on time."
    check("hedged code statement -> no violation", not any(x["rule"] == "R_CODE_MEANING"
                                                           for x in C.contract_violations(t, eo)))
    t = "The model estimates 62%. A credit score of 530 raises risk; no other factors matter."
    v = C.contract_violations(t, eo)
    check("completeness claim with unmentioned drivers -> R_COMPLETENESS", any(x["rule"] == "R_COMPLETENESS" for x in v), v)

    # ---------------- claims splitter
    cl = C.claims_of(GOOD)
    check("claims_of splits sentences", len(cl) == 4, [c["text"][:30] for c in cl])

    # ---------------- contract verifier with a fake judge
    def resp(payload):
        return lambda fam, msgs: json.dumps(payload)
    allsup = {"claims": [{"claim_id": f"c{k}", "status": "supported"} for k in range(1, 5)],
              "omissions": [], "findings": []}
    with fakes.patched(fakes.FakeGateway(resp(allsup))):
        r = verify_contract(cand(GOOD), eo, "qwen3_next")
    check("all supported -> accept", r.decision == "accept", r.reason)

    wrongdir = {"claims": [{"claim_id": "c3", "status": "contradicted", "rule": "R_DIRECTION",
                            "quote": "Past delays raise the risk score", "evidence_ids": ["a3"],
                            "reason": "a3 decreases risk"}] +
                          [{"claim_id": f"c{k}", "status": "supported"} for k in (1, 2, 4)]}
    with fakes.patched(fakes.FakeGateway(resp(wrongdir))):
        r = verify_contract(cand(t_rev), eo, "qwen3_next")
    check("valid objection -> reject", r.decision == "reject", r.reason)

    badobj = {"claims": [{"claim_id": "c4", "status": "contradicted", "rule": "R_CAUSAL",
                          "quote": "This describes the model, not a verified cause of default",
                          "evidence_ids": [], "reason": "causal claim"}] +
                        [{"claim_id": f"c{k}", "status": "supported"} for k in (1, 2, 3)]}
    with fakes.patched(fakes.FakeGateway(resp(badobj))):
        r = verify_contract(cand(GOOD), eo, "qwen3_next")
    check("only an invalid objection, no reassessment -> unresolved", r.decision == "unresolved", r.reason)
    seq = iter([json.dumps(badobj), json.dumps(allsup)])
    with fakes.patched(fakes.FakeGateway(lambda f, m: next(seq))):
        r = verify_contract(cand(GOOD), eo, "qwen3_next", reassess_family="glm47")
    check("invalid objection, reassessment supports -> accept", r.decision == "accept" and r.reassessed, r.reason)

    with fakes.patched(fakes.FakeGateway(resp(allsup))):
        r = verify_contract(cand(GOOD + " The DSCR is the strongest driver."), eo, "qwen3_next")
    check("contract detector gates before any LLM call", r.decision == "reject" and not r.calls, r.reason)

    with fakes.patched(fakes.FakeGateway(lambda f, m: '{"claims": [{"claim_id": "c1", "status": "suppor',
                                         finish_reason="length")):
        r = verify_contract(cand(GOOD), eo, "qwen3_next")
    check("truncated judge reply -> unresolved", r.decision == "unresolved", r.reason)

    with fakes.patched(fakes.FakeGateway(resp(allsup))):
        r = verify_contract(cand(GOOD, fam="qwen3_next"), eo, "qwen3_next")
    check("same-publisher judge refused -> unresolved", r.decision == "unresolved" and "config" in r.reason)

    # ---------------- comparators
    check("always_accept / always_reject", K.always_accept(cand(GOOD), eo).accepted is True
          and K.always_reject(cand(GOOD), eo).accepted is False)
    check("PT audit: clean text passes", K.pt_audit(cand(GOOD), eo).accepted is True,
          K.pt_audit(cand(GOOD), eo).detail)
    v = K.pt_audit(cand(t_rev), eo)
    check("PT audit: direction failure", v.accepted is False and ("direction", "PastDelays") in v.detail["failures"], v.detail)
    v = K.pt_audit(cand("The model estimates 62%. A DSCR of 0.54 raises risk."), eo)
    check("PT audit: top driver not covered", v.accepted is False and v.detail["failures"][-1][0] == "coverage")

    with fakes.patched(fakes.FakeGateway(resp({"TFC": 5, "TDC": 3, "rationale": "x"}))):
        v = K.zandi_judge(cand(GOOD), eo, "qwen3_next")
    check("Zandi: TDC below threshold -> reject", v.accepted is False and v.detail["TDC"] == 3)
    with fakes.patched(fakes.FakeGateway(resp({"confidence": 0.8, "verdict": "reject", "justification": "y",
                                               "error type": "NegateRelation"}))):
        v = K.mermigkis_verifier(cand(GOOD), eo, "qwen3_next")
    check("Mermigkis: reject with confidence -> score 0.2", v.accepted is False and abs(v.score - 0.2) < 1e-9)

    lock = {"notes": [{"criterion": "b", "note": "sign", "quotes": ["Past delays raise the risk score"]}]}
    verdict_ok = {"accepted": False, "violations": [{"type": "sign_reversal", "detail": "a3",
                                                     "quote": "Past delays raise the risk score"}]}
    verdict_unlocked = {"accepted": False, "violations": [{"type": "sign_reversal", "detail": "a3",
                                                           "quote": "something never noted"}]}
    seq = iter([json.dumps(lock), json.dumps(verdict_ok)])
    with fakes.patched(fakes.FakeGateway(lambda f, m: next(seq))):
        v = K.pbp_judge(cand(t_rev), eo, "qwen3_next")
    check("PbP: violation grounded in locked notes -> reject", v.accepted is False and not v.error, v.error)
    seq = iter([json.dumps(lock), json.dumps(verdict_unlocked)])
    with fakes.patched(fakes.FakeGateway(lambda f, m: next(seq))):
        v = K.pbp_judge(cand(t_rev), eo, "qwen3_next")
    check("PbP: violation outside locked notes -> unresolved", v.accepted is None and "locked" in (v.error or ""))

    import random
    rng = random.Random(1)
    stream = []
    for _ in range(2000):
        h = rng.random() < 0.8
        w = min(1, max(0, (0.75 if h else 0.35) + rng.gauss(0, 0.15)))
        stream.append((w, h))
    res = K.run_weak_strong(stream, alpha=0.1, beta=0.1, seed=3)
    check("weak-strong: thresholds ordered, some deferral", res["final_tau_a"] > res["final_tau_r"]
          and 0 < res["strong_rate"] < 1, res)
    check("weak-strong: false-accept near target 0.10", res["type1_false_accept"] <= 0.2, res["type1_false_accept"])

    items = ([{"group": "A", "loss": False, "s": True}] * 200 + [{"group": "A", "loss": True, "s": True}] * 2 +
             [{"group": "B", "loss": False, "s": True}] * 40 + [{"group": "B", "loss": True, "s": True}] * 6)
    pols = {"release_all": lambda it: True, "never": lambda it: False}
    pooled = K.select_release_policy(items, pols, eps=0.10, group_audited=False)
    grp = K.select_release_policy(items, pols, eps=0.10, group_audited=True)
    check("pooled bound passes where group-audited fails", pooled["chosen"] == "release_all" and grp["chosen"] is None,
          (pooled["chosen"], grp["chosen"]))

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

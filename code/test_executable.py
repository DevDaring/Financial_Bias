"""
Unit tests for witness.py (executable operations and witnesses), executable_verifier.py
(the proposed method and its ablations), the B2 tool baseline and the policy count fixes.
No API calls: every LLM response comes from fakes.FakeGateway.

    python test_executable.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators as K  # noqa: E402
import executable_verifier as EV  # noqa: E402
import fakes  # noqa: E402
import policy as PL  # noqa: E402
import witness as W  # noqa: E402
from evidence import EvidenceObject, EvidenceType, new_item  # noqa: E402
from generate import Candidate  # noqa: E402

RESULTS = []


def check(name, cond, info=""):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name:66} {info}")


def make_eo() -> EvidenceObject:
    """The Future_Plan §D1 worked example, extended: PD 0.12; balance -0.30 raw log-odds;
    utilization the strongest displayed driver; a fact-only field; an undocumented code;
    two dated bill observations."""
    eo = EvidenceObject(borrower_id="t-1", dataset="synthetic")
    eo.add(new_item("f1", EvidenceType.OBSERVED_FACT, source_field="Utilization", value=0.91, unit="ratio",
                    aliases=["utilization", "credit utilization"]))
    eo.add(new_item("f2", EvidenceType.OBSERVED_FACT, source_field="Balance", value=1000.0, unit="USD",
                    aliases=["balance", "account balance"]))
    eo.add(new_item("f3", EvidenceType.OBSERVED_FACT, source_field="Age", value=41, unit="years", aliases=["age"]))
    eo.add(new_item("f4", EvidenceType.OBSERVED_FACT, source_field="StatusCode", value=0.0,
                    value_label="code 0 (not defined by the source)", aliases=["status code", "repayment status"]))
    eo.add(new_item("f5", EvidenceType.OBSERVED_FACT, source_field="BILL_JUL", value=500.0, unit="USD",
                    observed_at="in July 2005", series="bill_amount", period_order=1,
                    aliases=["July bill"]))
    eo.add(new_item("f6", EvidenceType.OBSERVED_FACT, source_field="BILL_SEP", value=800.0, unit="USD",
                    observed_at="in September 2005", series="bill_amount", period_order=0,
                    aliases=["September bill"]))
    eo.add(new_item("a1", EvidenceType.MODEL_ATTRIBUTION, feature_group="Utilization", contribution=0.62,
                    direction="increases_risk", components=["f1"], explained_scale="raw log-odds margin",
                    aliases=["utilization", "credit utilization"]))
    eo.add(new_item("a2", EvidenceType.MODEL_ATTRIBUTION, feature_group="Balance", contribution=-0.30,
                    direction="decreases_risk", components=["f2"], explained_scale="raw log-odds margin",
                    aliases=["balance", "account balance"]))
    eo.add(new_item("a3", EvidenceType.MODEL_ATTRIBUTION, feature_group="BILL_SEP", contribution=0.20,
                    direction="increases_risk", components=["f6"], explained_scale="raw log-odds margin",
                    aliases=["September bill"]))
    eo.add(new_item("a4", EvidenceType.MODEL_ATTRIBUTION, feature_group="StatusCode", contribution=-0.10,
                    direction="decreases_risk", components=["f4"], explained_scale="raw log-odds margin",
                    aliases=["status code", "repayment status"]))
    eo.add(new_item("p1", EvidenceType.MODEL_PREDICTION, calibrated_pd=0.12, target_definition="default",
                    horizon="next month; not 12 months", aliases=["probability of default"]))
    return eo


EO = make_eo()


def status(text):
    tx = W.execute_text(text, EO)
    return tx.claims[0].status, tx


def ops(text):
    return [(o.op, o.status) for o in W.execute_text(text, EO).claims[0].ops]


# ---------------------------------------------------------------------------
def test_worked_example():
    s, _ = status("Balance reduces the model score relative to its reference; this does not establish a causal effect.")
    check("D1 valid: balance lowers the score + disclaimer -> support", s == W.SUPPORT, s)
    s, tx = status("Balance reduces PD by 30 percentage points.")
    check("D1 scale: -0.30 log-odds as 30 pp -> contradiction", s == W.CONTRA, ops("Balance reduces PD by 30 percentage points."))
    s, _ = status("Balance is the strongest displayed driver.")
    check("D1 rank: balance called strongest -> contradiction", s == W.CONTRA)
    s, _ = status("Utilization is the strongest driver of the score.")
    check("rank: true strongest -> support", s == W.SUPPORT)


def test_value():
    check("value: balance of 1,000 USD -> support", status("The balance was 1,000 USD.")[0] == W.SUPPORT)
    check("value: balance of 1,600 USD -> contradiction", status("The balance was 1,600 USD.")[0] == W.CONTRA)
    check("value: rounded 0.9 utilization -> support", status("Utilization is about 0.91.")[0] == W.SUPPORT)
    check("value: PD 12% -> support", status("The probability of default is 12%.")[0] == W.SUPPORT)
    check("value: PD 21% -> contradiction", status("The probability of default is 21%.")[0] == W.CONTRA)
    s, _ = status("The applicant, aged 41, has a balance that lowers the score.")
    check("value: a number of ANOTHER named field is not a contradiction", s != W.CONTRA, s)
    s, _ = status("Borrower t-1 has a balance that lowers the score.")
    check("value: identifiers are not values", s != W.CONTRA, s)


def test_direction():
    check("direction: utilization raises the score -> support",
          status("Utilization raises the model's risk score.")[0] == W.SUPPORT)
    check("direction: utilization lowers the score -> contradiction",
          status("Utilization lowers the model's risk score.")[0] == W.CONTRA)
    check("direction: two-noun object 'pushes the risk score down' read (1.1)",
          status("Utilization pushes the model's risk score down.")[0] == W.CONTRA)
    check("direction: negated phrase alone -> unresolved",
          status("Utilization does not lower the model's risk score.")[0] == W.UNRES)
    s, _ = status("Utilization raises the risk score, as does the balance.")
    check("direction: ellipsis 'as does' inherits direction -> contradiction", s == W.CONTRA, s)
    s, _ = status("The balance has no effect on the risk score.")
    check("direction: 'no effect' for -0.30 -> contradiction", s == W.CONTRA, s)
    s, _ = status("Age raises the model's risk score.")
    check("fact-as-driver: age (no attribution) given a direction -> contradiction", s == W.CONTRA,
          ops("Age raises the model's risk score."))
    s, _ = status("The applicant's employment history raises the risk score.")
    check("unbound factor with a direction -> unresolved, residual R_INVENTED",
          s == W.UNRES and "R_INVENTED" in W.execute_text("The applicant's employment history raises the risk score.",
                                                          EO).claims[0].residual)


def test_rank_scope():
    s, _ = status("The strongest risk-reducing factor is balance.")
    check("rank: restricted scope (risk-reducing) -> support", s == W.SUPPORT, s)
    s, _ = status("The strongest driver lowering risk is balance.")
    check("rank: participle scope ambiguous and readings disagree -> unresolved", s == W.UNRES, s)
    s, _ = status("Balance is the second largest driver.")
    check("rank: beyond first place -> unresolved", s == W.UNRES, s)


def test_scale_temporal_code_target():
    check("scale: labelled log-odds number -> support",
          status("Balance contributes -0.30 on the raw log-odds scale.")[0] == W.SUPPORT)
    s, _ = status("Balance adds -0.30 to the raw log-odds score, which is not a 30-point change in the probability.")
    check("scale: negated conversion is not a scale error", s != W.CONTRA, s)
    check("temporal: bill rose July->September -> support",
          status("The bill amount rose from the July bill to the September bill.")[0] in (W.SUPPORT, W.UNRES))
    s, _ = status("The balance has been rising over recent months.")
    check("temporal: change asserted on a one-observation field -> contradiction", s == W.CONTRA, s)
    s, _ = status("The status code is 0, which means the borrower paid duly.")
    check("code: meaning stated for undocumented code -> contradiction", s == W.CONTRA, s)
    s, _ = status("The status code is 0, a code the source does not document.")
    check("code: undocumented acknowledged -> not a contradiction", s != W.CONTRA, s)
    s, _ = status("The probability of default is 12% within the next 12 months.")
    check("target: 12-month horizon vs 'not 12 months' -> contradiction", s == W.CONTRA, s)
    s, _ = status("The probability of default is 12% for next month, not to a one-year horizon.")
    check("target: denied horizon several words after 'not' -> not a contradiction (1.2)", s != W.CONTRA, s)
    s, _ = status("The probability of default is 12%, with a payment delay of 2 months recorded.")
    check("target: a measured duration is not a horizon", s != W.CONTRA, s)


def test_text_level():
    t = "The probability of default is 12%. Balance lowers the model's risk score."
    tx = W.execute_text(t, EO)
    cov = [o for o in tx.text_ops if o.op == "COVERAGE"]
    check("coverage: strongest driver missing -> contradiction",
          any(o.status == W.CONTRA and "strongest" in o.detail for o in cov))
    t2 = "The probability of default is 12%. Utilization raises the model's risk score. No other factors matter."
    tx2 = W.execute_text(t2, EO)
    check("completeness: 'no other factors' with 3 unmentioned -> contradiction",
          any(o.op == "COMPLETENESS" and o.status == W.CONTRA for o in tx2.text_ops))


# ---------------------------------------------------------------------------
def wit(rule, quote, ids, claimed=None):
    return W.execute_witness(W.Witness(rule=rule, quote=quote, evidence_ids=ids, claimed_result=claimed),
                             TEXT, EO)


TEXT = ("The probability of default is 12% for next month. Utilization raises the model's risk score. "
        "The balance was 1,000 USD and it lowers the model's risk score. "
        "This does not mean that the balance causes default.")


def test_witnesses():
    check("witness: true value but objection claims mismatch -> disproved",
          wit("R_VALUE", "The balance was 1,000 USD", ["f2"]).outcome == "disproved")
    w = wit("R_VALUE", "The balance was 1,000 USD", ["f2"], claimed="5000")
    check("witness: false arithmetic is flagged", any("disagrees" in p for p in w.problems) and not w.blocks, w.problems)
    check("witness: cites another field -> malformed",
          wit("R_VALUE", "The balance was 1,000 USD", ["f3"]).outcome == "malformed")
    check("witness: quote not in text -> malformed",
          wit("R_VALUE", "The balance was 9,000 USD", ["f2"]).outcome == "malformed")
    check("witness: unknown evidence id -> malformed",
          wit("R_DIRECTION", "Utilization raises the model's risk score", ["zz"]).outcome == "malformed")
    check("witness: direction objection on a correct direction -> disproved",
          wit("R_DIRECTION", "Utilization raises the model's risk score", ["a1"]).outcome == "disproved")
    w = wit("R_CAUSAL", "the balance causes default", ["a2"])
    check("witness: partial quote from a disclaimer -> semantic_invalid", w.outcome == "semantic_invalid", w.problems)
    bad = "Utilization lowers the model's risk score."
    w = W.execute_witness(W.Witness(rule="R_DIRECTION", quote=bad, evidence_ids=["a1"]), bad, EO)
    check("witness: real sign reversal -> verified", w.outcome == "verified")
    w = W.execute_witness(W.Witness(rule="R_TARGET", quote="12% within the next 12 months", evidence_ids=["p1"]),
                          "The probability of default is 12% within the next 12 months.", EO)
    check("witness: TARGET is executed (1.1), wrong horizon -> verified", w.outcome == "verified", w.problems)
    c = "Increasing utilization would make default less likely."
    w = W.execute_witness(W.Witness(rule="R_CAUSAL", quote=c, evidence_ids=["a1"]), c, EO)
    check("witness: action-effect wording is a causal assertion (1.2)", w.outcome == "semantic_plausible", w.problems)


# ---------------------------------------------------------------------------
def cand(text, fam="nova_lite"):
    return Candidate(borrower_id="t-1", arm="grounded", text=text, claims=[], family=fam)


GOOD = ("The probability of default is 12% for next month. Utilization raises the model's risk score. "
        "Balance lowers the model's risk score. These are model associations, not causes.")
RESIDUAL = GOOD + " The applicant's employment history raises the risk score."


def responder_factory(kind):
    def r(family, messages):
        last = str(messages[-1]["content"])
        ids = sorted(set(__import__("re").findall(r'"claim_id": "(c\d+)"', last)))
        if kind == "support":
            return json.dumps({"claims": [{"claim_id": c, "status": "supported"} for c in ids], "omissions": []})
        if kind == "false_objection":      # objects to a correct direction: code must disprove it
            return json.dumps({"claims": [{"claim_id": "c2", "status": "contradicted", "witness": {
                "rule": "R_DIRECTION", "operation": "DIRECTION", "quote": "Utilization raises the model's risk score",
                "evidence_ids": ["a1"], "claimed_result": "decreases_risk"}}] +
                [{"claim_id": c, "status": "supported"} for c in ids if c != "c2"], "omissions": []})
        if kind == "invented":             # a valid semantic objection to the residual claim
            return json.dumps({"claims": [{"claim_id": c, "status": "supported"} for c in ids if c != "c5"] +
                              [{"claim_id": "c5", "status": "contradicted", "witness": {
                                  "rule": "R_INVENTED", "operation": "SEMANTIC",
                                  "quote": "employment history raises the risk score", "evidence_ids": []}}],
                              "omissions": []})
        return "{}"
    return r


def test_verifier():
    with fakes.patched(fakes.FakeGateway(responder_factory("support"))) as fk:
        v = EV.verify_executable(cand(GOOD), EO, "qwen3_next", mode="P")
        check("P: fully executed correct text -> accept with NO LLM call", v.decision == "accept" and not fk.calls,
              v.reason)
        v = EV.verify_executable(cand(GOOD.replace("Balance lowers", "Balance raises")), EO, "qwen3_next", mode="P")
        check("P: executed contradiction -> reject with NO LLM call", v.decision == "reject" and not fk.calls, v.reason)
        v = EV.verify_executable(cand(RESIDUAL), EO, "qwen3_next", mode="P")
        check("P: residual claim -> one LLM call; supported -> accept",
              v.decision == "accept" and len(fk.calls) == 1, (v.decision, v.reason, len(fk.calls)))
        v = EV.verify_executable(cand(RESIDUAL), EO, None, mode="A0")
        check("A0: residual claim, code only -> unresolved", v.decision == "unresolved")
    with fakes.patched(fakes.FakeGateway(responder_factory("false_objection"))):
        v = EV.verify_executable(cand(RESIDUAL), EO, "qwen3_next", mode="P")
        check("P: false objection to a code-supported claim is disproved -> accept", v.decision == "accept",
              (v.decision, v.reason, [w["outcome"] for w in v.witnesses]))
        v = EV.verify_executable(cand(RESIDUAL), EO, "qwen3_next", mode="A1")
        check("A1: same false objection blocks (no objection checking) -> reject", v.decision == "reject")
    with fakes.patched(fakes.FakeGateway(responder_factory("invented"))):
        v = EV.verify_executable(cand(RESIDUAL), EO, "qwen3_next", mode="P")
        check("P: valid semantic objection on the residual claim -> reject", v.decision == "reject",
              (v.reason, [w["outcome"] for w in v.witnesses]))
    with fakes.patched(fakes.FakeGateway(responder_factory("false_objection"))) as fk:
        v = EV.verify_executable(cand(RESIDUAL), EO, "qwen3_next", mode="D")
        check("D: the same false objection is a code-judge dispute -> defer (never release)",
              v.decision == "unresolved" and v.reason == "code_judge_dispute", (v.decision, v.reason))
    def reader(family, messages):
        sys_ = str(messages[0].get("content", ""))
        if sys_.startswith("You read sentences"):
            import re as _re
            pids = _re.findall(r'"pid": "([^"]+)"', str(messages[-1].get("content", "")))
            return json.dumps({"passages": [{"pid": p, "propositions": []} for p in pids], "coverage": []})
        return responder_factory("false_objection")(family, messages)
    with fakes.patched(fakes.FakeGateway(reader)):
        v = EV.verify_executable(cand(RESIDUAL), EO, "qwen3_next", mode="R", dispute_family="gptoss120b")
        check("R: a reader that asserts nothing cannot settle the dispute -> defer",
              v.decision == "unresolved" and v.reason == "dispute_unresolved", (v.decision, v.reason))
    import dispute as DP
    st, _ = DP.execute_proposition({"quote": "", "operation": "DIRECTION", "evidence_ids": [EO.by_type(
        EvidenceType.MODEL_ATTRIBUTION)[0].eid], "asserted": "increases" if (EO.by_type(
        EvidenceType.MODEL_ATTRIBUTION)[0].contribution or 0) < 0 else "decreases", "polarity": "asserted"}, EO, "")
    check("dispute reader: an independently bound wrong direction executes as a contradiction", st == "contradict", st)
    import verify as V
    n0 = V._signed_numbers("at 0 it")[0]
    check("1.5: a stated 0 is not a rounding of 0.157", not V._num_matches(n0, 0.157, mag_ok=True))
    check("1.5: a stated 0 still matches a recorded zero", V._num_matches(n0, 0.0, mag_ok=True))
    check("verifier modes registered", set(EV.MODES) == {"P", "A1", "A2", "PX", "A0", "D", "R", "R11"})


def test_tools_and_policy():
    r = K.run_tool(EO, {"tool": "compare_value", "args": {"evidence_id": "f2", "number": "1000"}})
    check("B2 tool compare_value: 1000 vs balance -> match", r["result"]["matches"] is True, r)
    r = K.run_tool(EO, {"tool": "strongest", "args": {}})
    check("B2 tool strongest -> a1", [x["evidence_id"] for x in r["result"]] == ["a1"], r)
    r = K.run_tool(EO, {"tool": "dated_values", "args": {"evidence_id": "a3"}})
    check("B2 tool dated_values: July then September", [x["evidence_id"] for x in r["result"]] == ["f5", "f6"], r)
    recs = [PL.Record("b1", "g", "x", "f", False, None, template_correct=None, template_informative=True),
            PL.Record("b2", "g", "x", "f", True, None, informative=True),
            PL.Record("b3", "g", "x", "f", True, False, informative=True),
            PL.Record("b4", "g", "x", "f", True, True, informative=True)]
    m = PL.evaluate(recs, PL.Policy("all", 0.0, False, False))
    g = m["g"]
    check("policy: unknown labels are not counted as useful", g.useful == 1 and g.useful_unknown == 2,
          (g.useful, g.useful_unknown))
    cert = PL.certify(m, policies_m=1, groups_g=1)
    per = cert["per_group"]["g"]
    check("policy: certify uses exact counts; unlabelled released counted as errors",
          per["n_released"] == 4 and per["k_violations"] == 3 and per["k_labelled_errors"] == 1, per)


if __name__ == "__main__":
    for fn in [test_worked_example, test_value, test_direction, test_rank_scope, test_scale_temporal_code_target,
               test_text_level, test_witnesses, test_verifier, test_tools_and_policy]:
        fn()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)

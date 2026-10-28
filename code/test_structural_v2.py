"""Regression tests for the claim-scoped structural checker (structural/2.0).

Each case encodes one failure found against the v1 adjudicated labels, or one
property the rubric (plan §5.2) requires. Run: python test_structural_v2.py
"""
from __future__ import annotations

import sys

from evidence import EvidenceObject, EvidenceType as ET, new_item
from generate import Candidate
from verify import structural_check


def eo_tw() -> EvidenceObject:
    eo = EvidenceObject(borrower_id="taiwan-7", dataset="taiwan",
                        prediction_time="after the September 2005 statement")
    eo.add(new_item("f1", ET.OBSERVED_FACT, source_field="PAY_0", value=-2.0,
                    unit="code", observed_at="in September 2005", series="repayment_status",
                    period_order=0, aliases=["most recent repayment status"], internal_name="PAY_0"))
    eo.add(new_item("f2", ET.OBSERVED_FACT, source_field="PAY_2", value=2.0,
                    unit="code", observed_at="in August 2005", series="repayment_status",
                    period_order=1, aliases=["august repayment status"], internal_name="PAY_1"))
    eo.add(new_item("f3", ET.OBSERVED_FACT, source_field="LIMIT_BAL", value=50000.0,
                    unit="NT dollars", aliases=["credit limit"], internal_name="LIMIT_BAL"))
    eo.add(new_item("f4", ET.OBSERVED_FACT, source_field="AGE", value=41.0, unit="years",
                    aliases=["age"], internal_name="AGE"))
    eo.add(new_item("a1", ET.MODEL_ATTRIBUTION, feature_group="PAY_2", contribution=0.61,
                    direction="increases_risk", components=["f2"],
                    aliases=["august repayment status"], internal_name="PAY_1"))
    eo.add(new_item("a2", ET.MODEL_ATTRIBUTION, feature_group="LIMIT_BAL", contribution=-0.33,
                    direction="decreases_risk", components=["f3"],
                    aliases=["credit limit"], internal_name="LIMIT_BAL"))
    eo.add(new_item("p1", ET.MODEL_PREDICTION, calibrated_pd=0.384,
                    target_definition="default payment", horizon="next month"))
    return eo


BASE = ("The model estimates a 38% probability of default. PAY_2 raises the risk score, "
        "while the credit limit of 50,000 lowers it.")

CASES = [
    # (name, text, should_accept, expected violation type or None)
    ("clean base", BASE, True, None),
    ("signed contribution kept", BASE + " LIMIT_BAL has a contribution of -0.33.", True, None),
    ("signed status value kept", BASE + " PAY_0 is -2.", True, None),
    ("sign of a value matters", BASE + " PAY_0 is 2.", False, "numeric_or_unit_error"),
    ("magnitude of contribution ok", BASE + " PAY_2 contributes 0.61 to the score.", True, None),
    ("rounding ok", "The estimated probability of default is about 38.4%. PAY_2 raises the "
                    "risk score and the credit limit lowers it.", True, None),
    ("value bound to wrong field", BASE + " The borrower's age is 50,000.", False, "numeric_or_unit_error"),
    ("respectively list", BASE + " PAY_0 and AGE are -2 and 41 respectively.", True, None),
    ("score language is not a trend", "The model estimates 38%. The risk score is increased by "
                                      "PAY_2 and decreased by the credit limit.", True, None),
    ("real trend flagged", BASE + " Income has declined recently.", False, "unsupported_trend"),
    ("undocumented code cannot be ranked", BASE + " Repayment status improved from 2 in PAY_2 "
                                                 "to -2 in PAY_0.", False, "unsupported_trend"),
    ("one-month change flagged", BASE + " The PAY_0 status has worsened recently.", False,
     "unsupported_trend"),
    ("negated causal is fine", BASE + " This does not mean the limit causes default.", True, None),
    ("causal flagged", BASE + " Raising the limit will reduce the risk of default.", False,
     "unsupported_causal_or_action"),
    ("adjective is not a direction", "The model estimates 38%. PAY_2 raises the risk score. "
                                     "A lower credit limit raises risk.", False, "sign_reversal"),
    ("sign reversal flagged", "The model estimates 38%. PAY_2 raises the risk score and the "
                              "credit limit also raises the risk score.", False, "sign_reversal"),
    ("omitted top driver", "The model estimates 38%. The credit limit lowers the risk score.",
     False, "omission_of_essential_driver"),
    ("alias counts as mention", "The model estimates 38%. The August repayment status raises the "
                                "risk score; the credit limit lowers it.", True, None),
    ("omitted probability", "PAY_2 raises the risk score while the credit limit lowers it.",
     False, "omission_of_essential_driver"),
    ("year from evidence is fine", BASE + " Statuses were recorded in August 2005.", True, None),
    ("period + noun names a monthly driver", "The model estimates 38%. The August 2005 repayment "
                                              "status raises the risk score; the credit limit lowers it.", True, None),
    ("wrong month does not name it", "The model estimates 38%. The September 2005 repayment status "
                                     "raises the risk score; the credit limit lowers it.", False,
     "omission_of_essential_driver"),
]


P0 = "The model estimates a 38% probability of default. PAY_2 raises the risk score. "
# Cases from the 2 Oct 2026 adversarial review (each reproduced a defect in 2.1)
REVIEW_CASES = [
    ("rev: attribution with month is not trend", "The model estimates a 38% probability of default. "
     "The August repayment status increased the risk score, while the credit limit lowers it.", True, None),
    ("rev: 'has been set' is not trend", BASE + " The credit limit has been set at 50,000.", True, None),
    ("rev: contrastive aside (rather than)", "The model estimates a 38% probability of default. PAY_2, rather "
     "than the credit limit, raises the risk score; the credit limit lowers it.", True, None),
    ("rev: contrastive aside (unlike)", "The model estimates a 38% probability of default. PAY_2, unlike the "
     "credit limit, raises the risk score; the credit limit lowers it.", True, None),
    ("rev: parenthetical 'not' is no negation", "The model estimates a 38% probability of default. PAY_2 "
     "(not paid) raises the risk score, while the credit limit lowers it.", True, None),
    ("rev: 'as a result' connector", BASE + " As a result, the model's estimate is 38%.", True, None),
    ("rev: 50k format", "The model estimates a 38% probability of default. PAY_2 raises the risk score, "
     "while the credit limit of NT$50k lowers it.", True, None),
    ("rev: range containing PD", "The model estimates a 38-39% probability of default. PAY_2 raises the "
     "risk score, while the credit limit lowers it.", True, None),
    ("rev: '38 in 100'", "The model estimates about 38 in 100 chance of default. PAY_2 raises the risk "
     "score, while the credit limit lowers it.", True, None),
    ("rev: '.38' decimal", "The estimated probability of default is .38. PAY_2 raises the risk score, "
     "while the credit limit lowers it.", True, None),
    ("rev: trend + 'adds to the risk'", BASE + " Repayment behaviour has deteriorated, which adds to the risk.",
     False, "unsupported_trend"),
    ("rev: trend + 'raises the risk score'", BASE + " Bill amounts are rising, which raises the risk score.",
     False, "unsupported_trend"),
    ("rev: field index is not a value", BASE + " PAY_0 is 0.", False, "numeric_or_unit_error"),
    ("rev: contribution 2.0 for PAY_2", P0 + "PAY_2 contributes 2.0 to the log-odds score.", False,
     "numeric_or_unit_error"),
    ("rev: coordinated plural subject", "The model estimates a 38% probability of default. PAY_2 and the "
     "credit limit both raise the risk score.", False, "sign_reversal"),
    ("rev: 'pushes the risk score up'", P0 + "The credit limit pushes the risk score up.", False, "sign_reversal"),
    ("rev: 'drives up'", P0 + "The credit limit drives up the predicted risk.", False, "sign_reversal"),
    ("rev: 'counts against'", P0 + "The credit limit counts against the borrower.", False, "sign_reversal"),
    ("rev: apposition keeps binding", P0 + "PAY_0, the most recent repayment status, is 2.", False,
     "numeric_or_unit_error"),
    ("rev: explicit + sign", BASE + " LIMIT_BAL contributes +0.33 to the score.", False, "numeric_or_unit_error"),
    ("rev: abbreviation does not split", P0 + "The credit limit (approx. 50,000) also raises the score.",
     False, "sign_reversal"),
    ("rev: 'while modest' aside", P0 + "The credit limit, while modest, raises the risk score too.", False,
     "sign_reversal"),
    ("rev: modal action claim", BASE + " Paying the August bill on time would reduce the borrower's risk of default.",
     False, "unsupported_causal_or_action"),
    ("rev: 'could lower the chance of default'", BASE + " Raising the credit limit could lower the chance of default.",
     False, "unsupported_causal_or_action"),
    ("rev: 'without' does not negate", BASE + " Without help this will lead to default.", False,
     "unsupported_causal_or_action"),
    ("rev: PD not allowed for other fields", BASE + " Age is 38.", False, "numeric_or_unit_error"),
    ("rev: 'Not surprisingly' is no negation", BASE + " Not surprisingly, income declined sharply.", False,
     "unsupported_trend"),
    ("rev: comparison direction checked", BASE + " Repayment status worsened from PAY_2 to PAY_0.", False,
     "unsupported_trend"),
    ("rev: PD omission needs a PD", "Of 38 fields, PAY_2 raises the risk score, while the credit limit lowers it.",
     False, "omission_of_essential_driver"),
]


REVIEW2_CASES = [
    ("r2: contraction negation", BASE + " The credit limit doesn't raise the risk score.", True, None),
    ("r2: contraction disclaimer", BASE + " This doesn't mean the limit causes default.", True, None),
    ("r2: pronoun object reversal", "The model estimates a 38% probability of default. PAY_2 raises the risk "
     "score, while the credit limit raises it too.", False, "sign_reversal"),
    ("r2: chance of default object", BASE + " The credit limit increases the chance of default.", False, "sign_reversal"),
    ("r2: possessive object", BASE + " The credit limit raises the borrower's risk.", False, "sign_reversal"),
    ("r2: more likely", BASE + " The credit limit makes default more likely.", False, "sign_reversal"),
    ("r2: driven up", BASE + " The risk score is driven up by the credit limit.", False, "sign_reversal"),
    ("r2: percent bound to ratio field", BASE + " The borrower's age is 200%.", False, "numeric_or_unit_error"),
    ("r2: unit text is not a value", P0 + "PAY_0 is 1.", False, "numeric_or_unit_error"),
    ("r2: respectively swapped", BASE + " PAY_0 and AGE are 41 and -2 respectively.", False, "numeric_or_unit_error"),
    ("r2: two-modifier object is not trend", "The model estimates a 38% probability of default. The August "
     "repayment status increased the predicted default risk, while the credit limit lowers it.", True, None),
    ("r2: 'risk-increasing' is not trend", BASE + " PAY_2 is a risk-increasing factor.", True, None),
    ("r2: leading contrast", "The model estimates a 38% probability of default. PAY_2 raises the risk score. "
     "Unlike PAY_2, the credit limit lowers the risk score.", True, None),
    ("r2: 'not only' is no negation", "The model estimates a 38% probability of default. PAY_2 not only raises "
     "the risk score but is the largest driver; the credit limit lowers it.", True, None),
    ("r2: disclaimer 'does not say'", BASE + " The model does not say that the credit limit causes default.", True, None),
    ("r2: ordinal", BASE + " The credit limit is the 2nd strongest driver.", True, None),
    ("r2: 'have grown' trend", BASE + " Bill amounts have grown steadily over the last months.", False, "unsupported_trend"),
    ("r2: advice", BASE + " To lower the risk, the borrower should pay the August bill on time.", False,
     "unsupported_causal_or_action"),
    ("r2: 'default risk' action", BASE + " Lowering the bill would reduce the default risk.", False,
     "unsupported_causal_or_action"),
]


REVIEW3_CASES = [
    ("r3: contribution as probability change", "The model estimates a 38% probability of default. PAY_2 raises the "
     "probability of default by 0.61, while the credit limit lowers it.", False, "numeric_or_unit_error"),
    ("r3: percentage points", BASE + " The credit limit lowers the probability of default by 1.6 percentage points.",
     False, "numeric_or_unit_error"),
    ("r3: value swapped with contribution", BASE + " The credit limit is 0.33.", False, "numeric_or_unit_error"),
    ("r3: 'decreased' trend", BASE + " Payments decreased steadily.", False, "unsupported_trend"),
    ("r3: 'risen' trend", BASE + " Bill amounts have risen.", False, "unsupported_trend"),
    ("r3: score change over time", BASE + " The borrower's default risk increased between August and September 2005.",
     False, "unsupported_trend"),
    ("r3: trend disclaimer", BASE + " The evidence does not show how these values changed over time.", True, None),
    ("r3: 'no information about the trend'", BASE + " No information about the trend in payments is available.", True, None),
    ("r3: unbound wrong PD", BASE + " The chance of default within the horizon is 62%.", False, "numeric_or_unit_error"),
    ("r3: 'cause of default' noun", BASE + " Late repayment is a major cause of default.", False,
     "unsupported_causal_or_action"),
    ("r3: modal outcome 'go down'", BASE + " If the borrower pays on time, the probability of default will go down.",
     False, "unsupported_causal_or_action"),
    ("r3: ellipsis 'and so does'", "The model estimates a 38% probability of default. PAY_2 raises the risk score, "
     "and so does the credit limit.", False, "sign_reversal"),
    ("r3: PD with 'risk'", "The model puts this borrower's risk at 38%. PAY_2 raises the risk score; the credit limit "
     "lowers it.", True, None),
    ("r3: credit score inverse polarity", "The model estimates a 38% probability of default. PAY_2 lowers the "
     "borrower's credit score, while the credit limit lowers the risk.", True, None),
    ("r3: neither/nor", "The model estimates a 38% probability of default. PAY_2 raises the risk score. Neither AGE "
     "nor the credit limit raises the risk score.", True, None),
]


REVIEW4_CASES = [
    ("r4: SHAP-positive phrasing reversal", P0 + "The credit limit has a positive SHAP contribution of 0.33.",
     False, "sign_reversal"),
    ("r4: passive with adverb", P0 + "The score is also raised slightly by the credit limit.", False, "sign_reversal"),
    ("r4: contribution 'by N' on the score", P0 + "The credit limit lowers the risk score by 0.33.", True, None),
    ("r4: log-odds margin is not a trend", BASE + " The credit limit decreased the log-odds margin.", True, None),
    ("r4: modal action on the score", BASE + " Paying the August bill on time would lower the risk score.", False,
     "unsupported_causal_or_action"),
    ("r4: recommendation", BASE + " We recommend paying the August bill early.", False, "unsupported_causal_or_action"),
    ("r4: 'gotten worse'", BASE + " Repayment has gotten worse over the past few months.", False, "unsupported_trend"),
    ("r4: 'cannot show a trend'", BASE + " The data cannot show a trend.", True, None),
    ("r4: 'impossible to infer'", BASE + " It is impossible to infer a trend from one statement.", True, None),
    ("r4: coordination needs no verb", "The model estimates a 38% probability of default. PAY_2 is the top driver, "
     "and the credit limit lowers the risk score.", True, None),
    ("r4: pair without 'respectively' (swapped)", P0 + "The credit limit and age are 41 and 50,000.", False,
     "numeric_or_unit_error"),
    ("r4: pair without 'respectively' (correct)", P0 + "The credit limit and age are 50,000 and 41.", True, None),
    ("r4: '1 in N' PD", "The model puts the chance of default at roughly 3 in 8. PAY_2 raises the risk score; the "
     "credit limit lowers it.", False, "omission_of_essential_driver"),
    ("r4: spelled-out PD", "The probability of default is about thirty-eight percent. PAY_2 raises the risk score; "
     "the credit limit lowers it.", True, None),
    ("r4: bare 'between' is not a time span", BASE + " The strongest factor increasing risk is PAY_2, which sits "
     "between the other drivers in size.", True, None),
    ("r4: documentation digits are not facts", BASE + " The borrower has missed 12 payments.", False,
     "numeric_or_unit_error"),
]


def eo_may() -> EvidenceObject:
    eo = eo_tw()
    eo.add(new_item("f5", ET.OBSERVED_FACT, source_field="PAY_5", value=1.0, unit="code",
                    observed_at="in May 2005", series="repayment_status", period_order=4,
                    aliases=["may repayment status"], internal_name="PAY_4"))
    eo.add(new_item("a3", ET.MODEL_ATTRIBUTION, feature_group="PAY_5", contribution=0.9,
                    direction="increases_risk", components=["f5"], internal_name="PAY_4"))
    return eo


MAY_CASES = [
    ("documented comparison, correct direction", "The model estimates a 38% probability of default. "
     "PAY_5 raises the risk score. Repayment status worsened from 1 in May 2005 to 2 in August 2005.",
     True, None),
    ("documented comparison, wrong direction", "The model estimates a 38% probability of default. "
     "PAY_5 raises the risk score. Repayment status improved from 1 in May 2005 to 2 in August 2005.",
     False, "unsupported_trend"),
    ("rev: modal 'may' is not May", "The model estimates a 38% probability of default. Repayment may "
     "matter: PAY_2 raises the risk score, while the credit limit lowers it.", False,
     "omission_of_essential_driver"),
    ("May 2005 names the May driver", "The model estimates a 38% probability of default. The repayment "
     "status in May 2005 raises the risk score; PAY_2 raises it and the credit limit lowers it.", True, None),
]


def eo_pay() -> EvidenceObject:
    """Monthly payment series + a derived aggregate (pay_max) as top driver."""
    eo = EvidenceObject(borrower_id="taiwan-9", dataset="taiwan", prediction_time="after Sep 2005")
    eo.add(new_item("f1", ET.OBSERVED_FACT, source_field="PAY_AMT2", value=66652.0, unit="NT dollars",
                    observed_at="in August 2005", series="payment_amount", period_order=1,
                    aliases=["august payment amount", "amount paid in august 2005"], internal_name="PAY_AMT2"))
    eo.add(new_item("f2", ET.OBSERVED_FACT, source_field="PAY_AMT1", value=28763.0, unit="NT dollars",
                    observed_at="in September 2005", series="payment_amount", period_order=0,
                    aliases=["september payment amount", "amount paid in september 2005"], internal_name="PAY_AMT1"))
    eo.add(new_item("f3", ET.OBSERVED_FACT, source_field="PAY_0", value=1.0, unit="code", value_label="payment delay of 1 month",
                    observed_at="in September 2005", series="repayment_status", period_order=0,
                    aliases=["most recent repayment status"], internal_name="PAY_0"))
    eo.add(new_item("f4", ET.OBSERVED_FACT, source_field="PAY_2", value=-1.0, unit="code", value_label="paid duly",
                    observed_at="in August 2005", series="repayment_status", period_order=1,
                    aliases=["august repayment status"], internal_name="PAY_1"))
    eo.add(new_item("f5", ET.OBSERVED_FACT, source_field="pay_max (derived)", value=1.0, unit="code",
                    value_label="payment delay of 1 month", components=["f3", "f4"],
                    aliases=["pay_max", "highest repayment status", "highest repayment status code"],
                    internal_name="pay_max"))
    eo.add(new_item("a1", ET.MODEL_ATTRIBUTION, feature_group="pay_max (derived)", contribution=0.8,
                    direction="increases_risk", components=["f5"],
                    aliases=["pay_max", "highest repayment status", "highest repayment status code"],
                    internal_name="pay_max"))
    eo.add(new_item("a2", ET.MODEL_ATTRIBUTION, feature_group="PAY_AMT2", contribution=-0.4,
                    direction="decreases_risk", components=["f1"],
                    aliases=["august payment amount", "amount paid in august 2005"], internal_name="PAY_AMT2"))
    eo.add(new_item("p1", ET.MODEL_PREDICTION, calibrated_pd=0.051, target_definition="default payment",
                    horizon="next month"))
    return eo


PB = "The model estimates a 5.1% probability of default. pay_max raises the risk score. "
PAY_CASES = [
    ("r3: number after month+noun is checked", PB + "The borrower paid 12,345 in August 2005.", False,
     "numeric_or_unit_error"),
    ("r3: 'August payment amount was 12,345'", PB + "The August payment amount was 12,345.", False,
     "numeric_or_unit_error"),
    ("r3: year-range merge hides wrong value", PB + "The amount paid fell from 66,652 in August 2005 to 2,876 in "
     "September 2005.", False, "numeric_or_unit_error"),
    ("r3: correct two-month comparison", PB + "The amount paid fell from 66,652 in August to 28,763 in September.",
     True, None),
    ("r3: plural 'payments' comparison", PB + "Payments dropped from 66,652 in August 2005 to 28,763 in "
     "September 2005.", True, None),
    ("r3: aggregate accepts only its own value", "The model estimates a 5.1% probability of default. The highest "
     "repayment status code is -1, which raises the risk score.", False, "numeric_or_unit_error"),
    ("r4: ratio wording binds to the ratio", PB + "The ratio of the August payment to the credit limit was 37%.",
     False, "numeric_or_unit_error"),
    ("r3: aggregate own value ok", "The model estimates a 5.1% probability of default. The highest repayment "
     "status code is 1, which raises the risk score.", True, None),
]


def main() -> int:
    bad = 0
    allcases = [(c, eo_tw) for c in CASES + REVIEW_CASES + REVIEW2_CASES + REVIEW3_CASES + REVIEW4_CASES] + [(c, eo_may) for c in MAY_CASES] + [(c, eo_pay) for c in PAY_CASES]
    for (name, text, ok, vt), make in allcases:
        v = structural_check(Candidate(borrower_id="taiwan-7", arm="test", text=text), make())
        types = [x["type"] for x in v.violations]
        passed = (v.accepted == ok) and (vt is None or vt in types)
        bad += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:<34} accepted={v.accepted} {types}")
    print(f"\n{len(allcases) - bad}/{len(allcases)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

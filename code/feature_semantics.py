"""
Source-validated field semantics for the evidence contract (plan §2 P0 rows
"Evidence construction and feature meanings", "Concept mapping asserts
unsupported semantics", "Taiwan field mapping and annotation glossary").

Every model input is described here with:
  display      the SOURCE name shown to producers, judges and annotators
  meaning      what the field records, from the dataset's own documentation
  unit         unit or encoding
  observed_at  when the value was recorded, relative to the decision
  decode       code -> label for encoded categoricals (only documented codes)
  series/order time-indexed fields: same series, 0 = most recent period
  components   for derived fields, the inputs needed to interpret them
  aliases      plain-language names an explanation may use for the field
  concept      NEUTRAL concept; value-dependent upgrades in concept_for()

Sources
  Taiwan   UCI #350 "default of credit card clients" (Yeh & Lien 2009),
           https://archive.ics.uci.edu/dataset/350 . Raw file columns X1..X23.
  HomeCredit  HomeCredit_columns_description.csv shipped with the Kaggle
           "Home Credit Default Risk" data (application_{train|test}.csv rows
           and bureau.csv rows), read at import time.

Nothing here is guessed. Where the source does not document a code or unit,
the text says so ("not documented"), so a producer cannot be licensed to fill
the gap.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

DATA = os.environ.get("SAARTHI_DATA", "/home/Debz/Hackathon/IDBI_Hackathon/Dataset")

# ---------------------------------------------------------------------------
# Concepts (v2): neutral by default. Only REPAYMENT_HISTORY_ADVERSE is an
# evaluative concept, and it is assigned only by a documented value predicate
# (a recorded late status / days overdue > 0), never by field name alone.
# ---------------------------------------------------------------------------
REASON_CONCEPTS_V2: Dict[str, str] = {
    "REPAYMENT_HISTORY_ADVERSE": "A recorded repayment status or overdue count shows a late payment.",
    "REPAYMENT_STATUS":          "A recorded repayment status, with no late payment recorded in this field.",
    "CREDIT_AMOUNT":             "The size or terms of the credit (limit, loan amount, instalment, term).",
    "BALANCE_AND_PAYMENTS":      "Recorded bill balances, payments made, or ratios between them.",
    "INCOME_AND_EMPLOYMENT":     "Recorded income, income type, occupation or employment duration.",
    "EXTERNAL_SCORE":            "A normalised score supplied by an external data source.",
    "BUREAU_RECORD":             "Records held by the credit bureau about other credits or enquiries.",
    "PERSONAL_PROFILE":          "Recorded personal attributes such as age, education, family status.",
    "REGION_AND_HOUSING":        "Recorded region ratings, address consistency or housing attributes.",
    "APPLICATION_DETAIL":        "Details of the application process, documents or contact flags.",
    "SOCIAL_CIRCLE_RECORD":      "Recorded counts of observations and defaults in the client's social surroundings.",
    "UNMAPPED_FEATURE":          "The feature has no documented business meaning.",
    "NO_SUPPORTED_ESTIMATE":     "No supported estimate is available.",
}


@dataclass
class FieldSpec:
    internal: str
    display: str
    meaning: str
    unit: Optional[str] = None
    observed_at: str = ""
    concept: str = "UNMAPPED_FEATURE"
    decode: Dict[str, str] = field(default_factory=dict)
    series: Optional[str] = None
    order: Optional[int] = None
    components: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    adverse_if: Optional[Callable[[float], bool]] = None
    equivalents: Optional[Callable[[float], Dict[str, float]]] = None
    special: Optional[str] = None
    # what a missing (NaN) value means when the source makes it determinate,
    # e.g. a bureau COUNT is missing because no bureau credit was reported:
    # (display value or None, label). None = genuinely not recorded.
    missing_means: Optional[tuple] = None


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


# ---------------------------------------------------------------------------
# Taiwan (UCI 350)
# ---------------------------------------------------------------------------
# The adapter renames the six repayment-status columns X6..X11 to PAY_0..PAY_5.
# The UCI/Kaggle names are PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 (there is
# no PAY_1). Internal names are kept so trained models are untouched; every
# producer/judge/annotator sees the SOURCE name and month.
TW_MONTHS = ["September 2005", "August 2005", "July 2005", "June 2005",
             "May 2005", "April 2005"]
TW_PAY_SOURCE = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TW_STATUS_DECODE = {
    "-1": "paid duly",
    "1": "payment delay of 1 month", "2": "payment delay of 2 months",
    "3": "payment delay of 3 months", "4": "payment delay of 4 months",
    "5": "payment delay of 5 months", "6": "payment delay of 6 months",
    "7": "payment delay of 7 months", "8": "payment delay of 8 months",
    "9": "payment delay of 9 months or more",
    "-2": "code -2 (not defined by UCI)",
    "0": "code 0 (not defined by UCI)",
}
TW_OBS = "across the April-September 2005 statements"
TW_STATUS_UNIT = "repayment-status code (-1 = paid duly; 1..9 = months of payment delay; -2 and 0 not documented)"


def _taiwan() -> Dict[str, FieldSpec]:
    S: Dict[str, FieldSpec] = {}
    S["LIMIT_BAL"] = FieldSpec(
        "LIMIT_BAL", "LIMIT_BAL",
        "Amount of the given credit, including the individual consumer's credit and family (supplementary) credit",
        "NT dollars", "date not documented by UCI", "CREDIT_AMOUNT",
        aliases=["credit limit", "limit balance", "amount of credit", "given credit", "credit amount"])
    S["EDUCATION"] = FieldSpec(
        "EDUCATION", "EDUCATION", "Education level of the card holder", "code",
        "as recorded by the bank", "PERSONAL_PROFILE",
        decode={"1": "graduate school", "2": "university", "3": "high school",
                "4": "others", "5": "code 5 (not documented by UCI)",
                "6": "code 6 (not documented by UCI)",
                "0": "code 0 (not documented by UCI)"},
        aliases=["education", "education level"])
    S["MARRIAGE"] = FieldSpec(
        "MARRIAGE", "MARRIAGE", "Marital status of the card holder", "code",
        "as recorded by the bank", "PERSONAL_PROFILE",
        decode={"1": "married", "2": "single", "3": "others",
                "0": "code 0 (not documented by UCI)"},
        aliases=["marital status", "marriage"])
    S["AGE"] = FieldSpec("AGE", "AGE", "Age of the card holder", "years",
                         "as recorded by the bank", "PERSONAL_PROFILE",
                         aliases=["age"])
    late = lambda v: v is not None and v >= 1
    for k in range(6):
        src = TW_PAY_SOURCE[k]
        S[f"PAY_{k}"] = FieldSpec(
            f"PAY_{k}", src, f"Repayment status in {TW_MONTHS[k]}", TW_STATUS_UNIT,
            f"in {TW_MONTHS[k]}", "REPAYMENT_STATUS", decode=dict(TW_STATUS_DECODE),
            series="repayment_status", order=k,
            aliases=[f"repayment status in {TW_MONTHS[k]}",
                     f"{TW_MONTHS[k].split()[0]} repayment status",
                     f"{TW_MONTHS[k].split()[0].lower()} repayment"]
            + (["most recent repayment status", "latest repayment status",
                "recent repayment status", "current repayment status"] if k == 0 else []),
            adverse_if=late)
        S[f"BILL_AMT{k+1}"] = FieldSpec(
            f"BILL_AMT{k+1}", f"BILL_AMT{k+1}", f"Amount of bill statement in {TW_MONTHS[k]}",
            "NT dollars", f"in {TW_MONTHS[k]}", "BALANCE_AND_PAYMENTS",
            series="bill_amount", order=k,
            aliases=[f"{TW_MONTHS[k].split()[0]} bill", f"bill amount in {TW_MONTHS[k]}",
                     f"bill statement in {TW_MONTHS[k]}"]
            + (["latest bill", "most recent bill", "recent bill"] if k == 0 else []))
        S[f"PAY_AMT{k+1}"] = FieldSpec(
            f"PAY_AMT{k+1}", f"PAY_AMT{k+1}", f"Amount paid in {TW_MONTHS[k]} (UCI: amount of previous payment)",
            "NT dollars", f"in {TW_MONTHS[k]}", "BALANCE_AND_PAYMENTS",
            series="payment_amount", order=k,
            aliases=[f"{TW_MONTHS[k].split()[0]} payment amount", f"amount paid in {TW_MONTHS[k]}",
                     f"payment made in {TW_MONTHS[k]}"]
            + (["latest payment", "most recent payment", "recent payment amount"] if k == 0 else []))
    S["pay_max"] = FieldSpec(
        "pay_max", "pay_max (derived)",
        "Largest numeric repayment-status code across the 6 monthly statuses PAY_0, PAY_2 ... PAY_6 "
        "(April-September 2005). A value of 1 or more is the longest recorded payment delay; how the "
        "undocumented codes -2 and 0 rank against -1 (paid duly) is not documented",
        TW_STATUS_UNIT, TW_OBS, "REPAYMENT_STATUS", decode=dict(TW_STATUS_DECODE),
        components=[f"PAY_{k}" for k in range(6)],
        aliases=["pay_max", "maximum repayment status", "highest repayment status",
                 "largest repayment status", "largest repayment status code",
                 "maximum repayment status code", "highest repayment status code",
                 "largest repayment delay", "maximum repayment delay", "longest payment delay",
                 "largest recorded delay"],
        adverse_if=late)
    S["pay_sum_late"] = FieldSpec(
        "pay_sum_late", "pay_sum_late (derived)",
        "Number of months, out of the 6 monthly statuses PAY_0, PAY_2 ... PAY_6, with a payment delay (code 1 or more)",
        "months (0-6)", TW_OBS, "REPAYMENT_STATUS",
        components=[f"PAY_{k}" for k in range(6)],
        aliases=["pay_sum_late", "number of late months", "months with a payment delay",
                 "late payments", "count of late months"],
        adverse_if=lambda v: v is not None and v >= 1)
    S["util_ratio"] = FieldSpec(
        "util_ratio", "util_ratio (derived)",
        "BILL_AMT1 divided by LIMIT_BAL: the September 2005 bill as a share of the credit limit",
        "ratio (1.0 = bill equals the limit; negative for a credit balance; can exceed 1)",
        "September 2005 bill; the date of the credit limit is not documented", "BALANCE_AND_PAYMENTS",
        components=["BILL_AMT1", "LIMIT_BAL"],
        aliases=["util_ratio", "utilisation", "utilization", "credit utilisation",
                 "credit utilization", "share of the limit used"],
        equivalents=lambda v: {"percent": round(v * 100, 1)})
    S["pay_ratio"] = FieldSpec(
        "pay_ratio", "pay_ratio (derived)",
        "PAY_AMT1 divided by BILL_AMT1 (amount paid in September 2005 divided by the September 2005 "
        "bill). UCI calls PAY_AMT the 'amount of previous payment', so this is not documented as the "
        "share of that bill that was paid",
        "ratio; missing when the bill is 0; can be negative (credit balance) or above 1",
        "in September 2005", "BALANCE_AND_PAYMENTS", components=["PAY_AMT1", "BILL_AMT1"],
        aliases=["pay_ratio", "payment ratio", "payment-to-bill ratio"],
        equivalents=lambda v: {"percent": round(v * 100, 1)},
        missing_means=(None, "undefined: the September 2005 bill is 0"))
    return S


TAIWAN_TARGET = ("Default payment (Yes = 1, No = 0) as documented by UCI #350; the widely "
                 "used copy names it 'default payment next month'")
TAIWAN_HORIZON = "the payment due after the September 2005 statement (next month); not 12 months"
TAIWAN_DECISION_TIME = "after the September 2005 statement"


# ---------------------------------------------------------------------------
# Home Credit (application_train + bureau aggregates built by the adapter)
# ---------------------------------------------------------------------------
HC_AMT_UNIT = "currency amount (currency not documented by the data provider)"
HC_OBS = "at the loan application"


def _hc_concept(name: str) -> str:
    n = name.upper()
    if n.startswith("EXT_SOURCE"):
        return "EXTERNAL_SCORE"
    if n.startswith(("BUREAU_", "AMT_REQ_CREDIT_BUREAU")):
        return "BUREAU_RECORD"
    if n in ("AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CREDIT_TERM",
             "NAME_CONTRACT_TYPE", "RATIO_CREDIT_INCOME", "RATIO_ANNUITY_INCOME"):
        return "CREDIT_AMOUNT"
    if n in ("AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "DAYS_EMPLOYED", "OCCUPATION_TYPE",
             "ORGANIZATION_TYPE", "FLAG_EMP_PHONE", "DAYS_EMPLOYED_PCT"):
        return "INCOME_AND_EMPLOYMENT"
    if n in ("DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS",
             "NAME_EDUCATION_TYPE", "NAME_TYPE_SUITE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
             "OWN_CAR_AGE"):
        return "PERSONAL_PROFILE"
    if n.startswith(("REGION_", "REG_", "LIVE_", "NAME_HOUSING")) or n.endswith(
            ("_AVG", "_MODE", "_MEDI")):
        return "REGION_AND_HOUSING"
    if n.startswith(("FLAG_", "DAYS_ID", "DAYS_REGISTRATION", "DAYS_LAST_PHONE",
                     "HOUR_APPR", "WEEKDAY_APPR", "OBS_", "DEF_")):
        return "APPLICATION_DETAIL"
    return "UNMAPPED_FEATURE"


def _days_equiv(v: float) -> Dict[str, float]:
    if v is None or v > 0:          # 365243 placeholder or after-application: no years
        return {}
    return {"years": round(-v / 365.25, 1), "abs_days": abs(v)}


def _alias(name: str) -> List[str]:
    low = name.lower()
    return [low, low.replace("_", " ")]


def _home_credit() -> Dict[str, FieldSpec]:
    S: Dict[str, FieldSpec] = {}
    path = f"{DATA}/home_credit/HomeCredit_columns_description.csv"
    desc: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="latin-1") as fh:
            for r in csv.DictReader(fh):
                if r["Table"].startswith("application"):
                    desc[r["Row"]] = r
    for name, r in desc.items():
        if name in ("SK_ID_CURR", "TARGET", "CODE_GENDER"):
            continue
        d = r["Description"].strip()
        spec = (r.get("Special") or "").strip()
        unit = None
        if name.startswith("AMT_REQ_CREDIT_BUREAU"):
            unit = "number of enquiries"
        elif name.startswith("AMT_"):
            unit = HC_AMT_UNIT
        elif name.startswith("DAYS_"):
            unit = "days relative to the application (negative = before the application)"
        elif spec == "normalized":
            unit = "normalised score (scale not documented)"
        elif name.startswith(("FLAG_", "REG_", "LIVE_")):
            unit = "flag (1 = yes, 0 = no)" if not name.startswith(("FLAG_OWN",)) else "Y/N"
        elif name.startswith(("CNT_", "OBS_", "DEF_")):
            unit = "count"
        S[name] = FieldSpec(name, name, d, unit, HC_OBS, _hc_concept(name),
                            aliases=_alias(name), special=spec or None,
                            equivalents=_days_equiv if name.startswith("DAYS_") else None)
    # documented-in-text aliases for the most common drivers
    extra = {
        "AMT_CREDIT": ["credit amount", "loan amount"],
        "AMT_ANNUITY": ["annuity", "loan annuity", "instalment", "installment"],
        "AMT_GOODS_PRICE": ["goods price", "price of the goods"],
        "AMT_INCOME_TOTAL": ["income", "total income"],
        "DAYS_BIRTH": ["age"],
        "DAYS_EMPLOYED": ["employment duration", "time in current employment", "employment length"],
        "DAYS_ID_PUBLISH": ["identity document change", "id document change"],
        "DAYS_REGISTRATION": ["registration change"],
        "DAYS_LAST_PHONE_CHANGE": ["phone change"],
        "EXT_SOURCE_1": ["external score 1", "external source 1"],
        "EXT_SOURCE_2": ["external score 2", "external source 2"],
        "EXT_SOURCE_3": ["external score 3", "external source 3"],
        "NAME_EDUCATION_TYPE": ["education"],
        "NAME_INCOME_TYPE": ["income type"],
        "NAME_FAMILY_STATUS": ["family status", "marital status"],
        "NAME_CONTRACT_TYPE": ["contract type"],
        "OCCUPATION_TYPE": ["occupation"],
        "ORGANIZATION_TYPE": ["organisation type", "organization type", "employer type"],
        "REGION_RATING_CLIENT": ["region rating"],
        "REGION_RATING_CLIENT_W_CITY": ["region rating with city", "region and city rating"],
        "OWN_CAR_AGE": ["car age", "age of the car"],
        "CNT_CHILDREN": ["number of children"],
        "FLAG_OWN_CAR": ["owns a car", "car ownership"],
        "FLAG_OWN_REALTY": ["owns a house or flat", "property ownership", "realty"],
    }
    for k, al in extra.items():
        if k in S:
            S[k].aliases += al
    # structural/2.7 (6 Oct 2026): phrasings seen in the M1 outputs that annotators
    # read as naming these fields ("external score from source 2", "document 9")
    for n in ("1", "2", "3"):
        k = f"EXT_SOURCE_{n}"
        if k in S:
            S[k].aliases += [f"external score from source {n}", f"score from external source {n}",
                             f"external source {n} score", f"external credit score {n}",
                             f"external score (source {n})", f"source {n} score"]
    for k in [x for x in S if x.startswith("FLAG_DOCUMENT_")]:
        n = k.rsplit("_", 1)[-1]
        S[k].aliases += [f"document {n}", f"document number {n}", f"flag document {n}",
                         f"document no. {n}", f"document #{n}"]
    if "DAYS_EMPLOYED" in S:
        S["DAYS_EMPLOYED"].special = (
            "The value 365243 appears in the data as a placeholder; it is not documented by "
            "the data provider and is not a real employment duration")
    if "EXT_SOURCE_1" in S:
        for k in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
            S[k].unit = "normalised score between 0 and 1 (direction not documented)"

    # adapter-derived fields (saarthi/training/adapters.py: load_home_credit)
    der = [
        ("ratio_credit_income", "AMT_CREDIT divided by AMT_INCOME_TOTAL", "ratio",
         ["AMT_CREDIT", "AMT_INCOME_TOTAL"], ["credit-to-income ratio", "credit to income ratio",
                                             "loan-to-income ratio", "loan to income"]),
        ("ratio_annuity_income", "AMT_ANNUITY divided by AMT_INCOME_TOTAL", "ratio",
         ["AMT_ANNUITY", "AMT_INCOME_TOTAL"], ["annuity-to-income ratio", "annuity to income",
                                              "instalment-to-income ratio", "payment-to-income ratio"]),
        ("credit_term", "AMT_CREDIT divided by AMT_ANNUITY (number of annuity payments needed to repay the credit amount; not the contractual term)",
         "ratio (count of annuity payments)", ["AMT_CREDIT", "AMT_ANNUITY"],
         ["credit term", "credit-to-annuity ratio", "number of annuity payments"]),
        ("days_employed_pct", "DAYS_EMPLOYED divided by DAYS_BIRTH (share of life spent in current employment)",
         "ratio", ["DAYS_EMPLOYED", "DAYS_BIRTH"], ["employment-to-age ratio", "share of life employed"]),
        ("bureau_n", "Number of the client's credits from other institutions reported to the Credit Bureau",
         "count", [], ["number of bureau credits", "bureau credits", "previous credits"]),
        ("bureau_days_credit_mean", "Mean, over bureau credits, of days before the current application that the client applied for that credit",
         "days relative to the application (negative = before)", [], ["average age of bureau credits", "mean bureau credit age"]),
        ("bureau_overdue_max", "Largest number of days past due on any bureau credit at the time of the current application",
         "days past due", [], ["days past due", "bureau overdue days", "overdue days", "days overdue"]),
        ("bureau_sum", "Sum of AMT_CREDIT_SUM (credit amount) over the client's bureau credits; credits with no recorded amount count as 0; amounts in different currencies are added as recorded", HC_AMT_UNIT, [],
         ["total bureau credit", "bureau credit amount", "total credit at other institutions"]),
        ("bureau_debt", "Sum of AMT_CREDIT_SUM_DEBT (recorded current debt) over the client's bureau credits; credits with no recorded debt count as 0, negative values (credit balances) are included, and amounts in different currencies are added as recorded", HC_AMT_UNIT, [],
         ["bureau debt", "recorded bureau debt"]),
        ("bureau_active_n", "Number of the client's bureau credits with status Active", "count", [],
         ["active bureau credits", "active credits", "number of active credits"]),
    ]
    for name, meaning, unit, comps, al in der:
        S[name] = FieldSpec(name, f"{name} (derived)", meaning, unit, HC_OBS, _hc_concept(name),
                            components=comps, aliases=[name, name.replace("_", " ")] + al)
    S["bureau_overdue_max"].adverse_if = lambda v: v is not None and v > 0
    for k in ("ratio_credit_income", "ratio_annuity_income", "credit_term", "days_employed_pct"):
        S[k].missing_means = None   # NaN here means an input is not recorded
    # adapter joins bureau aggregates; a client with no bureau rows gets NaN,
    # and bureau_active_n is NaN when no bureau credit is Active. Counts are
    # therefore determinate zeros; the other aggregates have no credit to
    # aggregate over.
    S["bureau_n"].missing_means = (0.0, "no credits reported to the bureau")
    S["bureau_active_n"].missing_means = (0.0, "no Active bureau credit reported")
    for k in ("bureau_days_credit_mean", "bureau_overdue_max", "bureau_sum", "bureau_debt"):
        S[k].missing_means = (None, "no credits reported to the bureau")
    for k in ("FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"):
        if k in S:
            S[k].unit = "category (as recorded)"
    for k, sp in S.items():
        if k.startswith(("OBS_", "DEF_")):
            sp.concept = "SOCIAL_CIRCLE_RECORD"
    if "days_employed_pct" in S:
        S["days_employed_pct"].special = "Meaningless when DAYS_EMPLOYED holds the 365243 placeholder"
    return S


HC_TARGET = ("Payment difficulties (1): late payment of more than X days on at least one of "
             "the first Y instalments of the loan; X and Y are not disclosed by the data provider")
HC_HORIZON = "the first Y instalments of the loan (Y not disclosed); not 12 months"
HC_DECISION_TIME = "at the loan application"


# ---------------------------------------------------------------------------
_CACHE: Dict[str, Dict[str, FieldSpec]] = {}


def specs(dataset: str) -> Dict[str, FieldSpec]:
    if dataset not in _CACHE:
        _CACHE[dataset] = {"taiwan": _taiwan, "home_credit": _home_credit}[dataset]()
    return _CACHE[dataset]


TARGETS = {"taiwan": (TAIWAN_TARGET, TAIWAN_HORIZON, TAIWAN_DECISION_TIME),
           "home_credit": (HC_TARGET, HC_HORIZON, HC_DECISION_TIME)}


def spec_for(dataset: str, internal: str) -> FieldSpec:
    s = specs(dataset).get(internal)
    if s is None:      # an input with no documentation: say so, never guess
        return FieldSpec(internal, internal, "No documented meaning for this field",
                         None, "", "UNMAPPED_FEATURE", aliases=_alias(internal))
    return s


def decode_value(spec: FieldSpec, raw) -> Optional[str]:
    """Documented label for an encoded value, or None if not encoded."""
    if not spec.decode:
        return None
    v = _f(raw)
    key = str(int(v)) if v is not None and v == int(v) else str(raw)
    return spec.decode.get(key, f"code {key} (not documented)")


def concept_for(spec: FieldSpec, raw) -> str:
    """Neutral concept unless a documented value predicate licenses more."""
    if spec.adverse_if is not None and spec.adverse_if(_f(raw)):
        return "REPAYMENT_HISTORY_ADVERSE"
    return spec.concept


def glossary_markdown(datasets=("taiwan", "home_credit")) -> str:
    """Annotator glossary generated from the same specs the pipeline uses."""
    out = ["# Feature Glossary (generated from the study's field definitions)\n",
           "Field names are the names used by the original data source. A field marked "
           "(derived) was computed by the study from the source fields listed with it.\n"]
    if "taiwan" in datasets:
        out += ["## Taiwan credit-card default data (UCI #350)\n",
                f"Outcome: {TAIWAN_TARGET}. Horizon: {TAIWAN_HORIZON}. "
                "Amounts are in NT dollars. Data cover April-September 2005.\n",
                "Repayment-status codes: -1 = paid duly; 1 = payment delay of one month; "
                "2 = two months; ... 9 = nine months or more. **-2 and 0 occur in the data "
                "but are not defined by UCI**. Secondary sources interpret them, but the data "
                "provider does not, so an explanation must not state a meaning for them. "
                "A value of 0 is a recorded value, not a missing one. "
                "There is no PAY_1 in the source data.\n",
                "| Field | Meaning | Unit / coding |", "|---|---|---|"]
        for s in specs("taiwan").values():
            out.append(f"| {s.display} | {s.meaning} | {s.unit or ''} |")
        out.append("")
    if "home_credit" in datasets:
        out += ["## Home Credit default risk data\n",
                f"Outcome: {HC_TARGET}. Horizon: {HC_HORIZON}. The currency of amounts is "
                "not documented. DAYS_ fields count days relative to the application "
                "(negative = before it); for example DAYS_BIRTH = -12,000 means about 32.9 "
                "years old.\n",
                "Only fields that can appear in the evidence are listed in full below; any "
                "other field is described in the evidence line itself.\n",
                "| Field | Meaning | Unit / coding |", "|---|---|---|"]
        for s in specs("home_credit").values():
            if s.concept in ("REGION_AND_HOUSING",) and s.internal.endswith(("_AVG", "_MODE", "_MEDI")):
                continue
            note = f" Note: {s.special}." if s.special and s.special not in ("normalized",) else ""
            out.append(f"| {s.display} | {s.meaning}{note} | {s.unit or ''} |")
        out.append("\nBuilding/housing fields ending _AVG, _MODE or _MEDI are normalised "
                   "statistics about the building where the client lives (documented only as "
                   "'normalized').\n")
    out += ["## Reading the evidence\n",
            "- A categorical field is shown with its recorded value and the documented label, "
            "e.g. `EDUCATION = 2 (university)`. The field name alone never tells you the value.",
            "- `not recorded` means the source has no value for this borrower. It is not zero.",
            "- **Risk score / raw log-odds margin**: the model's internal score before it is "
            "converted to a probability. Attribution numbers live on this scale and are not "
            "percentages.",
            "- **Calibrated probability of default (PD)**: the only percentage an explanation "
            "may state as the default estimate.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    for ds in ("taiwan", "home_credit"):
        sp = specs(ds)
        print(ds, len(sp), "fields;", sum(1 for s in sp.values() if s.concept == "UNMAPPED_FEATURE"),
              "unmapped")
    s = spec_for("taiwan", "PAY_1")
    print(s.display, "|", s.meaning, "|", decode_value(s, 0), "|", concept_for(s, 2))

# Milestone-1 Annotation Guidelines — Credit Explanation Verification (v2)

Changes since round 1 are marked **NEW**.

## 1. Your task

Each row shows **EVIDENCE** (what a credit-risk model produced for one borrower)
and an **EXPLANATION** written from it for a credit officer. Decide whether the
explanation says anything the evidence does not support. Assume the evidence is
correct and judge the explanation only against it. You are not judging the
model, the loan decision, or writing style.

**NEW — two datasets.** The `dataset` column says which one the row comes from:
Taiwan credit-card data or Home Credit loan applications. Field meanings are in
`FEATURE_GLOSSARY.md`, and every evidence line now also states its meaning,
unit and when it was recorded.

## 2. Reading the evidence

| Type | Meaning | The explanation MAY say |
|---|---|---|
| Observed fact `[f…]` | A recorded value, with meaning, unit and period | The value (sensibly rounded), its meaning |
| Model attribution `[a…]` | A feature RAISES or LOWERS the model's risk score; `value in [f…]` points to its recorded value | That it raises / lowers the risk score |
| Model prediction `[p…]` | The calibrated probability of default, its outcome and horizon | That estimate |
| Not available | Something that could not be computed | That no estimate is available |

Rules for reading:
1. **Attribution contributions are not percentages.** "+0.61 on the raw log-odds
   margin" means the feature pushes the score up. Writing "61% more risk" is a
   `numeric_or_unit_error`.
2. **NEW — encoded values.** A categorical value is shown with its documented
   label, e.g. `EDUCATION = 2 (university)`. A field name alone never tells you
   the borrower's category. Some codes are marked *undocumented*: an explanation
   that treats an undocumented code as if its meaning were certain has an error.
3. **NEW — missing is not zero.** `not recorded` means no value exists. Treating
   it as 0 is a `numeric_or_unit_error`.
4. **NEW — periods.** Taiwan repayment, bill and payment fields are monthly
   (April–September 2005). Home Credit `DAYS_` fields count days before the
   application (negative numbers; -12,000 days is about 32.9 years).

## 3. Labels (Y / N)

Mark **Y** if the explanation contains at least one instance.

- **invented_factor** — names a factor, fact or reason that is not in the evidence.
  A plain-language name for an evidence field is fine ("credit limit" for LIMIT_BAL).
- **sign_reversal** — says a feature raises risk when the evidence says it LOWERS it, or vice versa.
- **numeric_or_unit_error** — **NEW: check each number against the field it is attached to.**
  "AGE is 50,000" is an error even if 50,000 appears elsewhere in the evidence.
  Sensible rounding is fine (38.4% → 38%). Attribution numbers presented as percentages
  or probability changes are errors.
- **unsupported_trend** — **NEW wording.** Asserts a change over time that the
  evidence does not show. A comparison of two periods that are **both shown in the
  evidence** is allowed if it is stated correctly (e.g. "2 months late in August 2005 and
  paid duly in September 2005"). "The risk score is increased by X" describes the model,
  not a trend.
- **unsupported_causal_or_action** — claims a real-world cause, or that an action will
  change the outcome. "The model associates X with higher risk" is fine. A disclaimer
  such as "this describes the model, not a verified cause" is correct, not a causal claim.
- **omission_of_essential_driver** — leaves out the strongest driver (the attribution with
  the largest absolute contribution) or the probability estimate. Naming the driver only by
  a broad category ("repayment history") without identifying it counts as omission.
- **other_material_error** — anything else that would mislead a credit officer (wrong
  outcome or horizon, two features confused, an undocumented code stated as certain). Explain in `notes`.
- **ANY_MATERIAL_ERROR** — Y if any of the seven above is Y.

**NEW — UNRESOLVED.** Mark `UNRESOLVED = Y` when you cannot decide because the
evidence is insufficient, the rubric does not apply, or a feature is unclear. Choose
`unresolved_reason`. Still give your best guess in `ANY_MATERIAL_ERROR`. Do not use
UNRESOLVED for ordinary hard cases.

**NEW — reasonable_disagreement.** Y if a careful colleague could reasonably give a
different `ANY_MATERIAL_ERROR` label. This is about the item, not your confidence.

- **informative_for_credit_officer** — Y if a credit officer would learn the estimate and the main reason(s).
- **confidence** — High / Medium / Low.
- **error_quote** — the exact wrong words, if ANY = Y.
- **NEW — error_evidence_ids** — the evidence ids involved, e.g. `a2, f3`.
- **notes** — anything useful.

## 4. Borderline rules

1. Wording is lenient, facts are strict.
2. One wrong sentence is enough for Y.
3. Hedged claims still count ("income may have declined" is a trend claim).
4. Do not penalise correct disclaimers or statements that something is unavailable.
5. Some rows look alike or identical. Label each on its own.

## 5. Practical

- **NEW — practice gate.** Fill the PRACTICE sheet first and return it. Your main
  sheet is counted only if you get **at least 8 of the 10** ANY_MATERIAL_ERROR labels
  right **and fill ANY_MATERIAL_ERROR on every practice row** (a blank counts as not
  passing; if you mark UNRESOLVED, still give your best guess). You will get the
  practice answers after you return it.
- About 1–2 minutes per row after the first 20. Take a break every 45 minutes.
- Work independently. Do not use ChatGPT or any AI tool.
- A broken row (empty or unreadable): ANY = Y, other_material_error = Y, note "broken".


---

## Fresh-test addendum (round 2)

This round contains explanations for NEW borrowers. Some explanations are short and formulaic; some are
longer and written in free prose. Label every row on its own with the same rules as before.

Three extra columns (fill for EVERY row):

- **conveys_pd** (Y/N): the text states the model's probability of default correctly (rounding is fine).
- **conveys_top_driver_with_direction** (Y/N): the text names the attribution with the largest absolute
  contribution AND says correctly whether it raises or lowers the risk score.
- **useful_for_officer** (1-5): how useful the text is to a credit officer who must understand why the model
  scored this borrower as it did (1 = useless or misleading, 3 = adequate, 5 = clear and complete).
  Judge usefulness from the content, not the length or style.

ANY_MATERIAL_ERROR keeps its meaning: one material error anywhere makes it Y.

Some explanations in this round are written TO THE APPLICANT in everyday language ("you", "your payments").
Judge them by the same rules: everyday wording is fine; misstating a value, a direction, the strongest driver, the
probability, the field a number belongs to, or a code's undocumented meaning is a material error.

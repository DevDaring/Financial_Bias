"""
Build the E1 annotation package (blinded) from pilot records.

Output
  Submission/annotation_pack_v1.zip        -> send to annotators
  Submission/annotation_key_v1.jsonl       -> KEEP LOCAL. Unblinding key.

Items = natural pilot explanations + controlled corruptions with known answers
+ ~10% hidden duplicates for intra-annotator consistency. Annotators never see
model, arm, borrower group, or any checker/judge verdict.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import EvidenceItem, EvidenceObject, EvidenceType  # noqa: E402
from generate import Candidate  # noqa: E402
from verify import ErrorType, corrupt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "Submission"
PACK = SUB / "annotation_pack_v1"
SEED = 20260930

LABEL_COLS = [
    "invented_factor", "sign_reversal", "numeric_or_unit_error",
    "unsupported_trend", "unsupported_causal_or_action",
    "omission_of_essential_driver", "other_material_error",
    "ANY_MATERIAL_ERROR", "informative_for_credit_officer", "confidence",
    "error_quote", "notes",
]
TYPE_LABEL = {
    "observed_fact": "Observed fact",
    "model_attribution": "Model attribution",
    "model_prediction": "Model prediction",
    "hypothetical_action": "Hypothetical action",
    "unsupported": "Not available",
}


def eo_from_dict(d: dict) -> EvidenceObject:
    eo = EvidenceObject(borrower_id=d["borrower_id"], dataset=d["dataset"],
                        prediction_time=d.get("prediction_time"),
                        group_attributes=d.get("group_attributes", {}),
                        schema_version=d.get("schema_version", "evidence/1.0"),
                        provenance=d.get("provenance", {}))
    for it in d["items"]:
        it = dict(it)
        it["etype"] = EvidenceType(it["etype"])
        eo.add(EvidenceItem(**it))
    return eo


def render_evidence(eo: EvidenceObject) -> str:
    """Human-readable evidence, no borrower id, no group attribute."""
    lines = []
    for i in eo.items:
        t = TYPE_LABEL[i.etype.value]
        if i.etype is EvidenceType.OBSERVED_FACT:
            v = f"{i.value:,.2f}" if isinstance(i.value, float) else i.value
            lines.append(f"[{i.eid}] {t}: {i.source_field} = {v}"
                         f"{' ' + i.unit if i.unit else ''}")
        elif i.etype is EvidenceType.MODEL_ATTRIBUTION:
            arrow = "RAISES" if i.direction == "increases_risk" else "LOWERS"
            lines.append(f"[{i.eid}] {t}: {i.feature_group} {arrow} the model's "
                         f"risk score (contribution {i.contribution:+.4f} on the "
                         f"{i.explained_scale})")
        elif i.etype is EvidenceType.MODEL_PREDICTION:
            lines.append(f"[{i.eid}] {t}: estimated probability of default = "
                         f"{i.calibrated_pd:.1%}  (outcome: {i.target_definition}; "
                         f"horizon: {i.horizon})")
        elif i.etype is EvidenceType.HYPOTHETICAL_ACTION:
            lines.append(f"[{i.eid}] {t}: {i.raw_edits} -> modelled PD "
                         f"{i.rescored_pd}")
        else:
            lines.append(f"[{i.eid}] {t}: {i.reason}")
    return "\n".join(lines)


def blind_text(text: str, eo: EvidenceObject) -> str:
    return text.replace(eo.borrower_id, "this borrower")


def main() -> None:
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(SUB / "pilot_records.jsonl")]
    rows = [r for r in rows if r.get("text") and r.get("evidence")]

    items, key = [], []

    # 1. natural explanations (all arms, all producers)
    for r in rows:
        eo = eo_from_dict(r["evidence"])
        items.append({"eo": eo, "text": r["text"]})
        key.append({"source": "natural", "borrower_id": eo.borrower_id,
                    "group": eo.group_attributes.get("audit_group"),
                    "arm": r["arm"], "family": r["family"],
                    "model_id": r["model_id"], "injected_error": None,
                    "structural_accepted": r["structural_accepted"],
                    "judges": {k: v.get("accepted")
                               for k, v in r.get("judges", {}).items()}})

    # 2. controlled corruptions with a known answer (8 per error type)
    clean = [r for r in rows if r["structural_accepted"]]
    rng.shuffle(clean)
    for et in ErrorType:
        made = 0
        for r in clean:
            if made >= 8:
                break
            eo = eo_from_dict(r["evidence"])
            base = Candidate(borrower_id=eo.borrower_id, arm=r["arm"],
                             text=r["text"], family=r["family"])
            c, applied = corrupt(base, eo, et, rng)
            if not applied or c.text == r["text"]:
                continue
            items.append({"eo": eo, "text": c.text})
            key.append({"source": "corrupted", "borrower_id": eo.borrower_id,
                        "group": eo.group_attributes.get("audit_group"),
                        "arm": r["arm"], "family": r["family"],
                        "model_id": r["model_id"], "injected_error": et.value,
                        "base_text": r["text"]})
            made += 1

    # 3. hidden duplicates (~10%) for intra-annotator agreement
    n_dup = max(1, len(items) // 10)
    for j in rng.sample(range(len(items)), n_dup):
        items.append(dict(items[j]))
        key.append({**key[j], "duplicate_of_index": j})

    # shuffle and assign opaque ids
    order = list(range(len(items)))
    rng.shuffle(order)
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)

    sheet, jsonl, keyrows = [], [], []
    idx_to_id = {}
    for pos, i in enumerate(order, 1):
        iid = f"E1-{pos:04d}"
        idx_to_id[i] = iid
    for pos, i in enumerate(order, 1):
        iid = idx_to_id[i]
        eo, text = items[i]["eo"], blind_text(items[i]["text"], items[i]["eo"])
        ev = render_evidence(eo)
        sheet.append({"item_id": iid, "evidence": ev, "explanation": text})
        jsonl.append({"item_id": iid, "dataset": eo.dataset,
                      "evidence": [{k: v for k, v in x.to_dict().items() if k not in ("model_id", "transformer_id", "calibrator_id")} for x in eo.items],
                      "explanation": text})
        k = dict(key[i])
        if "duplicate_of_index" in k:
            k["duplicate_of"] = idx_to_id[k.pop("duplicate_of_index")]
        keyrows.append({"item_id": iid, **k})

    _write_xlsx(PACK / "E1_annotation_sheet.xlsx", sheet)
    (PACK / "E1_items.jsonl").write_text(
        "\n".join(json.dumps(x, default=str) for x in jsonl))
    _write_practice(PACK)
    (PACK / "ANNOTATION_GUIDELINES.md").write_text(GUIDE)
    (PACK / "FEATURE_GLOSSARY.md").write_text(GLOSSARY)
    (PACK / "README.txt").write_text(README.format(n=len(sheet)))

    (SUB / "annotation_key_v1.jsonl").write_text(
        "\n".join(json.dumps(x, default=str) for x in keyrows))

    zp = SUB / "annotation_pack_v1.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PACK.iterdir()):
            z.write(f, f"annotation_pack_v1/{f.name}")

    nat = sum(1 for k in keyrows if k["source"] == "natural" and "duplicate_of" not in k)
    cor = sum(1 for k in keyrows if k["source"] == "corrupted" and "duplicate_of" not in k)
    dup = sum(1 for k in keyrows if "duplicate_of" in k)
    print(f"items={len(sheet)}  natural={nat}  corrupted={cor}  duplicates={dup}")
    print(f"zip  -> {zp}")
    print(f"key  -> {SUB/'annotation_key_v1.jsonl'}  (DO NOT SEND)")


# ---------------------------------------------------------------------------
def _write_xlsx(path: Path, rows: list) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Annotate"
    head = ["item_id", "evidence", "explanation"] + LABEL_COLS
    ws.append(head)
    for r in rows:
        ws.append([r["item_id"], r["evidence"], r["explanation"]]
                  + [""] * len(LABEL_COLS))

    hdr = PatternFill("solid", fgColor="1F2937")
    lab = PatternFill("solid", fgColor="FEF3C7")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = hdr
        c.alignment = Alignment(wrap_text=True, vertical="center")
    widths = {"A": 11, "B": 70, "C": 70}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for j in range(4, 4 + len(LABEL_COLS)):
        L = ws.cell(row=1, column=j).column_letter
        ws.column_dimensions[L].width = 16
    ws.column_dimensions[ws.cell(row=1, column=3 + LABEL_COLS.index("error_quote") + 1).column_letter].width = 35
    ws.column_dimensions[ws.cell(row=1, column=3 + LABEL_COLS.index("notes") + 1).column_letter].width = 35
    for row in ws.iter_rows(min_row=2):
        for c in row[:3]:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[3:]:
            c.fill = lab
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "D2"

    n = len(rows) + 1
    yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    conf = DataValidation(type="list", formula1='"High,Medium,Low"', allow_blank=True)
    ws.add_data_validation(yn)
    ws.add_data_validation(conf)
    for name in LABEL_COLS[:9]:
        L = ws.cell(row=1, column=4 + LABEL_COLS.index(name)).column_letter
        yn.add(f"{L}2:{L}{n}")
    L = ws.cell(row=1, column=4 + LABEL_COLS.index("confidence")).column_letter
    conf.add(f"{L}2:{L}{n}")

    meta = wb.create_sheet("Annotator")
    for r in [["annotator_id", ""], ["background (e.g. credit analyst, finance grad)", ""],
              ["date started", ""], ["date finished", ""],
              ["total time spent (hours)", ""], ["used any AI tool? (must be N)", ""]]:
        meta.append(r)
    meta.column_dimensions["A"].width = 48
    meta.column_dimensions["B"].width = 30
    wb.save(path)


def _write_practice(pack: Path) -> None:
    ev = ("[f1] Observed fact: LIMIT_BAL = 200,000.00\n"
          "[f2] Observed fact: AGE = 41.00\n"
          "[a1] Model attribution: PAY_0 RAISES the model's risk score "
          "(contribution +0.6120 on the raw log-odds margin (not calibrated PD))\n"
          "[a2] Model attribution: LIMIT_BAL LOWERS the model's risk score "
          "(contribution -0.2210 on the raw log-odds margin (not calibrated PD))\n"
          "[p1] Model prediction: estimated probability of default = 38.0%  "
          "(outcome: taiwan documented default label; horizon: documented "
          "outcome horizon (NOT assumed 12 months))")
    cases = [
        ("P-1", "The model estimates a 38% probability of default. The recent "
                "repayment status (PAY_0) raises the risk score, while the credit "
                "limit of 200,000 lowers it.", "No error. ANY_MATERIAL_ERROR = N."),
        ("P-2", "The model estimates a 38% probability of default. PAY_0 raises "
                "the risk score and the credit limit of 200,000 also raises it.",
         "sign_reversal = Y (LIMIT_BAL LOWERS risk in the evidence)."),
        ("P-3", "The model estimates a 52% probability of default, driven mainly "
                "by PAY_0; the credit limit of 200,000 lowers the score.",
         "numeric_or_unit_error = Y (52% is not in the evidence; it is 38%)."),
        ("P-4", "The model estimates 38%. PAY_0 raises the score and the credit "
                "limit lowers it. The borrower's income has declined recently.",
         "unsupported_trend = Y AND invented_factor = Y (no income item; "
         "'declined recently' asserts change over time)."),
        ("P-5", "The model estimates 38%. The credit limit of 200,000 lowers the "
                "risk score.",
         "omission_of_essential_driver = Y (PAY_0, the largest driver, is missing)."),
        ("P-6", "The model estimates 38%. PAY_0 raises the risk score and the limit "
                "lowers it. Increasing the credit limit will reduce the risk of "
                "default.",
         "unsupported_causal_or_action = Y (a causal/action claim with no "
         "hypothetical-action evidence)."),
    ]
    txt = ["PRACTICE SET — do these first, then compare with the answers below.",
           "All six share the same evidence:\n", ev, "\n" + "=" * 70 + "\n"]
    for cid, expl, _ in cases:
        txt.append(f"{cid}\nEXPLANATION: {expl}\n")
    txt.append("\n" + "=" * 70 + "\nANSWERS (scroll only after attempting)\n" + "\n" * 12)
    for cid, _, ans in cases:
        txt.append(f"{cid}: {ans}")
    (pack / "PRACTICE_SET_with_answers.txt").write_text("\n".join(txt))


# ---------------------------------------------------------------------------
README = """SAARTHI — E1 Explanation Verification Annotation (v1)
=====================================================

Items to label: {n}

Files
  README.txt                      this file
  ANNOTATION_GUIDELINES.md        READ FULLY before starting
  FEATURE_GLOSSARY.md             what each dataset column means
  PRACTICE_SET_with_answers.txt   6 practice items with answers - do these first
  E1_annotation_sheet.xlsx        the work: fill the yellow columns
  E1_items.jsonl                  same items, machine-readable (you do not need it)

Steps
  1. Read ANNOTATION_GUIDELINES.md (about 20 minutes).
  2. Do the practice set, check your answers.
  3. Fill the "Annotator" tab of the spreadsheet.
  4. Label every row in the "Annotate" tab. Work independently.
  5. Return ONLY the spreadsheet, renamed E1_annotation_sheet_<annotator_id>.xlsx

Rules
  - Do not discuss items with other annotators until everyone has returned.
  - Do not use ChatGPT or any AI tool to help decide labels.
  - The material is for this study only. Do not share it.
"""

GUIDE = r"""# E1 Annotation Guidelines — Credit Explanation Verification

## 1. What you are doing

Each row shows **EVIDENCE** (what a credit-risk model produced for one borrower)
and an **EXPLANATION** (a short text written from that evidence for a credit officer).

Your task: **decide whether the explanation says anything the evidence does not
support.** You are checking the text against the evidence — nothing else.

You are **not** judging:
- whether the model is right about the borrower;
- whether the loan should be approved;
- writing style, grammar, or tone (unless it changes meaning).

Assume the evidence is correct. Judge the explanation only against it.

## 2. How to read the evidence

Each line starts with an id like `[f1]`, `[a2]`, `[p1]`.

| Type | Meaning | What the explanation MAY say |
|---|---|---|
| **Observed fact** `[f…]` | A recorded value, e.g. `LIMIT_BAL = 200,000` | State the value. |
| **Model attribution** `[a…]` | Whether a feature **RAISES** or **LOWERS** the model's risk score | Say it raises/lowers the risk score. |
| **Model prediction** `[p…]` | The estimated probability of default and the outcome it refers to | State that estimate. |
| **Hypothetical action** | A modelled "what if" scenario | Say that *under the model*, the score would change. |
| **Not available** | Something that could not be computed | Say no estimate is available. |

Two important points:

1. **Attribution contributions are NOT percentages.** "+0.61 on the raw log-odds
   margin" means the feature pushes the score up. It does **not** mean "increases
   default probability by 61%". An explanation that converts it into a percentage
   effect has a `numeric_or_unit_error`.
2. **The evidence is a single snapshot.** It contains no history. Any claim that
   something *changed over time* is unsupported.

Feature names come from public datasets; see `FEATURE_GLOSSARY.md`. A name like
`EDUCATION_5.0` means "the EDUCATION field has value 5". A name like `pay_max`
is a derived summary (defined in the glossary).

## 3. The labels (Y / N for each)

Mark **Y** if the explanation contains at least one instance. Otherwise **N**.

### invented_factor
The explanation names a risk factor, fact, or reason that **does not appear
anywhere in the evidence**.
- ✗ "…also reflects an unresolved legal dispute." (no such item)
- ✓ Plain-language restatement of an evidence item is fine: "a low credit limit"
  for `LIMIT_BAL`, "recent repayment status" for `PAY_0`.

### sign_reversal
The explanation says a feature **raises** risk when the evidence says it
**LOWERS** it, or vice versa.
- ✗ Evidence: LIMIT_BAL LOWERS. Text: "the credit limit increases the risk."

### numeric_or_unit_error
A number in the explanation **does not match** the evidence, or has the wrong
unit/scale.
- ✗ Evidence PD 38%, text says 52%.
- ✗ Converting an attribution (+0.61) into "61% higher risk".
- ✓ Sensible rounding is fine (38.4% → "about 38%"; 200,000.00 → "200,000").
- ✓ Repeating a number that appears in the evidence text is fine.

### unsupported_trend
The explanation asserts **change over time**: "declined", "rising", "has
worsened", "recently dropped", "continues to", "over the past months".
- ✓ Describing a level is fine: "a low credit limit", "a high balance".

### unsupported_causal_or_action
The explanation claims a real-world **cause** or that an **action will change
the outcome**, without a hypothetical-action item in the evidence.
- ✗ "Raising the limit **will reduce** the risk of default."
- ✗ "The borrower defaults **because of** …"
- ✓ "The model associates X with higher risk" is fine.

### omission_of_essential_driver
The explanation **leaves out the strongest driver** — the model attribution with
the largest absolute contribution — or leaves out the probability estimate.
- Leaving out minor drivers is fine.
- An empty or generic text ("risk is moderate") that names no driver = Y.

### other_material_error
Anything else that would **mislead a credit officer**, not covered above
(e.g. confuses two features, states the wrong outcome/horizon, says "12 months"
when the evidence says the horizon is NOT 12 months). Describe it in `notes`.

### ANY_MATERIAL_ERROR
**Y if any of the seven columns above is Y.** Otherwise N.
This is the single most important column. Fill it for every row.

### informative_for_credit_officer
**Y** if a credit officer reading only this text would learn the estimate and at
least the main reason(s). **N** if it is vague, generic, or empty.
(A text can be correct but uninformative, or informative but wrong — label both.)

### confidence
How sure you are of your labels for this row: High / Medium / Low.
Use Low freely — honest uncertainty is valuable.

### error_quote
If ANY_MATERIAL_ERROR = Y, copy the exact words that are wrong.

### notes
Anything useful: ambiguity, a feature you did not understand, a borderline call.

## 4. Decision rules for borderline cases

1. **Wording is lenient, facts are strict.** Paraphrase is fine; new facts,
   wrong numbers, flipped directions are not.
2. **When a feature name is unclear**, check the glossary. If still unclear, do
   not guess — mark Low confidence and write a note.
3. **Hedged claims still count.** "Income *may have* declined" is still an
   unsupported trend.
4. **One wrong sentence is enough** for Y, even if the rest is excellent.
5. **Do not penalise** mentioning that the model's estimate is not a verified
   cause, or that some information is unavailable — that is correct behaviour.
6. Some rows may look very similar or identical. **Label each one on its own.**

## 5. Practical

- Expected pace: about 1–2 minutes per row after the first 20.
- Take breaks every 45 minutes; fatigue causes errors.
- Work independently. Do not use AI tools.
- If a row is broken (empty text, unreadable), set ANY_MATERIAL_ERROR = Y,
  other_material_error = Y, and note "broken".

## 6. What happens to your labels

Two or more annotators label every item independently. Disagreements are
adjudicated. Your labels are the ground truth used to measure how well automatic
checkers work — so accuracy matters more than speed.
"""

GLOSSARY = """# Feature Glossary

## Taiwan credit-card default dataset (UCI #350)

Amounts are in New Taiwan dollars (NT$). Default outcome: payment default in the
following month (as documented by the dataset).

| Feature | Meaning |
|---|---|
| LIMIT_BAL | Credit limit granted (NT$), individual + family |
| EDUCATION | 1 = graduate school, 2 = university, 3 = high school, 4 = others, 5/6 = unknown |
| MARRIAGE | 1 = married, 2 = single, 3 = others, 0 = unknown |
| AGE | Age in years |
| PAY_0 | Repayment status in the most recent month: -2/-1/0 = paid duly or no use, 1 = one month late, 2 = two months late, … |
| PAY_2 … PAY_5 | Repayment status in earlier months (same coding). These are recorded values, not a trend. |
| BILL_AMT1 … BILL_AMT6 | Bill statement amount for months 1 (most recent) … 6 (NT$) |
| PAY_AMT1 … PAY_AMT6 | Amount paid in months 1 … 6 (NT$) |
| pay_max | Worst (largest) repayment-status value across PAY_0 … PAY_5 |
| pay_sum_late | Number of months with a late repayment status |
| util_ratio | BILL_AMT1 / LIMIT_BAL (how much of the limit is used) |
| pay_ratio | PAY_AMT1 / BILL_AMT1 (share of the latest bill that was paid) |

## Reading encoded names

- `EDUCATION_5.0` → the feature is "EDUCATION equals 5" (a yes/no indicator).
- `SEX` is **never** a model input; it is used only for auditing and is not shown.

## Terms

- **Risk score / raw log-odds margin**: the model's internal score before it is
  converted to a probability. Attribution numbers live on this scale.
- **Calibrated probability of default (PD)**: the model's probability estimate
  for the documented outcome. This is the only percentage the explanation may state.
"""


if __name__ == "__main__":
    main()

"""
Re-annotation pack R2: the E1 items whose label can depend on the restored
zeros (restore_evidence_v2.py), shown with v2 evidence.

Annotators labelled v1 evidence where every recorded zero appeared as
"PAY_0 = None". For items whose explanation touches repayment status, that can
change the correct label. They are re-labelled here against v2 evidence.

Also included, blinded and shuffled in:
  - anchors   : ~10% label-portable items. Their label must not move between
                rounds; disagreement measures annotator drift, not the restore.
  - duplicates: ~10% hidden repeats for intra-annotator consistency.

Output
  Submission/annotation_pack_r2.zip          -> send to annotators
  Submission/annotation_key_r2.jsonl         -> KEEP LOCAL. R2 id -> E1 id.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_annotation_pack as v1  # noqa: E402
from evidence import EvidenceItem, EvidenceObject, EvidenceType  # noqa: E402

SUB = v1.SUB
V2 = SUB / "annotation_pack_v2"
PACK = SUB / "annotation_pack_r2"
SEED = 20261002

README = """SAARTHI — E1 Re-annotation, round 2 (R2)
==========================================

Items to label: {n}

Why a second round
  In round 1 some repayment-status values were shown as "None" because of a
  software bug. The real recorded value was 0. The evidence in this round shows
  the correct values. Please label every item again from scratch, against the
  evidence shown here. Do not try to recall your round-1 answer.

  Reminder from the glossary: PAY_0 ... PAY_6 are repayment-status codes.
  -2, -1 and 0 all mean paid duly / no consumption / revolving credit used and
  paid as required; 1 = one month late, 2 = two months late, and so on.
  A value of 0 is a recorded value, not a missing one.

Files
  README.txt                      this file
  ANNOTATION_GUIDELINES.md        same rules as round 1 - reread section 2
  FEATURE_GLOSSARY.md             what each dataset column means
  R2_annotation_sheet.xlsx        the work: fill the yellow columns
  R2_items.jsonl                  same items, machine-readable (you do not need it)

Steps
  1. Reread ANNOTATION_GUIDELINES.md section 2 and the glossary entry for PAY_x.
  2. Fill the "Annotator" tab (your annotator code only).
  3. Label every row in the "Annotate" tab. Work independently.
  4. Return ONLY the spreadsheet, renamed R2_annotation_sheet_<annotator_id>.xlsx

Rules
  - Do not discuss items with other annotators until everyone has returned.
  - Do not use ChatGPT or any AI tool to help decide labels.
  - The material is for this study only. Do not share it.
"""


def eo_from_items(it: dict) -> EvidenceObject:
    eo = EvidenceObject(borrower_id="this borrower", dataset=it["dataset"])
    for d in it["evidence"]:
        d = dict(d)
        d["etype"] = EvidenceType(d["etype"])
        eo.add(EvidenceItem(**d))
    return eo


def main() -> None:
    rng = random.Random(SEED)
    items = {json.loads(l)["item_id"]: json.loads(l)
             for l in open(V2 / "E1_items_v2.jsonl") if l.strip()}
    sens = json.load(open(V2 / "label_sensitivity.json"))["items"]
    key = {json.loads(l)["item_id"]: json.loads(l)
           for l in open(SUB / "annotation_key_v1.jsonl") if l.strip()}

    # v1 duplicates are re-derived here; only originals enter R2
    orig = [i for i in items if "duplicate_of" not in key[i]]
    redo = sorted(i for i in orig if not sens[i]["label_portable"])
    portable = sorted(i for i in orig if sens[i]["label_portable"])
    anchors = rng.sample(portable, max(1, round(0.10 * len(redo))))
    base = [(i, "reannotate") for i in redo] + [(i, "anchor") for i in anchors]
    dups = rng.sample(range(len(base)), max(1, round(0.10 * len(base))))
    rows = base + [(base[j][0], "duplicate") for j in dups]
    rng.shuffle(rows)

    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    sheet, jsonl, keyrows, first = [], [], [], {}
    for pos, (e1, role) in enumerate(rows, 1):
        rid = f"R2-{pos:04d}"
        it = items[e1]
        sheet.append({"item_id": rid, "evidence": v1.render_evidence(eo_from_items(it)),
                      "explanation": it["explanation"]})
        jsonl.append({"item_id": rid, "dataset": it["dataset"],
                      "evidence": it["evidence"], "explanation": it["explanation"]})
        k = {"item_id": rid, "e1_item_id": e1, "role": role}
        if role != "duplicate":
            first[e1] = rid
        keyrows.append(k)
    # a duplicate can be shuffled ahead of its original: link after the pass
    for k in keyrows:
        if k["role"] == "duplicate":
            k["duplicate_of"] = first[k["e1_item_id"]]

    v1._write_xlsx(PACK / "R2_annotation_sheet.xlsx", sheet)
    (PACK / "R2_items.jsonl").write_text("\n".join(json.dumps(x) for x in jsonl) + "\n")
    (PACK / "ANNOTATION_GUIDELINES.md").write_text(v1.GUIDE)
    (PACK / "FEATURE_GLOSSARY.md").write_text(v1.GLOSSARY)
    (PACK / "README.txt").write_text(README.format(n=len(sheet)))
    (SUB / "annotation_key_r2.jsonl").write_text("\n".join(json.dumps(x) for x in keyrows) + "\n")

    zp = SUB / "annotation_pack_r2.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PACK.iterdir()):
            z.write(f, f"annotation_pack_r2/{f.name}")
    c = {r: sum(1 for k in keyrows if k["role"] == r) for r in ("reannotate", "anchor", "duplicate")}
    print(f"items={len(sheet)} {c}\nzip -> {zp}\nkey -> {SUB/'annotation_key_r2.jsonl'} (DO NOT SEND)")


if __name__ == "__main__":
    main()

"""
Reader parse-accuracy check (Improvement_Plan WS1.5): a human checks 150 structured readings that the independent
reader produced for disputed claims on the (development) locked-v1 set, stratified by operation. For each reading
the checker sees the evidence, the sentence and the reader's reading, and marks whether the field binding, the
operation, the asserted value/direction and the polarity are correct. Measures how often the reader's binding is
wrong, separately from any verdict.

    python build_parse_check_pack.py
Outputs: Submission/packs/Reader_parse_check_pack.zip; Submission/annotation_key_parse_check.jsonl (LOCAL)
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pack_m1 as PM  # noqa: E402
import feature_semantics as FS  # noqa: E402
from build_pack_m1 import SUB, assert_blind, blind, eo_from  # noqa: E402
from evidence_builder import render_evidence_v2  # noqa: E402

SEED = 20261019
README = """Reader parse check
==================

Each row shows a borrower's evidence, ONE sentence from an explanation, and how a computer program READ that
sentence: which evidence item(s) it thinks the words refer to, what kind of statement it is, what value or
direction the sentence asserts, and whether the sentence asserts or denies it. You do NOT judge whether the
sentence is true. You judge whether the program READ it correctly.

Fill the yellow columns with Y or N:
  BINDING_OK     the evidence id(s) are the items the quoted words refer to
  OPERATION_OK   the kind of statement is right (VALUE = a number for a field; DIRECTION = raises/lowers the risk
                 score; RANK = strongest driver; SCALE = log-odds shown as % or probability; CODE = meaning of a code;
                 TARGET = what is predicted or its horizon; TEMPORAL = change over time; NONE = says nothing about
                 the borrower; OTHER = anything else)
  ASSERTED_OK    the asserted value / direction / rank matches what the quoted words say
  POLARITY_OK    'asserted' vs 'negated' matches the words (watch for 'not', 'no', 'does not')
  NOTES          optional
Do not use any AI tool. Work alone.
"""


def main() -> int:
    rng = random.Random(SEED)
    recs = {json.loads(l)["item_id"]: json.loads(l) for l in open(SUB / "locked" / "fresh_narratives_v1.jsonl")}
    pool = defaultdict(list)
    for f in ("exec_R11__qwen3_next.jsonl", "exec_R11__gptoss120b.jsonl"):
        for l in open(SUB / "locked" / "fresh_narratives_v1_runs_v16" / f):
            r = json.loads(l)
            rr = ((r.get("detail") or {}).get("claim_status") or {}).get("_reinterpretation") or {}
            texts = {c["claim_id"]: c["text"] for c in ((r.get("detail") or {}).get("execution") or {}).get("claims", [])}
            for cid, c in (rr.get("claims") or {}).items():
                for p in c.get("propositions", []):
                    pool[p.get("operation") or "OTHER"].append((r["item_id"], texts.get(cid, ""), p, rr.get("family")))
    quota = {"DIRECTION": 40, "VALUE": 35, "RANK": 25, "NONE": 15, "OTHER": 10, "CODE": 8, "SCALE": 5, "TARGET": 5, "TEMPORAL": 2}
    pick = []
    for op, n in quota.items():
        pick += rng.sample(pool[op], min(n, len(pool[op])))
    rng.shuffle(pick)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "Check"
    ws.append(["item_id", "dataset", "evidence", "sentence", "quoted_words", "reading_operation", "reading_evidence_ids",
               "reading_asserted", "reading_polarity", "BINDING_OK", "OPERATION_OK", "ASSERTED_OK", "POLARITY_OK", "NOTES"])
    key = []
    for n, (iid, sent, p, fam) in enumerate(pick, 1):
        rec = recs[iid]
        eo = eo_from(rec["evidence"])
        sid = f"PC-{n:03d}"
        ws.append([sid, PM.DATASET_LABEL[rec["dataset"]], render_evidence_v2(eo), blind(sent, eo), blind(str(p.get("quote", "")), eo),
                   p.get("operation"), ", ".join(map(str, p.get("evidence_ids") or [])), str(p.get("asserted", "")),
                   p.get("polarity"), "", "", "", "", ""])
        key.append({"item_id": sid, "source_item_id": iid, "borrower_id": rec["borrower_id"], "reader": fam,
                    "operation": p.get("operation"), "exec_status": p.get("status")})
    hdr, fill = PatternFill("solid", fgColor="1F2937"), PatternFill("solid", fgColor="FEF3C7")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), hdr
    for col, w in zip("ABCDEFGHIJKLMN", (9, 12, 70, 50, 30, 12, 12, 16, 10, 10, 10, 10, 10, 25)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[9:]:
            c.fill = fill
    for row in ws.iter_rows(min_row=2, values_only=True):
        assert_blind(" ".join(str(v) for v in row if v), {k["borrower_id"] for k in key})
    meta = wb.create_sheet("Checker")
    meta.append(["checker_code", ""]); meta.append(["used any AI tool? (must be N)", ""])
    stage = SUB / "packs" / "Reader_parse_check_pack"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    wb.save(stage / "READER_parse_check_sheet.xlsx")
    (stage / "README.txt").write_text(README)
    (stage / "FEATURE_GLOSSARY.md").write_text(FS.glossary_markdown())
    z = SUB / "packs" / "Reader_parse_check_pack.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(stage.iterdir()):
            zf.write(f, f"Reader_parse_check_pack/{f.name}")
    (SUB / "annotation_key_parse_check.jsonl").write_text("\n".join(json.dumps(k) for k in key) + "\n")
    from collections import Counter
    print(z, len(key), dict(Counter(k["operation"] for k in key)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Development sets for the ReCheck 3 repair (Improvement_Plan WS1): the 299 adjudicated M1 narratives and the 400
locked-v1 narratives (labelled 9 Oct; development from now on). Labels only; no annotator information is read.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_m1 import M1, SUB, final_labels, read_sheet, yn  # noqa: E402


def m1_items():
    latest = {}
    for line in open(M1 / "records.jsonl"):
        r = json.loads(line)
        latest[(r["borrower_id"], r["arm"], r["family"])] = r
    out = []
    for i, (k, v, _b) in final_labels().items():
        rec = latest[(k["borrower_id"], k["arm"], k["family"])]
        out.append({"item_id": i, "set": "m1", "truth": v, "rec": rec, "dataset": rec["dataset"],
                    "group": k.get("group"), "borrower_id": k["borrower_id"], "family": rec["family"]})
    return out


def locked_items():
    key = [json.loads(l) for l in open(SUB / "annotation_key_locked_v1.jsonl")][1:]
    D = SUB / "Annotated" / "IDBI_Hack_Annotation"
    lab = {q: read_sheet(sorted(D.rglob(f"LOCKED_annotation_sheet_{q}*.xlsx"))[-1])["rows"] for q in ("AS", "AB", "NK", "SP")}
    adj = {r["item_id"]: r["ADJUDICATED_ANY"] == "Y" for r in csv.DictReader(open(SUB / "locked" / "adjudication_locked.csv"))}
    recs = {json.loads(l)["item_id"]: json.loads(l) for l in open(SUB / "locked" / "fresh_natural_v1.jsonl")}
    out = []
    for k in key:
        if k.get("duplicate_of") or k["arm"] != "grounded":
            continue
        rows = [lab[q][k["item_id"]] for q in k["raters"]]
        t = adj.get(k["item_id"], yn(rows[0]["ANY_MATERIAL_ERROR"]))
        rec = recs[k["source_item_id"]]
        out.append({"item_id": k["source_item_id"], "set": "locked", "truth": t, "rec": rec, "dataset": rec["dataset"],
                    "group": rec["group"], "borrower_id": rec["borrower_id"], "family": rec["family"]})
    return out


def all_dev():
    return m1_items() + locked_items()

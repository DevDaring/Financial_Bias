"""
Score the returned round-4 sheets (only the yellow answer columns; the Declaration tab is read only for the
no-AI-tool answer).

Task 1 -> Submission/locked/adjudication_fresh_v2_independent.csv (item_id, ADJUDICATED_ANY, basis), the format
          analyze_locked.py reads; then the prespecified comparison is re-scored with it:
              python analyze_locked.py --fresh-v2 --primary exec_P --baseline facts_judge \
                  --labels-dir ../../Submission/packs/Annotated/IDBI_Annotator_Checks \
                  --adjudication ../../Submission/locked/adjudication_fresh_v2_independent.csv
Task 2 -> Submission/mechanism/paraphrase_check.json (meaning preserved, per rewriter x style) and the list of
          rewrites a person judged to change the meaning (excluded in analyze_paraphrase.py when present)

    python score_round4.py --returned <folder with the returned sheets>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SUB = Path(__file__).resolve().parents[2] / "Submission"


def yn(x):
    s = str(x or "").strip().upper()
    return True if s.startswith("Y") else False if s.startswith("N") else None


def rows_of(path, sheet):
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True)[sheet]
    it = ws.iter_rows(values_only=True)
    head = list(next(it))
    return [dict(zip(head, r)) for r in it if r and r[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned", required=True)
    a = ap.parse_args()
    key = [json.loads(l) for l in open(SUB / "annotation_key_round4.jsonl")]
    k1 = {k["item_id"]: k for k in key if k["task"] == 1}
    k2 = {k["item_id"]: k for k in key if k["task"] == 2}
    root = Path(a.returned)
    adj = sorted(root.rglob("ADJUDICATION_sheet*.xlsx"))
    if adj:
        rows = rows_of(adj[-1], "Adjudicate")
        out = SUB / "locked" / "adjudication_fresh_v2_independent.csv"
        n = Counter()
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "ADJUDICATED_ANY", "basis"])
            for r in rows:
                v = yn(r.get("FINAL_MATERIAL_ERROR"))
                if r["item_id"] not in k1 or v is None or yn(r.get("CANNOT_DECIDE")):
                    n["undecided"] += 1
                    continue
                w.writerow([k1[r["item_id"]]["fresh_item_id"], "Y" if v else "N", "independent adjudicator (protocol)"])
                n["Y" if v else "N"] += 1
        print(out, dict(n), "of", len(k1))
    chk = sorted(root.rglob("PARAPHRASE_check_sheet*.xlsx"))
    if chk:
        rows = rows_of(chk[-1], "Check")
        res, changed = {}, []
        for r in rows:
            k = k2.get(r["item_id"])
            v = yn(r.get("SAME_MEANING"))
            if not k or v is None:
                continue
            c = res.setdefault(f"{k['rewriter']}|{k['style']}", Counter())
            c["same" if v else "changed"] += 1
            if not v:
                changed.append({"paraphrase_item_id": k["paraphrase_item_id"], "what": r.get("WHAT_CHANGED")})
        out = SUB / "mechanism" / "paraphrase_check.json"
        out.write_text(json.dumps({"by_cell": {k: dict(v) for k, v in res.items()}, "changed": changed}, indent=1))
        print(out, {k: dict(v) for k, v in res.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

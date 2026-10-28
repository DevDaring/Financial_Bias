"""
Import the author's returned adjudication workbook (build_adjudication_pack.py).

  sheet A_Disagreed_66 -> Submission/m1/adjudication_m1.csv  (ADJUDICATED_ANY, adjudicated_types, adjudication_note)
  sheet B_Contract_7   -> Submission/m1/contract_conflict_decisions.json

Validates every decision (Y / N / UNRESOLVED; error / correct / unresolved; known error types)
and refuses to write anything if a row is missing, reordered or invalid.

    .venv/bin/python import_adjudication.py path/to/M1_adjudication_workbook.xlsx
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_adjudication_pack import TYPES  # noqa: E402
from score_m1 import M1  # noqa: E402


def main(path: str) -> int:
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    adj_rows = list(csv.DictReader(open(M1 / "adjudication_m1.csv")))
    fields = list(adj_rows[0].keys())
    ws = wb["A_Disagreed_66"]
    hdr = [c.value for c in ws[1]]
    got = {r[0]: dict(zip(hdr, r)) for r in ws.iter_rows(min_row=2, values_only=True) if r[0]}
    problems = []
    if set(got) != {r["item_id"] for r in adj_rows}:
        problems.append("sheet A item ids differ from adjudication_m1.csv")
    for i, r in got.items():
        d = str(r.get("ADJUDICATED_ANY") or "").strip().upper()
        if d not in ("Y", "N", "UNRESOLVED"):
            problems.append(f"{i}: ADJUDICATED_ANY must be Y/N/UNRESOLVED, got {d!r}")
        ts = [t.strip() for t in str(r.get("adjudicated_types") or "").split(";") if t.strip()]
        if d == "Y" and not ts:
            problems.append(f"{i}: Y needs at least one error type")
        bad = [t for t in ts if t not in TYPES]
        if bad:
            problems.append(f"{i}: unknown error types {bad}")
        if d == "UNRESOLVED" and not str(r.get("adjudication_note") or "").strip():
            problems.append(f"{i}: UNRESOLVED needs a note")
    wsb = wb["B_Contract_7"]
    hb = [c.value for c in wsb[1]]
    conf = {r[0]: dict(zip(hb, r)) for r in wsb.iter_rows(min_row=2, values_only=True) if r[0]}
    for i, r in conf.items():
        d = str(r.get("AUTHOR_DECISION") or "").strip().lower()
        if d not in ("error", "correct", "unresolved"):
            problems.append(f"{i}: AUTHOR_DECISION must be error/correct/unresolved, got {d!r}")
        if d == "correct" and not str(r.get("rule_note") or "").strip():
            problems.append(f"{i}: 'correct' needs the rule you applied (rule_note)")
    if problems:
        print("NOT IMPORTED:\n  " + "\n  ".join(problems))
        return 1
    for r in adj_rows:
        g = got[r["item_id"]]
        r["ADJUDICATED_ANY"] = str(g["ADJUDICATED_ANY"]).strip().upper()
        r["adjudicated_types"] = ";".join(t.strip() for t in str(g.get("adjudicated_types") or "").split(";") if t.strip())
        r["adjudication_note"] = str(g.get("adjudication_note") or "").strip()
    with open(M1 / "adjudication_m1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(adj_rows)
    (M1 / "contract_conflict_decisions.json").write_text(json.dumps(
        {i: {"decision": str(r["AUTHOR_DECISION"]).strip().lower(), "rule_note": str(r.get("rule_note") or "").strip()}
         for i, r in conf.items()}, indent=1))
    print(f"imported {len(got)} adjudications and {len(conf)} contract-conflict decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

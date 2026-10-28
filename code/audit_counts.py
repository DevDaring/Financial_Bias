"""
Denominator audit (Future_Plan §B, P0): logs, unique outputs and borrowers are different
units. Prints and saves every count a table may use, so no number is reported against
the wrong denominator.

    python audit_counts.py   -> Submission/m1/denominators.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_m1 import M1, SUB, agreed_natural  # noqa: E402


def main() -> int:
    rows = [json.loads(l) for l in open(M1 / "records.jsonl")]
    latest = {}
    for r in rows:
        latest[(r["dataset"], r["borrower_id"], r["arm"], r["family"])] = r
    ok = [r for r in latest.values() if not r.get("error") and (r.get("text") or "").strip()]
    borrowers = {(r["dataset"], r["borrower_id"]) for r in latest.values()}
    key = [json.loads(l) for l in open(SUB / "annotation_key_m1.jsonl")]
    key = [k for k in key if not k.get("_meta")]
    agreed = agreed_natural()
    ag_b = Counter(k["borrower_id"] for k, _ in agreed.values())
    import csv
    adj = list(csv.DictReader(open(M1 / "adjudication_m1.csv")))
    out = {
        "records_log_rows": len(rows),
        "unique_dataset_borrower_arm_producer": len(latest),
        "unique_successful_outputs": len(ok),
        "unique_borrowers": len(borrowers),
        "borrowers_by_dataset": dict(Counter(d for d, _ in borrowers)),
        "outputs_by_arm": dict(Counter(r["arm"] for r in ok)),
        "outputs_by_producer": dict(Counter(r["family"] for r in ok)),
        "annotation_items": len(key),
        "annotation_items_by_source": dict(Counter(k.get("source") for k in key)),
        "primary_natural_items_double_labelled": sum(1 for k in key if k.get("source") == "natural"
                                                     and not k.get("duplicate_of") and len(k.get("raters", [])) == 2),
        "agreed_natural_items": len(agreed),
        "agreed_errors": sum(1 for _, v in agreed.values() if v),
        "agreed_borrowers": len(ag_b),
        "agreed_items_per_borrower": {"min": min(ag_b.values()), "max": max(ag_b.values()),
                                      "mean": round(sum(ag_b.values()) / len(ag_b), 2)},
        "agreed_by_dataset_group": dict(Counter(f"{k['dataset']}|{k.get('group')}" for k, _ in agreed.values())),
        "agreed_borrowers_by_dataset_group": dict(Counter(g for g, _ in {(f"{k['dataset']}|{k.get('group')}",
                                                                          k["borrower_id"]) for k, _ in agreed.values()})),
        "adjudication_items": len({r.get("item_id") for r in adj}),
        "adjudication_items_decided": len({r.get("item_id") for r in adj
                                           if str(r.get("ADJUDICATED_ANY") or "").strip()}),
        "note": "Items from one borrower are dependent (same evidence, up to 9 outputs). Binomial or Fisher tests "
                "on items treat them as independent and are invalid for borrower-level claims; use borrower-cluster "
                "resampling (replay_objections.py) or one output per borrower.",
    }
    (M1 / "denominators.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

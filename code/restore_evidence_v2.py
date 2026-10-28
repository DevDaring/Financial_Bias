"""
Evidence v2 for the E1 items: restore the zeros that evidence/1.0 dropped.

evidence/1.0 serialised items with `v not in (None, {}, [], False)`. Because
0 == False in Python, every recorded zero vanished. On the 184 E1 items this
removed 362 observed-fact values (PAY_0..PAY_3 only), and annotators saw
"PAY_0 = None". This script:

1. Restores each missing value from the raw Taiwan record and checks every
   value that WAS present against the same record (any mismatch aborts).
2. Splits items by whether the human label can depend on the restore.
   The producers wrote, the annotators labelled, and the judges were given the
   v1 evidence. A label is only portable to v2 evidence when the item lost no
   value, or its explanation never touches a restored field or repayment
   status. Everything else needs re-annotation against v2 evidence.

Outputs (local, under Submission/; never shipped with the key):
    annotation_pack_v2/E1_items_v2.jsonl   items with restored evidence
    annotation_pack_v2/label_sensitivity.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "saarthi" / "training"))

from run_pilot import load_corpus  # noqa: E402
from score_e1 import load_items, load_key  # noqa: E402

SUB = Path(__file__).resolve().parents[2] / "Submission"
OUT = SUB / "annotation_pack_v2"

#: Text that can depend on a repayment-status value. Deliberately broad: a
#: false "sensitive" only costs a re-annotation, a false "invariant" would
#: carry a possibly wrong label into qualification.
SENSITIVE = re.compile(
    r"\bPAY_[0-6]\b|repay|payment (status|history|record)|delinq|\blate\b|"
    r"overdue|arrear|on[- ]time|duly|missing|unavailable|not available|"
    r"not recorded|not provided|no data|\bnone\b|\bnull\b|unknown",
    re.I)


def main() -> int:
    key, items = load_key(), load_items()
    X, _, _, _ = load_corpus("taiwan")
    OUT.mkdir(exist_ok=True)

    restored_total, checked, sens = 0, 0, {}
    out_lines = []
    for iid, it in items.items():
        bid = key[iid]["borrower_id"]
        idx = int(bid.rsplit("-", 1)[1])
        row = X.iloc[idx]
        restored = []
        ev = []
        for e in it["evidence"]:
            e = dict(e)
            if e["etype"] == "observed_fact":
                raw = row[e["source_field"]]
                raw = float(raw) if isinstance(raw, (int, float, np.number)) else str(raw)
                if "value" not in e:
                    if raw != 0:
                        raise SystemExit(f"{iid} {e['source_field']}: dropped value "
                                         f"is {raw}, expected 0 - abort")
                    e["value"] = raw
                    restored.append(e["source_field"])
                else:
                    checked += 1
                    if e["value"] != raw:
                        raise SystemExit(f"{iid} {e['source_field']}: pack has "
                                         f"{e['value']}, raw has {raw} - abort")
            ev.append(e)
        restored_total += len(restored)
        hits = sorted({m.group(0).lower() for m in SENSITIVE.finditer(it["explanation"])})
        sens[iid] = {"restored_fields": restored,
                     "text_hits": hits,
                     "label_portable": (not restored) or (not hits)}
        out_lines.append(json.dumps({**it, "evidence": ev,
                                     "evidence_schema": "evidence/1.1"}))

    (OUT / "E1_items_v2.jsonl").write_text("\n".join(out_lines) + "\n")
    port = [i for i, s in sens.items() if s["label_portable"]]
    nat = {i for i, k in key.items() if k["source"] == "natural" and "duplicate_of" not in k}
    summary = {
        "items": len(items),
        "items_with_restored_values": sum(1 for s in sens.values() if s["restored_fields"]),
        "values_restored": restored_total,
        "present_values_checked_against_raw": checked,
        "label_portable_items": len(port),
        "label_portable_natural": len([i for i in port if i in nat]),
        "needs_reannotation": sorted(i for i, s in sens.items() if not s["label_portable"]),
    }
    (OUT / "label_sensitivity.json").write_text(
        json.dumps({"summary": summary, "items": sens}, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "needs_reannotation"}, indent=2))
    print("needs re-annotation:", len(summary["needs_reannotation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

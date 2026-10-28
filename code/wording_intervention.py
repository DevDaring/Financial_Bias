"""
Paired wording intervention (Future_Plan review §7; CPU only, no API calls).

Question: does the WORDING of a field reference, with evidence and substantive claims held fixed, change the
code layer's decision, and does a binding policy that helps one wording let wrong numbers through?

Cases: every CORRECT development narrative (adjudicated labels) that names at least one evidence field by its
documented source name (e.g. EXT_SOURCE_2, PAY_AMT1). Versions of the same narrative:
  source     as written
  alias      every source-name mention replaced by a documented plain-language alias of the SAME field
             (from the data dictionary, so meaning is preserved by construction)
  alias_bad  the alias version with ONE number attached to a renamed field replaced by the value of a
             different field in the same evidence (a binding error; must not be accepted)
Binding policies, crossed with the versions on identical cases:
  1.3        names and aliases only (family binding disabled)
  1.4        + numbered-family stems with value disambiguation (the current checker)
Outcome per version x policy: code-layer decision (auto-accept, reject, unresolved -> needs the LLM).
Withholding of correct text = not auto-accepted; error escape = alias_bad auto-accepted.
Reported per dataset x group with borrower-cluster bootstrap intervals for the paired alias-source difference.

    python wording_intervention.py   -> Submission/m1/wording_intervention.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as V  # noqa: E402
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from evidence import EvidenceType  # noqa: E402
from replay_same_transcript import records  # noqa: E402
from score_m1 import M1, final_labels  # noqa: E402

_ORIG_SECONDARY = W._secondary_mentions


def decide(text, eo, policy):
    W._secondary_mentions = _ORIG_SECONDARY if policy == "1.4" else (lambda span, eo, primary: [])
    try:
        tx = W.execute_text(W.clean(text), eo)
    finally:
        W._secondary_mentions = _ORIG_SECONDARY
    if tx.contradictions:
        return "reject"
    if not tx.residual_claims and all(o.status == W.SUPPORT for o in tx.text_ops):
        return "accept"
    return "unresolved"


def alias_of(it):
    al = sorted(a for a in (it.aliases or []) if " " in a and not re.search(r"_|\d{4}", a))
    return al[0] if al else None


def rewrite(text, eo):
    """Replace source names by aliases; return (new text, list of (alias, field eid, end offset))."""
    subs = []
    for it in eo.items:
        if it.etype not in (EvidenceType.MODEL_ATTRIBUTION, EvidenceType.OBSERVED_FACT):
            continue
        src = (it.feature_group or it.source_field or "").replace(" (derived)", "")
        al = alias_of(it)
        if not src or not al or "_" not in src:
            continue
        subs.append((src, al, it.eid))
    out, changed = text, []
    for src, al, eid in sorted(subs, key=lambda x: -len(x[0])):
        if re.search(rf"(?<![\w]){re.escape(src)}(?![\w])", out):
            out = re.sub(rf"(?<![\w]){re.escape(src)}(?![\w])", al, out)
            changed.append((al, eid))
    return out, changed


def corrupt(text, eo, changed, rng):
    """Replace the first number after a renamed field with another field's value."""
    vals = []
    for it in eo.by_type(EvidenceType.OBSERVED_FACT):
        if isinstance(it.value, (int, float)) and not isinstance(it.value, bool):
            vals.append((it.eid, float(it.value)))
    for al, eid in changed:
        m = re.search(re.escape(al) + r"[^.;]{0,40}?(-?\d[\d,]*(?:\.\d+)?)", text)
        if not m:
            continue
        own = float(m.group(1).replace(",", ""))
        others = [v for e, v in vals if abs(v - own) > max(0.05 * abs(own), 1e-6) and e != eid]
        if not others:
            continue
        nv = rng.choice(sorted(others))
        s = f"{nv:,.0f}" if abs(nv) >= 1000 and float(nv).is_integer() else (f"{nv:g}")
        return text[:m.start(1)] + s + text[m.end(1):]
    return None


def norm_group(g):
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(g or "").lower(), "other")


def main() -> int:
    rng = random.Random(20261007)
    fl, recs = final_labels(), records()
    cases = []
    for i, (k, truth, _b) in fl.items():
        if truth:
            continue
        rec = recs[(k["borrower_id"], k["arm"], k["family"])]
        if not (rec.get("text") or "").strip() or rec["arm"] == "template":
            continue
        eo = eo_from(rec["evidence"])
        alt, changed = rewrite(rec["text"], eo)
        if not changed:
            continue
        bad = corrupt(alt, eo, changed, rng)
        row = {"item_id": i, "borrower_id": k["borrower_id"], "dataset": k["dataset"],
               "group": norm_group(k.get("group")), "n_renamed": len(changed)}
        for pol in ("1.3", "1.4"):
            row[f"source|{pol}"] = decide(rec["text"], eo, pol)
            row[f"alias|{pol}"] = decide(alt, eo, pol)
            row[f"alias_bad|{pol}"] = decide(bad, eo, pol) if bad else None
        cases.append(row)

    def summary(rows):
        out = {"n": len(rows)}
        for v in ("source", "alias"):
            for pol in ("1.3", "1.4"):
                out[f"{v}|{pol}|withheld"] = round(sum(r[f"{v}|{pol}"] != "accept" for r in rows) / max(1, len(rows)), 4)
                out[f"{v}|{pol}|rejected"] = sum(r[f"{v}|{pol}"] == "reject" for r in rows)
        bad = [r for r in rows if r["alias_bad|1.4"] is not None]
        out["n_bad"] = len(bad)
        for pol in ("1.3", "1.4"):
            out[f"alias_bad|{pol}|accepted"] = sum(r[f"alias_bad|{pol}"] == "accept" for r in bad)
            out[f"alias_bad|{pol}|rejected"] = sum(r[f"alias_bad|{pol}"] == "reject" for r in bad)
        return out

    by = defaultdict(list)
    for r in cases:
        by[f"{r['dataset']}|{r['group']}"].append(r)
    # borrower-cluster bootstrap of the paired wording effect (alias - source withheld) under each policy
    B = 2000
    boot = {}
    for pol in ("1.3", "1.4"):
        for ds in ("taiwan", "home_credit"):
            rows = [r for r in cases if r["dataset"] == ds]
            clus = defaultdict(list)
            for r in rows:
                clus[r["borrower_id"]].append(r)
            cl = list(clus.values())
            eff = lambda rs: (sum((r[f"alias|{pol}"] != "accept") - (r[f"source|{pol}"] != "accept") for r in rs)  # noqa: E731
                              / max(1, len(rs)))
            bs = sorted(eff([r for c in (rng.choice(cl) for _ in cl) for r in c]) for _ in range(B))
            boot[f"{ds}|{pol}"] = {"alias_minus_source_withheld": round(eff(rows), 4),
                                   "ci95": [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)], "n": len(rows)}
    res = {"note": "DEVELOPMENT narratives, CPU replay of the code layer only (no LLM). Correct narratives that "
                   "name a field by its documented source name; alias = documented plain-language alias.",
           "overall": summary(cases), "by_stratum": {k: summary(v) for k, v in sorted(by.items())},
           "paired_effect_bootstrap": boot, "cases": cases}
    (M1 / "wording_intervention.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

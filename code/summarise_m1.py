"""
Pre-annotation summary of the milestone-1 run (no labels yet: these are
verifier SIGNALS, not error rates).

Re-scores every candidate with the current structural checker version (the
run stored the version it used), then reports acceptance by producer x arm,
by judge, judge-judge agreement, failures, route usage and token cost.
Writes Submission/m1/m1_summary.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack_m1 import eo_from  # noqa: E402
from generate import Candidate  # noqa: E402
from verify import STRUCTURAL_VERSION, structural_check  # noqa: E402

M1 = Path(__file__).resolve().parents[2] / "Submission" / "m1"


def main() -> None:
    latest = {}
    for l in open(M1 / "records.jsonl"):          # append-only: last row per item
        r = json.loads(l)
        latest[(r["borrower_id"], r["arm"], r["family"])] = r
    rows = list(latest.values())
    ok = [r for r in rows if not r.get("error") and (r.get("text") or "").strip()]
    for r in ok:
        v = structural_check(Candidate(borrower_id=r["borrower_id"], arm=r["arm"], text=r["text"],
                                       claims=r.get("claims") or [], family=r["family"]),
                             eo_from(r["evidence"]))
        r["structural_now"] = v.to_dict()
    cells = defaultdict(lambda: {"n": 0, "struct": 0, **{}})
    judges = sorted({j for r in ok for j in r["judges"]})
    out = {"structural_version": STRUCTURAL_VERSION, "records": len(rows), "ok": len(ok),
           "failed": Counter((r["family"], (r.get("error") or "")[:60]) for r in rows if r.get("error")),
           "by_cell": {}, "by_judge": {}, "judge_agreement": {}, "routes": Counter(),
           "structural_violation_types": Counter()}
    for r in ok:
        k = f"{r['dataset']}|{r['family']}|{r['arm']}"
        c = cells[k]
        c["n"] += 1
        c["struct"] += r["structural_now"]["accepted"]
        for j in judges:
            jv = r["judges"].get(j, {})
            c[f"{j}_acc"] = c.get(f"{j}_acc", 0) + bool(jv.get("accepted"))
            c[f"{j}_unres"] = c.get(f"{j}_unres", 0) + bool(jv.get("error"))
        out["routes"][f"{r['provider']}#{r['route_index']}"] += 1
        for x in r["structural_now"]["violations"]:
            out["structural_violation_types"][x["type"]] += 1
    out["by_cell"] = {k: {kk: (round(vv / v["n"], 3) if kk != "n" else vv) for kk, vv in v.items()}
                      for k, v in sorted(cells.items())}
    for j in judges:
        vs = [r["judges"].get(j, {}) for r in ok]
        out["by_judge"][j] = {"accept_rate": round(sum(bool(v.get("accepted")) for v in vs) / len(vs), 3),
                              "unresolved": sum(bool(v.get("error")) for v in vs)}
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            both = [r for r in ok if not r["judges"].get(a, {}).get("error")
                    and not r["judges"].get(b, {}).get("error")]
            agree = sum(r["judges"][a]["accepted"] == r["judges"][b]["accepted"] for r in both)
            out["judge_agreement"][f"{a}~{b}"] = round(agree / len(both), 3) if both else None
    tmpl = [r for r in ok if r["family"] == "template"]
    out["template_sanity"] = {"n": len(tmpl), "structural_accept": sum(r["structural_now"]["accepted"] for r in tmpl),
                              **{f"{j}_accept": sum(bool(r["judges"].get(j, {}).get("accepted")) for r in tmpl)
                                 for j in judges}}
    out = json.loads(json.dumps(out, default=lambda o: {str(k): v for k, v in o.items()} if isinstance(o, Counter) else str(o)))
    (M1 / "m1_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("structural_version", "records", "ok", "failed", "by_judge",
                                          "judge_agreement", "template_sanity", "routes",
                                          "structural_violation_types")}, indent=1))
    print("\ncell (dataset|producer|arm): n, structural accept, judge accepts")
    for k, v in out["by_cell"].items():
        print(f"  {k:<36} n={v['n']:<3} struct={v['struct']:<6} "
              + " ".join(f"{j}={v.get(j + '_acc')}" for j in judges))


if __name__ == "__main__":
    main()

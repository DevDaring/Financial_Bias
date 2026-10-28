"""
Zero-cost development evaluation of verifier signals on the M1 natural items
that both annotators labelled the same way (pre-adjudication).

    python eval_m1_offline.py

Uses only cached candidates and cached judge verdicts: no API calls. These
are DEVELOPMENT estimates (the contract/1.1 detectors were designed after
reading these labels); confirmatory numbers come from the locked evaluation.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack_m1 import eo_from  # noqa: E402
from contract import contract_violations  # noqa: E402
from generate import Candidate  # noqa: E402
from score_m1 import M1, agreed_natural  # noqa: E402
from verify import structural_check  # noqa: E402


def rates(rows, accept_fn):
    c = Counter()
    for truth, sig in rows:
        acc = accept_fn(sig)
        if acc is None:
            c["unresolved"] += 1
            continue
        c[("err" if truth else "ok", "acc" if acc else "rej")] += 1
    ne = c[("err", "acc")] + c[("err", "rej")]
    nc = c[("ok", "acc")] + c[("ok", "rej")]
    return {"a_false_accept": f"{c[('err', 'acc')]}/{ne}", "b_false_reject": f"{c[('ok', 'rej')]}/{nc}",
            "a": round(c[("err", "acc")] / ne, 3) if ne else None,
            "b": round(c[("ok", "rej")] / nc, 3) if nc else None, "unresolved": c["unresolved"]}


def main() -> int:
    latest = {}
    for line in open(M1 / "records.jsonl"):
        r = json.loads(line)
        latest[(r["borrower_id"], r["arm"], r["family"])] = r
    rows, det_hits = [], defaultdict(Counter)
    for i, (k, truth) in agreed_natural().items():
        rec = latest.get((k["borrower_id"], k["arm"], k["family"]))
        if rec is None:
            continue
        eo = eo_from(rec["evidence"])
        cand = Candidate(borrower_id=rec["borrower_id"], arm=rec["arm"], text=rec["text"],
                         claims=rec.get("claims") or [], family=rec["family"])
        s = structural_check(cand, eo)
        cv = contract_violations(rec["text"], eo)
        for v in cv:
            det_hits[v["rule"]]["err" if truth else "ok"] += 1
        j = {n: (None if v.get("error") else bool(v.get("accepted"))) for n, v in (rec.get("judges") or {}).items()}
        rows.append((truth, {"struct": s.accepted, "contract": not cv, **j}))
    out = {"n_items": len(rows), "n_errors": sum(t for t, _ in rows),
           "detector_hits_on_errors_vs_clean": {k: dict(v) for k, v in det_hits.items()}, "verifiers": {}}
    V = out["verifiers"]
    V["structural/2.6"] = rates(rows, lambda s: s["struct"])
    V["contract/1.1 detectors only"] = rates(rows, lambda s: s["contract"])
    V["structural+contract (deterministic gate)"] = rates(rows, lambda s: s["struct"] and s["contract"])
    for j in ("gpt56_luna", "grok420", "sonnet5"):
        V[j] = rates(rows, lambda s, j=j: s.get(j))
        V[f"structural -> {j}"] = rates(rows, lambda s, j=j: (s["struct"] and s[j]) if s.get(j) is not None else None)
        V[f"structural+contract -> {j}"] = rates(
            rows, lambda s, j=j: (s["struct"] and s["contract"] and s[j]) if s.get(j) is not None else None)
    (M1 / "offline_eval_dev.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

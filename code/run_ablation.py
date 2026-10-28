"""
Post-hoc ablations of the frozen ReCheck 3 on the fresh v2 narratives (reviewer questions 1-2). The frozen sources
(witness.py, executable_verifier.py) are NOT edited: each variant wraps them at run time, so the default path and
the recorded SHA-256 hashes are unchanged. Every variant uses the frozen configuration of the fresh test
(whole-narrative contract, judge and reassessor GPT-OSS-120B).

Variants
  gate_off_FIELD_IDENTITY  zero-call gate without the field-identity rule (no field-confusion contradiction,
                           no R_FIELD_IDENTITY routing)
  gate_off_NEGATION_SCOPE  ... without negation-scope routing
  gate_off_CODE_SCOPE      ... without undocumented-code-meaning routing
  gate_off_TARGET          ... without horizon/outcome routing
  gate_off_ALL             no zero-call gate (code-only release as in checker 1.5)
  repeat                   the frozen system again (run-to-run noise of the judge)
  reassess2                up to two reassessments instead of one
  A1                       objections unchecked (any objection rejects), as exec_A1
  pt_audit                 the Pratama-Tseng deterministic audit (no model call)

    python run_ablation.py --variant gate_off_TARGET --live
Output: Submission/locked/fresh_v2_ablation/<variant>/<method>__gptoss120b.jsonl (append-only, resumable)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators as K  # noqa: E402
import executable_verifier as EV  # noqa: E402
import witness as W  # noqa: E402

SUB = Path(__file__).resolve().parents[2] / "Submission"
PACK = SUB / "locked" / "fresh_v2_narratives.jsonl"
OUT = SUB / "locked" / "fresh_v2_ablation"
JUDGE = "gptoss120b"
RULES = ("FIELD_IDENTITY", "NEGATION_SCOPE", "CODE_SCOPE", "TARGET")
FROZEN_GATE = W.gate


def gate_without(rules):
    def gate(cl, eo, f, ops):
        extra, res = FROZEN_GATE(cl, eo, f, ops)
        if "FIELD_IDENTITY" in rules:
            extra = [o for o in extra if not str(o.detail).startswith("field identity")]
        res = [r for r in res if r.replace("R_", "") not in rules]
        return extra, res
    return gate


def _exec_P2_full(cand, eo, family=None, *, experiment="", reassess_family=None, claim_level=False, **_):
    """Frozen policy P, then one further reassessment for a narrative still unresolved after the first."""
    v = EV.verify_executable(cand, eo, family, mode="P", reassess_family=reassess_family, experiment=experiment,
                             claim_level=claim_level)
    if v.decision == "unresolved" and v.reason.endswith("unresolved after reassessment") and reassess_family:
        text = W.clean(cand.text)
        tx = W.execute_text(text, eo)
        if claim_level:
            tx.text_ops = []
        exec_claims, how = EV.MODES["P"]
        last = [w for w in v.witnesses if w.get("reassessment")] or v.witnesses
        dismissed = [w for w in last if w.get("outcome") not in ("verified", "semantic_plausible")]
        note = ("Two previous reviews raised objections that code recomputed and did not confirm: "
                + "; ".join(f"{w.get('rule')} '{(w.get('quote') or '')[:60]}': {w.get('outcome')}"
                            + (f" ({(w.get('execution') or {}).get('detail', '')[:80]})" if w.get("execution") else "")
                            for w in dismissed)[:1500]
                + " Re-assess; object only with a witness whose recomputation shows the error.")
        msgs = [{"role": "system", "content": EV.EV_SYS},
                {"role": "user", "content": EV._payload(text, eo, tx, exec_claims, note=note)}]
        item = f"{cand.borrower_id}|{cand.arm}|{cand.family}"
        try:
            d, info, call, raw = EV._call(cand, reassess_family, msgs, experiment, item + "|reassess2", 2500)
            v.calls.append(call); v.parse.append(info); v.raw.append(raw[:4000])
            if d is not None:
                ws = EV.check_witnesses([w for w in EV.W.witnesses_from_json(d)
                                         if not (claim_level and w.rule == "R_OMISSION")], text, eo, tx, how)
                v.witnesses += [dict(w.to_dict(), reassessment=2) for w in ws]
                dec, why, st = EV._resolve(d, tx, ws, exec_claims)
                if dec:
                    v.decision, v.reason, v.claim_status = dec, why + "_on_reassessment2", st
                else:
                    v.reason += "; unresolved after reassessment 2"
        except Exception as e:  # noqa: BLE001
            v.reason += f"; reassessment 2 failed: {type(e).__name__}"
    return K.CVerdict("exec_P2", v.accepted, family, detail=v.to_dict(), calls=v.calls,
                      error=None if v.decision != "unresolved" else v.reason)


def code_decision(r):
    from build_pack_m1 import eo_from
    eo = eo_from(r["evidence"])
    tx = W.execute_text(W.clean(r["text"]), eo)
    if tx.contradictions:
        return "reject"
    if not tx.residual_claims and all(o.status == W.SUPPORT for o in tx.text_ops):
        return "release"
    return "judge"


def changed_items(rules) -> list:
    """Narratives whose code-layer decision differs from the frozen one when these gate rules are off."""
    import json
    pack = [json.loads(l) for l in open(PACK)]
    W.gate = FROZEN_GATE
    base = {r["item_id"]: code_decision(r) for r in pack}
    W.gate = gate_without(rules)
    out = [r["item_id"] for r in pack if code_decision(r) != base[r["item_id"]]]
    return out


def continue_reassess2(workers: int) -> int:
    """One more reassessment for each frozen exec_P record left unresolved after its first reassessment."""
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from build_pack_m1 import eo_from
    from generate import Candidate
    pack = {json.loads(l)["item_id"]: json.loads(l) for l in open(PACK)}
    frozen = [json.loads(l) for l in open(SUB / "locked" / "fresh_v2_runs" / f"exec_P__{JUDGE}.jsonl")]
    todo = [r for r in frozen if str(r["detail"].get("reason", "")).endswith("unresolved after reassessment")]
    out = OUT / "reassess2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"exec_P2__{JUDGE}.jsonl"
    done = {json.loads(l)["item_id"] for l in open(path)} if path.exists() else set()
    lock = threading.Lock()
    exec_claims, how = EV.MODES["P"]

    def work(r):
        if r["item_id"] in done:
            return
        p = pack[r["item_id"]]
        eo = eo_from(p["evidence"])
        cand = Candidate(borrower_id=p["borrower_id"], arm=p.get("arm", "passage"), text=p["text"], family=p["family"])
        text = W.clean(cand.text)
        tx = W.execute_text(text, eo)
        last = [w for w in r["detail"].get("witnesses", []) if w.get("reassessment")]
        note = ("Two previous reviews raised objections that code recomputed and did not confirm: "
                + "; ".join(f"{w.get('rule')} '{(w.get('quote') or '')[:60]}': {w.get('outcome')}"
                            + (f" ({(w.get('execution') or {}).get('detail', '')[:80]})" if w.get("execution") else "")
                            for w in last)[:1500]
                + " Re-assess; object only with a witness whose recomputation shows the error.")
        msgs = [{"role": "system", "content": EV.EV_SYS},
                {"role": "user", "content": EV._payload(text, eo, tx, exec_claims, note=note)}]
        row = dict(r, method="exec_P2")
        try:
            d, info, call, raw = EV._call(cand, JUDGE, msgs, "fresh_v2_ablation",
                                          f"{cand.borrower_id}|{cand.arm}|{cand.family}|reassess2", 2500)
            row["calls"] = (r.get("calls") or []) + [call]
            row["n_calls"] = len(row["calls"])
            dec = None
            if d is not None:
                ws = EV.check_witnesses([w for w in W.witnesses_from_json(d)], text, eo, tx, how)
                dec, why, _st = EV._resolve(d, tx, ws, exec_claims)
                row["detail"] = dict(r["detail"], reassess2=[w.to_dict() for w in ws])
            if dec:
                row["accepted"] = dec == "accept"
                row["detail"] = dict(row["detail"], decision=dec, reason=why + "_on_reassessment2")
            else:
                row["detail"] = dict(row["detail"], reason=r["detail"]["reason"] + "; unresolved after reassessment 2")
        except Exception as e:  # noqa: BLE001
            row["detail"] = dict(r["detail"], reason=r["detail"]["reason"] + f"; reassessment 2 failed: {type(e).__name__}")
        with lock, open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    print("reassess2 done", len(todo), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    method = "exec_P"
    if a.variant.startswith("gate_off_"):
        rules = set(RULES) if a.variant == "gate_off_ALL" else {a.variant.replace("gate_off_", "")}
        assert rules <= set(RULES), a.variant
        W.gate = gate_without(rules)
    elif a.variant == "reassess2":
        return continue_reassess2(a.workers) if a.live else 2
    elif a.variant == "A1":
        method = "exec_A1"
    elif a.variant == "pt_audit":
        method = "pt_audit"
    elif a.variant != "repeat":
        raise SystemExit(f"unknown variant {a.variant}")
    import run_context as RC
    sys.argv = ["run_context.py", "--pack", str(PACK), "--judge", JUDGE, "--methods", method,
                "--out", str(OUT / a.variant), "--workers", str(a.workers), "--full-text", "--reassess", JUDGE]
    if a.variant.startswith("gate_off_"):
        # exact counterfactual: only narratives whose code decision changes are re-run; the others keep their
        # frozen outcome in the analysis
        ids = changed_items(rules)
        (OUT / a.variant).mkdir(parents=True, exist_ok=True)
        sub = OUT / a.variant / "changed_items.txt"
        sub.write_text("\n".join(ids) + "\n")
        sys.argv += ["--subset", str(sub)]
    if a.live:
        sys.argv.append("--live")
    return RC.main()


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)

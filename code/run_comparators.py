"""
Run verification methods on the cached M1 candidates and score them against the
annotators' agreed labels (development). DRY RUN BY DEFAULT: no API calls.

    python run_comparators.py --methods pt_audit contract zandi_judge --judge qwen3_next
    python run_comparators.py ... --live --confirm-spend      # paid run, after the estimate

Method names: always_accept always_reject pt_audit generic_judge2 generic_tool_judge (B2)
generic_second_look zandi_judge mermigkis_verifier pbp_judge contract (B3: contract-aware
verifier/1.0) exec_P (proposed) exec_A1 exec_A2 exec_PX exec_A0 (ablations; --reassess FAMILY).
Output: Submission/m1/comparators/<method>__<judge>[__dry].jsonl (append-only, resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators as K  # noqa: E402
import fakes  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from contract_verifier import verify_contract  # noqa: E402
from generate import Candidate  # noqa: E402
from score_m1 import M1, agreed_natural, final_labels  # noqa: E402

OUT = M1 / "comparators"
DISPUTE_FAMILY = None      # --dispute-family (exec_R); None = executable_verifier.DISPUTE_FAMILY
#: expected LLM calls per item (exec_*: only items with residual claims reach the LLM;
#: 0.63 of the 245 agreed M1 items did in the code-only development run, +20% reassessment)
LLM_METHODS = {"generic_judge2": 1, "zandi_judge": 1, "mermigkis_verifier": 1, "pbp_judge": 2, "contract": 1.2,
               "generic_tool_judge": 1.8, "generic_second_look": 2, "exec_P": 0.8, "exec_A1": 0.8,
               "exec_A2": 0.8, "exec_PX": 1.2, "exec_D": 0.6, "exec_R": 0.8, "exec_R11": 0.8,
               "generic_rubric": 1, "facts_judge": 1, "forced_tool_judge": 2.2, "claimwise_judge": 1,
               "factool_judge": 2, "parse_exec": 1}
#: USD per 1M tokens (input, output). OpenRouter list prices read 6 Oct 2026 are used as a
#: proxy for Bedrock (recheck the Bedrock price page before a live run); grok420 is derived
#: from the M1 actual spend ($1.42 for 323 calls), the others from M1 actuals likewise.
PRICE = {"qwen3_next": (0.10, 1.10), "glm47": (0.60, 2.20), "gptoss120b": (0.037, 0.17),
         "gemma27": (0.08, 0.45), "grok420": (1.25, 2.50), "grok45": (0.50, 1.50), "sonnet5": (2.00, 10.0),
         "gpt56_luna": (0.20, 1.20), "sarvam_105b": (0.0, 0.0)}


def dry_responder(family, messages):
    sys_ = str(messages[0].get("content", ""))
    last = str(messages[-1].get("content", ""))
    if "code_status" in last:                           # executable verifier
        import re as _re
        ids = sorted(set(_re.findall(r'"claim_id": "(c\d+)"', last)), key=lambda x: int(x[1:]))
        return json.dumps({"claims": [{"claim_id": c, "status": "supported"} for c in ids], "omissions": []})
    if sys_.startswith("You read sentences"):                    # dispute reinterpretation
        import re as _re
        pids = _re.findall(r'"pid": "([^"]+)"', last)
        return json.dumps({"passages": [{"pid": p, "propositions": []} for p in pids], "coverage": []})
    if "MUST first request tool" in sys_ and "Tool results" not in last and "tool_calls" not in last:
        return json.dumps({"tool_calls": [{"tool": "strongest", "args": {}}]})
    if sys_.startswith("You are given a credit-risk explanation"):          # FacTool extraction
        return json.dumps({"claims": [{"claim": "dry", "queries": ["p1"]}]})
    if '"factuality"' in sys_:
        return json.dumps({"claims": [{"claim": "dry", "reasoning": "", "error": "none", "factuality": True}],
                           "omissions": []})
    if '"claims":[{"claim_id":"c1","status":"supported|contradicted|insufficient","rule"' in sys_:
        n = last.count('"claim_id"')
        return json.dumps({"claims": [{"claim_id": f"c{k}", "status": "supported"} for k in range(1, n + 1)],
                           "omissions": []})
    if '{"accepted":true|false' in sys_:
        return json.dumps({"accepted": True, "violations": []})
    if sys_.startswith("You verify a credit-risk explanation"):
        n = last.count('"claim_id"')
        return json.dumps({"claims": [{"claim_id": f"c{k}", "status": "supported"} for k in range(1, n + 1)],
                           "omissions": [], "findings": []})
    if "Tabular Feature Coverage" in sys_:
        return json.dumps({"TFC": 4, "TDC": 4, "rationale": "dry run"})
    if sys_.startswith("## Role"):
        return json.dumps({"confidence": 0.8, "verdict": "accept", "justification": "dry run", "error type": "None"})
    if "TURN 2" in last:
        return json.dumps({"accepted": True, "violations": []})
    if sys_.startswith("You will assess"):
        return json.dumps({"notes": [{"criterion": "a", "note": "dry run", "quotes": []}]})
    return json.dumps({"accepted": True, "violations": []})


def load_items(which: str):
    latest = {}
    for line in open(M1 / "records.jsonl"):
        r = json.loads(line)
        latest[(r["borrower_id"], r["arm"], r["family"])] = r
    agreed = agreed_natural() if which == "agreed" else {i: (k, v) for i, (k, v, _b) in final_labels().items()}
    items = []
    if which in ("agreed", "final"):
        for i, (k, truth) in sorted(agreed.items()):
            rec = latest.get((k["borrower_id"], k["arm"], k["family"]))
            if rec:
                items.append((i, rec, truth, k))
    else:
        for (b, arm, fam), rec in sorted(latest.items()):
            if rec.get("error") or not (rec.get("text") or "").strip():
                continue
            items.append((f"{b}|{arm}|{fam}", rec, None, {"group": rec.get("group")}))
    return items


def run_one(method, cand, eo, judge, reassess):
    if method == "contract":
        v = verify_contract(cand, eo, judge, reassess_family=reassess, experiment="m1-comparators")
        d = v.to_dict()
        return {"accepted": v.accepted, "decision": v.decision, "reason": v.reason, "detail": d,
                "calls": v.calls}
    fn = K.METHODS[method]
    if method.startswith("exec_"):
        v = fn(cand, eo, family=(judge if method in LLM_METHODS else None), experiment="m1-comparators",
               reassess_family=reassess, dispute_family=DISPUTE_FAMILY)
    else:
        v = fn(cand, eo, family=judge, experiment="m1-comparators") if method in LLM_METHODS else fn(cand, eo)
    return {"accepted": v.accepted, "score": v.score, "detail": v.detail, "error": v.error, "calls": v.calls,
            "n_calls": len(v.calls or [])}


def with_deadline(fn, seconds):
    """Run fn with a wall-clock limit. Some LinkAPI calls keep the connection alive without ever
    answering, so the per-read client timeout never fires; such an item is recorded as unresolved."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    except FTE:
        return {"accepted": None, "error": f"wall_clock_timeout_{int(seconds)}s", "detail": {}, "calls": [],
                "score": None, "n_calls": 0}
    finally:
        ex.shutdown(wait=False)


def estimate(items, methods, judge):
    chars = sum(len(json.dumps(r["evidence"])) + len(r["text"]) for _, r, _, _ in items)
    p_tok = chars / 4 + 600 * len(items)                  # + system prompt
    est = {}
    for m in methods:
        if m not in LLM_METHODS:
            continue
        mult = LLM_METHODS[m]
        pin, pout = PRICE.get(judge, (1.0, 4.0))
        tin, tout = p_tok * mult, 500 * len(items) * mult
        est[m] = {"calls": round(len(items) * mult), "input_Mtok": round(tin / 1e6, 2),
                  "output_Mtok": round(tout / 1e6, 2), "usd": round(tin / 1e6 * pin + tout / 1e6 * pout, 2)}
    return est


def score(rows):
    c = Counter()
    for r in rows:
        if r["truth"] is None:
            continue
        if r["accepted"] is None:
            c["unresolved_err" if r["truth"] else "unresolved_ok"] += 1
            continue
        c[("err" if r["truth"] else "ok", "acc" if r["accepted"] else "rej")] += 1
    ne = c[("err", "acc")] + c[("err", "rej")] + c["unresolved_err"]
    nc = c[("ok", "acc")] + c[("ok", "rej")] + c["unresolved_ok"]
    return {"n_err": ne, "n_ok": nc, "false_accept": c[("err", "acc")], "false_reject": c[("ok", "rej")],
            "unresolved": c["unresolved_err"] + c["unresolved_ok"],
            "a": round(c[("err", "acc")] / ne, 3) if ne else None,
            "b": round(c[("ok", "rej")] / nc, 3) if nc else None,
            "coverage_on_clean": round(c[("ok", "acc")] / nc, 3) if nc else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["always_accept", "always_reject", "pt_audit"])
    ap.add_argument("--judge", default="qwen3_next")
    ap.add_argument("--reassess", default=None)
    ap.add_argument("--items", choices=["agreed", "final", "all"], default="agreed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--call-timeout", type=float, default=None,
                    help="per-request client timeout (s); reasoning models such as Grok 4.5 can exceed the 90 s default")
    ap.add_argument("--item-timeout", type=float, default=600.0,
                    help="wall-clock limit per item (s); an item over it is recorded unresolved (timeout)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. v1.4 (keeps earlier runs intact)")
    ap.add_argument("--dispute-family", default=None, help="independent reader for exec_R")
    a = ap.parse_args()
    global DISPUTE_FAMILY
    DISPUTE_FAMILY = a.dispute_family
    if a.call_timeout:
        import gateway as _G
        _G.GATEWAY.timeout = a.call_timeout
    items = load_items(a.items)
    if a.limit:
        items = items[: a.limit]
    est = estimate(items, a.methods, a.judge)
    if est:
        print("cost estimate (list prices; recheck):", json.dumps(est), "TOTAL $",
              round(sum(v["usd"] for v in est.values()), 2))
    if a.live and est and not a.confirm_spend:
        print("live LLM run needs --confirm-spend after reading the estimate")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    ctx = fakes.patched(fakes.FakeGateway(dry_responder)) if not a.live else _null()
    with ctx:
        for m in a.methods:
            judge = a.judge if (m in LLM_METHODS) else "none"
            path = OUT / f"{m}__{judge}{('__' + a.tag) if a.tag else ''}{'' if a.live else '__dry'}.jsonl"
            done = {}
            if path.exists():
                for line in open(path):
                    r = json.loads(line)
                    done[r["item_id"]] = r
            rows = list(done.values())
            todo = [(i, rec, truth, k) for i, rec, truth, k in items if i not in done]
            import threading
            from concurrent.futures import ThreadPoolExecutor
            lock = threading.Lock()

            def work(it, m=m, judge=judge, path=path):
                i, rec, truth, k = it
                eo = eo_from(rec["evidence"])
                cand = Candidate(borrower_id=rec["borrower_id"], arm=rec["arm"], text=rec["text"],
                                 claims=rec.get("claims") or [], family=rec["family"])
                t0 = time.time()
                out = with_deadline(lambda: run_one(m, cand, eo, a.judge, a.reassess), a.item_timeout)
                row = {"item_id": i, "method": m, "judge": judge, "truth": truth, "group": k.get("group"),
                       "dataset": rec["dataset"], "family": rec["family"], "arm": rec["arm"],
                       "seconds": round(time.time() - t0, 2), "live": a.live, **out}
                with lock:
                    with open(path, "a") as f:
                        f.write(json.dumps(row, default=str) + "\n")
                    rows.append(row)

            if a.workers > 1:
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    list(ex.map(work, todo))
            else:
                for it in todo:
                    work(it)
            summary[f"{m}__{judge}"] = score(rows)
            print(f"{m:20} {judge:12} {json.dumps(summary[f'{m}__{judge}'])}")
    (OUT / f"summary{('_' + a.tag) if a.tag else ''}{'' if a.live else '_dry'}.json").write_text(json.dumps(summary, indent=1))
    return 0


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)        # a hung provider thread must not keep the process alive

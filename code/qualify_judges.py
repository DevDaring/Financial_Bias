"""
Judge qualification against independent (adjudicated) labels - plan §6/E1:
"qualify each judge on separate, independently validated clean and erroneous
cases" before it is allowed to gate anything.

Three item sets, never pooled:

  natural_v1     120 natural items, v1 evidence. Labels were made against this
                 exact evidence (annotators saw "PAY_0 = None"), so this is the
                 like-for-like qualification. Primary selection set.
  natural_v2     the natural items whose label cannot depend on the zero
                 restore (restore_evidence_v2.py), with v2 evidence. Compared
                 PAIRED with the same items under v1: does the evidence bug
                 explain a judge's false rejections?
  challenge      48 controlled corruptions (8 per error type), v1 evidence.
                 Gives per-type recall, above all for invented_factor, which
                 the structural checker cannot catch. Planted prevalence: never
                 mixed into natural-error rates.

Publisher overlap is reported, not hidden: v1 natural items were written by
DeepSeek and Llama 3.1 8B, so Llama 4 Maverick on Llama text is same-publisher.

Selection rule (fixed before results were read):
  eligible  : unresolved <= 5% and false-reject b <= 0.20 on natural_v1
  primary   : lowest false-accept a on natural_v1 among eligible judges,
              ties broken by invented_factor recall on challenge
  second    : best eligible judge from a different publisher than primary

    python qualify_judges.py                      # roster + candidates
    python qualify_judges.py --judges sonnet5     # one judge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SUB = Path(__file__).resolve().parents[2] / "Submission"
os.environ.setdefault("SAARTHI_LLM_TRACE", str(SUB / "llm_trace_qualification.jsonl"))

import registry  # noqa: E402
from generate import Candidate  # noqa: E402
from score_e1 import load_adjudication, load_items, load_key, rebuild_evidence  # noqa: E402
from verify import semantic_check  # noqa: E402

OUT = SUB / "judge_qualification.json"
PROMPT = 1   # set from --prompt; judge/1 is the prompt qualified on 2 Oct 2026
V2_ITEMS = SUB / "annotation_pack_v2" / "E1_items_v2.jsonl"
SENS = SUB / "annotation_pack_v2" / "label_sensitivity.json"
PILOT_PUBLISHER = {"deepseek": "DeepSeek", "llama": "Meta", "template": None}
MAX_UNRESOLVED, MAX_B = 0.05, 0.20


def cp_interval(k: int, n: int, alpha: float = 0.05):
    if not n:
        return None
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def judge_one(judge: str, iid: str, it: dict, k: dict, set_name: str) -> dict:
    eo = rebuild_evidence(it)
    cand = Candidate(borrower_id=iid, arm=k.get("arm") or "", text=it["explanation"],
                     family=k.get("family") or "")
    base = {"item_id": iid, "judge": judge, "set": set_name}
    try:
        v = semantic_check(cand, eo, judge, enforce_independence=False,
                           experiment=f"judge_qualification:{set_name}:p{PROMPT}",
                           prompt_version=PROMPT)
    except Exception as e:  # noqa: BLE001 - one bad item must not kill the run
        return {**base, "accepted": False, "error": f"exception:{type(e).__name__}:{str(e)[:60]}",
                "violation_types": [], "latency_ms": 0, "model_id": None}
    return {**base, "accepted": v.accepted, "error": v.error,
            "violation_types": [x.get("type") for x in v.violations],
            "latency_ms": v.latency_ms, "model_id": v.judge_model_id,
            "parse_method": v.parse_method, "error_kind": v.error_kind,
            "route": f"{v.provider}:{v.judge_model_id}" if v.provider else None}


def confusion(rows):
    tp = sum(1 for r in rows if not r["accepted"] and r["truth"])
    tn = sum(1 for r in rows if r["accepted"] and not r["truth"])
    fa = sum(1 for r in rows if r["accepted"] and r["truth"])
    fr = sum(1 for r in rows if not r["accepted"] and not r["truth"])
    n = tp + tn + fa + fr
    pos, neg = tp + fa, tn + fr
    return {"n": n, "truth_error": pos, "truth_clean": neg,
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "false_accept_a": round(fa / pos, 4) if pos else None,
            "a_ci95": cp_interval(fa, pos),
            "false_reject_b": round(fr / neg, 4) if neg else None,
            "b_ci95": cp_interval(fr, neg),
            "balanced_accuracy": round(((tp / pos) + (tn / neg)) / 2, 4) if pos and neg else None,
            "accept_rate": round((tn + fa) / n, 4) if n else None}


def recall_by_type(rows, type_of):
    out = {}
    for r in rows:
        if not r["truth"]:
            continue
        for t in type_of(r):
            d = out.setdefault(t, [0, 0])
            d[1] += 1
            d[0] += 0 if r["accepted"] else 1
    return {t: {"caught": c, "n": n, "recall": round(c / n, 3)} for t, (c, n) in sorted(out.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="*",
                    default=list(dict.fromkeys(
                        [registry.PRIMARY_JUDGE, registry.SECOND_JUDGE,
                         "llama4_maverick", "gpt5_mini"] + registry.JUDGE_CANDIDATES)))
    ap.add_argument("--workers-per-judge", type=int, default=2,
                    help="kept low: Bedrock and LinkAPI throttle bursts")
    ap.add_argument("--prompt", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    global PROMPT, OUT
    PROMPT = a.prompt
    if PROMPT != 1:
        OUT = SUB / f"judge_qualification_prompt{PROMPT}.json"

    key, items, adj = load_key(), load_items(), load_adjudication()
    v2 = {json.loads(l)["item_id"]: json.loads(l) for l in open(V2_ITEMS) if l.strip()}
    sens = json.load(open(SENS))["items"]
    nondup = {i: k for i, k in key.items() if "duplicate_of" not in k and i in adj}
    nat = sorted(i for i, k in nondup.items() if k["source"] == "natural")
    nat_v2 = [i for i in nat if sens[i]["label_portable"]]
    chal = sorted(i for i, k in nondup.items() if k["source"] == "corrupted")
    jobs_for = lambda j: ([(j, i, items[i], "natural_v1") for i in nat]
                          + [(j, i, v2[i], "natural_v2") for i in nat_v2]
                          + [(j, i, items[i], "challenge") for i in chal])
    print(f"judges={a.judges}\nitems: natural_v1={len(nat)} natural_v2={len(nat_v2)} "
          f"challenge={len(chal)}  -> {len(jobs_for('x'))} calls per judge", flush=True)

    def run_judge(j):
        # Bedrock throttles Maverick even at a few parallel calls; keep it low
        # so its verdicts come from the primary route, not the fallback.
        on_bedrock = registry.get(j).usable_routes()[0].provider == "bedrock"
        w = a.workers_per_judge if on_bedrock else max(a.workers_per_judge, 4)
        with ThreadPoolExecutor(max_workers=w) as ex:
            res = list(ex.map(lambda t: judge_one(t[0], t[1], t[2], key[t[1]], t[3]),
                              jobs_for(j)))
        print(f"  done {j}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=len(a.judges)) as ex:
        raw = [r for res in ex.map(run_judge, a.judges) for r in res]

    report = {"note": ("natural_v1 = like-for-like with the labels; natural_v2 = "
                       "label-portable subset with zeros restored; challenge = planted errors"),
              "selection_rule": {"max_unresolved": MAX_UNRESOLVED, "max_b": MAX_B,
                                 "primary": "min a on natural_v1, tie -> invented_factor recall",
                                 "second": "best eligible from another publisher"},
              "n_items": {"natural_v1": len(nat), "natural_v2": len(nat_v2), "challenge": len(chal)},
              "judges": {}}
    for j in a.judges:
        fam = registry.get(j)
        primary_model = fam.usable_routes()[0].model_id if fam.usable_routes() else None
        rows = []
        for r in raw:
            if r["judge"] != j:
                continue
            k = key[r["item_id"]]
            rows.append({**r, "truth": adj[r["item_id"]]["any_error"],
                         "truth_types": adj[r["item_id"]]["error_types"],
                         "injected": k.get("injected_error"),
                         "group": k.get("group"),
                         "restored": bool(sens[r["item_id"]]["restored_fields"]),
                         "same_publisher": PILOT_PUBLISHER.get(k.get("family")) == fam.publisher})
        by = lambda s: [r for r in rows if r["set"] == s]
        ok = lambda rs: [r for r in rs if not r["error"]]
        n1 = by("natural_v1")
        unresolved = [r for r in rows if r["error"]]
        served = [r["model_id"] for r in rows if r["model_id"]]
        d = {
            "publisher": fam.publisher,
            "unresolved": len(unresolved),
            "unresolved_rate_natural_v1": round(len([r for r in n1 if r["error"]]) / len(n1), 4) if n1 else None,
            "unresolved_detail": sorted({(r["error"] or "")[:80] for r in unresolved})[:5],
            "off_primary_route_share": round(sum(m != primary_model for m in served) / len(served), 4) if served else None,
            "median_latency_ms": sorted(r["latency_ms"] for r in rows if r["latency_ms"])[len(served) // 2] if served else None,
            "natural_v1": confusion(ok(n1)),
            "natural_v1_cross_publisher": confusion([r for r in ok(n1) if not r["same_publisher"]]),
            "natural_v1_same_publisher": confusion([r for r in ok(n1) if r["same_publisher"]]),
            "natural_v1_items_with_restored_zeros": confusion([r for r in ok(n1) if r["restored"]]),
            "natural_v1_items_without": confusion([r for r in ok(n1) if not r["restored"]]),
            "natural_v1_by_group": {g: confusion([r for r in ok(n1) if r["group"] == g])
                                    for g in sorted({r["group"] for r in n1 if r["group"]})},
            "natural_v1_recall_by_type": recall_by_type(ok(n1), lambda r: r["truth_types"]),
            "portable_paired": {
                "v1": confusion([r for r in ok(n1) if r["item_id"] in set(nat_v2)]),
                "v2": confusion(ok(by("natural_v2")))},
            "challenge": confusion(ok(by("challenge"))),
            "challenge_recall_by_injected_type": recall_by_type(
                ok(by("challenge")), lambda r: [r["injected"]] if r["injected"] else []),
        }
        nv = d["natural_v1"]
        d["eligible"] = bool(d["unresolved_rate_natural_v1"] is not None
                             and d["unresolved_rate_natural_v1"] <= MAX_UNRESOLVED
                             and nv["false_reject_b"] is not None and nv["false_reject_b"] <= MAX_B)
        report["judges"][j] = d

    elig = [(j, d) for j, d in report["judges"].items() if d["eligible"]]
    rank = sorted(elig, key=lambda x: (x[1]["natural_v1"]["false_accept_a"],
                                       -(x[1]["challenge_recall_by_injected_type"]
                                         .get("invented_factor", {}).get("recall", 0))))
    primary = rank[0][0] if rank else None
    second = next((j for j, d in rank[1:] if d["publisher"] != report["judges"][primary]["publisher"]),
                  None) if primary else None
    report["selection"] = {"ranking": [j for j, _ in rank], "primary": primary, "second": second}
    # provenance: the exact prompt/parser that was qualified (run_m1 refuses on mismatch)
    import hashlib
    import json_parse as JP
    from verify import JUDGE_PROMPTS
    report["judge_prompt_version"] = PROMPT
    report["judge_prompt_sha256"] = hashlib.sha256(JUDGE_PROMPTS[PROMPT].encode()).hexdigest()
    report["parser_sys_sha256"] = hashlib.sha256(JP.PARSER_SYS.encode()).hexdigest()[:16]
    report["json_parse_sha256"] = hashlib.sha256(open(Path(__file__).parent / "json_parse.py", "rb").read()).hexdigest()[:12]
    report["served_model_ids"] = {j: sorted({r["model_id"] for r in raw if r["judge"] == j and r["model_id"]})
                                  for j in a.judges}
    # the run gate requires each judge to be served on exactly this route
    report["served_routes"] = {j: sorted({r["route"] for r in raw if r["judge"] == j and r.get("route")})
                               for j in a.judges}
    report["parse_methods"] = {j: dict(__import__("collections").Counter(
        r.get("parse_method", "") for r in raw if r["judge"] == j)) for j in a.judges}
    OUT.write_text(json.dumps({"summary": report, "raw": raw}, indent=2, default=str))

    print()
    hdr = f"{'judge':<16}{'pub':<10}{'unres':>6}{'acc':>7}{'a':>7}{'b':>7}{'balacc':>8}{'INVrec':>8}{'chalAcc':>8}{'b_v1→v2 (portable)':>22}  elig"
    print(hdr)
    for j, d in report["judges"].items():
        nv, ch, pp = d["natural_v1"], d["challenge"], d["portable_paired"]
        inv = d["challenge_recall_by_injected_type"].get("invented_factor", {}).get("recall")
        print(f"{j:<16}{d['publisher']:<10}{d['unresolved']:>6}{str(nv['accuracy']):>7}"
              f"{str(nv['false_accept_a']):>7}{str(nv['false_reject_b']):>7}{str(nv['balanced_accuracy']):>8}"
              f"{str(inv):>8}{str(ch['accuracy']):>8}{(str(pp['v1']['false_reject_b']) + '→' + str(pp['v2']['false_reject_b'])):>22}  {d['eligible']}")
    print(f"\nselection: {report['selection']}\n-> {OUT}")


if __name__ == "__main__":
    main()

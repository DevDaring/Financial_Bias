"""
Development analysis after the 7 Oct manuscript review (Future_Plan review §3, §5, §9.3, §11).

For each judge (qwen3_next, grok45, gemma27_vllm) and every method row file found:
  - all six correctness x action cells (correct/wrong x release/reject/defer), selective error,
    LLM calls, tokens, list USD, p50/p95 latency;
  - paired borrower-cluster bootstrap vs the reference method (ReCheck 2-R), on wrong-release rate and
    correct-release rate, with a one-sided 95% non-inferiority bound for correct release (margin 5 points);
  - utility U(lambda) = correct releases - lambda x wrong releases for lambda in 1, 2, 5, 10;
  - risk-coverage under ONE selection mechanism for every method: release iff the method accepts and the
    Kwok expected-score (Gemma, token log-probabilities) >= t; AURC and selective error at matched coverage.
    This is empirical; no finite-sample certificate is claimed (36 borrowers cannot support one).

    python analyze_v2.py   -> Submission/m1/analysis_v2.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_comparators import OUT, PRICE  # noqa: E402
from score_m1 import M1, final_labels  # noqa: E402

B = 2000
PRICE = dict(PRICE, gemma27_vllm=(0.0, 0.0))
JUDGES = ["qwen3_next", "grok45", "gemma27_vllm"]
#: (file stem pattern, display) - first existing file wins
METHODS = [
    ("always_accept", ["always_accept__none"]), ("pt_audit", ["pt_audit__none"]),
    ("exec_A0", ["exec_A0__none__v1.4", "exec_A0__none"]),
    ("generic_judge2", ["generic_judge2__{j}__v2", "generic_judge2__{j}"]),
    ("generic_rubric", ["generic_rubric__{j}__v2"]), ("facts_judge", ["facts_judge__{j}__v2"]),
    ("generic_tool_judge", ["generic_tool_judge__{j}__v2", "generic_tool_judge__{j}"]),
    ("forced_tool_judge", ["forced_tool_judge__{j}__v2"]), ("claimwise_judge", ["claimwise_judge__{j}__v2"]),
    ("factool_judge", ["factool_judge__{j}__v2"]),
    ("generic_second_look", ["generic_second_look__{j}__v2", "generic_second_look__{j}"]),
    ("zandi_judge", ["zandi_judge__{j}__v2", "zandi_judge__{j}"]),
    ("mermigkis_verifier", ["mermigkis_verifier__{j}__v2", "mermigkis_verifier__{j}"]),
    ("pbp_judge", ["pbp_judge__{j}__v2", "pbp_judge__{j}"]),
    ("exec_A1", ["exec_A1__{j}__v2", "exec_A1__{j}__v1.4", "exec_A1__{j}"]),
    ("exec_P", ["exec_P__{j}__v2", "exec_P__{j}__v1.4", "exec_P__{j}"]),
    ("exec_D", ["exec_D__{j}__v2"]), ("exec_R", ["exec_R__{j}__v2"]),
    ("exec_R_replay", ["replayR__{j}__gptoss120b"]),
    ("exec_R_replay_gemma_reader", ["replayR__{j}__gemma27_vllm"]),
    ("exec_R11_replay", ["replayR11__{j}__gptoss120b"]), ("exec_R11", ["exec_R11__{j}__v2"]),
]
REF = "exec_R"


def norm_group(g):
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(str(g or "").lower(), "other")


def load(stems, j):
    for s in stems:
        p = OUT / (s.format(j=j) + ".jsonl")
        if p.exists():
            rows = {}
            for line in open(p):
                r = json.loads(line)
                rows[r["item_id"]] = r
            return rows, p.name
    return None, None


def cells(rows, truth):
    c = defaultdict(int)
    for i, r in rows.items():
        if i not in truth:
            continue
        a = {True: "release", False: "reject", None: "defer"}[r.get("accepted")]
        c[f"{'wrong' if truth[i] else 'correct'}_{a}"] += 1
    return {k: c[k] for k in ("correct_release", "correct_reject", "correct_defer",
                              "wrong_release", "wrong_reject", "wrong_defer")}


def cost(rows, j):
    pin, pout = PRICE.get(j, (0, 0))
    calls = [c for r in rows.values() for c in (r.get("calls") or []) if isinstance(c, dict)]
    tin = sum(c.get("prompt_tokens") or 0 for c in calls)
    tout = sum(c.get("completion_tokens") or 0 for c in calls)
    usd = 0.0
    for c in calls:     # the dispute reader is priced at its own family
        fam = c.get("family") or j
        pi, po = PRICE.get(fam, (pin, pout)) if fam != "gptoss120b" else (0.037, 0.17)
        usd += ((c.get("prompt_tokens") or 0) * pi + (c.get("completion_tokens") or 0) * po) / 1e6
    lat = sorted((r.get("seconds") or 0) for r in rows.values() if r.get("seconds") is not None)
    q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None   # noqa: E731
    return {"calls": len(calls), "calls_per_item": round(len(calls) / max(1, len(rows)), 2), "input_tokens": tin,
            "output_tokens": tout, "usd_list": round(usd, 4), "latency_p50_s": q(0.5), "latency_p95_s": q(0.95)}


def main() -> int:
    fl = final_labels()
    truth = {i: v for i, (_k, v, _b) in fl.items()}
    borrower = {i: k["borrower_id"] for i, (k, _v, _b) in fl.items()}
    stratum = {i: (k["dataset"], norm_group(k.get("group"))) for i, (k, _v, _b) in fl.items()}
    strata = defaultdict(lambda: defaultdict(list))
    for i in truth:
        strata[stratum[i]][borrower[i]].append(i)
    base = {k: list(v.values()) for k, v in strata.items()}
    kw, _ = load(["kwok__gemma27_vllm"], "")
    kscore = {i: r.get("score") for i, r in (kw or {}).items() if r.get("score") is not None}
    res = {"note": "DEVELOPMENT: 299 adjudicated M1 narratives (36 borrowers). Methods frozen after this analysis "
                   "are evaluated on the context experiment and the locked natural test.", "judges": {}}
    rng = random.Random(20261007)
    for j in JUDGES:
        data, tab, subsets = {}, {}, {}
        for m, stems in METHODS:
            rows, src = load(stems, j)
            if not rows:
                continue
            rows = {i: r for i, r in rows.items() if i in truth}
            if len(rows) < 0.95 * len(truth):
                tab[m] = {"incomplete": f"{len(rows)}/{len(truth)}", "source": src}
                if len(rows) >= 90:
                    subsets[m] = rows
                continue
            data[m] = rows
            c = cells(rows, truth)
            rel = c["correct_release"] + c["wrong_release"]
            tab[m] = dict(c, source=src, n=len(rows), selective_error=round(c["wrong_release"] / rel, 4) if rel else None,
                          **cost(rows, j),
                          utility={str(l): c["correct_release"] - l * c["wrong_release"] for l in (1, 2, 5, 10)})
            gw = {}
            for ds in ("taiwan", "home_credit"):
                for grp in ("female", "male"):
                    its = [i for i in rows if not truth[i] and stratum[i] == (ds, grp)]
                    gw[f"{ds}|{grp}"] = [sum(rows[i].get("accepted") is not True for i in its), len(its)]
                gw[f"{ds}|gap_male_minus_female"] = round(gw[f"{ds}|male"][0] / max(1, gw[f"{ds}|male"][1])
                                                          - gw[f"{ds}|female"][0] / max(1, gw[f"{ds}|female"][1]), 4)
            tab[m]["withheld_correct_by_group"] = gw
            if m == "forced_tool_judge":
                tab[m]["tool_use_items"] = sum(1 for r in rows.values() if (r.get("detail") or {}).get("n_tools", 0) > 0)
            if m == "generic_tool_judge":
                tab[m]["tool_use_items"] = sum(1 for r in rows.values() if (r.get("detail") or {}).get("tools"))
        rp = M1 / f"replay_same_transcript_{j}.json"
        if rp.exists():           # dispute-to-defer on the SAME saved first-stage responses
            rr = json.load(open(rp))["rows"]
            data["exec_D_replay"] = {x["item_id"]: {"accepted": x["dispute"], "calls": [], "seconds": None} for x in rr}
            c = cells(data["exec_D_replay"], truth)
            rel = c["correct_release"] + c["wrong_release"]
            tab["exec_D_replay"] = dict(c, source=rp.name, n=len(rr), selective_error=round(c["wrong_release"] / rel, 4) if rel else None,
                                        calls=None, calls_per_item=0.0, usd_list=0.0,
                                        utility={str(l): c["correct_release"] - l * c["wrong_release"] for l in (1, 2, 5, 10)})
        # adapters audited (review §5.2): thresholds TUNED on these same development labels (optimistic for them)
        for m, grid in (("zandi_judge", [(a, b) for a in (2, 3, 4, 5) for b in (2, 3, 4, 5)]),
                        ("mermigkis_verifier", sorted({round(r.get("score"), 3) for r in data.get("mermigkis_verifier", {}).values()
                                                       if r.get("score") is not None}))):
            if m not in data:
                continue
            best = None
            for g in grid:
                rows2 = {}
                for i, r in data[m].items():
                    d = r.get("detail") or {}
                    if r.get("accepted") is None:
                        acc = None
                    elif m == "zandi_judge":
                        acc = (d.get("TFC") or 0) >= g[0] and (d.get("TDC") or 0) >= g[1]
                    else:
                        acc = (r.get("score") or 0) >= g
                    rows2[i] = dict(r, accepted=acc)
                c = cells(rows2, truth)
                u = c["correct_release"] - 2 * c["wrong_release"]
                if best is None or u > best[0]:
                    best = (u, g, rows2, c)
            data[m + "_tuned"] = best[2]
            rel = best[3]["correct_release"] + best[3]["wrong_release"]
            tab[m + "_tuned"] = dict(best[3], threshold=best[1], tuned_on="development labels, utility lambda=2",
                                     selective_error=round(best[3]["wrong_release"] / rel, 4) if rel else None,
                                     **cost(data[m], j),
                                     utility={str(l): best[3]["correct_release"] - l * best[3]["wrong_release"] for l in (1, 2, 5, 10)})
        ref = REF if REF in data else ("exec_R_replay" if "exec_R_replay" in data else None)
        paired = {}
        if ref:
            items = sorted(set(data[ref]))

            def rate(smp, m, wrong):
                rs = [data[m].get(i) for i in smp if truth[i] == wrong and i in data[m]]
                return sum(1 for r in rs if r and r.get("accepted") is True) / max(1, len(rs))
            for m in data:
                if m == ref:
                    continue
                out = {}
                for kind, wrong in (("wrong_release", True), ("correct_release", False)):
                    pt = rate(items, ref, wrong) - rate(items, m, wrong)
                    bs = []
                    for _ in range(B):
                        smp = [i for v in base.values() for b in (rng.choice(v) for _ in v) for i in b]
                        bs.append(rate(smp, ref, wrong) - rate(smp, m, wrong))
                    bs.sort()
                    out[kind] = {"ref_minus_X": round(pt, 4), "ci95": [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)],
                                 "one_sided_95_lower": round(bs[int(.05 * B)], 4)}
                out["correct_release_noninferior_5pts"] = out["correct_release"]["one_sided_95_lower"] > -0.05
                paired[m] = out
        # risk-coverage with the shared Kwok score
        rc = {}
        if kscore:
            ts = sorted({round(s, 3) for s in kscore.values()} | {-1.0})
            for m in list(data) + ["kwok_alone"]:
                pts = []
                for t in ts:
                    rel = [i for i in truth if i in kscore and kscore[i] >= t
                           and (m == "kwok_alone" or (data[m].get(i) or {}).get("accepted") is True)]
                    w = sum(1 for i in rel if truth[i])
                    ok = sum(1 for i in rel if not truth[i])
                    pts.append((ok / sum(1 for i in truth if not truth[i] and i in kscore), w / max(1, len(rel)), t))
                pts = sorted(set(pts))
                # AURC over correct-release coverage (step integral of selective error)
                aurc, prev = 0.0, 0.0
                for cov, risk, _t in pts:
                    aurc += (cov - prev) * risk
                    prev = cov
                at = {}
                for target in (0.5, 0.6, 0.7):
                    ok_pts = [p for p in pts if p[0] >= target]
                    at[str(target)] = round(min(p[1] for p in ok_pts), 4) if ok_pts else None
                rc[m] = {"aurc": round(aurc, 4), "max_coverage": round(max(p[0] for p in pts), 4),
                         "selective_error_at_coverage": at,
                         "curve": [[round(a, 4), round(b, 4)] for a, b, _ in pts]}
        sub = {}
        for m, rows in subsets.items():      # bounded shared subset: compare with the reference on the SAME items
            its = set(rows)
            sub[m] = {"n": len(its), "method": cells(rows, truth)}
            for r_ in ([ref] if ref else []) + ["generic_rubric", "facts_judge"]:
                if r_ in data:
                    sub[m][r_] = cells({i: data[r_][i] for i in its if i in data[r_]}, truth)
        res["judges"][j] = {"methods": tab, "reference": ref, "paired_ref_minus_X": paired, "risk_coverage_kwok": rc,
                            "subset_comparisons": sub}
    if kscore:
        pos = [kscore[i] for i in kscore if truth.get(i) is False]
        neg = [kscore[i] for i in kscore if truth.get(i) is True]
        res["kwok"] = {"n": len(kscore), "auroc_correct_vs_wrong": round(
            sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(1, len(pos) * len(neg)), 4)}
    (M1 / "analysis_v2.json").write_text(json.dumps(res, indent=1, default=str))
    for j, d in res["judges"].items():
        print("=====", j, "ref", d["reference"])
        print(f"{'method':22} {'C.rel':>5} {'C.rej':>5} {'C.def':>5} {'W.rel':>5} {'W.rej':>5} {'W.def':>5} {'selErr':>6} {'calls':>5} {'$':>6}")
        for m, v in d["methods"].items():
            if "incomplete" in v:
                print(f"{m:22} incomplete {v['incomplete']}")
                continue
            print(f"{m:22} {v['correct_release']:>5} {v['correct_reject']:>5} {v['correct_defer']:>5} {v['wrong_release']:>5} "
                  f"{v['wrong_reject']:>5} {v['wrong_defer']:>5} {str(v['selective_error']):>6} {v['calls_per_item']:>5} {v['usd_list']:>6}")
    if "kwok" in res:
        print("kwok", res["kwok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

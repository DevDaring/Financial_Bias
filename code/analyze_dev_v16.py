"""
ReCheck 3 selection on development data (Improvement_Plan WS1 Gate A): the 299 M1 + 400 locked-v1 narratives.
For each judge, executable/1.5 vs 1.6 (zero-call gate) for D / P / R11 against the same judge's generic baselines,
with paired borrower bootstraps within dataset x group strata.

    python analyze_dev_v16.py -> Submission/locked/dev_v16_analysis.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dev_sets import all_dev  # noqa: E402
from score_m1 import SUB  # noqa: E402

CMP = SUB / "m1" / "comparators"
LK = SUB / "locked"
B = 3000


def load_jsonl(p):
    return {json.loads(l)["item_id"]: json.loads(l).get("accepted") for l in open(p)} if p.exists() else None


def m1_file(m, j, tags):
    for t in tags:
        p = CMP / (f"{m}__{j}__{t}.jsonl" if t else f"{m}__{j}.jsonl")
        if p.exists():
            return p
    return None


SOURCES = {  # name -> (M1 file finder, locked file)
    "B1": lambda j: (m1_file("generic_judge2", j, ["v2", ""]), LK / "fresh_narratives_v1_runs" / f"generic_judge2__{j}.jsonl"),
    "B1R": lambda j: (m1_file("generic_rubric", j, ["v2"]), LK / "fresh_narratives_v1_runs" / f"generic_rubric__{j}.jsonl"),
    "B1F": lambda j: (m1_file("facts_judge", j, ["v2"]), LK / "fresh_narratives_v1_runs" / f"facts_judge__{j}.jsonl"),
    "P_1.5": lambda j: (m1_file("exec_P", j, ["v2", "v1.4", ""]), LK / "fresh_narratives_v1_runs" / f"exec_P__{j}.jsonl"),
    "D_1.5": lambda j: (m1_file("exec_D", j, ["v2"]), LK / "fresh_narratives_v1_runs" / f"exec_D__{j}.jsonl"),
    "R11_1.5": lambda j: (m1_file("exec_R11", j, ["v2"]), LK / "fresh_narratives_v1_runs" / f"exec_R11__{j}.jsonl"),
    "A0_1.6": lambda j: (CMP / "exec_A0__none__v16.jsonl", LK / "fresh_narratives_v1_runs_v16" / "exec_A0__qwen3_next.jsonl"),
    "D_1.6": lambda j: (m1_file("exec_D", j, ["v16"]), LK / "fresh_narratives_v1_runs_v16" / f"exec_D__{j}.jsonl"),
    "P_1.6": lambda j: (m1_file("exec_P", j, ["v16"]), LK / "fresh_narratives_v1_runs_v16" / f"exec_P__{j}.jsonl"),
    "R11_1.6": lambda j: (m1_file("exec_R11", j, ["v16"]), LK / "fresh_narratives_v1_runs_v16" / f"exec_R11__{j}.jsonl"),
}


def main() -> int:
    items = {x["item_id"]: x for x in all_dev()}
    truth = {i: x["truth"] for i, x in items.items()}
    strata = defaultdict(list)
    for i, x in items.items():
        strata[(x["set"], x["dataset"], str(x["group"]).lower()[:1])].append(i)
    rng = random.Random(20261017)
    draws = [[i for v in strata.values() for i in (rng.choice(v) for _ in v)] for _ in range(B)]
    res = {"n": len(truth), "n_wrong": sum(truth.values()), "judges": {}}
    for j in ("qwen3_next", "gemma27_vllm", "gptoss120b"):
        dec = {}
        for name, f in SOURCES.items():
            pm, pl = f(j)
            a, b = (load_jsonl(pm) if pm else None), load_jsonl(pl)
            if a is None or b is None:
                continue
            d = {**a, **b}
            if sum(1 for i in truth if i in d) < 0.97 * len(truth):
                continue
            dec[name] = d
        out = {"methods": {}, "paired": {}}
        for name, d in dec.items():
            cr = sum(1 for i, t in truth.items() if not t and d.get(i) is True)
            wr = sum(1 for i, t in truth.items() if t and d.get(i) is True)
            nok, nw = sum(1 for t in truth.values() if not t), sum(truth.values())
            out["methods"][name] = {"correct_release": cr, "wrong_release": wr, "of_correct": nok, "of_wrong": nw,
                                    "selective_error": round(wr / max(1, cr + wr), 4), "U2": cr - 2 * wr}
        base = "B1F" if "B1F" in dec else ("B1R" if "B1R" in dec else None)
        for name in dec:
            if not name.startswith(("D_", "P_", "R11_")) or base is None:
                continue
            o = {}
            for kind, w in (("wrong_release", True), ("correct_release", False)):
                def rate(its, d):
                    x = [i for i in its if truth[i] == w]
                    return sum(d.get(i) is True for i in x) / max(1, len(x))
                pt = rate(list(truth), dec[name]) - rate(list(truth), dec[base])
                bs = sorted(rate(s, dec[name]) - rate(s, dec[base]) for s in draws)
                o[kind] = {"diff": round(pt, 4), "ci95": [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)]}
            out["paired"][f"{name}_vs_{base}"] = o
        res["judges"][j] = out
    (LK / "dev_v16_analysis.json").write_text(json.dumps(res, indent=1))
    print("n", res["n"], "wrong", res["n_wrong"])
    for j, o in res["judges"].items():
        print("==", j)
        for name, v in o["methods"].items():
            print(f"  {name:9} correct {v['correct_release']:>4}/{v['of_correct']}  wrong {v['wrong_release']:>3}/{v['of_wrong']}"
                  f"  selErr {v['selective_error']:.3f}  U2 {v['U2']}")
        for k, v in o["paired"].items():
            print("   ", k, {kk: (vv["diff"], vv["ci95"]) for kk, vv in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Score the LOCKED natural test once the blinded labels return (Future_Plan review §4, §8). Prespecified:

  reference     agreement of the two independent raters; disagreements and UNRESOLVED go to adjudication
                (Submission/locked/adjudication_locked.csv, column ADJUDICATED_ANY) and are reported separately;
                primary analysis = items with a final label
  primary       ReCheck 2-R vs the strongest generic baseline chosen on DEVELOPMENT data (facts_judge, B1F):
                wrong-release rate difference (superiority) and correct-release difference with a one-sided
                95% non-inferiority bound at a 5-point margin; borrower = unit (one narrative per borrower),
                bootstrap within dataset x group strata
  secondary     every other frozen method; per stratum; producer; utility for lambda in 1, 2, 5, 10
  template      label-based correctness and the content floor (conveys_pd, conveys_top_driver_with_direction,
                useful_for_officer) of the narrative vs the template of the SAME borrower: what a lender actually
                sees after each verifier's decision (released narrative, else template)

Before labels exist, run with --label-free: decision rates per method and stratum only (no accuracy).

    python analyze_locked.py [--label-free]   -> Submission/locked/locked_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_m1 import SUB, find, read_sheet, yn  # noqa: E402

LK = SUB / "locked"
RUNS = LK / "fresh_narratives_v1_runs"
CFG = {"key": SUB / "annotation_key_locked_v1.jsonl", "sheet": "LOCKED_annotation_sheet_",
       "adjudication": LK / "adjudication_locked.csv", "records": LK / "fresh_narratives_v1.jsonl",
       "out": LK / "locked_analysis.json"}
RATERS = ["AS", "NK", "SP", "AB"]
PRIMARY, BASELINE = "exec_R", "facts_judge"
B = 4000


def runs():
    data = defaultdict(dict)
    for p in sorted(RUNS.glob("*.jsonl")):
        m, j = p.stem.split("__")
        for l in open(p):
            r = json.loads(l)
            data[(m, j)][r["item_id"]] = r
    return data


LABEL_QA: dict = {}
ADJ_MODE = "provided"   # provided | agreed_only | any_yes  (label sensitivity)


def labels(dirpath: Path):
    key = [json.loads(l) for l in open(CFG["key"])][1:]
    lab = defaultdict(dict)
    for q in RATERS:
        hits = sorted(dirpath.rglob(f"{CFG['sheet']}{q}*.xlsx"))
        if not hits:
            continue
        sh = read_sheet(hits[-1])        # labels only; the Annotator tab is never read
        for i, row in sh["rows"].items():
            lab[i][q] = row
    adj = {}
    ap = CFG["adjudication"]
    if ap.exists():
        for r in csv.DictReader(open(ap)):
            if (r.get("ADJUDICATED_ANY") or "").strip().upper() in ("Y", "N"):
                adj[r["item_id"]] = r["ADJUDICATED_ANY"].strip().upper() == "Y"
    out, dis, dup = {}, [], []
    for k in key:
        if k.get("duplicate_of"):
            dup.append(k)
            continue
        rows = [lab[k["item_id"]].get(q) for q in k["raters"]]
        if None in rows:
            continue
        a = [yn(x.get("ANY_MATERIAL_ERROR")) for x in rows]
        unres = any(yn(x.get("UNRESOLVED")) for x in rows)
        content = {c: [x.get(c) for x in rows] for c in ("conveys_pd", "conveys_top_driver_with_direction",
                                                         "useful_for_officer")}
        types = sorted({t for x in rows for t in TYPES if yn(x.get(t))})
        disagree = not (not unres and a[0] is not None and a[0] == a[1])
        if disagree and ADJ_MODE == "agreed_only":
            dis.append(k["item_id"])
            continue
        if disagree and ADJ_MODE == "any_yes":
            out[k["source_item_id"]] = dict(k, truth=any(x is True for x in a), basis="any_rater_yes", content=content,
                                            types=types)
        elif k["item_id"] in adj:
            out[k["source_item_id"]] = dict(k, truth=adj[k["item_id"]], basis="adjudicated", content=content, types=types)
        elif not unres and a[0] is not None and a[0] == a[1]:
            out[k["source_item_id"]] = dict(k, truth=a[0], basis="agreed", content=content, types=types)
        else:
            dis.append(k["item_id"])
    # label quality, from labels only
    from score_m1 import kappa
    qa = {}
    for arm in ("all", "grounded", "template"):
        x, y = [], []
        for k in key:
            if k.get("duplicate_of") or (arm == "grounded" and k["arm"] == "template") or \
                    (arm == "template" and k["arm"] != "template"):
                continue
            rows = [lab[k["item_id"]].get(q) for q in k["raters"]]
            if None in rows:
                continue
            x.append(yn(rows[0].get("ANY_MATERIAL_ERROR")))
            y.append(yn(rows[1].get("ANY_MATERIAL_ERROR")))
        qa[arm] = {"n": len(x), "agreement": round(sum(i == j for i, j in zip(x, y)) / max(1, len(x)), 4),
                   "kappa": kappa(x, y)}
    cons = [yn(lab[k["item_id"]][k["raters"][0]].get("ANY_MATERIAL_ERROR")) ==
            yn(lab[k["duplicate_of"]][k["raters"][0]].get("ANY_MATERIAL_ERROR")) for k in dup
            if k["item_id"] in lab and k["duplicate_of"] in lab]
    qa["duplicates"] = {"n": len(cons), "consistent": sum(cons)}
    qa["adjudicated"] = len(adj)
    LABEL_QA.update(qa)
    return out, dis, dup


TYPES = ["invented_factor", "sign_reversal", "numeric_or_unit_error", "unsupported_trend",
         "unsupported_causal_or_action", "omission_of_essential_driver", "other_material_error"]


def _mean(xs):
    xs = [float(x) for x in xs if x not in (None, "") and str(x).strip().replace(".", "", 1).isdigit()]
    return sum(xs) / len(xs) if xs else None


def extra_analyses(data, nar, tmpl, lab, recs):
    """Utility, escape by error type, template comparison, what the lender sees, group gaps."""
    out = {"utility": {}, "escape_by_type": {}, "group_withheld": {}, "shown": {}, "by_cell": {}}
    rng = random.Random(20261011)
    for (m, j), rows in data.items():
        c = defaultdict(int)
        for i, v in nar.items():
            if i in rows and rows[i].get("accepted") is True:
                c["wrong" if v["truth"] else "correct"] += 1
        out["utility"][f"{m}|{j}"] = {str(l): c["correct"] - l * c["wrong"] for l in (1, 2, 5)}
        cell = defaultdict(lambda: [0, 0, 0, 0])
        for i, v in nar.items():
            if i not in rows:
                continue
            cc = cell[v.get("cell") or v.get("family")]
            if v["truth"]:
                cc[0] += rows[i].get("accepted") is True
                cc[1] += 1
            else:
                cc[2] += rows[i].get("accepted") is True
                cc[3] += 1
        out["by_cell"][f"{m}|{j}"] = {k_: {"wrong_released": v_[0], "wrong": v_[1], "correct_released": v_[2],
                                           "correct": v_[3]} for k_, v_ in cell.items()}
        esc = defaultdict(int)
        for i, v in nar.items():
            if v["truth"] and i in rows and rows[i].get("accepted") is True:
                for t in v.get("types", []):
                    esc[t] += 1
        out["escape_by_type"][f"{m}|{j}"] = dict(esc)
        # what a lender sees: the released narrative, else the borrower's template
        useful, top, err, nshown = [], [], 0, 0
        for i, v in nar.items():
            # only borrowers whose template was labelled (all in locked v1; a 200-borrower sample in fresh v2)
            if i not in rows or v["borrower_id"] not in tmpl:
                continue
            nshown += 1
            shown = v if rows[i].get("accepted") is True else tmpl[v["borrower_id"]]
            err += bool(shown["truth"])
            u = _mean(shown["content"]["useful_for_officer"])
            if u is not None:
                useful.append(u)
            top.append(all(str(x).strip().upper() == "Y" for x in shown["content"]["conveys_top_driver_with_direction"]))
        out["shown"][f"{m}|{j}"] = {"error_rate": round(err / max(1, nshown), 4), "n": nshown,
                                    "mean_usefulness": round(sum(useful) / max(1, len(useful)), 3),
                                    "top_driver_conveyed": round(sum(top) / max(1, len(top)), 4)}
        # withheld correct by dataset x group, gap male - female, borrower bootstrap
        gaps = {}
        for ds in ("taiwan", "home_credit"):
            def wh(items):
                g = {}
                for grp in ("m", "f"):
                    its = [i for i in items if not nar[i]["truth"] and nar[i]["dataset"] == ds
                           and str(nar[i]["group"]).lower()[:1] == grp and i in rows]
                    g[grp] = sum(rows[i].get("accepted") is not True for i in its) / max(1, len(its))
                return g["m"] - g["f"]
            its = [i for i in nar if nar[i]["dataset"] == ds]
            bs = sorted(wh([rng.choice(its) for _ in its]) for _ in range(2000))
            gaps[ds] = {"gap_male_minus_female": round(wh(its), 4), "ci95": [round(bs[50], 4), round(bs[1949], 4)]}
        out["group_withheld"][f"{m}|{j}"] = gaps
    # templates versus narratives (all borrowers, independent of any verifier)
    def summ(d):
        return {"n": len(d), "error_rate": round(sum(v["truth"] for v in d) / max(1, len(d)), 4),
                "conveys_pd": round(sum(all(str(x).upper() == "Y" for x in v["content"]["conveys_pd"]) for v in d) / max(1, len(d)), 4),
                "top_driver": round(sum(all(str(x).upper() == "Y" for x in v["content"]["conveys_top_driver_with_direction"]) for v in d) / max(1, len(d)), 4),
                "mean_usefulness": round(sum(_mean(v["content"]["useful_for_officer"]) or 0 for v in d) / max(1, len(d)), 3)}
    out["template_vs_narrative"] = {"narrative_all": summ(list(nar.values())),
                                    "narrative_correct": summ([v for v in nar.values() if not v["truth"]]),
                                    "template": summ(list(tmpl.values()))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-free", action="store_true")
    ap.add_argument("--labels-dir", default=str(SUB / "Annotated" / "IDBI_Hack_Annotation"))
    ap.add_argument("--fresh-v2", action="store_true", help="score the fresh v2 test instead of locked v1")
    ap.add_argument("--adj-mode", default="provided", choices=["provided", "agreed_only", "any_yes"])
    ap.add_argument("--primary", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--adjudication", default=None, help="alternative adjudication csv (fresh v2 only)")
    a = ap.parse_args()
    global RUNS, PRIMARY, BASELINE, ADJ_MODE
    ADJ_MODE = a.adj_mode
    if a.fresh_v2:
        RUNS = LK / "fresh_v2_runs"
        CFG.update(key=SUB / "annotation_key_fresh_v2.jsonl", sheet="FRESH_annotation_sheet_",
                   adjudication=LK / "adjudication_fresh_v2.csv", records=LK / "fresh_natural_v2_final.jsonl",
                   out=LK / ("fresh_v2_analysis.json" if a.adj_mode == "provided" else f"fresh_v2_analysis_{a.adj_mode}.json"))
        if a.adjudication:
            CFG.update(adjudication=Path(a.adjudication), out=LK / "fresh_v2_analysis_independent_adjudication.json")
    PRIMARY = a.primary or PRIMARY
    BASELINE = a.baseline or BASELINE
    data = runs()
    recs = {json.loads(l)["item_id"]: json.loads(l) for l in open(CFG["records"])}
    res = {"n_narratives": len(recs), "methods": {}}
    for (m, j), rows in sorted(data.items()):
        by = defaultdict(lambda: [0, 0, 0])
        for i, r in rows.items():
            s = recs[i]["stratum"]
            by[s][{True: 0, False: 1, None: 2}[r.get("accepted")]] += 1
        calls = [c for r in rows.values() for c in (r.get("calls") or []) if isinstance(c, dict)]
        res["methods"][f"{m}|{j}"] = {"n": len(rows), "release": sum(r.get("accepted") is True for r in rows.values()),
                                      "reject": sum(r.get("accepted") is False for r in rows.values()),
                                      "defer": sum(r.get("accepted") is None for r in rows.values()),
                                      "by_stratum": {k: dict(zip(("release", "reject", "defer"), v)) for k, v in by.items()},
                                      "calls_per_item": round(len(calls) / max(1, len(rows)), 2)}
    if not a.label_free:
        lab, dis, dup = labels(Path(a.labels_dir))
        nar = {i: v for i, v in lab.items() if v["arm"] != "template"}
        tmpl = {v["borrower_id"]: v for v in lab.values() if v["arm"] == "template"}
        res["label_quality"] = LABEL_QA
        res["errors_by_producer"] = {f: [sum(v["truth"] for v in nar.values() if v["family"] == f),
                                         sum(1 for v in nar.values() if v["family"] == f)]
                                     for f in sorted({v["family"] for v in nar.values()})}
        res["labels"] = {"narratives_labelled": len(nar), "templates_labelled": len(tmpl), "to_adjudicate": len(dis),
                         "narrative_error_rate": round(sum(v["truth"] for v in nar.values()) / max(1, len(nar)), 4),
                         "template_error_rate": round(sum(v["truth"] for v in tmpl.values()) / max(1, len(tmpl)), 4)}
        strata = defaultdict(list)
        for i, v in nar.items():
            strata[v["stratum"]].append(i)
        rng = random.Random(20261010)
        for (m, j), rows in data.items():
            c = defaultdict(int)
            for i, v in nar.items():
                if i in rows:
                    c[("wrong" if v["truth"] else "correct", {True: "release", False: "reject", None: "defer"}[rows[i].get("accepted")])] += 1
            res["methods"][f"{m}|{j}"]["cells"] = {f"{x}_{y}": c[(x, y)] for x in ("correct", "wrong")
                                                   for y in ("release", "reject", "defer")}
            # what the lender sees: released narrative, else the template of the same borrower
            shown_err = sum((v["truth"] if rows.get(i, {}).get("accepted") is True else
                             tmpl.get(v["borrower_id"], {}).get("truth", False)) for i, v in nar.items() if i in rows)
            res["methods"][f"{m}|{j}"]["shown_text_error_rate"] = round(shown_err / max(1, len(nar)), 4)
        for j in sorted({j for _m, j in data}):
            if (PRIMARY, j) not in data or (BASELINE, j) not in data:
                continue
            A, Bm = data[(PRIMARY, j)], data[(BASELINE, j)]

            def rate(items, rows, wrong):
                its = [i for i in items if i in rows and nar[i]["truth"] == wrong]
                return sum(rows[i].get("accepted") is True for i in its) / max(1, len(its))
            out = {}
            for kind, wrong in (("wrong_release", True), ("correct_release", False)):
                items = list(nar)
                pt = rate(items, A, wrong) - rate(items, Bm, wrong)
                bs = sorted(rate(s, A, wrong) - rate(s, Bm, wrong) for s in
                            ([i for v in strata.values() for i in (rng.choice(v) for _ in v)] for _ in range(B)))
                out[kind] = {"R_minus_baseline": round(pt, 4), "ci95": [round(bs[int(.025 * B)], 4), round(bs[int(.975 * B)], 4)],
                             "one_sided_95": [round(bs[int(.05 * B)], 4), round(bs[int(.95 * B)], 4)]}
            out["superiority_wrong_release"] = out["wrong_release"]["one_sided_95"][1] < 0
            out["noninferior_correct_release_5pts"] = out["correct_release"]["one_sided_95"][0] > -0.05
            res.setdefault("primary", {})[j] = out
    if not a.label_free:
        res.update(extra_analyses(data, nar, tmpl, lab, recs))
    CFG["out"].write_text(json.dumps(res, indent=1, default=str))
    for k, v in res["methods"].items():
        print(f"{k:34} rel {v['release']:>4} rej {v['reject']:>4} def {v['defer']:>4} calls {v['calls_per_item']}",
              v.get("cells", ""))
    if "primary" in res:
        print(json.dumps(res["primary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

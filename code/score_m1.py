"""
Audit and score the returned milestone-1 annotation sheets.

    python score_m1.py [--dir ../../Submission/SAARTHI_Annotated]

1. Integrity: every rater returned exactly their items; item text unchanged
   against the pack; labels complete and internally consistent.
2. Rater quality: hidden-duplicate consistency, planted-error recall and
   clean-control specificity on the challenge items, label patterns.
3. Agreement: raw agreement and Cohen's kappa per rater pair on natural items.
4. Adjudication list: every primary item where the two raters disagree on
   ANY_MATERIAL_ERROR or either marked it UNRESOLVED (CSV for the author).
5. Preliminary verifier scores on the AGREED natural items only (development
   data; final numbers wait for adjudication).

Writes Submission/m1/annotation_scores.json and Submission/m1/adjudication_m1.csv.
The annotation key stays local; nothing here is sent to raters.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SUB = Path(__file__).resolve().parents[2] / "Submission"
M1 = SUB / "m1"
TYPES = ["invented_factor", "sign_reversal", "numeric_or_unit_error", "unsupported_trend",
         "unsupported_causal_or_action", "omission_of_essential_driver", "other_material_error"]
RATERS = ["AS", "NK", "SP"]


def yn(v):
    s = str(v or "").strip().upper()[:1]
    return {"Y": True, "N": False}.get(s)


def read_sheet(path: Path) -> dict:
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    ws = wb["Annotate"]
    head = [c.value for c in ws[1]]
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        d = dict(zip(head, r))
        if d.get("item_id"):
            rows[str(d["item_id"]).strip()] = d
    # the Annotator tab (annotator identity, dates, time) is confidential and is never read
    return {"rows": rows, "meta": {}}


def find(dirpath: Path, stem: str) -> Path:
    c = sorted(dirpath.glob(f"{stem}*.xlsx"))
    if not c:
        raise SystemExit(f"no sheet matching {stem}* in {dirpath}")
    return c[-1]                                    # _v2 sorts after the plain name


def kappa(a, b) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def final_labels(dirpath: Path | None = None) -> dict:
    """item_id -> (key row, ANY_MATERIAL_ERROR, basis) for every primary natural item after author
    adjudication (7 Oct 2026): rater agreement, the adjudicated decision for disagreements, and the
    author's decision on the contract-vs-label conflicts (which overrides an agreed label).
    UNRESOLVED adjudications are left out and reported."""
    import csv
    out = {i: (k, v, "agreed") for i, (k, v) in agreed_natural(dirpath).items()}
    key = {r["item_id"]: r for r in map(json.loads, open(SUB / "annotation_key_m1.jsonl")) if not r.get("_meta")}
    for r in csv.DictReader(open(M1 / "adjudication_m1.csv")):
        if r.get("source") != "natural":
            continue
        d = (r.get("ADJUDICATED_ANY") or "").strip().upper()
        if d in ("Y", "N"):
            out[r["item_id"]] = (key[r["item_id"]], d == "Y", "adjudicated")
    cp = M1 / "contract_conflict_decisions.json"
    if cp.exists():
        for i, v in json.load(open(cp)).items():
            if v["decision"] in ("error", "correct") and i in out:
                out[i] = (out[i][0], v["decision"] == "error", "contract_conflict")
    return out


def agreed_natural(dirpath: Path | None = None) -> dict:
    """item_id -> (key row, agreed ANY_MATERIAL_ERROR) for primary natural items
    where both raters agree and neither marked UNRESOLVED (pre-adjudication)."""
    d = dirpath or (SUB / "SAARTHI_Annotated")
    key = {r["item_id"]: r for r in map(json.loads, open(SUB / "annotation_key_m1.jsonl")) if not r.get("_meta")}
    lab = defaultdict(dict)
    for r in RATERS:
        for i, row in read_sheet(find(d, f"M1_annotation_sheet_{r}"))["rows"].items():
            lab[i][r] = row
    out = {}
    for i, k in key.items():
        if k["source"] != "natural" or k.get("duplicate_of") or len(k["raters"]) != 2:
            continue
        rows = [lab[i].get(r) for r in k["raters"]]
        if None in rows or any(yn(x.get("UNRESOLVED")) for x in rows):
            continue
        a1, a2 = (yn(x.get("ANY_MATERIAL_ERROR")) for x in rows)
        if a1 is not None and a1 == a2:
            out[i] = (k, a1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(SUB / "SAARTHI_Annotated"))
    a = ap.parse_args()
    d = Path(a.dir)
    key_rows = [json.loads(l) for l in open(SUB / "annotation_key_m1.jsonl")]
    key = {r["item_id"]: r for r in key_rows if not r.get("_meta")}

    sheets, packs = {}, {}
    for r in RATERS:
        sheets[r] = read_sheet(find(d, f"M1_annotation_sheet_{r}"))
        packs[r] = read_sheet(SUB / f"annotation_pack_m1_{r}" / f"M1_annotation_sheet_{r}.xlsx")

    out = {"integrity": {}, "raters": {}, "agreement": {}, "challenge": {}, "preliminary": {}}
    labels = defaultdict(dict)                      # item -> rater -> row
    for r in RATERS:
        rows, pack = sheets[r]["rows"], packs[r]["rows"]
        expected = {i for i, k in key.items() if r in k["raters"]}
        got = set(rows)
        changed = [i for i in got & set(pack)
                   if str(rows[i].get("explanation") or "") != str(pack[i].get("explanation") or "")
                   or str(rows[i].get("evidence") or "") != str(pack[i].get("evidence") or "")]
        blank = [i for i in got if yn(rows[i].get("ANY_MATERIAL_ERROR")) is None
                 and yn(rows[i].get("UNRESOLVED")) is not True]
        incons = []
        for i in got:
            row = rows[i]
            anyv = yn(row.get("ANY_MATERIAL_ERROR"))
            typ = [t for t in TYPES if yn(row.get(t))]
            if anyv is True and not typ:
                incons.append((i, "ANY=Y but no error type marked"))
            if anyv is False and typ:
                incons.append((i, f"ANY=N but types marked {typ}"))
        notes = [str(rows[i].get("notes") or "").strip() for i in got if rows[i].get("notes")]
        out["integrity"][r] = {
            "sheet": find(d, f"M1_annotation_sheet_{r}").name,
            "expected_items": len(expected), "returned_items": len(got),
            "missing": sorted(expected - got)[:20], "unexpected": sorted(got - expected)[:20],
            "text_changed": changed[:20], "n_text_changed": len(changed),
            "blank_label": blank[:20], "n_blank": len(blank),
            "n_inconsistent": len(incons), "inconsistent": incons[:15],
            "dates": {k: str(v) for k, v in sheets[r]["meta"].items()}}
        any_y = sum(1 for i in got if yn(rows[i].get("ANY_MATERIAL_ERROR")))
        out["raters"][r] = {
            "any_error_rate": round(any_y / max(1, len(got)), 3),
            "unresolved": sum(1 for i in got if yn(rows[i].get("UNRESOLVED"))),
            "types_used": dict(Counter(t for i in got for t in TYPES if yn(rows[i].get(t)))),
            "confidence": dict(Counter(str(rows[i].get("confidence") or "").strip() for i in got)),
            "informative_N": sum(1 for i in got if yn(rows[i].get("informative_for_credit_officer")) is False),
            "notes": len(notes), "distinct_notes": len(set(notes)),
            "most_repeated_note": Counter(notes).most_common(1)[0] if notes else None}
        for i in got:
            labels[i][r] = rows[i]

    # hidden duplicates: the same rater's two labels of one explanation
    for r in RATERS:
        pairs = [(i, k["duplicate_of"]) for i, k in key.items() if k.get("duplicate_of") and r in k["raters"]]
        same = [yn(labels[i][r].get("ANY_MATERIAL_ERROR")) == yn(labels[o][r].get("ANY_MATERIAL_ERROR"))
                for i, o in pairs if r in labels.get(i, {}) and r in labels.get(o, {})]
        out["raters"][r]["duplicate_consistency"] = f"{sum(same)}/{len(same)}"

    # challenge items: planted truth is known
    for r in RATERS:
        hit = Counter()
        tot = Counter()
        type_hit = Counter()
        for i, k in key.items():
            if k["source"] != "challenge" or r not in labels.get(i, {}):
                continue
            row = labels[i][r]
            err = k.get("injected_error")
            anyv = yn(row.get("ANY_MATERIAL_ERROR"))
            cls = err or "clean_control"
            tot[cls] += 1
            hit[cls] += int((anyv is True) if err else (anyv is False))
            if err and yn(row.get(err)):
                type_hit[err] += 1
        out["challenge"][r] = {c: f"{hit[c]}/{tot[c]}" for c in sorted(tot)}
        out["challenge"][r]["_correct_type_named"] = f"{sum(type_hit.values())}/{sum(v for c, v in tot.items() if c != 'clean_control')}"
        errs = sum(v for c, v in tot.items() if c != "clean_control")
        out["challenge"][r]["_planted_recall"] = round(sum(hit[c] for c in tot if c != "clean_control") / max(1, errs), 3)
        out["challenge"][r]["_clean_specificity"] = round(hit["clean_control"] / max(1, tot["clean_control"]), 3)

    # agreement on primary natural items (two raters each), and the adjudication list
    pair_lab = defaultdict(lambda: ([], []))
    adjud, agreed = [], {}
    for i, k in key.items():
        if k.get("duplicate_of") or len(k["raters"]) != 2:
            continue
        r1, r2 = k["raters"]
        if r1 not in labels.get(i, {}) or r2 not in labels.get(i, {}):
            continue
        l1, l2 = labels[i][r1], labels[i][r2]
        a1, a2 = yn(l1.get("ANY_MATERIAL_ERROR")), yn(l2.get("ANY_MATERIAL_ERROR"))
        u = yn(l1.get("UNRESOLVED")) or yn(l2.get("UNRESOLVED"))
        if k["source"] == "natural" and a1 is not None and a2 is not None:
            pa, pb = pair_lab[(r1, r2)]
            pa.append(a1); pb.append(a2)
        if a1 is None or a2 is None or a1 != a2 or u:
            adjud.append({"item_id": i, "source": k["source"], "dataset": k["dataset"],
                          "planted_error": k.get("injected_error") or "",
                          f"rater1": r1, "r1_ANY": l1.get("ANY_MATERIAL_ERROR"),
                          "r1_types": ";".join(t for t in TYPES if yn(l1.get(t))),
                          "r1_quote": l1.get("error_quote") or "", "r1_notes": l1.get("notes") or "",
                          "rater2": r2, "r2_ANY": l2.get("ANY_MATERIAL_ERROR"),
                          "r2_types": ";".join(t for t in TYPES if yn(l2.get(t))),
                          "r2_quote": l2.get("error_quote") or "", "r2_notes": l2.get("notes") or "",
                          "explanation": l1.get("explanation") or "", "evidence": l1.get("evidence") or "",
                          "ADJUDICATED_ANY": "", "adjudicated_types": "", "adjudication_note": ""})
        else:
            agreed[i] = a1
    allp = [x for v in pair_lab.values() for x in v[0]], [x for v in pair_lab.values() for x in v[1]]
    out["agreement"] = {f"{p[0]}+{p[1]}": {"n": len(v[0]),
                                           "raw": round(sum(x == y for x, y in zip(*v)) / max(1, len(v[0])), 3),
                                           "kappa": kappa(*v)} for p, v in sorted(pair_lab.items())}
    out["agreement"]["all_pairs"] = {"n": len(allp[0]),
                                     "raw": round(sum(x == y for x, y in zip(*allp)) / max(1, len(allp[0])), 3),
                                     "kappa": kappa(*allp)}
    out["agreement"]["to_adjudicate"] = len(adjud)
    out["agreement"]["agreed_primary"] = len(agreed)

    # preliminary verifier scores on agreed natural items (development only)
    try:
        from build_pack_m1 import eo_from
        from generate import Candidate
        from verify import STRUCTURAL_VERSION, structural_check
        latest = {}
        for line in open(M1 / "records.jsonl"):
            rec = json.loads(line)
            latest[(rec["borrower_id"], rec["arm"], rec["family"])] = rec
        conf = defaultdict(Counter)
        by_prod = defaultdict(lambda: [0, 0])
        for i, truth in agreed.items():
            k = key[i]
            if k["source"] != "natural":
                continue
            rec = latest.get((k["borrower_id"], k["arm"], k["family"]))
            if rec is None:
                continue
            bp = by_prod[f"{k['family']}|{k['arm']}"]
            bp[0] += int(truth); bp[1] += 1
            s = structural_check(Candidate(borrower_id=rec["borrower_id"], arm=rec["arm"], text=rec["text"],
                                           claims=rec.get("claims") or [], family=rec["family"]),
                                 eo_from(rec["evidence"]))
            verdicts = {STRUCTURAL_VERSION: s.accepted}
            for j, v in (rec.get("judges") or {}).items():
                if not v.get("error"):
                    verdicts[j] = bool(v.get("accepted"))
            for name, acc in verdicts.items():
                conf[name][("err" if truth else "ok", "acc" if acc else "rej")] += 1
        res = {}
        for name, c in conf.items():
            ne, nc = c[("err", "acc")] + c[("err", "rej")], c[("ok", "acc")] + c[("ok", "rej")]
            res[name] = {"false_accept_a": f"{c[('err', 'acc')]}/{ne}", "false_reject_b": f"{c[('ok', 'rej')]}/{nc}",
                         "a": round(c[("err", "acc")] / ne, 3) if ne else None,
                         "b": round(c[("ok", "rej")] / nc, 3) if nc else None}
        out["preliminary"] = {"note": "agreed natural items only, before adjudication: development estimates",
                              "verifiers": res,
                              "error_rate_by_producer_arm": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_prod.items())}}
    except Exception as e:  # noqa: BLE001 - scoring the sheets must not depend on this step
        out["preliminary"] = {"error": f"{type(e).__name__}: {e}"}

    (M1 / "annotation_scores.json").write_text(json.dumps(out, indent=1, default=str))
    with open(M1 / "adjudication_m1.csv", "w", newline="") as f:
        if adjud:
            w = csv.DictWriter(f, fieldnames=list(adjud[0]))
            w.writeheader()
            w.writerows(adjud)
    print(json.dumps({k: out[k] for k in ("integrity", "raters", "challenge", "agreement")}, indent=1, default=str))
    print(json.dumps(out["preliminary"], indent=1, default=str))
    print(f"\nwrote {M1 / 'annotation_scores.json'} and {M1 / 'adjudication_m1.csv'} ({len(adjud)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

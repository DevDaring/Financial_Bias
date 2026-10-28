"""
Blinded annotation pack for the fresh evaluation v2 (Improvement_Plan WS4.5). Same design as the locked v1 pack:
every narrative (774) and a sample of 200 templates, two raters each (AS, NK, SP, AB in a balanced rotation of all
six pairs), a borrower's narrative and template to disjoint pairs, 6% hidden duplicates, raters blind to producer,
prompt style, verifier and arm. Disagreements go to an independent adjudicator (not an author), blind to verifiers.

    python build_fresh_pack_v2.py
Outputs: Submission/packs/Fresh_annotation_pack_<R>.zip (send)
         Submission/annotation_key_fresh_v2.jsonl (KEEP LOCAL)
         Submission/locked/fresh_natural_v2_final.jsonl (one current row per borrower x arm) + manifest
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter

sys.path.insert(0, __import__("os").path.dirname(__file__))
import build_locked_pack as BL  # noqa: E402
import build_pack_m1 as PM  # noqa: E402
import feature_semantics as FS  # noqa: E402
from build_pack_m1 import SUB, assert_blind, blind, eo_from  # noqa: E402
from evidence_builder import render_evidence_v2  # noqa: E402

SEED = 20261016
README = BL.README.replace("Locked-test explanation annotation", "Fresh-test explanation annotation (round 2)") \
    .replace("LOCKED_annotation_sheet_", "FRESH_annotation_sheet_")
ADDENDUM = BL.ADDENDUM.replace("## Locked-test addendum (October 2026)", "## Fresh-test addendum (round 2)") + """
Some explanations in this round are written TO THE APPLICANT in everyday language ("you", "your payments").
Judge them by the same rules: everyday wording is fine; misstating a value, a direction, the strongest driver, the
probability, the field a number belongs to, or a code's undocumented meaning is a material error.
"""


def main() -> int:
    keyp = SUB / "annotation_key_fresh_v2.jsonl"
    if keyp.exists() and "--force" not in sys.argv:
        print(f"{keyp} exists: built once")
        return 1
    rng = random.Random(SEED)
    latest = {}
    for l in open(SUB / "locked" / "fresh_natural_v2.jsonl"):
        r = json.loads(l)
        k = (r["borrower_id"], r["arm"])
        if k not in latest or not r.get("error"):
            latest[k] = r
    final = list(latest.values())
    with open(SUB / "locked" / "fresh_natural_v2_final.jsonl", "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    nar = [r for r in final if r["arm"] != "template" and not r.get("error") and (r.get("text") or "").strip()]
    fails = [r for r in final if r["arm"] != "template" and (r.get("error") or not (r.get("text") or "").strip())]
    tmpl = {r["borrower_id"]: r for r in final if r["arm"] == "template"}
    by_stratum = {}
    for r in nar:
        by_stratum.setdefault(r["stratum"], []).append(r["borrower_id"])
    tsample = set()
    for s, bids in sorted(by_stratum.items()):
        tsample |= set(rng.sample(sorted(bids), 50))
    man = {"narratives": len(nar), "producer_failures_excluded": len(fails),
           "by_cell": Counter(r["cell"] for r in nar), "failures_by_cell": Counter(r["cell"] for r in fails),
           "retries": sum(1 for r in final if r.get("retry")), "templates_sampled": len(tsample),
           "retry_rule": "one regeneration with a new seed after an unparseable output (prespecified)"}
    (SUB / "locked" / "fresh_natural_v2_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    bids = sorted({r["borrower_id"] for r in nar})
    rng.shuffle(bids)
    nar_by = {r["borrower_id"]: r for r in nar}
    items, key, assign = [], [], []
    for n, b in enumerate(bids):
        pair = BL.PAIRS[n % 6]
        for r, pr in ((nar_by[b], pair), (tmpl[b] if b in tsample else None, BL.COMPLEMENT[pair])):
            if r is None:
                continue
            eo = eo_from(r["evidence"])
            items.append({"eo": eo, "text": blind(r["text"], eo), "dataset": r["dataset"]})
            key.append({"source_item_id": r["item_id"], "borrower_id": b, "dataset": r["dataset"], "group": r["group"],
                        "stratum": r["stratum"], "pd_band": r["pd_band"], "arm": r["arm"], "family": r["family"],
                        "cell": r.get("cell") if r["arm"] != "template" else "template",
                        "evidence_fingerprint": r["evidence_fingerprint"],
                        "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()[:16]})
            assign.append(pr)
    order = list(range(len(items)))
    rng.shuffle(order)
    per = {q: [i for i in order if q in assign[i]] for q in BL.RATERS}
    slots = [("item", i, None) for i in order]
    for q in BL.RATERS:
        for i in rng.sample(per[q], max(1, round(BL.DUP_RATE * len(per[q])))):
            slots.append(("dup", i, q))
    ids = [f"FR-{n:04d}" for n in range(1, len(slots) + 1)]
    rng.shuffle(ids)
    iid, keyrows, dups = {}, [], {q: [] for q in BL.RATERS}
    for (kind, i, q), sid in zip(slots, ids):
        if kind == "item":
            iid[i] = sid
            keyrows.append({"item_id": sid, "raters": list(assign[i]), **key[i]})
    for (kind, i, q), sid in zip(slots, ids):
        if kind == "dup":
            dups[q].append((sid, i))
            keyrows.append({"item_id": sid, "raters": [q], "duplicate_of": iid[i], **key[i]})
    out = SUB / "packs"
    for f in out.glob("Fresh_annotation_pack_*"):
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    old = list(PM.LABEL_COLS)
    PM.LABEL_COLS[:] = old[:-3] + BL.EXTRA + old[-3:]
    glossary = FS.glossary_markdown()
    all_b = {k["borrower_id"] for k in key}
    load = {}
    try:
        for q in BL.RATERS:
            pack = out / f"Fresh_annotation_pack_{q}"
            pack.mkdir(parents=True)
            rows = [(iid[i], i) for i in per[q]] + dups[q]
            rng.shuffle(rows)
            sheet = [{"item_id": rid, "dataset": PM.DATASET_LABEL[items[i]["dataset"]],
                      "evidence": render_evidence_v2(items[i]["eo"]), "explanation": items[i]["text"]} for rid, i in rows]
            for row in sheet:
                assert_blind(row["evidence"] + " " + row["explanation"], all_b)
            PM.write_sheet(pack / f"FRESH_annotation_sheet_{q}.xlsx", sheet, q)
            (pack / "ANNOTATION_GUIDELINES.md").write_text(PM.GUIDE + ADDENDUM)
            (pack / "FEATURE_GLOSSARY.md").write_text(glossary)
            (pack / "README.txt").write_text(README.format(rater=q, n=len(sheet)))
            with zipfile.ZipFile(out / f"Fresh_annotation_pack_{q}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(pack.iterdir()):
                    z.write(f, f"Fresh_annotation_pack_{q}/{f.name}")
            load[q] = len(sheet)
    finally:
        PM.LABEL_COLS[:] = old
    keyp.write_text("\n".join(json.dumps(k, default=str) for k in [{"_meta": True, "seed": SEED}] + keyrows) + "\n")
    c = Counter((r, k["arm"]) for k in keyrows if "duplicate_of" not in k for r in k["raters"])
    print("unique items", len(order), Counter(k["arm"] for k in key), "per rater", load)
    print("rater x arm", sorted(c.items()))
    print(json.dumps(man, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

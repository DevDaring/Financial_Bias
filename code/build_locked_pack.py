"""
Blinded annotation pack for the LOCKED natural test (Future_Plan review §4, §8).

Items: the 400 fresh narratives AND the 400 deterministic templates (the actual fallback), one per borrower.
Raters: AS, NK, SP, AB (independent; the author KD only adjudicates). Every item is labelled by two raters.
A borrower's narrative and template go to COMPLEMENTARY rater pairs, so no rater sees both versions of one
borrower. Raters do not see method, model, verdict or which rows are templates. 6% hidden duplicates.
Extra columns for the template comparison (content floor, review §8): conveys_pd,
conveys_top_driver_with_direction, useful_for_officer (1-5).

    python build_locked_pack.py
Outputs: Submission/packs/Locked_annotation_pack_<R>.zip (send)
         Submission/annotation_key_locked_v1.jsonl (KEEP LOCAL; never send)
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pack_m1 as PM  # noqa: E402
import feature_semantics as FS  # noqa: E402
from build_pack_m1 import SUB, blind, assert_blind, eo_from  # noqa: E402
from evidence_builder import render_evidence_v2  # noqa: E402

SEED = 20261010
RATERS = ["AS", "NK", "SP", "AB"]
PAIRS = [("AS", "NK"), ("SP", "AB"), ("AS", "SP"), ("NK", "AB"), ("AS", "AB"), ("NK", "SP")]
COMPLEMENT = {p: tuple(r for r in ("AS", "NK", "SP", "AB") if r not in p) for p in PAIRS}
DUP_RATE = 0.06
EXTRA = ["conveys_pd", "conveys_top_driver_with_direction", "useful_for_officer"]

ADDENDUM = """

---

## Locked-test addendum (October 2026)

This round contains explanations for NEW borrowers. Some explanations are short and formulaic; some are
longer and written in free prose. Label every row on its own with the same rules as before.

Three extra columns (fill for EVERY row):

- **conveys_pd** (Y/N): the text states the model's probability of default correctly (rounding is fine).
- **conveys_top_driver_with_direction** (Y/N): the text names the attribution with the largest absolute
  contribution AND says correctly whether it raises or lowers the risk score.
- **useful_for_officer** (1-5): how useful the text is to a credit officer who must understand why the model
  scored this borrower as it did (1 = useless or misleading, 3 = adequate, 5 = clear and complete).
  Judge usefulness from the content, not the length or style.

ANY_MATERIAL_ERROR keeps its meaning: one material error anywhere makes it Y.
"""

README = """Locked-test explanation annotation ({rater})
============================================

Your annotator code: {rater}
Items in your sheet: {n}

Files
  README.txt                       this file
  ANNOTATION_GUIDELINES.md         the same rules as round 1 plus a short addendum at the end (READ the addendum)
  FEATURE_GLOSSARY.md              what each field means, both datasets
  LOCKED_annotation_sheet_{rater}.xlsx   fill the yellow columns

Steps
  1. Read the addendum at the end of ANNOTATION_GUIDELINES.md (5 minutes) and re-read section 4.
  2. Fill the Annotator tab (your code only) and label every row.
  3. Return the sheet unchanged in name.

Rules
  - Work independently; do not discuss items until everyone has returned.
  - Do not use ChatGPT or any AI tool.
  - The material is for this study only. Do not share it.
"""


def main() -> int:
    keyp = SUB / "annotation_key_locked_v1.jsonl"
    if keyp.exists() and "--force" not in sys.argv:
        print(f"{keyp} exists: the pack was built (and may have been sent); not rebuilding")
        return 1
    rng = random.Random(SEED)
    recs = [json.loads(l) for l in open(SUB / "locked" / "fresh_natural_v1.jsonl")]
    recs = [r for r in recs if (r.get("text") or "").strip() and not r.get("error")]
    by_b = {}
    for r in recs:
        by_b.setdefault(r["borrower_id"], {})[r["arm"]] = r
    bids = sorted(by_b)
    rng.shuffle(bids)
    items, key, assign = [], [], []
    for n, b in enumerate(bids):
        pair = PAIRS[n % 6]          # every rater sees both arms equally often
        for arm, pr in (("grounded", pair), ("template", COMPLEMENT[pair])):
            r = by_b[b].get(arm)
            if r is None:
                continue
            eo = eo_from(r["evidence"])
            items.append({"eo": eo, "text": blind(r["text"], eo), "dataset": r["dataset"]})
            key.append({"source_item_id": r["item_id"], "borrower_id": b, "dataset": r["dataset"], "group": r["group"],
                        "stratum": r["stratum"], "pd_band": r["pd_band"], "arm": arm, "family": r["family"],
                        "model_id": r.get("model_id"), "evidence_fingerprint": r["evidence_fingerprint"],
                        "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()[:16]})
            assign.append(pr)
    order = list(range(len(items)))
    rng.shuffle(order)
    per_rater = {q: [i for i in order if q in assign[i]] for q in RATERS}
    slots = [("item", i, None) for i in order]
    for q in RATERS:
        for i in rng.sample(per_rater[q], max(1, round(DUP_RATE * len(per_rater[q])))):
            slots.append(("dup", i, q))
    ids = [f"LK-{n:04d}" for n in range(1, len(slots) + 1)]
    rng.shuffle(ids)
    iid, keyrows, dups = {}, [], {q: [] for q in RATERS}
    for (kind, i, q), sid in zip(slots, ids):
        if kind == "item":
            iid[i] = sid
            keyrows.append({"item_id": sid, "raters": list(assign[i]), **key[i]})
    for (kind, i, q), sid in zip(slots, ids):
        if kind == "dup":
            dups[q].append((sid, i))
            keyrows.append({"item_id": sid, "raters": [q], "duplicate_of": iid[i], **key[i]})
    out = SUB / "packs"
    for f in out.glob("Locked_annotation_pack_*"):
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    old_cols = list(PM.LABEL_COLS)
    PM.LABEL_COLS[:] = old_cols[:-3] + EXTRA + old_cols[-3:]
    glossary = FS.glossary_markdown()
    all_b = {k["borrower_id"] for k in key}
    load = {}
    try:
        for q in RATERS:
            pack = out / f"Locked_annotation_pack_{q}"
            pack.mkdir(parents=True)
            rows = [(iid[i], i) for i in per_rater[q]] + dups[q]
            rng.shuffle(rows)
            sheet = [{"item_id": rid, "dataset": PM.DATASET_LABEL[items[i]["dataset"]],
                      "evidence": render_evidence_v2(items[i]["eo"]), "explanation": items[i]["text"]}
                     for rid, i in rows]
            for row in sheet:
                assert_blind(row["evidence"] + " " + row["explanation"], all_b)
            PM.write_sheet(pack / f"LOCKED_annotation_sheet_{q}.xlsx", sheet, q)
            (pack / "ANNOTATION_GUIDELINES.md").write_text(PM.GUIDE + ADDENDUM)
            (pack / "FEATURE_GLOSSARY.md").write_text(glossary)
            (pack / "README.txt").write_text(README.format(rater=q, n=len(sheet)))
            with zipfile.ZipFile(out / f"Locked_annotation_pack_{q}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(pack.iterdir()):
                    z.write(f, f"Locked_annotation_pack_{q}/{f.name}")
            load[q] = len(sheet)
    finally:
        PM.LABEL_COLS[:] = old_cols
    meta = {"_meta": True, "seed": SEED, "raters": RATERS, "source": "Submission/locked/fresh_natural_v1.jsonl",
            "frozen_before_eval": json.load(open(SUB / "frozen_before_eval.json"))["source_sha256_16"]}
    keyp.write_text("\n".join(json.dumps(k, default=str) for k in [meta] + keyrows) + "\n")
    print("unique items", len(order), Counter(k["arm"] for k in key), "per-rater", load,
          "rating-hours at 1.5 min:", round(sum(load.values()) * 1.5 / 60, 1))
    print("send:", ", ".join(f"Locked_annotation_pack_{q}.zip" for q in RATERS), "| KEEP LOCAL:", keyp.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

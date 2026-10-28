"""
Anonymous review archive (Improvement_Plan WS6.5). Builds a LOCAL zip; publishing it (e.g. anonymous.4open.science
or Zenodo) is a separate step that needs the authors' approval.

Included: all research code; frozen verifier records and hashes; evidence objects and narratives (derived from the
public Taiwan and Home Credit datasets); every verifier output; the context packs; the blinded annotation materials
(guidelines, glossary); anonymous per-item FINAL labels (no rater codes, no rater notes, no annotator metadata).
Excluded: credentials (.env), annotation keys and raw annotation sheets, backups and logs, LLM call traces,
IDBI documents, raw datasets, any Amex-derived data.

    python build_review_archive.py
Output: Submission/review_archive/ReCheck_review_archive.zip
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dev_sets import locked_items, m1_items  # noqa: E402
from score_m1 import SUB  # noqa: E402

ROOT = SUB.parent
CODE = Path(__file__).resolve().parent
OUT = SUB / "review_archive"
EXCLUDE = re.compile(r"annotation_key|annotator|Annotated|\.bak|purge_log|\.env|llm_trace|kaggle|IDBI_Documents|amex|"
                     r"_superseded|\.log$|__pycache__|practitioner_key|adjudication_.*\.csv", re.I)

README = """# ReCheck: review artifact (anonymous)

Contents
- `code/`: every script of the study (Python 3.11; requirements: openai, boto3, lightgbm, shap, pandas, numpy,
  openpyxl, matplotlib). Model routes and keys are read from a `.env` that is NOT included.
- `frozen/`: freeze records with SHA-256 prefixes of the verifier sources and of the fresh-test outputs
  (`frozen_v3_prespec.json`), the earlier freeze (`frozen_before_eval.json`), GPU cost.
- `data/m1/`: development narratives with evidence (`records.jsonl`) and all verifier outputs (`comparators/`).
- `data/locked/`: locked-v1 and fresh-v2 narratives with evidence, all verifier outputs (`*_runs*/`), analyses.
- `data/mechanism/`: controlled context packs (labels by construction) and runs.
- `labels/`: anonymous final labels per item (1 = material error) and the label basis; no rater information.
- `materials/`: blinded annotation guidelines and glossary as given to raters.

Reproduce the tables
    cd code
    python analyze_v2.py; python analyze_context.py --tag v1; python analyze_context.py --tag v2 --ref exec_R11
    python analyze_isolation.py; python analyze_dev_v16.py; python analyze_locked.py [--fresh-v2]
    python analyze_utility.py; python make_paper_assets_v3.py; python fill_paper.py
(the label-reading steps use the anonymous files in labels/ in place of the raw sheets; see labels/README.md)
"""


def label_files(z):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["item_id", "material_error", "basis"])
    for x in m1_items():
        w.writerow([x["item_id"], int(x["truth"]), "final"])
    z.writestr("labels/labels_m1_development.csv", buf.getvalue())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["item_id", "material_error"])
    for x in locked_items():
        w.writerow([x["item_id"], int(x["truth"])])
    z.writestr("labels/labels_locked_v1_narratives.csv", buf.getvalue())
    # fresh v2: final labels as used for the primary analysis (item, label, basis only; no rater codes or notes)
    import analyze_locked as AL
    AL.ADJ_MODE = "provided"
    AL.CFG.update(key=SUB / "annotation_key_fresh_v2.jsonl", sheet="FRESH_annotation_sheet_",
                  adjudication=SUB / "locked" / "adjudication_fresh_v2.csv")
    lab, _dis, _dup = AL.labels(SUB / "packs" / "Annotated" / "IDBI_Annotator_Checks")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["item_id", "arm", "material_error", "basis"])
    for i, v in sorted(lab.items()):
        w.writerow([i, "template" if v["arm"] == "template" else "narrative", int(v["truth"]), v["basis"]])
    z.writestr("labels/labels_fresh_v2.csv", buf.getvalue())
    z.writestr("labels/README.md", "Final labels: two independent raters per item; disagreements adjudicated "
                                    "(see the paper; basis = agreed or adjudicated). Rater identities, notes and metadata are not released.\n")


def add_tree(z, src: Path, arc: str):
    for p in sorted(src.rglob("*")):
        if p.is_file() and not EXCLUDE.search(str(p)):
            z.write(p, f"{arc}/{p.relative_to(src)}")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    zp = OUT / "ReCheck_review_archive.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README)
        for p in sorted(CODE.glob("*.py")):
            z.write(p, f"code/{p.name}")
        for f in ("frozen_before_eval.json", "frozen_v3_prespec.json", "gpu_cost.json"):
            if (SUB / f).exists():
                z.write(SUB / f, f"frozen/{f}")
        for f in ("records.jsonl", "analysis_v2.json", "label_sensitivity.json", "wording_intervention.json",
                  "label_lineage.json", "replay_same_transcript_qwen3_next.json", "replay_same_transcript_grok45.json",
                  "substrate_taiwan.json", "substrate_home_credit.json", "roles_taiwan.json", "roles_home_credit.json"):
            if (SUB / "m1" / f).exists():
                z.write(SUB / "m1" / f, f"data/m1/{f}")
        add_tree(z, SUB / "m1" / "comparators", "data/m1/comparators")
        add_tree(z, SUB / "locked", "data/locked")
        add_tree(z, SUB / "mechanism", "data/mechanism")
        for f in ("ANNOTATION_GUIDELINES.md", "FEATURE_GLOSSARY.md"):
            p = SUB / "packs" / "Fresh_annotation_pack_AS" / f
            if p.exists():
                z.write(p, f"materials/{f}")
        label_files(z)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    bad = [n for n in names if EXCLUDE.search(n)]
    print(zp, len(names), "files,", round(zp.stat().st_size / 1e6, 1), "MB; excluded-pattern hits:", bad[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

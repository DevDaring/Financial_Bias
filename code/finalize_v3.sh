#!/bin/bash
# Score the fresh test with the PRESPECIFIED primary (frozen_v3_prespec.json) and rebuild the v3 manuscript.
# Run after the FRESH_annotation_sheet_* files are under Submission/Annotated/ and the independent adjudicator's
# decisions are in Submission/locked/adjudication_fresh_v2.csv (item_id, ADJUDICATED_ANY).
set -e
cd "$(dirname "$0")"
../../.venv/bin/python analyze_locked.py --fresh-v2 --primary exec_P --baseline facts_judge --labels-dir ../../Submission/packs/Annotated/IDBI_Annotator_Checks
../../.venv/bin/python make_paper_assets_v3.py
../../.venv/bin/python fill_paper.py            # fails if any token is still missing
cd ../../Submission/ESWA_Submission && latexmk -pdf -interaction=nonstopmode ESWA_IDBI_Hackathon.tex > /dev/null
grep -c "^! " ESWA_IDBI_Hackathon.log || true

# ReCheck: code and results

Code, frozen records, model outputs and final labels for the paper

> Koushik Deb and Abhinaba Basu. *ReCheck: executing claims and judge objections to verify large-language-model explanations of credit risk.* Submitted to *Expert Systems* (Wiley), 2026.

ReCheck checks LLM-written explanations of credit-risk predictions. Deterministic code recomputes every claim it can compute from the model's evidence: values, directions and rankings. An LLM judge assesses the remaining claims. Code also checks every objection the judge raises. When a dispute remains, the explanation is withheld and an audited template is shown instead.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Every script of the study (Python 3.11). Requirements: openai, boto3, lightgbm, shap, pandas, numpy, openpyxl, matplotlib. `code/README_code.md` describes the modules. |
| `frozen/` | Freeze records: SHA-256 prefixes of the verifier sources and of the fresh-test outputs (`frozen_v3_prespec.json`), the earlier freeze (`frozen_before_eval.json`), and GPU use. |
| `data/m1/` | Development narratives with their evidence (`records.jsonl`) and all verifier outputs (`comparators/`). |
| `data/locked/` | Locked-test v1 and fresh-test v2 narratives with evidence, all verifier outputs (`*_runs*/`), and the analyses. `fresh_v2_analysis_independent_adjudication.json` is the primary analysis reported in the paper. |
| `data/mechanism/` | The controlled context and paraphrase packs (labels by construction) and their runs. |
| `labels/` | Final per-item labels (1 = material error). The fresh-test labels use the independent adjudication reported in the paper. |
| `materials/` | The blinded annotation guidelines and field glossary given to raters. |

## Reproducing the tables

```
cd code
python analyze_v2.py
python analyze_context.py --tag v1
python analyze_context.py --tag v2 --ref exec_R11
python analyze_isolation.py
python analyze_dev_v16.py
python analyze_locked.py --fresh-v2
python analyze_utility.py
python make_paper_assets_v3.py
```

The label-reading steps take their labels from `labels/` in place of the raw annotation sheets. Rerunning the verifiers needs model access. Model routes and API keys are read from a `.env` file, which is not included.

## Data

The evidence objects and narratives are derived from two public datasets:
- the Default of Credit Card Clients dataset (Yeh & Lien, 2009);
- the Home Credit Default Risk dataset (Kaggle, 2018).

The raw datasets are not redistributed here.

Each item in `labels/` keeps only its final label and its basis (agreed or adjudicated). Rater identities, rater codes, notes, timings and all other annotator information are excluded, and so are credentials and LLM call traces.

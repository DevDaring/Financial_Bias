# research_llm — experiment code for the ESWA study

Implements the method in [`Submission/Future_Plan.md`](../../Submission/Future_Plan.md):
*evidence-constrained credit explanations with group-audited selective verification*.

This is **research code**, separate from the production app in `saarthi/`. It
does not import the production gateway, because the production gateway has the
P0 defect this package exists to fix.

```
providers.py   transports only (endpoint + keys). NOT a research identity.
registry.py    model FAMILIES, pinned model ids, routes, fidelity grading
gateway.py     call a family; fallback never crosses families; JSONL trace
evidence.py    typed evidence contract (§4.1)
generate.py    template / grounded / typed arms + E2 cue injection (§4.2)
verify.py      structural, semantic and combined checkers + corruptions (§4.2)
policy.py      group metrics, release policies, certificate, decomposition (§4.3–4.5)
substrate.py   E0 predictor with all four P0 leakage fixes
run_pilot.py   the bounded pilot from §9, end to end
contract.py          contract/1.1 rules, evidence validation, objection checks, four detectors (§4.1-4.2)
contract_verifier.py earlier contract-aware verifier (B3 in Future_Plan §E1)
witness.py           executable/1.2: typed operations + objection witnesses, recomputed by code (Future_Plan §D1)
executable_verifier.py  the proposed method P and ablations A0/A1/A2/PX (Future_Plan §D2)
comparators.py       B1 generic, B2 same-tool, B2' second-look, five reimplemented studies, controls, release selection
replay_objections.py zero-cost replay of cached judge objections through execution; cluster bootstrap; conflicts list
build_mechanism_pack.py / eval_mechanism.py   E2 mechanism packs (v1 dev, v2 held-out scored once) and scoring
audit_counts.py      denominators (log rows, outputs, borrowers, strata)
feasibility.py       Clopper-Pearson sample sizes + certification simulation (Future_Plan §F)
run_comparators.py   run methods on cached M1 candidates; DRY RUN by default (no API calls)
score_m1.py          audit + score returned M1 annotation sheets; adjudication list
eval_m1_offline.py   zero-cost development scores of cached verifier signals
fakes.py             offline gateway for tests and dry runs
test_contract.py     38 unit tests (contract, verifier, comparators)
test_executable.py   56 unit tests (operations, witnesses, verifier modes, B2 tools, policy count fixes)
```

## The defect this fixes

Production `routes.py` falls back `deepseek → mistral → openrouter:llama`. A
rate-limit on one provider therefore changes the **model family** mid-run. For
a controlled experiment that is a confound: the treatment label says one model,
the wire says another, and nothing records the difference.

Here, a *family* owns one model and several *routes* that serve **that same
model**. Fallback moves between routes; it never moves between families. If
every route for a family fails, the call raises `FamilyExhausted` — a failed
observation is recorded as failed, never answered by a different model.

```python
from gateway import GATEWAY
r = GATEWAY.complete("llama", msgs)      # a family, not a provider
r.family, r.model_id, r.route_index, r.substituted
# 'llama', 'meta-llama/llama-3.1-8b-instruct', 0, False
```

## Routing

OpenRouter is the universal **same-model** backstop. Llama and Gemma lead on
NanoGPT, per project decision.

| Family | Publisher | Routes (fallback order) |
|---|---|---|
| `deepseek` | DeepSeek | deepseek → nanogpt → openrouter |
| `mistral` | Mistral AI | mistral → openrouter |
| `llama` | Meta | **nanogpt** → openrouter (both `meta-llama/llama-3.1-8b-instruct`) |
| `llama70` | Meta | **nanogpt** → openrouter |
| `gemma` | Google | openrouter `google/…` → nanogpt `unsloth/…` ⚠︎ |
| `gemini` | Google | gemini (native only) |
| `gpt4omini` | OpenAI | nanogpt → openrouter (dated snapshot) |
| `gptoss` | OpenAI | nanogpt → openrouter |
| `grok` | xAI | xai → nanogpt → openrouter |

⚠︎ **Gemma is not weight-identical across routes.** NanoGPT serves
`unsloth/gemma-3-4b-it`, a repack; OpenRouter serves Google's
`google/gemma-3-4b-it`. The registry grades that route `VARIANT` and strict
mode refuses it, so a repack can never silently stand in for the reference
build. Pass `allow_variant=True` to accept it, and report it as a different
model.

NanoGPT uses `NanoGPT_88_API_Key` first. Note `.env` also defines
`NanoGPT_Base_URL=https://api.nanogpt.com/v1`, which does not resolve;
`Nano_GPT_Base_URL` is the working endpoint.

## Judge independence

Publisher, not family. Gemini and Gemma are both Google, so a Gemma judge on a
Gemini explanation is not an independent check — `judged_by()` raises
`JudgeIndependenceError`. This caught a real error in the first draft of the
role assignment.

```
producers: deepseek (DeepSeek) · llama (Meta) · mistral (Mistral AI)
judges   : gemini (Google) · gpt4omini (OpenAI) · grok (xAI)
→ six distinct publishers, 9 cross-publisher pairs
```

## P0 leakage fixes (`substrate.py`)

| Plan item | Fix | Asserted by |
|---|---|---|
| P0-1 preprocessing fitted on the whole frame | `Fit()` fits inside the training fold, then frozen | medians differ fit-fold vs all |
| P0-2 global prior may have seen local test rows | off by default; refit per outer fold if enabled | — |
| P0-3 fairness audit sees fitting rows | subgroup rates from held-out predictions only | `n_test == split_sizes['test']` |
| P0-4 small files share calibration and test folds | disjoint folds; **refuses** below 400 rows / 40 positives | raises `InsufficientData` |

`python substrate.py` runs these assertions.

## Metrics that refuse to flatter (`policy.py`)

All five are always reported, because four of them can be gamed alone:

* `C_g` coverage · `R_g` selective error · `H_g` exposure · `U_g` useful rate · burden
* `R_g` is **`None`** when nothing is released — never `0`. Zero released
  narratives is undefined error, not perfect error.
* Selection reports **infeasible** rather than relaxing ε after seeing results.

Certificate is simultaneous Clopper–Pearson with `δ/(MG)` allocation, and
reproduces the plan's worked numbers exactly: `n=100, k=0, α=0.025 → 3.62%`,
and `n ≥ 72` to certify below 5%.

## Running

```bash
python providers.py                 # which transports are reachable
python registry.py                  # families, routes, publisher independence
python substrate.py                 # P0 assertions + model/calibrator grid
python verify.py                    # corruption detection matrix
python policy.py                    # policy sweep on synthetic records
python gateway.py report            # trace summary: anything off primary route?

python run_pilot.py --dataset taiwan --n 24            # live
python run_pilot.py --dataset sba --n 8 --dry-run      # no LLM calls
```

Artifacts land in `Submission/`: `pilot_report.json`, `pilot_records.jsonl`,
`llm_trace.jsonl` (one line per **attempt**, with the model id actually sent).

## What is NOT implemented, deliberately

The pilot labels violations with the **structural checker as a proxy** so the
plumbing can be exercised. E1 requires *independent annotation* — the plan is
explicit that the production judge must never supply the labels used to score
it. Every report is stamped `LABELS: PROXY`. No number from `run_pilot.py` is a
publishable result.

Also not implemented: the recourse rework (P1), the E2 cue study runner, and
the human-annotation interface.

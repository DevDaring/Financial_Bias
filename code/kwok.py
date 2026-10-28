"""
LLM-as-a-Verifier (Kwok et al. 2026, arXiv:2607.05391), adapted: a continuous verification score from the
EXPECTATION over the score-token distribution, with criteria decomposition and repeated evaluation.

  score(x) = mean over criteria c and repeats k of  E[s | x, c, k] rescaled to [0, 1],
  E[s] = sum_s s * p(s) / sum_s p(s),  s in 1..9 (single-token digits; first generated token's top-20 log-probs)

Criteria (the twelve contract rules grouped; fixed before any scoring):
  numbers   R_VALUE, R_SCALE
  roles     R_DIRECTION, R_FACT_AS_DRIVER, R_RANKING
  support   R_INVENTED, R_CAUSAL, R_TEMPORAL, R_CODE_MEANING, R_TARGET, R_COMPLETENESS
  coverage  R_OMISSION
Repeats: three fixed permutations of the evidence item order (temperature 0).
Departures: Kwok et al. score solution quality for ranking candidates; here the score is the probability-
weighted rating that an explanation is free of material error, used for selective release.

Requires a self-hosted OpenAI-compatible server that returns log-probabilities (vLLM): VAST_VLLM_URL/KEY.

    python kwok.py --items final --workers 32     -> Submission/m1/comparators/kwok__gemma27_vllm.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as C  # noqa: E402

CRITERIA = {
    "numbers": ["R_VALUE", "R_SCALE"],
    "roles": ["R_DIRECTION", "R_FACT_AS_DRIVER", "R_RANKING"],
    "support": ["R_INVENTED", "R_CAUSAL", "R_TEMPORAL", "R_CODE_MEANING", "R_TARGET", "R_COMPLETENESS"],
    "coverage": ["R_OMISSION"],
}
SYS = ("You verify a credit-risk explanation against the model evidence it must rest on. The evidence lists "
       "recorded facts, the model's attributions (contributions on the raw log-odds scale, with a direction) and "
       "the calibrated probability of default. Plain-language restatement, sensible rounding, documented unit "
       "conversions and disclaimers that deny causation are fine.")
ASK = ("Consider ONLY these rules:\n{rules}\n\nOn a scale of 1 to 9, how certain are you that the explanation "
       "breaks NONE of these rules? 1 = it certainly breaks one, 9 = it certainly breaks none. "
       "Reply with a single digit.")
_RULES = {ln.split(":", 1)[0].strip(): ln for ln in C.rule_text().splitlines() if ":" in ln}
REPEATS = 3


def _client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ["VAST_VLLM_URL"], api_key=os.environ["VAST_VLLM_KEY"], timeout=300,
                  max_retries=2)


def expected_digit(top) -> tuple:
    ps = {}
    for t in top:
        tok = t.token.strip()
        if len(tok) == 1 and tok in "123456789":
            ps[int(tok)] = ps.get(int(tok), 0.0) + math.exp(t.logprob)
    z = sum(ps.values())
    if z <= 0:
        return None, 0.0
    return sum(s * p for s, p in ps.items()) / z, z


def score_item(cli, model: str, evidence: dict, text: str, seed: int) -> dict:
    out = {"criteria": {}, "mass": []}
    for k in range(REPEATS):
        ev = json.loads(json.dumps(evidence, default=str))
        if k and isinstance(ev.get("items"), list):
            random.Random(seed * 31 + k).shuffle(ev["items"])
        user = json.dumps({"evidence": ev, "explanation": text}, default=str)[:24000]
        for c, rules in CRITERIA.items():
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": user + "\n\n" + ASK.format(rules="\n".join(_RULES[r] for r in rules))}]
            r = cli.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=1,
                                            logprobs=True, top_logprobs=20)
            lp = r.choices[0].logprobs.content[0].top_logprobs if r.choices[0].logprobs else []
            e, mass = expected_digit(lp)
            out["criteria"].setdefault(c, []).append(None if e is None else (e - 1) / 8)
            out["mass"].append(round(mass, 4))
            out.setdefault("tokens", 0)
            out["tokens"] += (r.usage.prompt_tokens or 0) if r.usage else 0
    per = {c: (sum(v for v in vs if v is not None) / max(1, sum(v is not None for v in vs))) for c, vs in out["criteria"].items()}
    out["per_criterion"] = per
    out["score_mean"] = sum(per.values()) / len(per)
    out["score_min"] = min(per.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="final")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--source", default="m1", help="m1 | path to a jsonl of {item_id, evidence, text, truth?}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from run_comparators import OUT, load_items
    if a.source == "m1":
        items = [(i, rec["evidence"], rec["text"], truth, {"group": k.get("group"), "dataset": rec["dataset"]})
                 for i, rec, truth, k in load_items(a.items)]
        dst = Path(a.out) if a.out else OUT / "kwok__gemma27_vllm.jsonl"
    else:
        items = [(r["item_id"], r["evidence"], r["text"], r.get("truth"), {"group": r.get("group"),
                  "dataset": r.get("dataset")}) for r in map(json.loads, open(a.source))]
        dst = Path(a.out)
    from build_pack_m1 import eo_from
    done = {json.loads(l)["item_id"] for l in open(dst)} if dst.exists() else set()
    cli = _client()
    lock = threading.Lock()
    todo = [x for x in items if x[0] not in done]
    print(len(todo), "items to score")

    def work(x):
        i, ev, text, truth, meta = x
        t0 = time.time()
        try:
            evd = eo_from(ev).for_generator() if a.source == "m1" else ev
            s = score_item(cli, a.model, evd, text, seed=int(hashlib.md5(i.encode()).hexdigest(), 16) % 10000)
            row = {"item_id": i, "method": "kwok_verifier", "judge": "gemma27_vllm", "truth": truth, **meta,
                   "score": s["score_mean"], "detail": s, "seconds": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            row = {"item_id": i, "method": "kwok_verifier", "judge": "gemma27_vllm", "truth": truth, **meta,
                   "score": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        with lock, open(dst, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)

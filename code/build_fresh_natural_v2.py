"""
Fresh evaluation set v2 (Improvement_Plan WS4): narratives for NEW locked-evaluation borrowers, never used before
(the 400 borrowers of fresh_natural_v1 are excluded). Generated before the repaired verifier is frozen: generation
does not depend on any verifier, and these texts are not read or scored until the freeze record exists.

Design: 200 borrowers per dataset x audit-group stratum (800), equal over the stratum's three PD bands; four
producer cells assigned in rotation within each stratum x band (50 per cell per stratum):
  nova_lite / grounded      (seen producer, seen prompt)
  ministral_8b / grounded   (seen producer, seen prompt)
  glm47 / grounded          (UNSEEN producer: Z.AI, never a producer or judge in this study)
  ministral_8b / applicant  (NEW prompt style: written to the applicant in everyday language)
Templates are rendered for every borrower; a 200-borrower sample goes to annotation.

    python build_fresh_natural_v2.py --per-stratum 200
Output: Submission/locked/fresh_natural_v2.jsonl (+ manifest)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack_m1 import SUB  # noqa: E402

OUT = SUB / "locked"
SEED = 20261012
CELLS = [("nova_lite", "grounded"), ("ministral_8b", "grounded"), ("glm47", "grounded"), ("ministral_8b", "applicant")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    path = OUT / "fresh_natural_v2.jsonl"
    import joblib
    from evidence_builder import GroupedShap, build_evidence_v2
    from generate import PROMPT_VERSION, generate, render_template_v2
    from run_m1 import load_corpus, model_hash, role_of
    used = {json.loads(l)["borrower_id"] for l in open(OUT / "fresh_natural_v1.jsonl")}
    done, failed = {}, set()
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            if not r.get("error"):
                done[(r["borrower_id"], r["arm"])] = r
            else:
                failed.add((r["borrower_id"], r["arm"]))
    failed -= set(done)
    m1 = SUB / "m1"
    man = {"version": "fresh-natural/2.0", "seed": SEED, "cells": CELLS, "per_stratum": a.per_stratum,
           "excluded_v1_borrowers": len(used), "role": "locked_evaluation", "strata": {},
           "note": "generated before the repaired verifier is frozen; not read or scored until then"}
    jobs = []
    for ds in ("taiwan", "home_credit"):
        X, y, groups, gname, orig = load_corpus(ds)
        sub = joblib.load(m1 / f"substrate_{ds}.joblib")
        mh = model_hash(sub)
        sg = GroupedShap(sub)
        test = np.array(sub.metrics["_test_index"])
        lk = [i for i in test if role_of(f"{ds}-{orig[i]}") == "locked_evaluation" and f"{ds}-{orig[i]}" not in used]
        pdv = sub.predict_pd(X.iloc[lk])
        fr = pd.DataFrame({"i": lk, "group": groups.iloc[lk].values, "pd": pdv})
        fr = fr[fr.group != "other"]
        for g, d in fr.groupby("group"):
            q1, q2 = (float(x) for x in d.pd.quantile([1 / 3, 2 / 3]))
            d = d.assign(band=pd.cut(d.pd, [-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"]))
            per = [a.per_stratum // 3 + (1 if k < a.per_stratum % 3 else 0) for k in range(3)]
            pick = pd.concat([d[d.band == b].sample(n, random_state=SEED) for b, n in zip(["low", "medium", "high"], per)])
            man["strata"][f"{ds}|{g}"] = {"pool": int(len(d)), "sampled": int(len(pick))}
            for k, r in enumerate(pick.itertuples()):
                i = int(r.i)
                bid = f"{ds}-{orig[i]}"
                fam, arm = CELLS[k % len(CELLS)]
                eo = build_evidence_v2(ds, bid, X.iloc[[i]], sub, sg, str(r.group))
                base = {"borrower_id": bid, "dataset": ds, "group": str(r.group), "stratum": f"{ds}|{r.group}",
                        "role": "locked_evaluation", "pd_band": str(r.band), "pd": float(r.pd), "cell": f"{fam}/{arm}",
                        "substrate_model_hash": mh}
                for arm_, f_ in ((arm, fam), ("template", "template")):
                    if (bid, arm_) not in done:
                        jobs.append((eo, arm_, f_, base))
    lock = threading.Lock()

    def work(job):
        eo, arm, fam, base = job
        # prespecified retry rule: ONE regeneration with a new seed after an unparseable output
        retry = (base["borrower_id"], arm) in failed
        cand = (render_template_v2(eo) if arm == "template" else
                generate(eo, arm, fam, temperature=0.2, seed=SEED + (1 if retry else 0),
                         experiment="fresh-natural-v2", prompt_version=2))
        rec = dict(base, arm=arm, family=fam, model_id=cand.model_id, provider=cand.provider, error=cand.error,
                   text=cand.text, claims=cand.claims, finish_reason=cand.finish_reason,
                   prompt_version=PROMPT_VERSION, evidence=eo.to_dict(include_groups=True),
                   evidence_fingerprint=eo.fingerprint(), prompt_tokens=cand.prompt_tokens,
                   completion_tokens=cand.completion_tokens, retry=retry,
                   item_id="FN2-" + hashlib.sha256(f"{base['borrower_id']}|{arm}".encode()).hexdigest()[:10])
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return arm, bool(cand.text)

    print(len(jobs), "jobs", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(work, jobs))
    from collections import Counter
    man["results"] = Counter(f"{arm}:{ok}" for arm, ok in res)
    (OUT / "fresh_natural_v2_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps(man["results"]), json.dumps(man["strata"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

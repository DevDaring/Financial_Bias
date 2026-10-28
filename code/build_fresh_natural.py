"""
Locked natural test (Future_Plan review §4): complete natural narratives for FRESH borrowers from the
locked-evaluation role (never sampled, inspected or used for development, calibration or rules).

Design (lean option of review §9.1; scalable with --per-stratum):
  strata      dataset (Taiwan, Home Credit) x audit group (two levels) = 4
  per stratum N borrowers (default 100), equal across the PD tertile bands of that stratum
  producers   nova_lite and ministral_8b (the two mid-error M1 producers, both on Bedrock), assigned
              50:50 within each stratum x band by a fixed alternation; ONE grounded narrative per borrower
              (prompt v2, temperature 0.2, the same producer settings as M1)
  template    the deterministic fallback actually shown when a narrative is withheld, one per borrower
The same M1 substrate model (joblib, hash-checked) produces the evidence. Verifiers are frozen BEFORE this
file is generated (manifest records their source hashes); labels come from blinded human annotation.

    python build_fresh_natural.py --per-stratum 100
Output: Submission/locked/fresh_natural_v1.jsonl (+ manifest)
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
from build_context_pack import freeze_record  # noqa: E402
from build_pack_m1 import SUB  # noqa: E402

OUT = SUB / "locked"
SEED = 20261009
PRODUCERS = ["nova_lite", "ministral_8b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fresh_natural_v1.jsonl"
    import joblib
    from evidence_builder import GroupedShap, build_evidence_v2
    from generate import PROMPT_VERSION, generate, render_template_v2
    from run_m1 import load_corpus, model_hash, role_of
    import witness as W
    import dispute as DP
    m1 = SUB / "m1"
    done = {}
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            done[(r["borrower_id"], r["arm"])] = r
    man = {"version": "fresh-natural/1.0", "seed": SEED, "producers": PRODUCERS, "per_stratum": a.per_stratum,
           "frozen_verifiers": {"executable": W.EXEC_VERSION, "dispute": DP.DISPUTE_VERSION,
                                "source_sha256_16": freeze_record()},
           "role": "locked_evaluation", "strata": {}}
    jobs = []
    for ds in ("taiwan", "home_credit"):
        X, y, groups, gname, orig = load_corpus(ds)
        sub = joblib.load(m1 / f"substrate_{ds}.joblib")
        mh = model_hash(sub)
        if mh != json.load(open(m1 / f"substrate_{ds}.json"))["model_hash"]:
            raise SystemExit(f"{ds}: substrate hash differs from M1")
        sg = GroupedShap(sub)
        test = np.array(sub.metrics["_test_index"])
        lk = [i for i in test if role_of(f"{ds}-{orig[i]}") == "locked_evaluation"]
        pdv = sub.predict_pd(X.iloc[lk])
        fr = pd.DataFrame({"i": lk, "group": groups.iloc[lk].values, "pd": pdv})
        fr = fr[fr.group != "other"]
        for g, d in fr.groupby("group"):
            q1, q2 = (float(x) for x in d.pd.quantile([1 / 3, 2 / 3]))
            d = d.assign(band=pd.cut(d.pd, [-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"]))
            per = [a.per_stratum // 3 + (1 if k < a.per_stratum % 3 else 0) for k in range(3)]
            pick = pd.concat([d[d.band == b].sample(n, random_state=SEED) for b, n in zip(["low", "medium", "high"], per)])
            man["strata"][f"{ds}|{g}"] = {"pool": int(len(d)), "sampled": int(len(pick)),
                                          "band_cuts": [round(q1, 5), round(q2, 5)]}
            for k, r in enumerate(pick.itertuples()):
                i = int(r.i)
                bid = f"{ds}-{orig[i]}"
                fam = PRODUCERS[k % 2]
                eo = build_evidence_v2(ds, bid, X.iloc[[i]], sub, sg, str(r.group))
                base = {"borrower_id": bid, "dataset": ds, "group": str(r.group), "stratum": f"{ds}|{r.group}",
                        "role": "locked_evaluation", "pd_band": str(r.band), "pd": float(r.pd),
                        "substrate_model_hash": mh}
                for arm, f in (("grounded", fam), ("template", "template")):
                    if (bid, arm) not in done:
                        jobs.append((eo, arm, f, base))
    lock = threading.Lock()

    def work(job):
        eo, arm, fam, base = job
        cand = (render_template_v2(eo) if arm == "template" else
                generate(eo, arm, fam, temperature=0.2, seed=SEED, experiment="fresh-natural", prompt_version=2))
        rec = dict(base, arm=arm, family=fam, model_id=cand.model_id, provider=cand.provider, error=cand.error,
                   text=cand.text, claims=cand.claims, finish_reason=cand.finish_reason,
                   prompt_version=PROMPT_VERSION, evidence=eo.to_dict(include_groups=True),
                   evidence_fingerprint=eo.fingerprint(), prompt_tokens=cand.prompt_tokens,
                   completion_tokens=cand.completion_tokens,
                   item_id="FN-" + hashlib.sha256(f"{base['borrower_id']}|{arm}".encode()).hexdigest()[:10])
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return arm, bool(cand.text)

    print(len(jobs), "jobs", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(work, jobs))
    from collections import Counter
    man["results"] = Counter(f"{arm}:{ok}" for arm, ok in res)
    (OUT / "fresh_natural_manifest_v1.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps(man["results"]), json.dumps(man["strata"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Milestone-1 generation run (plan §5.3) on the repaired pipeline.

    python run_m1.py                       # full run (resumable)
    python run_m1.py --per-corpus 2 --dry-run   # template only, no LLM calls

What is new against run_pilot.py (kept unchanged for the saved v1 pilot):
  evidence/2.0       evidence_builder.py (source names, meanings, periods,
                     driver-only facts, grouped SHAP with additivity check)
  substrate prep v2  numeric detection over non-missing rows, all category
                     levels kept, NaN kept for LightGBM
  borrower roles     test-fold borrowers are split ONCE into development /
                     calibration / locked evaluation by a hash of their id.
                     Only development borrowers are sampled here; the other
                     two roles are untouched until their experiments.
  producers          registry.MAIN_PRODUCERS x {grounded, typed} + template
  verifiers          structural/2.0 and every judge in registry.MAIN_JUDGES
                     with judge/2.0, on every candidate (so the primary vs
                     second judge choice can be made later, with data)

Outputs (Submission/m1/)
  records.jsonl          one row per candidate, with full evidence
  roles_<dataset>.json   role of every test-fold borrower
  substrate_<dataset>.json  model metrics, subgroup rates, provenance
  llm_trace_m1.jsonl     every LLM attempt
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "saarthi" / "training"))
OUT = HERE.parents[1] / "Submission" / "m1"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SAARTHI_LLM_TRACE", str(OUT / "llm_trace_m1.jsonl"))

import registry  # noqa: E402
from evidence_builder import GroupedShap, build_evidence_v2  # noqa: E402
from generate import PROMPT_VERSION, generate, reparse as reparse_candidate  # noqa: E402
from substrate import build  # noqa: E402
from generate import Candidate  # noqa: E402
from verify import STRUCTURAL_VERSION, semantic_check, structural_check  # noqa: E402

SEED = 20261002
CAP = 60_000
ROLE_CUTS = (("development", 0.20), ("calibration", 0.60), ("locked_evaluation", 1.00))
JUDGE_PROMPT = 2


def _prior_inspected() -> set:
    """Borrowers already inspected in the v1 pilot (annotated, used to select
    judges and tune the checker). They are development data for good: they may
    never enter the calibration or locked-evaluation roles, and are not
    resampled for M1 either (plan §5.3)."""
    out = set()
    pr = HERE.parents[1] / "Submission" / "pilot_records.jsonl"
    if not pr.exists():
        raise SystemExit(f"{pr} is missing: without it the 24 v1-pilot borrowers would leak "
                         "into the calibration and locked-evaluation roles")
    for l in open(pr):
        out.add(json.loads(l)["borrower_id"])
    if len(out) != 24:
        raise SystemExit(f"expected 24 v1-pilot borrowers, found {len(out)}")
    return out


def _superseded_inspected() -> set:
    """Borrowers whose outputs were inspected in the superseded first M1 attempt
    (development role; flagged per record so analyses can exclude them)."""
    out = set()
    for f in (HERE.parents[1] / "Submission" / "m1").glob("records_superseded*.jsonl"):
        for l in open(f):
            try:
                out.add(json.loads(l)["borrower_id"])
            except Exception:  # noqa: BLE001
                pass
    return out


PRIOR = _prior_inspected()
SUPERSEDED = _superseded_inspected()


def role_of(bid: str) -> str:
    if bid in PRIOR:
        return "development_v1_inspected"
    u = int(hashlib.sha256(f"m1-role:{bid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    for name, cut in ROLE_CUTS:
        if u < cut:
            return name
    return ROLE_CUTS[-1][0]


def model_hash(sub) -> str:
    """Content hash of the fitted predictor + calibrator + preprocessing, so a
    later calibration/locked run can prove it uses the same model."""
    h = hashlib.sha256()
    h.update(sub.model.booster_.model_to_string().encode())
    cal = sub.calibrator
    for attr in ("X_thresholds_", "y_thresholds_"):
        if hasattr(cal, attr):
            h.update(np.asarray(getattr(cal, attr)).tobytes())
    h.update(sub.prep.transformer_id.encode())
    return h.hexdigest()[:16]


def _versions() -> dict:
    out = {}
    for m in ("lightgbm", "sklearn", "shap", "numpy", "pandas", "openai", "boto3", "json_repair"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def run_config(a) -> dict:
    """Everything that defines the condition; resume refuses on any change."""
    import generate as G
    import json_parse as JP
    import verify as V
    from evidence_builder import TOP_K
    used = {"run_m1.py", "registry.py", "providers.py", "gateway.py", "evidence.py", "evidence_builder.py",
            "feature_semantics.py", "substrate.py", "generate.py", "verify.py", "json_parse.py"}
    code = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12]
            for f in sorted(HERE.glob("*.py")) if f.name in used}
    adp = HERE.parents[1] / "saarthi" / "training" / "adapters.py"
    code["adapters.py"] = hashlib.sha256(adp.read_bytes()).hexdigest()[:12]
    return {
        "producer_prompt_version": PROMPT_VERSION, "structural_version": STRUCTURAL_VERSION,
        "judge_prompt_version": JUDGE_PROMPT, "producers": list(a.producers),
        "judges": list(a.judges), "arms": list(a.arms), "per_corpus": a.per_corpus,
        "datasets": list(a.datasets), "parser_family": registry.PARSER_FAMILY,
        "prompts": {"grounded_v2": G._GROUNDED_SYS_V2, "typed_v2": G._TYPED_SYS_V2,
                    "judge": V.JUDGE_PROMPTS[JUDGE_PROMPT]},
        "prompt_hash": hashlib.sha256((G._GROUNDED_SYS_V2 + G._TYPED_SYS_V2
                                       + V.JUDGE_PROMPTS[JUDGE_PROMPT]).encode()).hexdigest()[:12],
        "judge_prompt_sha256": hashlib.sha256(V.JUDGE_PROMPTS[JUDGE_PROMPT].encode()).hexdigest(),
        "parser_sys_sha256": hashlib.sha256(JP.PARSER_SYS.encode()).hexdigest()[:16],
        "seed": SEED, "role_cuts": ROLE_CUTS, "cap": CAP, "top_k": TOP_K,
        "temperature": 0.2, "producer_max_tokens": 1500, "judge_max_tokens": 1500,
        "producer_routes": {f: [str(r) for r in registry.get(f).routes] for f in a.producers},
        "packages": _versions(),
        "code_sha256": code,
    }


# A failure that is a property of the PRODUCER (bad or truncated JSON) is an
# observation and is kept; only transport failures are retried on resume.
MODEL_FAILURES = ("unparseable_", "truncated_", "filtered_", "ambiguous_")
UNGUARDED = ("code_sha256", "code_history", "judge_routes", "route_history", "judge_set_history",
             "structural_version")       # deterministic, re-scorable; stored per record


def load_corpus(dataset: str):
    """X, y, audit group and ORIGINAL row ids, aligned after capping."""
    import adapters
    c = adapters.load(dataset)
    X, y = c.native.reset_index(drop=True), c.y.reset_index(drop=True)
    prot = c.protected.reset_index(drop=True)
    orig = np.arange(len(X))
    if len(X) > CAP:
        keep = np.sort(np.random.RandomState(20260502).choice(len(X), CAP, replace=False))
        X, y, prot, orig = (X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True),
                            prot.iloc[keep].reset_index(drop=True), orig[keep])
    col = prot.columns[0]
    g = prot[col].astype(str)
    top = g.value_counts().head(2).index.tolist()
    return X, y, g.where(g.isin(top), "other"), col, orig


def sample_development(dataset, X, groups, orig, sub, per_corpus, seed):
    test = np.array(sub.metrics["_test_index"])
    bids = [f"{dataset}-{orig[i]}" for i in test]
    roles = {b: role_of(b) for b in bids}
    counts = pd.Series(list(roles.values())).value_counts().to_dict()
    (OUT / f"roles_{dataset}.json").write_text(json.dumps(
        {"rule": "v1-pilot borrowers -> development_v1_inspected; otherwise "
                 "sha256('m1-role:'+borrower_id) uniform; <0.20 development, <0.60 calibration, "
                 "else locked_evaluation", "counts": counts, "roles": roles}, indent=0))
    dev = [i for i, b in zip(test, bids) if roles[b] == "development"]
    pdv = sub.predict_pd(X.iloc[dev])
    fr = pd.DataFrame({"i": dev, "group": groups.iloc[dev].values, "pd": pdv})
    fr = fr[fr.group != "other"]
    # bands on PD VALUES (equal PDs always share a band; isotonic PDs have ties)
    q1, q2 = (float(x) for x in fr.pd.quantile([1 / 3, 2 / 3]))
    if q1 >= q2:
        raise SystemExit(f"{dataset}: PD tertile cut points coincide ({q1}); cannot form 3 bands")
    fr["band"] = pd.cut(fr.pd, [-np.inf, q1, q2, np.inf], right=True, labels=["low", "medium", "high"])
    cuts = {"low": f"pd <= {q1:.5f}", "medium": f"{q1:.5f} < pd <= {q2:.5f}", "high": f"pd > {q2:.5f}"}
    cells = fr.groupby(["group", "band"], observed=True)
    if per_corpus % cells.ngroups:
        raise SystemExit(f"--per-corpus {per_corpus} is not a multiple of the {cells.ngroups} "
                         "group x PD-band cells; choose a multiple so every cell is equal")
    per = per_corpus // cells.ngroups
    pick = cells.apply(lambda d: d.sample(min(len(d), per), random_state=seed)).reset_index(drop=True)
    # design weights (plan §7.3): population of each cell in the development pool
    pop = cells.size().to_dict()
    design = {"cell_population": {f"{g}|{b}": int(n) for (g, b), n in pop.items()},
              "per_cell_sampled": per, "pd_band_rule": cuts,
              "realised_per_cell": {f"{g}|{b}": int(n) for (g, b), n in
                                    pick.groupby(["group", "band"], observed=True).size().items()},
              "weight": {f"{g}|{b}": round(int(n) / max(1, int(pick[(pick.group == g) & (pick.band == b)].shape[0])), 3)
                         for (g, b), n in pop.items()}}
    return pick, counts, design


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["taiwan", "home_credit"])
    ap.add_argument("--per-corpus", type=int, default=18)
    ap.add_argument("--producers", nargs="*", default=registry.MAIN_PRODUCERS)
    ap.add_argument("--judges", nargs="*", default=registry.MAIN_JUDGES)
    ap.add_argument("--arms", nargs="*", default=["template", "grounded", "typed"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--defer-judges", action="store_true",
                    help="produce and structurally check now; judges run on a later resume "
                         "(e.g. once AWS Bedrock access to the judge models is granted)")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume although code changed (recorded in run_config code_history)")
    a = ap.parse_args()
    t0 = time.time()

    global OUT
    if a.dry_run:      # never mix a dry run into the real records
        OUT = OUT / "dry_run"
        OUT.mkdir(parents=True, exist_ok=True)

    # one process per output folder (a second resume would race on the records)
    import fcntl
    lock_fh = open(OUT / ".run.lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another run_m1.py is already running on {OUT}")

    cfg = json.loads(json.dumps(run_config(a), default=str))   # tuples -> lists, as stored
    cfg["judge_routes"] = {f: [str(r) for r in registry.get(f).routes]
                           for f in list(a.judges) + [registry.PARSER_FAMILY]}
    cfg_path = OUT / "run_config.json"
    rec_path = OUT / "records.jsonl"
    if rec_path.exists() and cfg_path.exists():
        old = json.load(open(cfg_path))
        diff = [k for k in cfg if k not in UNGUARDED and old.get(k) != cfg.get(k)]
        if "judges" in diff:
            # the judge set may change only while NO verdict exists yet (all deferred)
            any_real = False
            for l in open(rec_path):
                try:
                    r0 = json.loads(l)
                except Exception:  # noqa: BLE001
                    continue
                if any(v.get("error") != "deferred" for v in (r0.get("judges") or {}).values()):
                    any_real = True
                    break
            if not any_real:
                diff.remove("judges")
                cfg["judge_set_history"] = old.get("judge_set_history", []) + [old.get("judges")]
                print(f"[note] judge set changed {old.get('judges')} -> {cfg['judges']} "
                      "(no verdicts existed; recorded)", flush=True)
        if diff:
            raise SystemExit(f"resume refused: run condition changed in {diff}. "
                             f"Archive {rec_path} first.")
        cfg["code_history"] = list(old.get("code_history", []))
        cfg.setdefault("judge_set_history", list(old.get("judge_set_history", [])))
        cfg["route_history"] = list(old.get("route_history", []))
        if old.get("judge_routes") and old["judge_routes"] != cfg["judge_routes"]:
            cfg["route_history"].append(old["judge_routes"])      # allowed, recorded
        if old.get("code_sha256") != cfg["code_sha256"]:
            changed = [f for f in cfg["code_sha256"] if old["code_sha256"].get(f) != cfg["code_sha256"][f]]
            if not a.allow_code_change:
                raise SystemExit(f"resume refused: code changed since this run started {changed}. "
                                 f"Archive the records, or pass --allow-code-change to record it.")
            print(f"[warn] code changed since this run started: {changed} (recorded)", flush=True)
            cfg["code_history"].append(old["code_sha256"])
    cfg["cfg_hash"] = hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != "cfg_hash"},
                                                sort_keys=True).encode()).hexdigest()[:12]
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # checks that need no network run for EVERY mode, so a deferred run can
    # always be judged later
    if registry.PARSER_FAMILY in a.producers or registry.PARSER_FAMILY in a.judges:
        raise SystemExit("the parser family must not be a producer or a study judge")
    panel = registry.assert_independent_panel(a.producers, a.judges, require_full_matrix=True)
    print(f"[gate] independent panel: {panel['producers']} x {panel['judges']}", flush=True)

    judging = not a.dry_run and not a.defer_judges
    if not a.dry_run:
        from providers import prune_dead_keys
        prune_dead_keys(["bedrock", "openrouter", "linkapi", "anthropic", "openai_direct", "xai_direct"])
    if judging:
        # the judges must have passed qualification with THIS prompt and parser,
        # on the SAME route they will be served on now (each judge is pinned)
        import gateway as GW
        qp = HERE.parents[1] / "Submission" / f"judge_qualification_prompt{JUDGE_PROMPT}.json"
        qs = json.load(open(qp))["summary"] if qp.exists() else {}
        q = qs.get("judges", {})
        bad = [j for j in a.judges if not q.get(j, {}).get("eligible")]
        if bad:
            raise SystemExit(f"judges {bad} are not qualified on judge prompt {JUDGE_PROMPT} ({qp.name})")
        if "judge_prompt_sha256" not in qs:
            raise SystemExit(f"{qp.name} predates the provenance fields; re-run "
                             f"qualify_judges.py --prompt {JUDGE_PROMPT}")
        if qs["judge_prompt_sha256"] != cfg["judge_prompt_sha256"]:
            raise SystemExit("the judge prompt changed after qualification; re-run qualify_judges.py")
        if qs.get("parser_sys_sha256") != cfg["parser_sys_sha256"]:
            raise SystemExit("the parser prompt changed after qualification; re-run qualify_judges.py")
        for j in a.judges:
            r = GW.GATEWAY.complete(j, [{"role": "user", "content": "Reply with exactly: ok"}],
                                    max_tokens=50, role="preflight", experiment="m1:preflight")
            route = f"{r.provider}:{r.model_id}"
            served = qs.get("served_routes", {}).get(j, [])
            if served != [route]:
                raise SystemExit(f"judge {j} would be served on {route}, but was qualified on {served}; "
                                 f"re-run qualify_judges.py on this route")
            GW.PINNED_ROUTES[j] = (r.provider, r.model_id)
            print(f"[gate] judge {j} pinned to {route}", flush=True)

    # ---- resume ---------------------------------------------------------
    done, rejudge, reparse = set(), {}, {}
    if rec_path.exists():
        # The file is append-only; on load the LAST row per (borrower, arm,
        # family) is the current one. Successful rows and MODEL failures are
        # final. A row whose parser was unavailable is RE-PARSED from its stored
        # response (the producer is not re-sampled). Other transport failures
        # are re-produced. Rows whose judges failed by transport (or were
        # deferred) keep their candidate; only those judges are re-run. Nothing
        # is removed from the file before its replacement is written.
        latest, corrupt = {}, []
        for l in open(rec_path):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:  # noqa: BLE001
                corrupt.append(l if l.endswith("\n") else l + "\n")
                continue
            latest[(r["borrower_id"], r["arm"], r["family"])] = r
        if corrupt:
            with open(OUT / "records_corrupt.jsonl", "a") as fh:
                fh.writelines(corrupt)
        failed = []
        for key, r in latest.items():
            err = str(r.get("error") or "")
            if err.startswith("parser_unavailable_"):
                reparse[key] = r
                done.add(key)
                continue
            if err and not err.startswith(MODEL_FAILURES):
                failed.append(json.dumps(r) + "\n")       # re-produce; outage logged
                continue
            done.add(key)
            if judging and not err:             # a failed candidate (no text) is never judged
                jv = r.get("judges") or {}
                if jv and all(v.get("error") == "deferred" for v in jv.values()):
                    r["judges"] = {}                      # placeholders of an older judge set
                    bad_j = list(a.judges)
                else:
                    bad_j = [j for j, v in jv.items()
                             if v.get("error") and (v.get("error_kind") == "transport"
                                                    or str(v["error"]).startswith(("deferred", "FamilyExhausted")))]
                    bad_j += [j for j in a.judges if j not in jv]
                if bad_j:
                    rejudge[key] = (r, bad_j)
        if failed:
            with open(OUT / "records_failed.jsonl", "a") as fh:
                fh.writelines(failed)
        # compact: one current row per key (atomic), failed keys dropped from the
        # current view only after they are safely logged above
        tmp = rec_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            for key, r in latest.items():
                if key in done:
                    fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, rec_path)
    wlock = threading.Lock()

    def write(rec):
        rec["cfg_hash"] = cfg["cfg_hash"]
        rec["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with wlock, open(rec_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    from build_pack_m1 import eo_from
    import joblib      # *.joblib is gitignored; reused by build_pack_m1
    for ds in a.datasets:
        X, y, groups, gname, orig = load_corpus(ds)
        sub = build(X, y, name=f"{ds}-m1", model_kind="gbdt", calibrator_kind="isotonic",
                    groups=groups, threshold=0.5, prep_version=2)
        mh = model_hash(sub)
        sj = OUT / f"substrate_{ds}.json"
        if done and sj.exists() and json.load(open(sj)).get("model_hash") != mh:
            raise SystemExit(f"resume refused: the rebuilt {ds} model differs from the one this run "
                             f"started with ({mh} vs {json.load(open(sj)).get('model_hash')})")
        shap_g = GroupedShap(sub)
        joblib.dump(sub, OUT / f"substrate_{ds}.joblib")
        pick, counts, design = sample_development(ds, X, groups, orig, sub, a.per_corpus, SEED)
        sj.write_text(json.dumps({
            "dataset": ds, "audit_group_field": gname, "model_id": sub.model_id,
            "calibrator_id": sub.calibrator_id, "transformer_id": sub.prep.transformer_id,
            "metrics": {k: v for k, v in sub.metrics.items() if k != "_test_index"},
            "subgroup": sub.subgroup, "splits": sub.split_sizes, "role_counts": counts,
            "model_hash": mh, "sampling_design": design,
            "sampled": [{"borrower_id": f"{ds}-{orig[int(r.i)]}", "group": r.group,
                         "band": str(r.band), "pd": round(float(r.pd), 4)} for r in pick.itertuples()],
        }, indent=2, default=str))
        print(f"[{ds}] AUC={sub.metrics['auc']} ECE={sub.metrics['ece']} roles={counts} "
              f"sampled={len(pick)} model={mh} ({time.time()-t0:.0f}s)", flush=True)

        jobs = []
        for r in pick.itertuples():
            i = int(r.i)
            bid = f"{ds}-{orig[i]}"
            eo = build_evidence_v2(ds, bid, X.iloc[[i]], sub, shap_g, str(r.group))
            for arm in a.arms:
                fams = ["template"] if arm == "template" else ([] if a.dry_run else a.producers)
                for fam in fams:
                    key = (bid, arm, fam)
                    if key in reparse:
                        jobs.append(("reparse", eo, key, str(r.band), float(r.pd)))
                    elif key in rejudge:
                        jobs.append(("rejudge", eo, key, str(r.band), float(r.pd)))
                    elif key not in done:
                        jobs.append(("produce", eo, key, str(r.band), float(r.pd)))
        kinds = Counter(j[0] for j in jobs)
        print(f"[{ds}] jobs {dict(kinds)} (resume kept {len(done)})", flush=True)

        def judge_all(cand, eo, names):
            return {j: semantic_check(cand, eo, j, experiment=f"m1:{ds}",
                                      prompt_version=JUDGE_PROMPT).to_dict() for j in names}

        def record(cand, eo, band, pdv, fam, arm, judges, st):
            return {
                "borrower_id": eo.borrower_id, "dataset": ds, "group": eo.group_attributes["audit_group"],
                "role": "development", "pd_band": band, "pd": pdv, "arm": arm,
                "family": "template" if arm == "template" else fam, "model_id": cand.model_id,
                "provider": cand.provider, "route_index": cand.route_index, "error": cand.error,
                "text": cand.text, "claims": cand.claims, "raw": cand.raw,
                "parse_method": cand.parse_method, "parse_info": cand.parse_info,
                "call_id": cand.call_id, "finish_reason": cand.finish_reason,
                "seed": SEED, "temperature": 0.2,
                "previously_inspected": eo.borrower_id in SUPERSEDED,
                "substrate_model_hash": mh, "run_prompt_hash": cfg["prompt_hash"],
                "evidence": eo.to_dict(include_groups=True), "evidence_fingerprint": eo.fingerprint(),
                "prompt_version": PROMPT_VERSION, "structural_version": STRUCTURAL_VERSION,
                "judge_prompt_version": JUDGE_PROMPT,
                "structural": st.to_dict() if st else None, "judges": judges,
                "latency_ms": cand.latency_ms, "prompt_tokens": cand.prompt_tokens,
                "completion_tokens": cand.completion_tokens}

        def deferred(names):
            return {j: {"accepted": False, "error": "deferred", "error_kind": "transport"} for j in names}

        def run(job):
            kind, eo_new, (bid, arm, fam), band, pdv = job
            if kind in ("rejudge", "reparse"):
                rec = rejudge[(bid, arm, fam)][0] if kind == "rejudge" else reparse[(bid, arm, fam)]
                eo = eo_from(rec["evidence"])        # judge what the PRODUCER saw
                if eo.fingerprint() != rec["evidence_fingerprint"]:
                    raise SystemExit(f"{bid}|{arm}|{fam}: stored evidence fingerprint mismatch")
            if kind == "rejudge":
                rec, names = rejudge[(bid, arm, fam)]
                cand = Candidate(borrower_id=bid, arm=arm, text=rec["text"], claims=rec.get("claims") or [],
                                 family=rec["family"])
                rec["judges"].update(judge_all(cand, eo, names))
                rec.setdefault("rejudged", []).extend(names)
                write(rec)
                return arm, fam, True
            if kind == "reparse":
                cand = reparse_candidate(rec, eo, experiment=f"m1:{ds}")
            else:
                eo = eo_new
                cand = generate(eo, arm, "" if fam == "template" else fam, temperature=0.2,
                                seed=SEED, experiment=f"m1:{ds}", prompt_version=2)
            st = structural_check(cand, eo) if cand.ok else None
            if not cand.ok or a.dry_run:
                judges = {}
            elif a.defer_judges:
                judges = deferred(a.judges)
            else:
                judges = judge_all(cand, eo, a.judges)
            write(record(cand, eo, band, pdv, fam, arm, judges, st))
            return arm, fam, cand.ok

        n = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for arm, fam, ok in ex.map(run, jobs):
                n += 1
                if n % 20 == 0 or n == len(jobs):
                    print(f"  [{ds}] {n}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    latest = {}
    for l in open(rec_path):
        try:
            r = json.loads(l)
            latest[(r["borrower_id"], r["arm"], r["family"])] = r
        except Exception:  # noqa: BLE001
            pass
    rows = list(latest.values())
    ok = [r for r in rows if not r.get("error")]
    pend = sum(1 for r in rows for v in (r.get("judges") or {}).values() if v.get("error") == "deferred")
    print(f"\nrecords: {len(rows)} ({len(ok)} ok, {len(rows)-len(ok)} model failures; "
          f"{pend} judge verdicts deferred) in {time.time()-t0:.0f}s -> {rec_path}")
    lock_fh.close()          # releases the run lock
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

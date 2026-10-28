"""
Score the executable layer (no LLM, zero cost) on the E2 mechanism pack.

Per item: the claim's executed status (support / contradiction / unresolved) and the
outcome of its witness (true witness on corruptions, forged witness on correct versions).

Reported separately for DEV and HELDOUT wording, by relation and by version:
  detection     corruptions whose claim is executed as a contradiction
  false alarm   correct versions (original / paraphrase / provocation) executed as contradiction
  settled       share of items the code decides (support or contradiction) without an LLM
  witness       true witnesses verified (or semantic-plausible); forged witnesses NOT verified
Constant acceptance or rejection fails by construction (half the witnesses are forged,
a quarter of claims are corrupted).

    python eval_mechanism.py              -> Submission/mechanism/eval_code_layer.json        (v1: development)
    python eval_mechanism.py --pack v2    -> Submission/mechanism/eval_code_layer_v2_once.json (fresh held-out, scored once)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import witness as W  # noqa: E402
from build_pack_m1 import eo_from  # noqa: E402
from build_mechanism_pack import OUT  # noqa: E402


def claim_status(text: str, eo) -> str:
    tx = W.execute_text(text, eo)
    st = [c.status for c in tx.claims]
    if W.CONTRA in st:
        return W.CONTRA
    if st and all(s in (W.SUPPORT, "nonmaterial") for s in st) and W.SUPPORT in st:
        return W.SUPPORT
    return W.UNRES


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="v1", choices=["v1", "v2"])
    tag = ap.parse_args().pack
    rows = [json.loads(l) for l in open(OUT / f"mechanism_pack_{tag}.jsonl")]
    res = []
    for r in rows:
        eo = eo_from(r["evidence"])
        st = claim_status(r["text"], eo)
        wj = r["witness"]
        w = W.Witness(rule=wj["rule"], quote=wj["quote"] or "", evidence_ids=wj["evidence_ids"] or [],
                      operation=wj.get("operation"), claimed_result=wj.get("claimed_result"))
        W.execute_witness(w, r["text"], eo)
        ok_w = (w.outcome in ("verified", "semantic_plausible")) if r["witness_expect"] != "not_verified" \
            else (w.outcome not in ("verified", "semantic_plausible"))
        res.append({"item_id": r["item_id"], "relation": r["relation"], "version": r["version"],
                    "split": r["template_split"], "label": r["label"], "status": st,
                    "witness_kind": wj["forgery"], "witness_outcome": w.outcome, "witness_ok": ok_w,
                    "witness_problems": w.problems[:2]})

    def summary(sel):
        err = [x for x in sel if x["label"] == "error"]
        ok = [x for x in sel if x["label"] == "correct"]
        tw = [x for x in sel if x["witness_kind"] == "true"]
        fw = [x for x in sel if x["witness_kind"] != "true"]
        return {"n": len(sel),
                "detection": f"{sum(x['status'] == W.CONTRA for x in err)}/{len(err)}",
                "missed_as_support": f"{sum(x['status'] == W.SUPPORT for x in err)}/{len(err)}",
                "false_alarm": f"{sum(x['status'] == W.CONTRA for x in ok)}/{len(ok)}",
                "correct_settled_support": f"{sum(x['status'] == W.SUPPORT for x in ok)}/{len(ok)}",
                "true_witness_verified": f"{sum(x['witness_ok'] for x in tw)}/{len(tw)}",
                "forged_witness_rejected": f"{sum(x['witness_ok'] for x in fw)}/{len(fw)}"}

    out = {"note": "code layer only (executable/1.0); labels by construction, not yet independently validated; "
                   "DEV wording was available while the checker was developed, HELDOUT wording was not used to tune it",
           "pack": tag, "checker": W.EXEC_VERSION,
           "overall": {s: summary([x for x in res if x["split"] == s]) for s in ("dev", "heldout", "v2")
                       if any(x["split"] == s for x in res)},
           "by_relation": {f"{rel}|{s}": summary([x for x in res if x["relation"] == rel and x["split"] == s])
                           for rel in sorted({x["relation"] for x in res}) for s in ("dev", "heldout", "v2")
                           if any(x["relation"] == rel and x["split"] == s for x in res)},
           "by_version": {f"{v}|{s}": dict(Counter(x["status"] for x in res if x["version"] == v and x["split"] == s))
                          for v in ("original", "paraphrase", "corruption", "provocation") for s in ("dev", "heldout", "v2")
                          if any(x["version"] == v and x["split"] == s for x in res)},
           "forgeries": {k: f"{sum(x['witness_ok'] for x in res if x['witness_kind'] == k)}/"
                            f"{sum(1 for x in res if x['witness_kind'] == k)}"
                         for k in ("true", "wrong_field", "false_arithmetic", "partial_quote")},
           "failures": [x for x in res if (x["label"] == "correct" and x["status"] == W.CONTRA) or not x["witness_ok"]][:60]}
    once = OUT / f"eval_code_layer_{tag}_once.json"
    # a held-out pack is scored ONCE; every later run is development and never overwrites that record
    dest = OUT / "eval_code_layer.json" if tag == "v1" else (once if not once.exists() else
                                                              OUT / f"eval_code_layer_{tag}_DEVELOPMENT.json")
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("overall", "forgeries")}, indent=1))
    for k, v in out["by_relation"].items():
        print(f"{k:18} det={v['detection']:6} miss={v['missed_as_support']:6} fa={v['false_alarm']:6} "
              f"sup={v['correct_settled_support']:6} tw={v['true_witness_verified']:6} fw={v['forged_witness_rejected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

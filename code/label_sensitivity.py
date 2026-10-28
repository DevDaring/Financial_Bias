"""
Label sensitivity (Future_Plan review §4): key development comparisons under three reference sets.
  final      agreed + adjudicated + author's contract-conflict decisions (the paper's labels)
  agreed     only items both raters agreed on (no adjudication, no relabelling)
  reverted   final, but the contract-triggered relabellings reverted to the raters' agreed label
    python label_sensitivity.py -> Submission/m1/label_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v2 import METHODS, load  # noqa: E402
from score_m1 import M1, agreed_natural, final_labels  # noqa: E402

KEY = ["generic_judge2", "generic_rubric", "facts_judge", "exec_P", "exec_D", "exec_R", "exec_R_replay", "exec_R11_replay"]


def main() -> int:
    fl = {i: v for i, (_k, v, _b) in final_labels().items()}
    ag = {i: v for i, (_k, v) in agreed_natural().items()}
    basis = {i: b for i, (_k, _v, b) in final_labels().items()}
    rev = {i: (ag[i] if basis[i] == "contract_conflict" and i in ag else v) for i, v in fl.items()}
    sets = {"final": fl, "agreed": ag, "reverted": rev}
    out = {"n": {k: len(v) for k, v in sets.items()}, "errors": {k: sum(v.values()) for k, v in sets.items()}, "judges": {}}
    stems = dict(METHODS)
    for j in ("qwen3_next", "gemma27_vllm", "grok45"):
        res = {}
        for m in KEY:
            rows, _ = load(stems[m], j)
            if not rows:
                continue
            res[m] = {}
            for name, lab in sets.items():
                cr = sum(1 for i, v in lab.items() if not v and (rows.get(i) or {}).get("accepted") is True)
                wr = sum(1 for i, v in lab.items() if v and (rows.get(i) or {}).get("accepted") is True)
                res[m][name] = {"correct_released": cr, "wrong_released": wr}
        out["judges"][j] = res
    (M1 / "label_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["n"]), json.dumps(out["errors"]))
    for j, r in out["judges"].items():
        for m, v in r.items():
            print(j, m, {k: f"{x['correct_released']}/{x['wrong_released']}" for k, x in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

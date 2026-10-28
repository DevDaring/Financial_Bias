"""
E1 scoring: the first real numbers for the paper, not proxies.

Joins three LOCAL-ONLY files (never shipped to annotators, never committed):
  Submission/annotation_key_v1.jsonl             item -> borrower/group/arm/family/
                                                  injected_error/structural_accepted/judges
  Submission/annotation_pack_v1/E1_items.jsonl   item -> evidence + explanation text
  Submission/Annotated/E1_adjudication_proposed_KD.csv
                                                  item -> adjudicated ground truth (reviewed by
                                                  the author; majority of 3 annotators, or
                                                  adjudicated where they split)

Produces, per plan Section 4.4 / 6 (E1):
  - structural checker vs ground truth: confusion matrix, by SOURCE (natural/corrupted),
    by GROUP (-> the new b'_g), and per-injected-error-type recall
  - LLM judges vs ground truth on natural items only, by group (-> a_g, b_g per judge)
  - the p_g / a_g / b_g / b'_g decomposition with REAL labels, replacing the proxy
    decomposition in pilot_report.json

Nothing in the output or in this script contains an annotator's name, background,
or any content from the Annotator identity tab. Annotators are referred to only
by the codes already used in the author's own adjudication report (AS, NK, SP).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import EvidenceItem, EvidenceObject, EvidenceType  # noqa: E402
from generate import Candidate  # noqa: E402
from verify import structural_check  # noqa: E402

SUB = Path(__file__).resolve().parents[2] / "Submission"
KEY = SUB / "annotation_key_v1.jsonl"
ITEMS = SUB / "annotation_pack_v1" / "E1_items.jsonl"
ADJ = SUB / "Annotated" / "E1_adjudication_proposed_KD.csv"
OUT = SUB / "e1_report.json"

# CSV short codes -> the ErrorType vocabulary used elsewhere in the codebase
CODE_TO_TYPE = {
    "INV": "invented_factor", "SIGN": "sign_reversal", "NUM": "numeric_or_unit_error",
    "TREND": "unsupported_trend", "CAUS": "unsupported_causal_or_action",
    "OMIT": "omission_of_essential_driver", "OTH": "other_material_error",
}


def load_key() -> Dict[str, dict]:
    return {json.loads(l)["item_id"]: json.loads(l)
            for l in open(KEY) if l.strip()}


def load_items() -> Dict[str, dict]:
    return {json.loads(l)["item_id"]: json.loads(l)
            for l in open(ITEMS) if l.strip()}


def load_adjudication() -> Dict[str, dict]:
    with open(ADJ, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[r["item_id"]] = {
            "any_error": r["final_ANY"].strip().upper() == "Y",
            "error_types": [CODE_TO_TYPE[t] for t in r["final_error_types"].split("+") if t],
            "informative": r["final_informative"].strip().upper() == "Y",
            "confidence": r["adjudication_confidence"],
            "needs_author_decision": r["needs_author_decision"].strip() == "1",
            "raw": {k: r.get(k) for k in ("AS_ANY", "NK_ANY", "SP_ANY", "source")},
        }
    return out


def rebuild_evidence(it: dict) -> EvidenceObject:
    eo = EvidenceObject(borrower_id=it["item_id"], dataset=it["dataset"],
                        schema_version=it.get("evidence_schema", "evidence/1.0"))
    for d in it["evidence"]:
        d = dict(d)
        d["etype"] = EvidenceType(d["etype"])
        eo.add(EvidenceItem(**d))
    return eo


# ---------------------------------------------------------------------------
@dataclass
class Row:
    item_id: str
    source: str                   # natural | corrupted
    group: Optional[str]
    arm: Optional[str]
    family: Optional[str]
    injected_error: Optional[str]
    truth_any: bool
    truth_types: List[str]
    truth_informative: bool
    struct_pred: bool             # True = structural checker ACCEPTS (no violation)
    struct_viol_types: List[str]
    judges: Dict[str, Optional[bool]]   # from the pilot run, natural items only


def build_rows() -> List[Row]:
    key, items, adj = load_key(), load_items(), load_adjudication()
    rows: List[Row] = []
    for iid, k in key.items():
        if iid not in items or iid not in adj:
            continue
        it, a = items[iid], adj[iid]
        eo = rebuild_evidence(it)
        cand = Candidate(borrower_id=iid, arm=k.get("arm") or "unknown",
                         text=it["explanation"], family=k.get("family") or "")
        v = structural_check(cand, eo)
        rows.append(Row(
            item_id=iid, source=k["source"], group=k.get("group"),
            arm=k.get("arm"), family=k.get("family"),
            injected_error=k.get("injected_error"),
            truth_any=a["any_error"], truth_types=a["error_types"],
            truth_informative=a["informative"],
            struct_pred=v.accepted,
            struct_viol_types=[vv["type"] for vv in v.violations],
            judges=(k.get("judges") or {}) if k["source"] == "natural" else {}))
    return rows


# ---------------------------------------------------------------------------
def confusion(rows: List[Row], predict_ok) -> dict:
    """predict_ok(row) -> True if the checker says ACCEPT (no error).
    truth_any True = ground truth says there IS a material error.
    Returns counts in the checker's frame: accept/reject x truth."""
    tp = fp = tn = fn = skipped = 0
    for r in rows:
        p = predict_ok(r)
        if p is None:
            skipped += 1
            continue
        if p and not r.truth_any:
            tn += 1          # correctly accepted a correct explanation
        elif p and r.truth_any:
            fn += 1          # ACCEPTED an erroneous one -> false accept (a)
        elif (not p) and not r.truth_any:
            fp += 1          # REJECTED a correct one -> false reject (b)
        else:
            tp += 1          # correctly rejected an erroneous one
    n = tp + fp + tn + fn
    return {
        "n": n, "skipped": skipped,
        "correctly_rejected_error": tp, "correctly_accepted_correct": tn,
        "false_accept_of_error": fn, "false_reject_of_correct": fp,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "false_accept_rate": round(fn / (tp + fn), 4) if (tp + fn) else None,  # a
        "false_reject_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,  # b or b'
    }


def by_group(rows: List[Row], predict_ok) -> Dict[str, dict]:
    out = {}
    for g in sorted({r.group for r in rows if r.group}):
        out[g] = confusion([r for r in rows if r.group == g], predict_ok)
    return out


def per_error_type_recall(rows: List[Row]) -> Dict[str, dict]:
    """On CORRUPTED items only, where the injected error is ground truth:
    does the structural checker's violation list contain that type?"""
    out = {}
    for et, code in CODE_TO_TYPE.items():
        sub = [r for r in rows if r.source == "corrupted" and r.injected_error == code]
        if not sub:
            continue
        caught = sum(1 for r in sub if code in r.struct_viol_types or not r.struct_pred)
        out[code] = {"n": len(sub), "caught_by_structural": caught,
                     "recall": round(caught / len(sub), 4)}
    return out


def decomposition(rows: List[Row]) -> Dict[str, dict]:
    """p_g / a_g / b_g / b'_g, implemented coverage identity, per group.
    Only NATURAL items carry a judge verdict, so a_g/b_g use that subset;
    b'_g (structural) uses every item with a group label."""
    out = {}
    for g in sorted({r.group for r in rows if r.group}):
        grp = [r for r in rows if r.group == g]
        nat = [r for r in grp if r.source == "natural"]
        p_g = round(sum(1 for r in grp if r.truth_any) / len(grp), 4) if grp else None

        struct_c = confusion(grp, lambda r: r.struct_pred)
        bprime = struct_c["false_reject_rate"]

        judge_rows = {}
        for jname in ("gemini", "gpt4omini"):
            def pred(r, jn=jname):
                v = r.judges.get(jn)
                return None if v is None else bool(v)
            jc = confusion([r for r in nat if r.judges.get(jname) is not None], pred)
            judge_rows[jname] = {"a_false_accept": jc["false_accept_rate"],
                                 "b_false_reject": jc["false_reject_rate"],
                                 "n": jc["n"]}

        out[g] = {"n_total": len(grp), "n_natural": len(nat),
                  "p_candidate_error": p_g,
                  "structural_checker": {"b_prime_false_reject": bprime,
                                         "accuracy": struct_c["accuracy"],
                                         "n": struct_c["n"]},
                  "judges": judge_rows}
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    rows = build_rows()
    print(f"rows joined: {len(rows)} / 184 expected", flush=True)

    struct_overall = confusion(rows, lambda r: r.struct_pred)
    struct_natural = confusion([r for r in rows if r.source == "natural"],
                               lambda r: r.struct_pred)
    struct_corrupted = confusion([r for r in rows if r.source == "corrupted"],
                                 lambda r: r.struct_pred)
    struct_by_group = by_group(rows, lambda r: r.struct_pred)
    recall_by_type = per_error_type_recall(rows)

    judge_overall = {}
    for j in ("gemini", "gpt4omini"):
        nat = [r for r in rows if r.source == "natural" and r.judges.get(j) is not None]
        judge_overall[j] = confusion(nat, lambda r, jn=j: bool(r.judges[jn]))

    decomp = decomposition(rows)

    # Arm x family on NATURAL items only: the paper's central cell.
    # (DeepSeek and the template sit at a 0%-error floor either way; Llama 8B
    # is where "does typed evidence help" is actually testable here.)
    nat_rows = [r for r in rows if r.source == "natural"]
    arm_family: Dict[str, dict] = {}
    for arm in sorted({r.arm for r in nat_rows if r.arm}):
        for fam in sorted({r.family for r in nat_rows if r.arm == arm and r.family}):
            sub = [r for r in nat_rows if r.arm == arm and r.family == fam]
            if not sub:
                continue
            arm_family[f"{arm}/{fam}"] = {
                "n": len(sub),
                "error_rate_adjudicated": round(sum(r.truth_any for r in sub) / len(sub), 4)}

    struct_false_reject_by_family = {}
    for fam in sorted({r.family for r in nat_rows if r.family}):
        sub = [r for r in nat_rows if r.family == fam]
        clean = [r for r in sub if not r.truth_any]
        fr = sum(1 for r in clean if not r.struct_pred)
        struct_false_reject_by_family[fam] = {
            "n_truth_clean": len(clean), "struct_false_rejects": fr,
            "b_prime": round(fr / len(clean), 4) if clean else None}

    # Producer capability vs REAL ground truth, NATURAL ITEMS ONLY.
    # Corrupted items inherit their base candidate's `family` tag (a template
    # corruption still reads family="template"), so mixing sources here would
    # silently attribute injected synthetic errors to the producer. This was
    # caught during review: an earlier pass showed "template" at 48% error
    # before this restriction - that number was the bug, not a finding.
    nat_rows = [r for r in rows if r.source == "natural"]
    by_fam = {}
    for fam in sorted({r.family for r in nat_rows if r.family}):
        sub = [r for r in nat_rows if r.family == fam]
        by_fam[fam] = {"n": len(sub),
                       "error_rate_adjudicated": round(sum(r.truth_any for r in sub) / len(sub), 4)}

    report = {
        "note": "Ground truth = adjudicated labels (3 annotators, author-reviewed). "
                "Replaces the structural-check PROXY used in pilot_report.json.",
        "arm_x_family_natural_only": arm_family,
        "structural_false_reject_by_family_bprime": struct_false_reject_by_family,
        "n_items": len(rows),
        "n_items_any_error": sum(r.truth_any for r in rows),
        "overall_adjudicated_error_rate": round(sum(r.truth_any for r in rows) / len(rows), 4),
        "structural_checker": {
            "overall": struct_overall, "natural_items_only": struct_natural,
            "corrupted_items_only": struct_corrupted,
            "false_reject_rate_by_group_bprime": struct_by_group,
            "recall_by_injected_error_type": recall_by_type,
        },
        "llm_judges_natural_items_only": judge_overall,
        "decomposition_by_group": decomp,
        "producer_error_rate_adjudicated": by_fam,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== structural checker ===")
    print(f"  overall      n={struct_overall['n']:<4} acc={struct_overall['accuracy']}  "
          f"false_accept(a)={struct_overall['false_accept_rate']}  "
          f"false_reject(b')={struct_overall['false_reject_rate']}")
    print(f"  natural only n={struct_natural['n']:<4} acc={struct_natural['accuracy']}  "
          f"false_reject(b')={struct_natural['false_reject_rate']}")
    print(f"  corrupted    n={struct_corrupted['n']:<4} acc={struct_corrupted['accuracy']}  "
          f"(ground truth = injected error; checker SHOULD reject)")
    print(f"\n  false_reject (b') by group:")
    for g, d in struct_by_group.items():
        print(f"    {g:<8} n={d['n']:<4} b'={d['false_reject_rate']}  a={d['false_accept_rate']}")
    print(f"\n  recall by injected error type:")
    for et, d in recall_by_type.items():
        print(f"    {et:<32} {d['caught_by_structural']}/{d['n']} = {d['recall']}")

    print(f"\n=== LLM judges (natural items only, vs adjudicated truth) ===")
    for j, d in judge_overall.items():
        print(f"  {j:<10} n={d['n']:<4} acc={d['accuracy']}  "
              f"false_accept(a)={d['false_accept_rate']}  false_reject(b)={d['false_reject_rate']}")

    print(f"\n=== decomposition by group ===")
    for g, d in decomp.items():
        print(f"  {g}: p_g={d['p_candidate_error']}  "
              f"struct b'_g={d['structural_checker']['b_prime_false_reject']}")
        for j, jd in d["judges"].items():
            print(f"      {j:<10} a_g={jd['a_false_accept']}  b_g={jd['b_false_reject']}  n={jd['n']}")

    print(f"\n=== producer error rate (adjudicated, not proxy; natural items only) ===")
    for fam, d in by_fam.items():
        print(f"  {fam:<10} n={d['n']:<4} error_rate={d['error_rate_adjudicated']}")

    print(f"\n=== arm x family (the central cell) ===")
    for k, d in arm_family.items():
        print(f"  {k:<20} n={d['n']:<4} error_rate={d['error_rate_adjudicated']}")

    print(f"\n=== structural false-reject rate (b') by family, truth-clean items only ===")
    for fam, d in struct_false_reject_by_family.items():
        print(f"  {fam:<10} n_clean={d['n_truth_clean']:<4} false_rejects={d['struct_false_rejects']:<4} b'={d['b_prime']}")

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

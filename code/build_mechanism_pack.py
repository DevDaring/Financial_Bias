"""
E2 mechanism benchmark (Future_Plan §E2): claim-level semantic contrasts with forged witnesses.

For each base claim (one evidence relation of one fresh borrower) four versions:

  original      a correct statement of the relation
  paraphrase    a valid rewording (different syntax and names)
  corruption    a genuine semantic error of the relation's type
  provocation   a CORRECT statement that invites a plausible but false objection
                (a disclaimer containing causal words, a rounded value, an
                acknowledged undocumented code, a restricted-scope rank, a
                negated scale conversion, a supported change over time)

and one witness per item: a TRUE witness for every corruption (expected: verified,
or semantic_plausible for causal/target content) and a FORGED witness for every
correct version (expected: not verified), cycling three forgeries:
  wrong_field       the quoted words are right, the cited evidence is another field
  false_arithmetic  the witness reports a wrong evidence value for a correct claim
  partial_quote     the quote drops the negation or disclaimer that makes the claim correct

Relations: VALUE, DIRECTION, RANK, SCALE, TEMPORAL, CODE, CAUSAL, PD (value / target).

Template split. Every relation has DEV wording (field source names; used while the
executable checker was developed) and HELDOUT wording (plain-language aliases,
different syntax), assigned per base claim by a fixed hash, never edited after the
first build. Labels are correct BY CONSTRUCTION from the evidence object and must still
be validated independently (needs_independent_validation = true) before any claim.
Templates were written by the method developers: an annotator-written held-out set is
the stronger test and is listed as a protocol item.

Borrowers: development-role borrowers NOT used in the M1 pack, the practice set or the
v1/superseded pilots - never calibration or locked-evaluation borrowers.

    python build_mechanism_pack.py --per-dataset 30           # 60 borrowers, 120 base claims, 480 items
    python build_mechanism_pack.py --per-dataset 30 --v2      # fresh held-out pack (written 7 Oct, after
                                                              # executable/1.1 was frozen; scored once)
Output: Submission/mechanism/mechanism_pack_v1.jsonl (+ manifest.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pack_m1 import SUB, _mutate_value, _numeric_ok, _all_values  # noqa: E402
from evidence import EvidenceObject, EvidenceType  # noqa: E402
from evidence_builder import _fmt_value  # noqa: E402

OUT = SUB / "mechanism"
SEED = 20261007
PACK_VERSION = "mechanism/1.0"
RELATIONS = ["VALUE", "DIRECTION", "RANK", "SCALE", "TEMPORAL", "CODE", "CAUSAL", "PD"]
RULE = {"VALUE": "R_VALUE", "DIRECTION": "R_DIRECTION", "RANK": "R_RANKING", "SCALE": "R_SCALE",
        "TEMPORAL": "R_TEMPORAL", "CODE": "R_CODE_MEANING", "CAUSAL": "R_CAUSAL", "PD": "R_VALUE"}


def split_of(key: str) -> str:
    return "heldout" if int(hashlib.sha256(f"mech-split:{key}".encode()).hexdigest()[:8], 16) % 2 else "dev"


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------
def src_name(it) -> str:
    return (it.feature_group or it.source_field or "").replace(" (derived)", "")


def plain_name(it, rng: random.Random) -> str:
    """A plain-language documented alias (held-out wording), else the source name."""
    al = [a for a in (it.aliases or []) if " " in a and not re.search(r"_|\d{4}", a)]
    return rng.choice(sorted(al)) if al else src_name(it)


def name(it, split: str, rng: random.Random) -> str:
    return src_name(it) if split == "dev" else plain_name(it, rng)


def short_unit(f) -> str:
    u = re.split(r"\s*[(;]", f.unit or "")[0].strip()
    return "" if u.lower() in ("ratio", "") else " " + u


def when(f) -> str:
    return f" {f.observed_at}" if f.observed_at and f.observed_at.startswith(("in ", "at ", "across ")) else ""


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# ---------------------------------------------------------------------------
# relation builders: each returns dict(version -> (text, witness)) or None
# ---------------------------------------------------------------------------
def _attrs(eo):
    return sorted(eo.by_type(EvidenceType.MODEL_ATTRIBUTION), key=lambda a: -abs(a.contribution or 0))


def _own_fact(eo, a):
    for c in a.components or []:
        f = eo.get(c)
        if f is not None and f.etype is EvidenceType.OBSERVED_FACT:
            return f
    return None


def _w(rule, quote, ids, claimed=None, op=None, kind="true"):
    return {"rule": rule, "quote": quote, "evidence_ids": ids, "claimed_result": claimed,
            "operation": op, "forgery": kind}


def rel_value(eo, a, split, rng):
    f = _own_fact(eo, a)
    if f is None or not _numeric_ok(f):
        return None
    nm, v = name(a, split, rng), _fmt_value(f.value)
    bad = _mutate_value(f.value, rng, _all_values(eo))
    if bad is None:
        return None
    rv = f.value
    rounded = (f"{round(rv, -1):,.0f}" if abs(rv) >= 1000 else
               (f"{rv:.2f}" if abs(rv) < 10 else f"{rv:.1f}"))
    if split == "dev":
        o = f"{nm} (recorded value {v}{short_unit(f)}{when(f)}) is one of the displayed drivers."
        p = f"The displayed driver {nm} has a recorded value of {v}{short_unit(f)}."
        c = o.replace(f"recorded value {v}", f"recorded value {_fmt_value(float(bad))}")
        q = f"{nm} is about {rounded}{short_unit(f)}, rounded."
    else:
        o = f"The recorded {nm} stands at {v}{short_unit(f)}."
        p = f"For this borrower, {nm} was {v}{short_unit(f)}{when(f)}."
        c = o.replace(f" {v}", f" {_fmt_value(float(bad))}", 1)
        q = f"Rounded, the {nm} comes to roughly {rounded}{short_unit(f)}."
    if split == "v2":
        o = f"{cap(nm)}: {v}{short_unit(f)}."
        p = f"Its {nm} is recorded as {v}{short_unit(f)}{when(f)}."
        c = f"{cap(nm)}: {_fmt_value(float(bad))}{short_unit(f)}."
        q = f"{cap(nm)} of approximately {rounded}{short_unit(f)}."
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q,
            "_ids": [f.eid, a.eid], "_true": f.value, "_wrong_field": _other_id(eo, [f.eid, a.eid]),
            "_error_type": "numeric_or_unit_error", "_neg": None}


def rel_direction(eo, a, split, rng):
    if abs(a.contribution or 0) < 0.01:
        return None
    nm = name(a, split, rng)
    up = a.direction == "increases_risk"
    v, vo = ("raises", "lowers") if up else ("lowers", "raises")
    if split == "dev":
        o = f"{nm} {v} the model's risk score."
        p = f"Relative to the model's reference, {nm} {v} the risk score."
        c = f"{nm} {vo} the model's risk score."
        q = f"{nm} does not {vo[:-1]} the model's risk score; {nm} {v} it."
    else:
        o = f"The {nm} pushes the model's risk score {'up' if up else 'down'}."
        p = f"Compared with the model's reference, the {nm} {'adds to' if up else 'pulls down'} the risk score."
        c = f"The {nm} pushes the model's risk score {'down' if up else 'up'}."
        q = (f"The {nm} {v} the model's risk score; this is an association inside the model, "
             f"not a cause of {'higher' if up else 'lower'} default.")
    if split == "v2":
        inc, dec = ("increases", "decreases") if up else ("decreases", "increases")
        o = f"{cap(nm)} {inc} the estimated risk."
        p = f"Relative to the reference, {nm} makes default {'more' if up else 'less'} likely in the model."
        c = f"{cap(nm)} {dec} the estimated risk."
        q = f"{cap(nm)} does not {dec[:-1]} the estimated risk; it {inc} it."
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q, "_ids": [a.eid],
            "_true": a.direction, "_wrong_field": _other_id(eo, [a.eid] + list(a.components or [])),
            "_error_type": "sign_reversal",
            "_neg": {"dev": f"does not {vo[:-1]}", "heldout": "not a cause"}.get(split, "does not")}


def rel_rank(eo, a, split, rng):
    at = _attrs(eo)
    if len(at) < 2 or a.eid != at[0].eid or abs(at[0].contribution or 0) - abs(at[1].contribution or 0) <= 0.01:
        return None
    top, sec = at[0], at[1]
    nt, ns = name(top, split, rng), name(sec, split, rng)
    other = [x for x in at if (x.direction != top.direction)]
    if split == "dev":
        o = f"The strongest driver of the model's risk score is {nt}."
        p = f"{nt} has the largest contribution of all the displayed drivers."
        c = f"The strongest driver of the model's risk score is {ns}."
    else:
        o = f"Of the displayed factors, {nt} is the main driver of the score."
        p = f"The biggest single contribution to the risk score comes from {nt}."
        c = f"Of the displayed factors, {ns} is the main driver of the score."
    if split == "v2":
        o = f"{cap(nt)} contributes most to the score."
        p = f"Among the displayed factors, none moves the score as much as {nt}."
        c = f"{cap(ns)} contributes most to the score."
    if other:
        od = max(other, key=lambda x: abs(x.contribution or 0))
        word = "risk-reducing" if od.direction == "decreases_risk" else "risk-increasing"
        q = (f"The strongest {word} factor is {name(od, split, rng)}." if split != "v2" else
             f"Among factors {'lowering' if od.direction == 'decreases_risk' else 'raising'} the risk, "
             f"the largest is {name(od, split, rng)}.")
        q_ids = [od.eid]
    else:
        q = f"{ns} is one of the drivers, but the strongest driver is {nt}."
        q_ids = [top.eid]
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q, "_ids": [top.eid],
            "_q_ids": q_ids, "_true": top.eid, "_wrong_field": sec.eid, "_error_type": "other_material_error",
            "_neg": None}


def rel_scale(eo, a, split, rng):
    c = a.contribution or 0
    if abs(c) < 0.05:
        return None
    nm = name(a, split, rng)
    pct = round(100 * abs(c))
    v = "raises" if c > 0 else "lowers"
    if split == "dev":
        o = f"{nm} contributes {c:+.2f} on the raw log-odds scale of the model."
        p = f"On the model's raw log-odds margin, {nm} accounts for {c:+.2f}."
        cc = f"{nm} {v} the probability of default by {pct} percentage points."
        q = f"{nm} adds {c:+.2f} to the raw log-odds score, which is not a {pct}-point change in the probability."
    else:
        o = f"The {nm} shifts the raw score by {abs(c):.2f} log-odds units."
        p = f"In log-odds terms, the {nm} moves the model's raw score by {abs(c):.2f}."
        cc = f"The {nm} {v} the chance of default by {pct}%."
        q = f"The {nm} moves the raw score by {abs(c):.2f} in log-odds; it should not be read as {pct}% of probability."
    if split == "v2":
        o = f"{cap(nm)} has a SHAP value of {c:+.2f} on the log-odds margin."
        p = f"The attribution for {nm} is {c:+.2f} in raw margin units."
        cc = f"{cap(nm)} {'adds' if c > 0 else 'removes'} {pct}% {'to' if c > 0 else 'from'} the default probability."
        q = f"The {c:+.2f} for {nm} is in log-odds, not a {pct} percentage-point shift in probability."
    return {"original": o, "paraphrase": p, "corruption": cc, "provocation": q, "_ids": [a.eid],
            "_true": c, "_wrong_field": _other_id(eo, [a.eid]), "_error_type": "numeric_or_unit_error",
            "_neg": {"dev": "not a", "heldout": "should not"}.get(split, "not a")}


def rel_temporal(eo, a, split, rng):
    facts = eo.by_type(EvidenceType.OBSERVED_FACT)
    f = _own_fact(eo, a)
    if f is None or not f.series:
        return None
    ser = sorted([x for x in facts if x.series == f.series and isinstance(x.value, (int, float))
                  and x.period_order is not None], key=lambda x: -x.period_order)
    if len(ser) < 2 or ser[0].value == ser[-1].value or f.series == "repayment_status":
        return None
    old, new = ser[0], ser[-1]
    noun = {"bill_amount": "bill amount", "payment_amount": "amount paid"}.get(f.series, f.series.replace("_", " "))
    mo = lambda x: (x.observed_at or "").replace("in ", "").strip()   # noqa: E731
    up = new.value > old.value
    tv, fv = ("rose", "fell") if up else ("fell", "rose")
    if split == "dev":
        o = f"The {noun} {tv} from {mo(old)} to {mo(new)}."
        p = f"Between {mo(old)} and {mo(new)}, the {noun} {tv}."
        c = f"The {noun} {fv} from {mo(old)} to {mo(new)}."
        q = f"Over the statements shown, the {noun} {'went up' if up else 'went down'} from {mo(old)} to {mo(new)}."
    else:
        o = f"From {mo(old)} to {mo(new)} the {noun} {'increased' if up else 'decreased'}."
        p = f"The {noun} in {mo(new)} was {'higher' if up else 'lower'} than in {mo(old)}, so it {tv}."
        c = f"From {mo(old)} to {mo(new)} the {noun} {'decreased' if up else 'increased'}."
        q = f"The {noun} has {'risen' if up else 'fallen'} over time, from {mo(old)} to {mo(new)}."
    if split == "v2":
        o = f"The {noun} {'increased' if up else 'decreased'} between {mo(old)} and {mo(new)}."
        p = f"Comparing {mo(old)} with {mo(new)}, the {noun} {'went up' if up else 'went down'}."
        c = f"The {noun} {'decreased' if up else 'increased'} between {mo(old)} and {mo(new)}."
        q = f"The {noun} has {'grown' if up else 'shrunk'} from {mo(old)} to {mo(new)}, as the dated values show."
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q, "_ids": [old.eid, new.eid],
            "_true": new.value - old.value, "_wrong_field": _other_id(eo, [old.eid, new.eid] + [x.eid for x in ser]),
            "_error_type": "unsupported_trend", "_neg": None}


def rel_code(eo, a, split, rng):
    cands = [x for x in eo.by_type(EvidenceType.OBSERVED_FACT)
             if x.value_label and re.search(r"not defined|undocumented|not documented", x.value_label, re.I)]
    if not cands:
        return None
    f = cands[0]
    code = _fmt_value(f.value)
    nm = name(f, split, rng) if split == "heldout" else src_name(f)
    if split == "dev":
        o = f"{nm} is code {code}{when(f)}, a code the data source does not define."
        p = f"The data source leaves code {code} for {nm} undocumented."
        c = f"{nm} is code {code}{when(f)}, which means the borrower paid duly."
        q = f"{nm} is code {code}; because that code is undocumented it cannot be read as on-time payment."
    else:
        o = f"The {nm} shows code {code}, whose meaning is not documented by the source."
        p = f"Code {code} for the {nm} has no documented meaning."
        c = f"The {nm} shows code {code}, meaning no payment delay."
        q = f"The {nm} is code {code}; the source does not say whether that means no delay."
    if split == "v2":
        o = f"Code {code} recorded for {nm} has no documented meaning in the source."
        p = f"For {nm}, the value is code {code}, which the source leaves undefined."
        c = f"Code {code} for {nm} indicates timely payment."
        q = f"Although code {code} is often read as on-time payment, the source does not document it."
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q, "_ids": [f.eid],
            "_true": f.value_label, "_wrong_field": _other_id(eo, [f.eid]), "_error_type": "other_material_error",
            "_neg": {"dev": "cannot be read", "heldout": "does not say"}.get(split, "does not document")}


def rel_causal(eo, a, split, rng):
    if abs(a.contribution or 0) < 0.01:
        return None
    nm = name(a, split, rng)
    up = a.direction == "increases_risk"
    hl = "higher" if up else "lower"
    if split == "dev":
        o = f"{nm} is associated with a {hl} model risk score."
        p = f"In the model, {nm} goes with a {hl} risk score."
        c = (f"Reducing {nm} will lower the borrower's risk of default." if up
             else f"Raising {nm} will lower the borrower's risk of default.")
        q = f"This does not mean that changing {nm} would cause a different default outcome."
    else:
        o = f"The model links the {nm} to a {hl} risk score."
        p = f"A {hl} score accompanies the {nm} in this model."
        c = (f"If the borrower cut the {nm}, the chance of default would fall." if up
             else f"If the borrower increased the {nm}, the chance of default would fall.")
        q = f"The link between the {nm} and the score is not evidence that the {nm} causes default."
    if split == "v2":
        o = f"{cap(nm)} is linked to a {hl} score in the model."
        p = f"The model associates {nm} with a {hl} risk estimate."
        c = (f"Paying down {nm} would reduce the likelihood of default." if up
             else f"Increasing {nm} would make default less likely.")
        q = f"Changing {nm} would not necessarily change whether the borrower defaults."
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q, "_ids": [a.eid],
            "_true": None, "_wrong_field": _other_id(eo, [a.eid]), "_error_type": "unsupported_causal_or_action",
            "_neg": {"dev": "does not mean", "heldout": "not evidence"}.get(split, "would not necessarily")}


def rel_pd(eo, a, split, rng):
    p_ = eo.prediction
    if p_ is None or p_.calibrated_pd is None:
        return None
    pd = p_.calibrated_pd
    pdt = f"{pd:.1%}"
    bad = f"{min(0.95, pd + rng.choice([0.07, 0.11, 0.15])):.1%}"
    hz = "next month" if eo.dataset == "taiwan" else "the first loan instalments"
    if split == "dev":
        o = f"The model's estimated probability of default is {pdt}."
        p = f"The calibrated default probability comes to {pdt}."
        c = f"The model's estimated probability of default is {bad}."
        q = f"The model's estimated probability of default is {pdt} for {hz}, not over 12 months."
    else:
        o = f"According to the model, the chance of default is {pdt}."
        p = f"The model puts the borrower's default probability at {pdt}."
        c = f"According to the model, the chance of default is {pdt} within the next 12 months."
        q = f"The model gives a {pdt} default probability for {hz}; it is not a 12-month estimate."
    if split == "v2":
        o = f"Estimated default probability: {pdt}."
        p = f"The model assigns a {pdt} probability of default."
        c = f"The model assigns a {pdt} probability of default over the next year."
        q = f"The {pdt} estimate refers to {hz}, not to a one-year horizon."
    et = "numeric_or_unit_error" if split == "dev" else "other_material_error"
    return {"original": o, "paraphrase": p, "corruption": c, "provocation": q, "_ids": [p_.eid],
            "_true": pd, "_wrong_field": _other_id(eo, [p_.eid]), "_error_type": et,
            "_rule": "R_VALUE" if split == "dev" else "R_TARGET",
            "_neg": {"dev": "not over", "heldout": "not a"}.get(split, "not to")}


BUILDERS = {"VALUE": rel_value, "DIRECTION": rel_direction, "RANK": rel_rank, "SCALE": rel_scale,
            "TEMPORAL": rel_temporal, "CODE": rel_code, "CAUSAL": rel_causal, "PD": rel_pd}


def _other_id(eo, avoid) -> Optional[str]:
    xs = [i.eid for i in eo.items if i.eid not in avoid and i.etype in (EvidenceType.OBSERVED_FACT,
                                                                       EvidenceType.MODEL_ATTRIBUTION)]
    return xs[-1] if xs else None


# ---------------------------------------------------------------------------
def items_for(eo: EvidenceObject, rel: str, base_id: str, rng: random.Random, split: Optional[str] = None) -> List[dict]:
    split = split or split_of(base_id)
    at = _attrs(eo)
    a = at[0] if rel == "RANK" else (at[rng.randrange(len(at))] if at else None)
    if rel in ("VALUE", "TEMPORAL"):
        rng.shuffle(at := list(at))
    out = None
    for cand in ([a] if rel in ("RANK", "PD", "CODE") else at):
        out = BUILDERS[rel](eo, cand, split, rng)
        if out:
            break
    if not out:
        return []
    rule = out.get("_rule", RULE[rel])
    items = []
    forgeries = ["wrong_field", "false_arithmetic", "partial_quote"]
    for k, ver in enumerate(["original", "paraphrase", "corruption", "provocation"]):
        text = out[ver]
        ids = out.get("_q_ids", out["_ids"]) if ver == "provocation" else out["_ids"]
        if ver == "corruption":
            w = _w(rule, text, out["_ids"], out["_true"], kind="true")
            expect = "semantic_plausible" if rule in ("R_CAUSAL", "R_TARGET") else "verified"
        else:
            kind = forgeries[(int(hashlib.sha256(f"{base_id}{ver}".encode()).hexdigest()[:6], 16)) % 3]
            if kind == "wrong_field" and out["_wrong_field"]:
                w = _w(rule, text, [out["_wrong_field"]], kind=kind)
            elif kind == "partial_quote" and out.get("_neg") and out["_neg"] in text and ver == "provocation":
                tail = text.split(out["_neg"], 1)[1].strip(" ,;.")
                w = _w(rule, tail or text, ids, kind=kind)
            else:
                wrong = out["_true"]
                if isinstance(wrong, (int, float)):
                    wrong = round(float(wrong) * 1.7 + 1, 4)
                w = _w(rule, text, ids, claimed=wrong, kind="false_arithmetic")
            expect = "not_verified"
        items.append({"item_id": f"{base_id}-{ver[:4]}", "base_id": base_id,
                      "pack_version": PACK_VERSION if split != "v2" else PACK_VERSION.replace("1.0", "2.0"),
                      "borrower_id": eo.borrower_id, "dataset": eo.dataset,
                      "group": (eo.group_attributes or {}).get("sex") or (eo.group_attributes or {}).get("gender")
                      or next(iter((eo.group_attributes or {}).values()), None),
                      "relation": rel, "version": ver, "template_split": split, "text": text,
                      "label": "error" if ver == "corruption" else "correct",
                      "error_type": out["_error_type"] if ver == "corruption" else None,
                      "rule": rule, "witness": w, "witness_expect": expect,
                      "needs_independent_validation": True,
                      "evidence": eo.to_dict(include_groups=False)})
    return items


def used_borrowers() -> set:
    used = set()
    for line in open(SUB / "annotation_key_m1.jsonl"):
        r = json.loads(line)
        used.add(r.get("borrower_id"))
        used |= set(r.get("withheld_templates") or [])
    pk = SUB / "practice_key_m1.json"
    if pk.exists():
        for bid in re.findall(r"(?:taiwan|home_credit)-\d+", pk.read_text()):
            used.add(bid)
    return {u for u in used if u}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="fresh held-out pack: new borrowers (v1 borrowers excluded), new seed, v2 wording only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "v2" if a.v2 else "v1"
    path = OUT / f"mechanism_pack_{tag}.jsonl"
    if path.exists() and not a.force:
        print(f"{path} exists; the pack is frozen once built (use --force only before any evaluation)")
        return 1
    from build_pack_m1 import extra_borrowers
    rng = random.Random(SEED + (1 if a.v2 else 0))
    exclude = used_borrowers()
    if a.v2:
        exclude |= {json.loads(l)["borrower_id"] for l in open(OUT / "mechanism_pack_v1.jsonl")}
    rows, k = [], 0
    for ds in ("taiwan", "home_credit"):
        rels = [r for r in RELATIONS if not (ds == "home_credit" and r in ("TEMPORAL", "CODE"))]
        bs = extra_borrowers(ds, exclude, a.per_dataset * 2, rng)
        made = 0
        for eo, _claims in bs:
            if made >= a.per_dataset:
                break
            got = 0
            for _ in range(len(rels)):
                rel = rels[k % len(rels)]
                k += 1
                pre = "MY" if a.v2 else "MX"
                its = items_for(eo, rel, f"{pre}-{ds[:2]}-{eo.borrower_id.split('-')[-1]}-{rel}", rng,
                                split="v2" if a.v2 else None)
                if its:
                    rows += its
                    got += 1
                if got == 2:
                    break
            made += 1
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    from collections import Counter
    man = {"pack_version": rows[0]["pack_version"] if rows else PACK_VERSION, "seed": SEED + (1 if a.v2 else 0),
           "checker_frozen_before_build": "executable/1.1" if a.v2 else None, "n_items": len(rows),
           "n_base_claims": len({r['base_id'] for r in rows}), "n_borrowers": len({r['borrower_id'] for r in rows}),
           "by_relation": Counter(r["relation"] for r in rows if r["version"] == "original"),
           "by_split": Counter(r["template_split"] for r in rows if r["version"] == "original"),
           "by_dataset": Counter(r["dataset"] for r in rows if r["version"] == "original"),
           "excluded_borrowers": len(exclude),
           "note": "labels by construction; needs independent validation before use as reference truth"}
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps(man, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

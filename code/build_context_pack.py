"""
Controlled context experiment (Future_Plan review §6): does a verifier's accuracy on ONE focal claim depend on
the claim's context? 80 base cases x 3 contexts x 2 truth conditions = 480 passages.

  alone    the focal claim by itself
  claims   the focal claim among three other CORRECT claims about the same borrower (probability estimate,
           directions of two other attributions)
  filler   the focal claim among neutral prose of about the same length that asserts nothing about the
           borrower (claim count fixed, tokens increased)

Truth: correct focal claim vs a single-error counterpart, built from the evidence (labels by construction).
Error relations, balanced: VALUE (wrong number bound to a field), RANK (second driver called strongest),
SCALE (log-odds contribution presented as a probability change), NEGATION (a negated direction that
is wrong), REFERENCE (a recorded fact with no attribution given a direction), TEMPORAL (Taiwan only:
reversed change over time). The focal claim's position (1-4) cycles across base cases and is the same
in both multi-sentence contexts.

Writer: Llama 4 Maverick (Meta) on Bedrock, a family used in no judge or reader role. It sees only a
one-line specification, never the evidence, so it cannot "repair" an error specification. Every
realization is format-checked against its specification; failures are rewritten (at most 3 times) or
the base case is dropped. Borrowers: development-role borrowers never used by any earlier pack.

Freeze: the verifier versions are recorded BEFORE the pack is generated (manifest), and the pack is
built once (no --force after any evaluation).

    python build_context_pack.py --per-dataset 40
Output: Submission/mechanism/context_pack_v1.jsonl, context_manifest_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gateway as G  # noqa: E402
import json_parse as JP  # noqa: E402
from build_mechanism_pack import _attrs, _own_fact, plain_name, short_unit, src_name, used_borrowers  # noqa: E402
from build_pack_m1 import SUB, _all_values, _mutate_value, _numeric_ok  # noqa: E402
from evidence import EvidenceType  # noqa: E402
from evidence_builder import _fmt_value  # noqa: E402

OUT = SUB / "mechanism"
SEED = 20261008
WRITER = "llama4_maverick"
VERSION = "context/1.0"
UPW = r"\b(rais|increas|push(?:es|ed)? up|add|higher|elevat|boost)"
DNW = r"\b(lower|decreas|reduc|cut|offset|pull(?:s|ed)? down|less)"

WRITER_SYS = ("You write single sentences for a credit-model explanation. Write exactly what the specification "
              "says, in natural plain English, as one sentence. Do not add any other facts, numbers, hedges or "
              "comments. Use the field name given in quotes (you may drop the quotes). "
              'Reply with JSON only: {"sentence": "..."}')
FILLER_SYS = ("You write neutral framing text for an automated credit-model summary. Write {n} sentences (about "
              "{w} words in total) on how a reader should use such a summary: that it describes a statistical "
              "model's scoring, that a credit officer reviews it, that it is not advice, and similar general points. "
              "Do NOT mention any borrower, field, amount, number, percentage, month, or whether anything raises or "
              'lowers risk. Reply with JSON only: {{"sentences": ["...", "..."]}}')


def _dir(a):
    return "raises" if (a.contribution or 0) > 0 else "lowers"


def _num_in(text, v) -> bool:
    t = text.replace(",", "")
    s = _fmt_value(v).replace(",", "")
    return s in t or (s.endswith(".0") and s[:-2] in t)


def specs(eo, rng: random.Random, rel: str):
    """(correct spec, error spec, checks, ids, error_type) or None."""
    at = _attrs(eo)
    if len(at) < 3:
        return None
    nm = lambda it: plain_name(it, rng)          # noqa: E731
    if rel == "VALUE":
        for a in at:
            f = _own_fact(eo, a)
            if f is None or not _numeric_ok(f):
                continue
            bad = _mutate_value(f.value, rng, _all_values(eo))
            if bad is None:
                continue
            n, u = nm(a), short_unit(f)
            return (f'State that the borrower\'s "{n}" is {_fmt_value(f.value)}{u} and that it {_dir(a)} the model\'s risk score.',
                    f'State that the borrower\'s "{n}" is {_fmt_value(float(bad))}{u} and that it {_dir(a)} the model\'s risk score.',
                    (lambda t, v=f.value: _num_in(t, v)), (lambda t, v=float(bad): _num_in(t, v)),
                    [f.eid, a.eid], "numeric_or_unit_error", a.eid)
        return None
    if rel == "RANK":
        a1, a2 = at[0], at[1]
        if abs(abs(a1.contribution or 0) - abs(a2.contribution or 0)) < 0.02:
            return None
        n1, n2 = nm(a1), nm(a2)
        sup = r"\b(strongest|largest|biggest|main|most important|primary|leading|top)\b"
        return (f'State that "{n1}" is the strongest driver of the model\'s risk score for this borrower.',
                f'State that "{n2}" is the strongest driver of the model\'s risk score for this borrower.',
                (lambda t: bool(re.search(sup, t, re.I))), (lambda t: bool(re.search(sup, t, re.I))),
                [a1.eid, a2.eid], "other_material_error", a1.eid)
    if rel == "SCALE":
        a = next((x for x in at if abs(x.contribution or 0) >= 0.05), None)
        if a is None:
            return None
        c = a.contribution or 0
        n = nm(a)
        return (f'State that "{n}" moves the model\'s raw log-odds score by {c:+.2f}.',
                f'State that "{n}" {"raises" if c > 0 else "lowers"} the borrower\'s probability of default by {round(100 * abs(c))} percentage points.',
                (lambda t: "log" in t.lower() and f"{abs(c):.2f}" in t),
                (lambda t: bool(re.search(r"%|percent", t, re.I))),
                [a.eid], "numeric_or_unit_error", a.eid)
    if rel == "NEGATION":
        a = next((x for x in at if abs(x.contribution or 0) >= 0.05), None)
        if a is None:
            return None
        n = nm(a)
        good, wrong = ("lower", "raise") if (a.contribution or 0) > 0 else ("raise", "lower")
        return (f'State that "{n}" does not {good} the model\'s risk score (use the word "not").',
                f'State that "{n}" does not {wrong} the model\'s risk score (use the word "not").',
                (lambda t: bool(re.search(r"\bnot\b|n't", t))), (lambda t: bool(re.search(r"\bnot\b|n't", t))),
                [a.eid], "sign_reversal", a.eid)
    if rel == "REFERENCE":
        owned = {c for x in at for c in (x.components or [])}
        facts = [f for f in eo.by_type(EvidenceType.OBSERVED_FACT)
                 if f.eid not in owned and _numeric_ok(f) and not any(
                     src_name(x).lower() == str(f.source_field or "").lower() for x in at)]
        if not facts:
            return None
        f = rng.choice(sorted(facts, key=lambda x: x.eid))
        n, u = plain_name(f, rng), short_unit(f)
        d = rng.choice(["raises", "lowers"])
        return (f'State that the borrower\'s "{n}" was recorded as {_fmt_value(f.value)}{u}, as a plain fact (say nothing about risk).',
                f'State that the borrower\'s "{n}" of {_fmt_value(f.value)}{u} {d} the model\'s risk score.',
                (lambda t: _num_in(t, f.value) and not re.search(UPW + "|" + DNW, t, re.I)),
                (lambda t, d=d: _num_in(t, f.value) and bool(re.search(UPW if d == "raises" else DNW, t, re.I))),
                [f.eid], "other_material_error", None)
    if rel == "TEMPORAL":
        facts = eo.by_type(EvidenceType.OBSERVED_FACT)
        for a in at:
            f = _own_fact(eo, a)
            if f is None or not f.series or f.series == "repayment_status":
                continue
            ser = sorted([x for x in facts if x.series == f.series and isinstance(x.value, (int, float))
                          and x.period_order is not None], key=lambda x: -x.period_order)
            if len(ser) < 2 or ser[0].value == ser[-1].value:
                continue
            old, new = ser[0], ser[-1]
            noun = {"bill_amount": "bill amount", "payment_amount": "amount paid"}.get(f.series, f.series.replace("_", " "))
            mo = lambda x: (x.observed_at or "").replace("in ", "").strip()   # noqa: E731
            up = new.value > old.value
            w1, w2 = ("increased", "decreased") if up else ("decreased", "increased")
            return (f"State that the {noun} {w1} from {mo(old)} to {mo(new)}.",
                    f"State that the {noun} {w2} from {mo(old)} to {mo(new)}.",
                    (lambda t, u=up: bool(re.search(UPW if u else DNW, t, re.I) or re.search(r"\b(rose|grew|went up)\b" if u else r"\b(fell|dropped|declined|went down)\b", t, re.I))),
                    (lambda t, u=up: bool(re.search(DNW if u else UPW, t, re.I) or re.search(r"\b(fell|dropped|declined|went down)\b" if u else r"\b(rose|grew|went up)\b", t, re.I))),
                    [old.eid, new.eid], "unsupported_trend", a.eid)
        return None
    return None


def context_specs(eo, rng, focal_attr, rel):
    """Three CORRECT context claims: probability estimate + directions of two other attributions."""
    pred = next((x for x in eo.items if x.calibrated_pd is not None), None)
    at = [x for x in _attrs(eo) if x.eid != focal_attr and abs(x.contribution or 0) >= 0.05]
    if rel == "RANK":
        at = [x for x in at if x.eid != _attrs(eo)[0].eid and x.eid != _attrs(eo)[1].eid]
    if pred is None or len(at) < 2:
        return None
    pd = f"{100 * pred.calibrated_pd:.1f}%"
    out = [(f"State that the model estimates a {pd} probability of default for this borrower.",
            lambda t, p=pd: p.rstrip("%") in t)]
    for a in at[:2]:
        n = plain_name(a, rng)
        d = _dir(a)
        out.append((f'State that "{n}" {d} the model\'s risk score.',
                    lambda t, d=d: bool(re.search(UPW if d == "raises" else DNW, t, re.I)) and not re.search(r"\bnot\b|n't", t)))
    return out


def write(spec, check, item):
    for k in range(3):
        msgs = [{"role": "system", "content": WRITER_SYS}, {"role": "user", "content": spec}]
        r = G.GATEWAY.complete(WRITER, msgs, temperature=0.7 if k else 0.3, max_tokens=200, want_json=True,
                               experiment="context-pack", item_id=f"{item}|w{k}", role="writer")
        d, _ = JP.parse_local(r.text, validate=lambda o: isinstance(o, dict) and isinstance(o.get("sentence"), str))
        if d and check(d["sentence"].strip()):
            s = d["sentence"].strip()
            return s if s.endswith((".", "!", "?")) else s + "."
    return None


def filler(n, words, item):
    for k in range(3):
        msgs = [{"role": "user", "content": FILLER_SYS.format(n=n, w=words)}]
        r = G.GATEWAY.complete(WRITER, msgs, temperature=0.7, max_tokens=400, want_json=True,
                               experiment="context-pack", item_id=f"{item}|f{k}", role="writer")
        d, _ = JP.parse_local(r.text, validate=lambda o: isinstance(o, dict) and isinstance(o.get("sentences"), list))
        if d:
            ss = [str(s).strip() for s in d["sentences"] if str(s).strip()][:n]
            if len(ss) == n and not any(re.search(r"\d|%|" + UPW + "|" + DNW, s, re.I) for s in ss):
                return [s if s.endswith(".") else s + "." for s in ss]
    return None


def freeze_record() -> dict:
    here = Path(__file__).resolve().parent
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest()[:16]
            for f in ("witness.py", "executable_verifier.py", "dispute.py", "comparators2.py", "contract.py")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=40)
    ap.add_argument("--tag", default="v1", help="v2: replication pack on new borrowers (v1 borrowers excluded)")
    a = ap.parse_args()
    path = OUT / f"context_pack_{a.tag}.jsonl"
    if path.exists():
        print(f"{path} exists; built once, never rebuilt after evaluation")
        return 1
    import witness as W
    import dispute as DP
    man = {"version": VERSION, "seed": SEED, "writer": WRITER, "frozen_verifiers": {
        "executable": W.EXEC_VERSION, "dispute": DP.DISPUTE_VERSION, "source_sha256_16": freeze_record()}}
    from build_pack_m1 import extra_borrowers
    rng = random.Random(SEED + (0 if a.tag == "v1" else 1000))
    man["seed"] = SEED + (0 if a.tag == "v1" else 1000)
    exclude = used_borrowers()
    for prev in OUT.glob("context_pack_v*.jsonl"):
        exclude |= {json.loads(l)["borrower_id"] for l in open(prev)}
    for p in ("mechanism_pack_v1.jsonl", "mechanism_pack_v2.jsonl", "mechanism_pack_v3_annotator.jsonl"):
        if (OUT / p).exists():
            exclude |= {json.loads(l).get("borrower_id") for l in open(OUT / p)}
    rows, dropped, k = [], [], 0
    for ds in ("taiwan", "home_credit"):
        rels = ["VALUE", "RANK", "SCALE", "NEGATION", "REFERENCE"] + (["TEMPORAL"] if ds == "taiwan" else [])
        made = 0
        for eo, _cl in extra_borrowers(ds, exclude, a.per_dataset * 3, rng):
            if made >= a.per_dataset:
                break
            rel = rels[made % len(rels)]
            sp = specs(eo, rng, rel)
            if sp is None:
                dropped.append((eo.borrower_id, rel, "no spec"))
                continue
            good, bad, chk_g, chk_b, ids, etype, focal_attr = sp
            ctx = context_specs(eo, rng, focal_attr, rel)
            if ctx is None:
                dropped.append((eo.borrower_id, rel, "no context"))
                continue
            base = f"{'CX' if a.tag == 'v1' else 'CY'}-{ds[:2]}-{eo.borrower_id.split('-')[-1]}-{rel}"
            fg, fb = write(good, chk_g, base + "|good"), write(bad, chk_b, base + "|bad")
            cs = [write(s, c, base + f"|ctx{j}") for j, (s, c) in enumerate(ctx)]
            if not fg or not fb or None in cs:
                dropped.append((eo.borrower_id, rel, "writer failed checks"))
                continue
            words = len(" ".join(cs).split())
            fl = filler(3, words, base)
            if not fl:
                dropped.append((eo.borrower_id, rel, "filler failed checks"))
                continue
            pos = k % 4
            k += 1
            for truth, focal in ((False, fg), (True, fb)):
                for cname, others in (("alone", []), ("claims", cs), ("filler", fl)):
                    sents = list(others)
                    sents.insert(min(pos, len(sents)), focal)
                    rows.append({"item_id": f"{base}|{'err' if truth else 'ok'}|{cname}", "base_id": base,
                                 "borrower_id": eo.borrower_id, "dataset": ds, "group": (eo.group_attributes or {}).get("audit_group"),
                                 "relation": rel, "error_type": etype if truth else None, "truth": truth,
                                 "context": cname, "focal_position": min(pos, len(sents) - 1), "focal_text": focal,
                                 "text": " ".join(sents), "context_sentences": others, "evidence_ids": ids,
                                 "evidence": eo.to_dict(include_groups=True), "pack_version": VERSION,
                                 "labels": "by construction; needs independent validation"})
            made += 1
            print(base, "ok", flush=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    from collections import Counter
    man.update({"n_items": len(rows), "n_base": len({r["base_id"] for r in rows}),
                "by_relation": Counter(r["relation"] for r in rows if r["context"] == "alone" and not r["truth"]),
                "by_dataset": Counter(r["dataset"] for r in rows if r["context"] == "alone" and not r["truth"]),
                "dropped": dropped, "excluded_borrowers": len(exclude)})
    (OUT / f"context_manifest_{a.tag}.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: v for k, v in man.items() if k != "dropped"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

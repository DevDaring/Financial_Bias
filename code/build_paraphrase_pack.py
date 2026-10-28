"""
Paraphrase and adversarial-wording stress test (reviewer: robustness of the binder to paraphrase diversity).

Source: the single-sentence ("alone") items of both context packs: 160 base cases x {correct, single-error} = 320
focal sentences with labels by construction. Each is rewritten by two models used in no other role in this
experiment (Kimi K2.5, GLM-4.7; the original writer was Llama 4 Maverick, the judge is GPT-OSS-120B) in two styles:

  paraphrase   plain rewording that keeps the meaning
  adversarial  everyday names instead of field codes, passive voice, reordered clauses, hedged phrasing,
               synonyms for directions; meaning kept

A rewrite is kept only if it keeps every number of the original (as a multiset), keeps the presence or absence of
a negation, and keeps the direction word class (raises / lowers / neither); otherwise it is regenerated once and
then dropped. A random sample is checked by a person for meaning preservation (annotator pack, task 2).

    python build_paraphrase_pack.py
Output: Submission/mechanism/paraphrase_pack.jsonl (+ paraphrase_manifest.json)
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gateway as G  # noqa: E402
import json_parse as JP  # noqa: E402

SUB = Path(__file__).resolve().parents[2] / "Submission"
MECH = SUB / "mechanism"
WRITERS = ("kimi_k25", "glm47")
STYLES = {
    "paraphrase": "Rewrite the sentence in different words. Keep exactly the same meaning, every number, every "
                  "direction (raises / lowers) and any negation. Do not add or remove information.",
    "adversarial": "Rewrite the sentence the way a hurried loan officer would say it: use everyday names instead "
                   "of field codes (for example 'the September bill' instead of BILL_AMT1), passive voice or a "
                   "reordered clause, and a different verb for the direction. Keep exactly the same meaning, every "
                   "number, every direction and any negation. Do not add or remove information.",
}
SYS = "You rewrite one sentence. Reply with JSON only: {\"sentence\": \"...\"}."
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
NEG = re.compile(r"\bnot\b|n't\b|\bno\b|\bnever\b|\bneither\b|\bnor\b", re.I)
UP = re.compile(r"\b(rais|increas|push(?:es|ed|ing)? up|lift|add(?:s|ed)? to|heighten|elevat|boost|drive[sn]? up|"
                r"higher risk|more risk)", re.I)
DOWN = re.compile(r"\b(lower|reduc|decreas|cut|push(?:es|ed|ing)? down|pull(?:s|ed)? down|less risk|"
                  r"diminish|drive[sn]? down|eas(?:es|ed|ing))", re.I)


def nums(s):
    return Counter(n.replace(",", "").rstrip(".") for n in NUM.findall(s))


def direction(s):
    return ("up" if UP.search(s) else "") + ("down" if DOWN.search(s) else "")


def keeps(orig, new):
    return nums(orig) == nums(new) and bool(NEG.search(orig)) == bool(NEG.search(new)) and \
        direction(orig) == direction(new) and 0.5 < len(new) / max(1, len(orig)) < 2.5


def rewrite(item, fam, style):
    for k in range(2):
        r = G.GATEWAY.complete(fam, [{"role": "system", "content": SYS},
                                     {"role": "user", "content": STYLES[style] + "\n\nSentence: " + item["text"]}],
                               temperature=0.4 if k else 0.2, max_tokens=300, want_json=True, role="writer",
                               experiment="paraphrase-pack", item_id=f"{item['item_id']}|{fam}|{style}|{k}")
        d, _ = JP.parse_local(r.text, validate=lambda o: isinstance(o, dict) and isinstance(o.get("sentence"), str))
        if d and keeps(item["text"], d["sentence"].strip()):
            s = d["sentence"].strip()
            return s if s.endswith((".", "!", "?")) else s + "."
    return None


def main() -> int:
    src = []
    for v in ("v1", "v2"):
        src += [json.loads(l) for l in open(MECH / f"context_pack_{v}.jsonl") if json.loads(l)["context"] == "alone"]
    jobs = [(it, fam, st) for it in src for fam in WRITERS for st in STYLES]
    out = MECH / "paraphrase_pack.jsonl"
    rows, dropped = [], Counter()

    def work(j):
        it, fam, st = j
        try:
            return j, rewrite(it, fam, st)
        except Exception as e:  # noqa: BLE001
            return j, None
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(work, jobs))
    for it in src:   # the originals, re-run with the same judge as the rewrites
        rows.append(dict(it, item_id=it["item_id"] + "|orig", source_id=it["item_id"], rewriter=None, style="original"))
    for (it, fam, st), s in res:
        if s is None:
            dropped[(fam, st)] += 1
            continue
        rows.append(dict(it, item_id=f"{it['item_id']}|{fam}|{st}", source_id=it["item_id"], rewriter=fam, style=st, family=fam,
                         text=s, focal_text=s))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    man = {"n_sources": len(src), "n_rows": len(rows), "kept": Counter(f"{r['rewriter']}|{r['style']}" for r in rows),
           "dropped": {f"{a}|{b}": n for (a, b), n in dropped.items()}, "writers": WRITERS, "seed": 20261013}
    (MECH / "paraphrase_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps(man, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

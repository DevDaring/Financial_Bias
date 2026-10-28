"""
Assemble the ESWA paper: fill every {{TOKEN}} in body_part2.tex (and the abstract/highlights) from
the analysis files, then write ESWA_IDBI_Hackathon.tex. Aborts if any token is left unfilled.

    python fill_paper.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_m1 import M1, SUB, final_labels  # noqa: E402

ESWA = SUB / "ESWA_Submission"
N = json.load(open(ESWA / "paper_numbers.json"))
T = {}

PRICE = {  # USD per 1M tokens (input, output), list prices validated against bills (Research_Plan §9.9)
    "nova_lite": (0.06, 0.24), "nemotron_nano": (0.05, 0.20), "kimi_k25": (0.45, 2.25), "ministral_8b": (0.15, 0.15),
    "gpt56_luna": (0.20, 1.20), "sonnet5": (2.0, 10.0), "grok420": (1.25, 2.50), "qwen3_next": (0.10, 1.10),
    "gemma27": (0.08, 0.45), "gpt5_mini": (0.125, 1.0), "glm47": (0.60, 2.20), "gptoss120b": (0.037, 0.17),
    "grok47": (3.75, 3.75), "sarvam_105b": (0.35, 0.88), "llama4_maverick": (0.24, 0.97), "deepseek_v32": (0.62, 1.85),
    "gemini31_pro": (2.0, 12.0), "grok": (1.25, 2.50),
    # early pilot families (Sep 2026, OpenRouter/Gemini routes; approximate list prices, < 400 calls in total)
    "gemini": (0.30, 2.50), "gpt4omini": (0.15, 0.60), "deepseek": (0.27, 1.10), "llama": (0.02, 0.05),
    "gemma27_vllm": (0.0, 0.0)}   # self-hosted: costed by GPU time (gpu_cost.json)
GROK45 = {"linkapi_grok": (0.50, 1.50), "linkapi_grokbuild": (0.10, 0.30)}


def pct(x):
    return f"{100 * x:.1f}"


def main() -> int:
    fl = final_labels()
    nerr = sum(v for _k, v, _b in fl.values())
    T["NFINAL"], T["NERR"], T["NOK"] = len(fl), nerr, len(fl) - nerr
    for jk, judge in (("Q", "qwen3_next"), ("G", "grok45")):
        res = N["main"].get(judge, {})
        ana = json.load(open(M1 / f"phaseA_{judge}_v1.4_final.json"))
        for m, r in res.items():
            T[f"{jk}.{m}.EA"] = r["err_acc"]
            T[f"{jk}.{m}.CR"] = r["corr_rel"]
            T[f"{jk}.{m}.CJ"] = r["corr_rej"]
            T[f"{jk}.{m}.DF"] = r["deferred"]
            T[f"{jk}.{m}.SE"] = "--" if r["sel_err"] is None else pct(r["sel_err"])
            T[f"{jk}.{m}.CALLS"] = f"{r['calls'] / len(fl):.2f}"
            if m in ana["methods"]:
                T[f"{jk}.{m}.USD"] = f"{ana['methods'][m]['usd_list']:.2f}"
        for m, d in N["paired"].get(judge, {}).items():
            T[f"{jk}.d.{m}.EA"] = f"{100 * d['errors_accepted']['P_minus_X']:+.1f}"
            T[f"{jk}.d.{m}.CR"] = f"{100 * d['released_correct']['P_minus_X']:+.1f}"
        tot = N["objections"].get(judge + "_totals", {})
        for o in ("verified", "semantic_plausible", "disproved", "malformed", "unexecutable", "semantic_invalid"):
            T[f"{jk}.obj.{o}"] = tot.get(o, 0)
        T[f"{jk}.obj.total"] = sum(tot.values())
    rel = [abs(100 * N["paired"]["qwen3_next"][m]["errors_accepted"]["P_minus_X"])
           for m in ("zandi_judge", "mermigkis_verifier", "pbp_judge", "ws_cascade_human", "ws_cascade_grok")
           if m in N["paired"]["qwen3_next"]]
    T["RELMIN"], T["RELMAX"] = f"{min(rel):.0f}", f"{max(rel):.0f}"
    gen = [100 * N["main"][j][m]["err_acc"] / N["main"][j][m]["n_err"] for j in ("qwen3_next", "grok45")
           for m in ("generic_judge2", "generic_tool_judge", "generic_second_look") if m in N["main"][j]]
    T["GENMIN"], T["GENMAX"] = f"{min(gen):.0f}", f"{max(gen):.0f}"
    # self-reported confidence
    mv = [json.loads(l) for l in open(M1 / "comparators" / "mermigkis_verifier__qwen3_next.jsonl")]
    T["CONF_HIGH"] = sum(1 for r in mv if r["item_id"] in fl and (r.get("score") or 0) >= 0.95)
    # tool use in B2 (Qwen)
    b2 = [json.loads(l) for l in open(M1 / "comparators" / "generic_tool_judge__qwen3_next.jsonl")]
    T["Q.B2TOOLS"] = sum(1 for r in b2 if r["item_id"] in fl and (r.get("detail") or {}).get("tools"))
    # held-out v3
    v3 = N["heldout_v3"]
    any_ = next(iter(v3.values()))
    T["V3N"] = any_["n_err"] + any_["n_ok"]
    T["V3NE"], T["V3NO"] = any_["n_err"], any_["n_ok"]
    for k, r in v3.items():
        for f in ("err_acc", "err_rej", "ok_acc", "ok_rej", "prov_rej", "deferred"):
            T[f"V3.{k}.{f}"] = r[f]
    # access gaps (Home Credit)
    acc = N["access"]

    def gap(k, ds):
        wf, nf = acc[k][f"{ds}|female"]
        wm, nm = acc[k][f"{ds}|male"]
        return f"{100 * (wm / nm - wf / nf):+.1f}"
    T["ACC.A0.HC"] = gap("exec_A0|qwen3_next", "home_credit")
    T["ACC.P.HC"] = gap("exec_P|qwen3_next", "home_credit")
    T["ACC.PG.HC"] = gap("exec_P|grok45", "home_credit")
    T["A0SETTLED"] = N["main"]["qwen3_next"]["exec_A0"]["err_acc"] + N["main"]["qwen3_next"]["exec_A0"]["corr_rel"] + \
        (N["main"]["qwen3_next"]["exec_A0"]["n_err"] - N["main"]["qwen3_next"]["exec_A0"]["err_acc"]) + \
        N["main"]["qwen3_next"]["exec_A0"]["corr_rej"] - N["main"]["qwen3_next"]["exec_A0"]["deferred"] + \
        sum(0 for _ in [0])
    T["A0SETTLED"] = len(fl) - N["main"]["qwen3_next"]["exec_A0"]["deferred"]
    # total API spend from the full call trace
    usd, unknown = 0.0, Counter()
    for line in open(SUB / "llm_trace.jsonl"):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not r.get("ok"):
            continue
        fam = r.get("family")
        pr = GROK45.get(r.get("provider")) if fam == "grok45" else PRICE.get(fam)
        if pr is None:
            unknown[fam] += 1
            continue
        usd += ((r.get("prompt_tokens") or 0) * pr[0] + (r.get("completion_tokens") or 0) * pr[1]) / 1e6
    if unknown:
        print("WARNING unpriced families in trace:", dict(unknown))
    T["TOTALUSD"] = f"{usd:.0f}"
    N["fill_tokens"] = {k: str(v) for k, v in T.items()}

    v3 = (ESWA / "body_part1_v3.tex").exists() and "--v2" not in sys.argv and "--v1" not in sys.argv
    v2 = (ESWA / "body_part1_v2.tex").exists() and "--v1" not in sys.argv
    if v2 or v3:
        T.update(json.load(open(ESWA / "paper_tokens_v2.json")))
    sfx = "_v3" if v3 else ("_v2" if v2 else "")
    body1 = (ESWA / f"body_part1{sfx}.tex").read_text()
    body2 = (ESWA / f"body_part2{sfx}.tex").read_text()
    fp = ESWA / f"frontmatter{sfx}.tex"
    front = fp.read_text() if fp.exists() else ""
    text = front + "\n%BODYSPLIT%\n" + body1 + "\n" + body2
    sp = ESWA / f"supplement{sfx}.tex"     # supplementary tables, filled from the same tokens
    if sp.exists():
        text += "\n%SUPPSPLIT%\n" + sp.read_text()
    text = text.replace("%NFINAL%", "{{NFINAL}}").replace("%NERR%", "{{NERR}}")

    def sub(m):
        k = m.group(1)
        if k not in T:
            raise KeyError(k)
        return str(T[k])
    missing = sorted({k for k in re.findall(r"\{\{([^}]+)\}\}", text) if k not in T})
    if missing:
        print("UNFILLED TOKENS:", missing)
        if "--draft" not in sys.argv:
            return 1
        for k in missing:
            T[k] = "??"
    text = re.sub(r"\{\{([^}]+)\}\}", sub, text)
    text, _, supp = text.partition("\n%SUPPSPLIT%\n")
    if supp:
        # the supplement is a list of tables under headings: pin each table where it is input
        for name in set(re.findall(r"\\tabinput\{tables/([^}]+)\}", supp)):
            src = ESWA / "tables" / name
            if src.exists():
                (ESWA / "tables" / f"supp_{name}").write_text(src.read_text().replace("\\begin{table}[!htb]", "\\begin{table}[H]"))
                supp = supp.replace(f"\\tabinput{{tables/{name}}}", f"\\tabinput{{tables/supp_{name}}}")
        supp = supp.replace("\\begin{table}[!htb]", "\\begin{table}[H]")
        (ESWA / "ESWA_Supplementary.tex").write_text(supp)
    front_f, body = text.split("\n%BODYSPLIT%\n", 1)
    shell = (ESWA / "ESWA_IDBI_Hackathon.tex").read_text()
    if "%BODY%" not in shell:
        shell = (ESWA / "paper_shell.tex").read_text()
    else:
        (ESWA / "paper_shell.tex").write_text(shell)
    out = shell.replace("%BODY%", body)
    if front_f.strip():
        out = out.replace("%ABSTRACT%\n\n%HIGHLIGHTS%", front_f.strip())
    (ESWA / "ESWA_IDBI_Hackathon.tex").write_text(out)
    (ESWA / "paper_numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print("filled", len(T), "tokens; wrote ESWA_IDBI_Hackathon.tex; total API USD", T["TOTALUSD"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Assets for manuscript v3 (Improvement_Plan WS6). Runs the v2 asset builder, then adds:
  tab_dev699.tex        ReCheck selection on 699 development narratives (Gate A), per judge
  tab_isolation.tex     same-transcript policies (identical first-stage judge responses) and the reader ledger
  tab_matched.tex       error among released narratives at matched release volume (locked v1)
  fig_context.pdf       context experiment with 95% Wilson intervals (overwrites the v2 figure)
and tokens DV.*, ISO.*, MC.*, GROKDEV.*, FLOW.*.
    python make_paper_assets_v3.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_assets_v2 as A  # noqa: E402
from score_m1 import SUB  # noqa: E402

T = A.T
LK = SUB / "locked"
JN = {"gptoss120b": ("O", "GPT-OSS-120B"), "qwen3_next": ("Q", "Qwen3-Next"), "gemma27_vllm": ("M", "Gemma 3 27B"),
      "grok45": ("G", "Grok 4.5")}
DVL = {"B1": "B1 generic judge", "B1R": "B1R aligned rubric", "B1F": "B1F precomputed facts",
       "P_1.5": "ReCheck-P, checker 1.5", "D_1.5": "ReCheck-D, checker 1.5", "R11_1.5": "ReCheck-R11, checker 1.5",
       "A0_1.6": "Code only, checker 1.6", "D_1.6": "ReCheck-D, checker 1.6", "P_1.6": "ReCheck-P, checker 1.6",
       "R11_1.6": "ReCheck-R11, checker 1.6"}


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def signed(x):
    return f"${100 * x:+.1f}$"


def ci(c):
    return f"[${100 * c[0]:+.1f}$, ${100 * c[1]:+.1f}$]"


def dev699():
    D = A.jl(LK / "dev_v16_analysis.json")
    if not D:
        return
    T["DV.N"], T["DV.NERR"], T["DV.NOK"] = D["n"], D["n_wrong"], D["n"] - D["n_wrong"]
    rows = []
    for j in ("gptoss120b", "qwen3_next", "gemma27_vllm"):
        o = D["judges"].get(j)
        if not o:
            continue
        k, nm = JN[j]
        first = True
        for m in ("B1", "B1R", "B1F", "P_1.5", "D_1.5", "A0_1.6", "D_1.6", "P_1.6", "R11_1.6"):
            v = o["methods"].get(m)
            if not v:
                continue
            key = m.replace(".", "")
            T[f"DV.{k}.{key}.cr"], T[f"DV.{k}.{key}.wr"] = v["correct_release"], v["wrong_release"]
            T[f"DV.{k}.{key}.se"] = pct(v["selective_error"])
            rows.append(f"{nm if first else ''} & {DVL[m]} & {v['correct_release']} & {v['wrong_release']} & "
                        f"{pct(v['selective_error'])} & ${v['U2']}$ \\\\")
            first = False
        for name, pv in o["paired"].items():
            key = name.split("_vs_")[0].replace(".", "")
            for kind in ("wrong_release", "correct_release"):
                T[f"DV.{k}.{key}.d.{kind}"] = signed(pv[kind]["diff"])
                T[f"DV.{k}.{key}.d.{kind}.CI"] = ci(pv[kind]["ci95"])
        rows.append("\\midrule")
    rows = rows[:-1]
    A.tex_table("tab_dev699.tex", 6, "Judge & Verifier & Correct & Wrong & Err.\\,\\% & $U_2$ \\\\", rows,
                f"Development selection on {D['n']} narratives ({D['n_wrong']} wrong): explanations released, error "
                "among released explanations and utility $U_2$. Checker 1.6 adds the zero-call gate.", "tab:dev699",
                size="\\scriptsize", spec="@{}llrrrr@{}")
    g = A.jl(LK / "grok_dev_v16.json")
    if g:
        T["GROKDEV.P.cr"], T["GROKDEV.P.wr"] = g["P_1.6_grok"]["correct"], g["P_1.6_grok"]["wrong"]


def isolation():
    I = A.jl(LK / "isolation_analysis.json")
    if not I:
        return
    rows = []
    lab = {"none": "any objection blocks", "execute": "objections executed", "P": "+ one reassessment (P)",
           "D": "unconfirmed objection defers (D)", "R": "+ independent reader (R)", "R11": "+ reader, framing allowed (R11)"}
    for j in ("qwen3_next", "gemma27_vllm"):
        rj = I["replay"].get(j)
        if not rj or "cells" not in rj:
            continue
        k, nm = JN[j]
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{nm}}}}} \\\\")
        for p in ("none", "execute", "D", "P", "R", "R11"):
            c = rj["cells"][p]
            d = rj["vs_D"].get(p)
            dw = "--" if d is None else f"{signed(d['wrong_release']['diff'])}"
            dc = "--" if d is None else f"{signed(d['correct_release']['diff'])}"
            rows.append(f"\\quad {lab[p]} & {c.get('correct_release', 0)} & {c.get('wrong_release', 0)} & {dc} & {dw} \\\\")
            T[f"ISO.{k}.{p}.cr"], T[f"ISO.{k}.{p}.wr"] = c.get("correct_release", 0), c.get("wrong_release", 0)
            if d:
                for kind in ("wrong_release", "correct_release"):
                    T[f"ISO.{k}.{p}.d.{kind}"] = signed(d[kind]["diff"])
                    T[f"ISO.{k}.{p}.d.{kind}.CI"] = ci(d[kind]["ci95"])
        for led in ("ledger_R", "ledger_R11"):
            for f_, v_ in rj.get(led, {}).items():
                T[f"ISO.{k}.{led}.{f_}"] = v_
    hdr = ("& \\multicolumn{2}{c}{Released} & \\multicolumn{2}{c}{vs D, points} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
           "Same first-stage response, processed by & Correct & Wrong & Correct & Wrong \\\\")
    A.tex_table("tab_isolation.tex", 5, hdr, rows,
                f"Locked-v1 narratives ({I['n']}, {I['n_wrong']} wrong): the same saved first-stage judge responses "
                "processed by each objection and dispute policy (checker 1.5).", "tab:isolation", size="\\scriptsize",
                spec="@{}lrrrr@{}")
    rows = []
    order = [("facts_judge|grok45", "Grok 4.5 + facts (B1F)"), ("facts_judge|gptoss120b", "GPT-OSS-120B + facts (B1F)"),
             ("generic_rubric|gptoss120b", "GPT-OSS-120B aligned rubric (B1R)"),
             ("both(facts_judge)|qwen+gptoss", "Two judges must both accept (B1F)"),
             ("exec_D|qwen3_next", "ReCheck-D, Qwen"), ("exec_P|qwen3_next", "ReCheck-P, Qwen"),
             ("facts_judge|qwen3_next", "Qwen + facts (B1F)"), ("parse_exec|gptoss120b", "Parser + executor, no judge")]
    for key, nm in order:
        m = I["matched"].get(key)
        v = I["methods"].get(key)
        if not m or not v:
            continue
        rows.append(f"{nm} & {v['correct_release']} & {v['wrong_release']} & {pct(v['selective_error'])} & "
                    f"{m['k']} & {m['wrong']} & {pct(m['error_rate'])} \\\\")
        kk = key.replace("|", ".").replace("(", "").replace(")", "").replace("+", "")
        T[f"MC.{kk}.cr"], T[f"MC.{kk}.wr"], T[f"MC.{kk}.mwr"] = v["correct_release"], v["wrong_release"], m["wrong"]
        T[f"MC.{kk}.se"], T[f"MC.{kk}.mse"] = pct(v["selective_error"]), pct(m["error_rate"])
    T["MC.k"] = I["methods"].get("exec_D|qwen3_next", {}).get("released")
    hdr = ("& \\multicolumn{3}{c}{As released} & \\multicolumn{3}{c}{Matched volume} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
           "Verifier & Correct & Wrong & Err.\\,\\% & $k$ & Wrong & Err.\\,\\% \\\\")
    A.tex_table("tab_matched.tex", 7, hdr, rows,
                "Locked-v1 narratives: error among released explanations as released and at a matched release volume "
                "$k$ (each method's releases ranked by one shared score, the share of claims the code layer supports).",
                "tab:matched", size="\\scriptsize", spec="@{}lrrrrrr@{}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def context_ci_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pack = [json.loads(l) for l in open(SUB / "mechanism" / "context_pack_v1.jsonl")]
    truth = {r["item_id"]: (r["truth"], r["context"]) for r in pack}
    runs = SUB / "mechanism" / "context_v1_runs"
    judges = [("qwen3_next", "Qwen3-Next"), ("gemma27_vllm", "Gemma 3 27B"), ("grok45", "Grok 4.5")]
    sty = {"generic_rubric": ("#c0392b", "o", "B1R aligned rubric"), "exec_R": ("#1f618d", "s", "ReCheck-R"),
           "exec_R11": ("#117a65", "^", "ReCheck-R11")}
    f, ax = plt.subplots(2, 3, figsize=(9.6, 5.2), sharey=True)
    for c, (j, nm) in enumerate(judges):
        for r, wrong in enumerate((True, False)):
            a = ax[r][c]
            for n_, (m, (col, mk, lab)) in enumerate(sty.items()):
                p = runs / f"{m}__{j}.jsonl"
                if not p.exists():
                    continue
                rows = [json.loads(l) for l in open(p)]
                ys, lo, hi = [], [], []
                for ctx in ("alone", "claims", "filler"):
                    rr = [x for x in rows if truth[x["item_id"]] == (wrong, ctx)]
                    k = sum(x.get("accepted") is True for x in rr)
                    w = wilson(k, len(rr))
                    ys.append(100 * k / max(1, len(rr)))
                    lo.append(ys[-1] - 100 * w[0])
                    hi.append(100 * w[1] - ys[-1])
                xs = [x + (n_ - 1) * 0.08 for x in range(3)]
                a.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=col, label=lab, capsize=2, lw=1.2, ms=4)
            a.set_xticks(range(3))
            a.set_xticklabels(["alone", "+claims", "+filler"])
            a.grid(alpha=0.3)
            if r == 0:
                a.set_title(nm, fontsize=10)
            if c == 0:
                a.set_ylabel("Wrong passages released (%)" if wrong else "Correct passages released (%)", fontsize=9)
    ax[0][0].legend(fontsize=7, loc="upper left")
    f.tight_layout()
    f.savefig(A.FIG / "fig_context.pdf")


def flow():
    T.update({"FLOW.gen": 324, "FLOW.ok": 323, "FLOW.withheld": 24, "FLOW.nat": 299, "FLOW.agreed": 238,
              "FLOW.adj": 54, "FLOW.conflict": 7})
    m = A.jl(LK / "fresh_natural_v2_manifest.json")
    if m:
        T["FV.gen.nar"], T["FV.gen.fail"], T["FV.gen.retry"] = m["narratives"], m["producer_failures_excluded"], m["retries"]


def verdicts():
    """'met' / 'not met' words for the prespecified tests, taken from the analysis files (never typed by hand)."""
    fresh_src = "fresh_v2_analysis_independent_adjudication.json" if (LK / "fresh_v2_analysis_independent_adjudication.json").exists() else "fresh_v2_analysis.json"
    for src, P in (("locked_analysis.json", "LK"), (fresh_src, "FV")):
        L = A.jl(LK / src)
        if not L or "primary" not in L:
            continue
        for j, d in L["primary"].items():
            k = JN[j][0]
            for kind in ("wrong_release", "correct_release"):
                T[f"{P}.{k}.P.{kind}.abs"] = f"{abs(100 * d[kind]['R_minus_baseline']):.1f}"
            T[f"{P}.{k}.P.SUP"] = "met" if d.get("superiority_wrong_release") else "not met"
            T[f"{P}.{k}.P.NI"] = "met" if d.get("noninferior_correct_release_5pts") else "not met"
            T[f"{P}.{k}.P.BOTH"] = "met" if (d.get("superiority_wrong_release") and
                                              d.get("noninferior_correct_release_5pts")) else "not met"


CELLS = [("nova_lite/grounded", "Nova Lite"), ("ministral_8b/grounded", "Ministral 8B"),
         ("glm47/grounded", "GLM-4.7 (unseen producer)"), ("ministral_8b/applicant", "Ministral 8B, applicant style")]


def fresh_cells():
    L = A.jl(LK / "fresh_v2_analysis.json")
    if not L or "by_cell" not in L:
        return
    rows = []
    for ck, nm in CELLS:
        cells = []
        for m in ("facts_judge|gptoss120b", "exec_P|gptoss120b"):
            v = L["by_cell"].get(m, {}).get(ck)
            if not v:
                cells += ["--", "--"]
                continue
            cells += [f"{v['wrong_released']}/{v['wrong']}", f"{v['correct_released']}/{v['correct']}"]
            key = ck.replace("/", "_")
            T[f"FV.CELL.{m.split('|')[0]}.{key}.wr"], T[f"FV.CELL.{m.split('|')[0]}.{key}.nw"] = v["wrong_released"], v["wrong"]
            T[f"FV.CELL.{m.split('|')[0]}.{key}.cr"], T[f"FV.CELL.{m.split('|')[0]}.{key}.nc"] = v["correct_released"], v["correct"]
        rows.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{2}{c}{B1F (GPT-OSS-120B)} & \\multicolumn{2}{c}{ReCheck-P (GPT-OSS-120B)} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
           "Producer cell & Wrong rel. & Correct rel. & Wrong rel. & Correct rel. \\\\")
    A.tex_table("tab_fresh_cells.tex", 5, hdr, rows, "Fresh test by producer cell: wrong and correct narratives released "
                "(of the cell's wrong and correct narratives).", "tab:freshcells", size="\\scriptsize")


def placeholders():
    """Draft builds only: visible placeholder tables until the fresh-test labels return."""
    for f, lab, cap in (("tab_fresh_main.tex", "tab:freshmain", "Fresh test: outcomes (awaiting labels)."),
                        ("tab_fresh_cells.tex", "tab:freshcells", "Fresh test by producer cell (awaiting labels).")):
        p = A.TAB / f
        L = A.jl(LK / "fresh_v2_analysis.json") or {}
        if "labels" not in L:
            p.write_text("\\begin{table}[!htb]\n\\centering\n\\caption{" + cap + "}\n\\label{" + lab +
                         "}\n\\fbox{\\parbox{0.8\\linewidth}{\\centering PLACEHOLDER: generated from "
                         "fresh\\_v2\\_analysis.json when the labels return.}}\n\\end{table}\n")


def fresh_sensitivity():
    for mode in ("agreed_only", "any_yes"):
        L = A.jl(LK / f"fresh_v2_analysis_{mode}.json")
        if not L or "primary" not in L:
            continue
        d = L["primary"].get("gptoss120b")
        if not d:
            continue
        for kind in ("wrong_release", "correct_release"):
            T[f"FVS.{mode}.{kind}"] = signed(d[kind]["R_minus_baseline"])
            T[f"FVS.{mode}.{kind}.CI"] = ci(d[kind]["ci95"])
        T[f"FVS.{mode}.NIOS"] = signed(d["correct_release"]["one_sided_95"][0])
        T[f"FVS.{mode}.n"] = L["labels"]["narratives_labelled"]


def review_checks():
    R = A.jl(LK / "review_checks.json")
    if not R:
        return
    for k, v in R["independent"].items():
        T[f"RC.ind.{k}.agree"], T[f"RC.ind.{k}.n"] = v.get("agree", 0), v.get("agree", 0) + v.get("differ", 0)
    li = R["locked_with_independent_labels"]
    T["RC.ind.changed"] = li["n_changed"]
    for j, k in (("qwen3_next", "Q"), ("gemma27_vllm", "M")):
        for kind in ("wrong_release", "correct_release"):
            T[f"RC.ind.{k}.{kind}"] = signed(li[j][kind]["diff"])
            T[f"RC.ind.{k}.{kind}.CI"] = ci(li[j][kind]["ci95"])
    a = R["context_check"]["ALL"]
    T["RC.ctx.match"], T["RC.ctx.n"] = a.get("match", 0), sum(a.values())
    pc = R["parse_check"]
    for f in ("BINDING_OK", "OPERATION_OK", "ASSERTED_OK", "POLARITY_OK", "ALL_FOUR_OK"):
        v = pc.get(f, {})
        n = v.get("ok", 0) + v.get("wrong", 0)
        T[f"RC.parse.{f}"] = pct(v.get("ok", 0) / max(1, n), 0)
        T["RC.parse.n"] = n
    for op in ("NONE", "SCALE", "TARGET"):
        v = pc.get(f"OPERATION_OK|{op}", {})
        T[f"RC.parse.op.{op}"] = f"{v.get('ok', 0)}/{v.get('ok', 0) + v.get('wrong', 0)}"


def practitioner():
    P = A.jl(LK / "practitioner_analysis.json")
    if not P:
        return
    T["PR.n"] = P["n_case_answers"]
    for c, v in P["conditions"].items():
        for k, x in v.items():
            if x:
                T[f"PR.{c}.{k}"] = pct(x["mean"], 0) if k not in ("useful", "minutes") else f"{x['mean']:.1f}"
                T[f"PR.{c}.{k}.n"] = x["n"]


def wording():
    runs = SUB / "mechanism" / "wording_v4_items_runs"
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(SUB / "mechanism" / "wording_v4_items.jsonl")}
    if not runs.exists():
        return
    rows = []
    T["WV.nerr"] = sum(1 for x in items.values() if x["truth"])
    T["WV.nok"] = sum(1 for x in items.values() if not x["truth"])
    lab = {"exec_A0": "Code only, checker 1.6", "exec_P": "ReCheck-P", "facts_judge": "B1F precomputed facts",
           "generic_rubric": "B1R aligned rubric"}
    for p in sorted(runs.glob("*.jsonl")):
        m, j = p.stem.split("__")
        d = {json.loads(l)["item_id"]: json.loads(l).get("accepted") for l in open(p)}
        if len(d) < len(items):
            continue
        we = sum(1 for i, x in items.items() if x["truth"] and d.get(i) is True)
        cr = sum(1 for i, x in items.items() if not x["truth"] and d.get(i) is True)
        k = JN[j][0]
        T[f"WV.{k}.{m}.wr"], T[f"WV.{k}.{m}.cr"] = we, cr
        # wrong sentences released because code alone found every claim supported (no judge call)
        T[f"WV.{k}.{m}.codepass"] = sum(1 for l in open(p) if (r := json.loads(l))["truth"] and r.get("accepted") is True
                                        and (r.get("detail") or {}).get("reason") == "all_claims_executed_supported")
        rows.append((k, f"{JN[j][1]} & {lab.get(m, m)} & {we}/{T['WV.nerr']} & {cr}/{T['WV.nok']} \\\\"))
        rel = {}
        for i, x in items.items():
            r_ = rel.setdefault(x["relation"], [0, 0, 0, 0])
            if x["truth"]:
                r_[0] += d.get(i) is True
                r_[1] += 1
            else:
                r_[2] += d.get(i) is True
                r_[3] += 1
        for r_, v in rel.items():
            T[f"WV.{k}.{m}.{r_}.wr"], T[f"WV.{k}.{m}.{r_}.nw"] = v[0], v[1]
    rows = [r for _k, r in sorted(rows, key=lambda x: "OQMG".index(x[0]))]
    A.tex_table("tab_wording.tex", 4, "Judge & Verifier & Wrong released & Correct released \\\\", rows,
                "Sentences written by an independent writer in six kinds of wording (related fields, negation, units, "
                "months covered, undocumented meanings, framing), labelled by the writer's intent.", "tab:wording",
                size="\\scriptsize", spec="@{}llrr@{}")


def utility():
    """Net benefit U_lambda = correct released - lambda x wrong released (Section 3), from analyze_utility.py."""
    U = A.jl(LK / "fresh_v2_utility.json")
    if not U:
        return
    short = {"gptoss120b": "O", "qwen3_next": "Q", "gemma27_vllm": "M", "grok45": "G"}
    alt = {"facts_judge": "B1F", "generic_rubric": "B1R", "generic_judge2": "B1", "exec_D": "D", "exec_A0": "A0",
           "exec_R11": "R11"}
    for key, v in U["comparisons"].items():
        a, b = key.split(" vs ")
        j = short[a.split("|")[1]]
        m2, j2 = b.split("|")
        tag = f"UT.{j}.{alt.get(m2, m2)}" + ("" if short[j2] == j else f".{short[j2]}")
        for lam in U["lambdas"]:
            x = v[f"lambda_{lam}"]
            T[f"{tag}.l{lam}"] = f"{x['diff_per_100']:+.1f}".replace("-", "$-$")
            T[f"{tag}.l{lam}.CI"] = f"[{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]".replace("-", "$-$")
        if v["dC"] * v["dW"] > 0:   # the two verifiers swap order at lambda = dC / dW
            T[f"{tag}.cross"] = f"{v['dC'] / v['dW']:.1f}"
        if "break_even_lambda" in v:
            T[f"{tag}.lstar"] = f"{v['break_even_lambda']['point']:.2f}"
            T[f"{tag}.lstar.up"] = f"{v['break_even_lambda']['upper95']:.1f}"
    for k, usd in U.get("usd_per_1000", {}).items():
        m, j = k.split("|")
        T[f"UT.USD.{short[j]}.{alt.get(m, 'P' if m == 'exec_P' else m)}"] = f"{usd:.2f}"
    for k, sec in U.get("seconds_per_item", {}).items():
        m, j = k.split("|")
        T[f"UT.SEC.{short[j]}.{alt.get(m, 'P' if m == 'exec_P' else m)}"] = f"{sec:.0f}"
    T["UT.COSTX"] = f"{U['usd_per_1000']['facts_judge|grok45'] / U['usd_per_1000']['exec_P|gptoss120b']:.0f}"
    T["UT.SECX"] = f"{U['seconds_per_item']['facts_judge|grok45'] / U['seconds_per_item']['exec_P|gptoss120b']:.0f}"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("gptoss120b", "GPT-OSS-120B judge"), ("qwen3_next", "Qwen3-Next judge")]
    sty = [("facts_judge", "#c0392b", "vs B1F precomputed facts"), ("generic_rubric", "#7d3c98", "vs B1R aligned rubric"),
           ("generic_judge2", "#b9770e", "vs B1 generic judge"), ("facts_judge|grok45", "#555555", "vs Grok 4.5 + B1F")]
    f, ax = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
    for a, (j, title) in zip(ax, panels):
        for m2, col, lab in sty:
            other = m2 if "|" in m2 else f"{m2}|{j}"
            v = U["comparisons"].get(f"exec_P|{j} vs {other}")
            if not v:
                continue
            g = v["grid"]
            xs = [r[0] for r in g]
            a.plot(xs, [r[1] for r in g], color=col, lw=1.4, label=lab, ls="--" if "grok" in m2 else "-")
            a.fill_between(xs, [r[2] for r in g], [r[3] for r in g], color=col, alpha=0.12, lw=0)
        a.axhline(0, color="black", lw=0.8)
        a.set_title(title, fontsize=10)
        a.set_xlabel("$\\lambda$: cost of a wrong release relative to a withheld correct narrative", fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("ReCheck-P minus alternative\n(net benefit per 100 narratives)", fontsize=8)
    ax[0].legend(fontsize=7, loc="upper left")
    f.tight_layout()
    f.savefig(A.FIG / "fig_utility.pdf")


def _m(x):
    return f"{x:+.1f}".replace("-", "$-$")


def _ci(c):
    return f"[{c[0]:.1f}, {c[1]:.1f}]".replace("-", "$-$")


def review2():
    """Ablations, residual-risk attribution, coverage, fairness and label rules (analyze_review2.py), and the
    paraphrase stress test (analyze_paraphrase.py)."""
    R = A.jl(LK / "review2_analysis.json")
    if R:
        ab = R["ablation"]
        rows = []
        lab = [("frozen", "Frozen \\ewv{}-P"), ("gate_off_FIELD_IDENTITY", "without the field-identity rule"),
               ("gate_off_NEGATION_SCOPE", "without the negation rule"),
               ("gate_off_CODE_SCOPE", "without the code-meaning rule"), ("gate_off_TARGET", "without the horizon rule"),
               ("gate_off_ALL", "without the zero-call gate"), ("reassess2", "a second reassessment"),
               ("A1_objections_unchecked", "objections not executed (any objection rejects)"),
               ("strict_semantic", "withhold judge-supported causal or invented claims"),
               ("repeat", "the frozen system run again"), ("pt_audit", "Pratama--Tseng audit (no model)")]
        base = None
        for k, nm in lab:
            if k == "frozen":
                v = next(iter(x for x in ab.values() if isinstance(x, dict) and "vs_B1F" in x))
                fz = v["vs_frozen_P"]
                c = fz["counts"]
                # frozen counts: the variant's counts minus its difference are not needed; take from strict's base
                continue
            if k not in ab:
                continue
            a, b = ab[k]["vs_frozen_P"], ab[k]["vs_B1F"]
            c = a["counts"]
            tag = f"RV.ABL.{k}"
            T[f"{tag}.wr"], T[f"{tag}.cr"] = c["wrong_released"], c["correct_released"]
            T[f"{tag}.dwr"], T[f"{tag}.dwr.CI"] = _m(a["wrong_release"]["diff"]), _ci(a["wrong_release"]["ci95"])
            T[f"{tag}.dcr"], T[f"{tag}.dcr.CI"] = _m(a["correct_release"]["diff"]), _ci(a["correct_release"]["ci95"])
            T[f"{tag}.U2"], T[f"{tag}.U2.CI"] = _m(b["U2"]["diff"]), _ci(b["U2"]["ci95"])
            T[f"{tag}.wrB1F"], T[f"{tag}.wrB1F.CI"] = _m(b["wrong_release"]["diff"]), _ci(b["wrong_release"]["ci95"])
            T[f"{tag}.crB1F"] = _m(b["correct_release"]["diff"])
            rows.append(f"{nm} & {c['wrong_released']} & {c['correct_released']} & {_m(a['wrong_release']['diff'])} "
                        f"{_ci(a['wrong_release']['ci95'])} & {_m(a['correct_release']['diff'])} "
                        f"{_ci(a['correct_release']['ci95'])} & {_m(b['U2']['diff'])} \\\\")
        n = next(iter(ab[k] for k in ab if isinstance(ab[k], dict) and "vs_B1F" in ab[k]))["vs_B1F"]["counts"]
        P0 = R["labels"]["provided"]["counts"]
        T["RV.P.wr"], T["RV.P.cr"] = P0["wrong_released"], P0["correct_released"]
        rows.insert(0, f"Frozen \\ewv{{}}-P & {P0['wrong_released']} & {P0['correct_released']} & -- & -- & "
                       f"{_m(R['labels']['provided']['U2']['diff'])} \\\\ \\midrule")
        A.tex_table("tab_ablation.tex", 6,
                    "Variant & Wrong & Correct & $\\Delta$ wrong (pts) & $\\Delta$ correct (pts) & $U_2$ vs B1F \\\\",
                    rows, "Fresh-test ablations (GPT-OSS-120B); rate differences from frozen \\ewv{}-P with 95\\% intervals.",
                    "tab:ablation", size="\\scriptsize",
                    spec="@{}>{\\raggedright\\arraybackslash}Xrr>{\\raggedright\\arraybackslash}p{0.85in}>{\\raggedright\\arraybackslash}p{0.85in}r@{}")
        for k, v in ab.get("changed_items", {}).items():
            T[f"RV.ABL.{k}.changed"] = v
        at = R["attribution"]["by_path"]
        code = sum(v for k, v in at.items() if k.startswith("code alone"))
        sem = sum(v for k, v in at.items() if "semantic" in k)
        gate = sum(v for k, v in at.items() if "gate-routed" in k)
        rest = sum(at.values()) - code - sem - gate
        T.update({"RV.ATTR.total": sum(at.values()), "RV.ATTR.code": code, "RV.ATTR.sem": sem, "RV.ATTR.gate": gate,
                  "RV.ATTR.rest": rest})
        cov = R["coverage"]
        T["RV.COV.claims"], T["RV.COV.settled"] = cov["n_claims"], cov["settled_by_code_pct"]
        for k, v in cov["residual_reasons"].items():
            T[f"RV.COV.{k}"] = v
        g = R["fairness"]["gaps_ReCheck-P"]
        for k, v in g.items():
            ds, nm, _ = k.split("|")
            T[f"RV.GAP.{ds}.{nm}"], T[f"RV.GAP.{ds}.{nm}.CI"] = _m(v["diff"]), _ci(v["ci95"])
        T["RV.FAIR.maxabs"] = f"{max(abs(v['diff']) for v in g.values()):.1f}"
        T["RV.FAIR.allci0"] = "all" if all(v["ci95"][0] <= 0 <= v["ci95"][1] for v in g.values()) else "not all"
        L = R["labels"]
        loo = L["leave_one_pair_out"].values()
        T["RV.LOO.wr.min"] = _m(min(v["wrong_release"]["diff"] for v in loo))
        T["RV.LOO.wr.max"] = _m(max(v["wrong_release"]["diff"] for v in loo))
        T["RV.LOO.cr.min"] = _m(min(v["correct_release"]["diff"] for v in loo))
        T["RV.LOO.cr.max"] = _m(max(v["correct_release"]["diff"] for v in loo))
        T["RV.LOO.npairs"] = len(L["leave_one_pair_out"])
        if "first_author" in L:
            T["RV.FA.wr"], T["RV.FA.cr"] = _m(L["first_author"]["wrong_release"]["diff"]), _m(L["first_author"]["correct_release"]["diff"])
        T["RV.ALLYES.wr"], T["RV.ALLYES.wr.CI"] = _m(L["all_yes"]["wrong_release"]["diff"]), _ci(L["all_yes"]["wrong_release"]["ci95"])
        T["RV.ALLYES.cr"], T["RV.ALLYES.cr.CI"] = _m(L["all_yes"]["correct_release"]["diff"]), _ci(L["all_yes"]["correct_release"]["ci95"])
        # supplementary tables
        fr = R["fairness"]
        frows = []
        for meth in ("ReCheck-P", "B1F"):
            for ds, nm in (("taiwan", "Taiwan"), ("home_credit", "Home Credit")):
                for grp, x in fr[meth][ds].items():
                    frows.append(f"{meth.replace('ReCheck', chr(92) + 'ewv{}')} & {nm} & {grp} & {x['n']} & {x['wrong_released']} & "
                                 f"{x['correct_withheld']} & {x['deferred']} \\\\")
        A.tex_table("tab_s_fairness.tex", 7, "Verifier & Dataset & Group & $n$ & Wrong released (\\%) & "
                    "Correct withheld (\\%) & Deferred (\\%) \\\\", frows,
                    "Fresh test by dataset and recorded sex (GPT-OSS-120B judge).", "tab:sfair", size="\\scriptsize")
        arows = [f"{k.replace(' | -', '')} & {v} \\\\" for k, v in at.items()]
        A.tex_table("tab_s_attribution.tex", 2, "Path to release | claim the judge assessed & Wrong narratives \\\\",
                    arows, "Fresh test: the wrong narratives released by \\ewv{}-P (GPT-OSS-120B), by the path that "
                    "released them.", "tab:sattr", size="\\scriptsize", spec="@{}lr@{}")
        lrows = [f"{nm} & {v['n']} & {_m(v['wrong_release']['diff'])} {_ci(v['wrong_release']['ci95'])} & "
                 f"{_m(v['correct_release']['diff'])} {_ci(v['correct_release']['ci95'])} \\\\"
                 for nm, v in [("independent adjudication (primary)", L["provided"])]
                 + ([("first-author adjudication", L["first_author"])] if "first_author" in L else [])
                 + [("error only if both raters flag one", L["all_yes"])]
                 + sorted((f"without rater {k}", v) for k, v in L["leave_one_pair_out"].items())]
        A.tex_table("tab_s_labels.tex", 4, "Label rule & $n$ & $\\Delta$ wrong release & $\\Delta$ correct release \\\\",
                    lrows, "Fresh test, \\ewv{}-P minus B1F (GPT-OSS-120B) under further label rules; rater pairs are "
                    "anonymous indices.", "tab:slabels", size="\\scriptsize", spec="@{}lrll@{}")
        crow = [f"{k.replace('_', chr(92) + '_')} & {v} \\\\" for k, v in cov["residual_reasons"].items()]
        A.tex_table("tab_s_coverage.tex", 2, "Reason a claim was sent to the judge & Claims \\\\", crow,
                    f"Fresh test: of {cov['n_claims']} claims, {cov['settled_by_code_pct']}\\% were settled by code "
                    "(supported or contradicted); the others went to the judge for these reasons (a claim can have "
                    "several).", "tab:scov", size="\\scriptsize", spec="@{}lr@{}")
    Pp = A.jl(SUB / "mechanism" / "paraphrase_analysis.json")
    if Pp and Pp.get("methods"):
        mn = {"exec_P": "\\ewv{}-P", "facts_judge": "B1F", "generic_rubric": "B1R"}
        prow = []
        for m, v in Pp["methods"].items():
            for st in ("original", "paraphrase", "adversarial"):
                if st not in v:
                    continue
                x = v[st]
                for k in ("escape", "retention", "flip"):
                    if k in x:
                        T[f"PP.{m}.{st}.{k}"] = f"{x[k]:.1f}"
                prow.append(f"{mn[m]} & {st} & {x['escape']:.1f} {_ci(x['escape_ci'])} & {x['retention']:.1f} "
                            f"{_ci(x['retention_ci'])} & {x.get('flip', '--')} \\\\")
        T["PP.excluded"] = Pp["excluded_by_human_check"]
        A.tex_table("tab_paraphrase.tex", 5, "Verifier & Wording & Wrong released (\\%) & Correct released (\\%) & "
                    "Decision flips (\\%) \\\\", prow,
                    "Paraphrase stress test (GPT-OSS-120B judge): single sentences with labels by construction, as "
                    "written and rewritten by two unseen models, plainly or adversarially; flips are decisions that "
                    "differ from those on the original sentence.", "tab:paraphrase", size="\\scriptsize",
                    spec="@{}llllr@{}")
    man = A.jl(SUB / "mechanism" / "paraphrase_manifest.json")
    if man:
        kept = man["kept"]
        T["PP.n.src"] = man["n_sources"]
        T["PP.n.para"] = sum(v for k, v in kept.items() if k.endswith("paraphrase"))
        T["PP.n.adv"] = sum(v for k, v in kept.items() if k.endswith("adversarial"))
        T["PP.n.drop"] = sum(man["dropped"].values())


def review3():
    """Second review: per-operation statistics, token accounting, escalation router (analyze_review3.py) and the
    alias perturbation of the code layer (analyze_alias.py)."""
    R = A.jl(LK / "review3_analysis.json")
    if R:
        rows = []
        for op, v in R["operations"].items():
            prec = "--" if v["precision"] is None else f"{v['precision']:.0f}"
            rows.append(f"{op.replace('_', chr(92) + '_')} & {v['executed']} & {v['support']} & {v['contradiction']} & "
                        f"{v['narratives_contradicted']} & {prec} & {v['share_of_wrong_narratives']:.1f} \\\\")
            T[f"OP.{op}.n"], T[f"OP.{op}.prec"] = v["executed"], prec
            T[f"OP.{op}.nc"], T[f"OP.{op}.share"] = v["narratives_contradicted"], f"{v['share_of_wrong_narratives']:.0f}"
        A.tex_table("tab_s_ops.tex", 7, "Operation & Executed & Supported & Contradicted & Narratives contradicted & "
                    "Of which wrong (\\%) & Wrong narratives caught (\\%) \\\\", rows,
                    "Fresh test, \\ewv{}-P with GPT-OSS-120B: how often each operation ran and what it found. A "
                    "narrative is contradicted by an operation if at least one of its executions is a contradiction; "
                    "the last column is the share of all wrong narratives so caught.", "tab:sops", size="\\scriptsize",
                    spec="@{}lrrrrrr@{}")
        lab = {"exec_P": "\\ewv{}-P", "exec_D": "\\ewv{}-D", "exec_R11": "\\ewv{}-R11", "facts_judge": "B1F",
               "generic_rubric": "B1R", "generic_judge2": "B1"}
        rows = []
        for j, nm in (("gptoss120b", "GPT-OSS-120B"), ("qwen3_next", "Qwen3-Next"), ("gemma27_vllm", "Gemma 3 27B"),
                      ("grok45", "Grok 4.5")):
            first = True
            for m in ("exec_P", "exec_D", "exec_R11", "facts_judge", "generic_rubric", "generic_judge2"):
                v = R["tokens"].get(f"{m}|{j}")
                if not v:
                    continue
                usd = f"{v['usd_per_1000']:.2f}" if "usd_per_1000" in v else "GPU"
                rows.append(f"{nm if first else ''} & {lab[m]} & {v['calls_per_narrative']:.2f} & "
                            f"{v['input_tokens_per_call']:,} & {v['output_tokens_per_call']:,} & {v['tokens_per_narrative']:,} & "
                            f"{v['truncated_calls_pct']:.1f} & {usd} \\\\")
                first = False
            rows.append("\\midrule")
        A.tex_table("tab_s_tokens.tex", 8, "Judge & Verifier & Calls & Input/call & Output/call & Tokens/narrative & "
                    "Truncated (\\%) & US\\$/1000 \\\\", rows[:-1],
                    "Fresh test: model calls and tokens per narrative, as logged for every call (reassessments and "
                    "retried calls included), and cost at the list prices given in Section~S15. Gemma ran on a rented GPU.",
                    "tab:stokens", size="\\scriptsize", spec="@{}llrrrrrr@{}")
        t = R["tokens"]["exec_P|gptoss120b"]
        T["TOK.P.in"], T["TOK.P.out"], T["TOK.P.nar"] = f"{t['input_tokens_per_call']:,}", f"{t['output_tokens_per_call']:,}", f"{t['tokens_per_narrative']:,}"
        e = R["escalation"]
        T["ESC.n"] = e["deferred"]
        for k, v in e["counts"].items():
            T[f"ESC.{k}.cr"], T[f"ESC.{k}.wr"] = v["correct"], v["wrong"]
        for k, v in e["usd_per_1000"].items():
            T[f"ESC.usd.{k}"] = f"{v:.2f}"
        for k, v in e["seconds_per_narrative"].items():
            T[f"ESC.sec.{k}"] = f"{v:.0f}"
        for cmp in ("router_vs_B1F", "router_vs_Grok", "router_vs_ReCheck-P"):
            for nm in ("wrong_release", "correct_release", "U2"):
                T[f"ESC.{cmp}.{nm}"], T[f"ESC.{cmp}.{nm}.CI"] = _m(e[cmp][nm]["diff"]), _ci(e[cmp][nm]["ci95"])
        rt = R.get("residual_types", {})
        T["RES.total"] = sum(rt.values())
        for k, v in rt.items():
            T[f"RES.{k}"] = v
    Al = A.jl(LK / "alias_perturbation.json")
    if Al:
        fz = Al["frozen"]
        T["AL.aliases"], T["AL.fz.rel"], T["AL.fz.relw"] = Al["aliases"], fz.get("release", 0), fz.get("release|wrong", 0)
        T["AL.perbor"] = round(Al["aliases"] / Al["n"])
        T["AL.fz.routed"] = fz.get("claims_routed", 0)
        rows = [f"none (frozen) & {fz.get('release', 0)} & {fz.get('release|wrong', 0)} & {fz.get('reject', 0)} & "
                f"{fz.get('judge', 0)} & {fz.get('claims_routed', 0)} \\\\"]
        for p, runs in Al["perturbed"].items():
            def mean(k):
                return sum(r.get(k, 0) for r in runs) / len(runs)
            def rng(k):
                xs = [r.get(k, 0) for r in runs]
                return f"{min(xs)}" if min(xs) == max(xs) else f"{min(xs)}--{max(xs)}"
            tag = f"AL.{int(100 * float(p))}"
            for k in ("release", "release|wrong", "reject", "judge", "claims_routed"):
                T[f"{tag}.{k.replace('|', '_')}"] = rng(k)
            for r in runs:
                r["new_wrong_release"] = sum(v for k, v in r.get("moves_wrong", {}).items() if k.endswith("->release"))
                r["lost_reject_wrong"] = r.get("moves_wrong", {}).get("reject->judge", 0)
            T[f"{tag}.newwrong"] = rng("new_wrong_release")
            T[f"{tag}.lostrej"] = rng("lost_reject_wrong")
            rows.append(f"{int(100 * float(p))}\\% & {rng('release')} & {rng('release|wrong')} & {rng('reject')} & "
                        f"{rng('judge')} & {rng('claims_routed')} \\\\")
        A.tex_table("tab_s_alias.tex", 6, "Aliases removed & Released by code & of which wrong & Rejected by code & "
                    "Sent to judge & Claims sent to judge \\\\", rows,
                    "Fresh test, code layer only: each evidence item's plain-language aliases removed at random with "
                    "the given probability (five draws; ranges shown; field names kept).", "tab:salias",
                    size="\\scriptsize", spec="@{}lrrrrr@{}")


def adjudication():
    """Independent adjudication of the fresh-test disagreements (protocol-correct labels)."""
    src = LK / "fresh_v2_analysis_independent_adjudication.json"
    A_ = A.jl(src)
    if not A_:
        return
    T["ADJ.N"] = A_["labels"]["narratives_labelled"]
    T["ADJ.ERRPCT"] = f"{100 * A_['labels']['narrative_error_rate']:.1f}"
    cells = A_["methods"]
    for mkey, tag in (("exec_P|gptoss120b", "ADJ.O.exec_P"), ("facts_judge|gptoss120b", "ADJ.O.facts_judge"),
                      ("exec_P|qwen3_next", "ADJ.Q.exec_P"), ("facts_judge|qwen3_next", "ADJ.Q.facts_judge")):
        v = cells.get(mkey)
        if not v:
            continue
        c = v["cells"]
        T[f"{tag}.wrong_release"] = c["wrong_release"]
        T[f"{tag}.correct_release"] = c["correct_release"]
    n_wrong = cells["exec_P|gptoss120b"]["cells"]["wrong_release"] + cells["exec_P|gptoss120b"]["cells"]["wrong_reject"] + cells["exec_P|gptoss120b"]["cells"]["wrong_defer"]
    n_correct = cells["exec_P|gptoss120b"]["cells"]["correct_release"] + cells["exec_P|gptoss120b"]["cells"]["correct_reject"] + cells["exec_P|gptoss120b"]["cells"]["correct_defer"]
    T["ADJ.NERR"] = n_wrong
    T["ADJ.NOK"] = n_correct
    for j, k in (("gptoss120b", "O"), ("qwen3_next", "Q")):
        d = A_["primary"].get(j)
        if not d:
            continue
        for kind in ("wrong_release", "correct_release"):
            T[f"ADJ.{k}.P.{kind}.abs"] = f"{abs(100 * d[kind]['R_minus_baseline']):.1f}"
            T[f"ADJ.{k}.P.{kind}.CI"] = ci(d[kind]["ci95"])
            T[f"ADJ.{k}.P.{kind}.OS"] = signed(d[kind]["one_sided_95"][0])
        T[f"ADJ.{k}.P.SUP"] = "met" if d.get("superiority_wrong_release") else "not met"
        T[f"ADJ.{k}.P.NI"] = "met" if d.get("noninferior_correct_release_5pts") else "not met"
        T[f"ADJ.{k}.P.BOTH"] = "met" if (d.get("superiority_wrong_release") and
                                           d.get("noninferior_correct_release_5pts")) else "not met"
    T["ADJ.nadj"] = 94


def runs3():
    """Three runs of ReCheck-P and B1F, and the B1F majority-vote baseline (analyze_runs3.py)."""
    R = A.jl(LK / "runs3_analysis.json")
    if not R:
        return
    for nm, k in (("ReCheck-P", "P"), ("B1F", "B1F")):
        r = R["runs"][nm]
        for f in ("wrong_mean", "wrong_sd", "correct_mean", "correct_sd"):
            T[f"R3.{k}.{f}"] = f"{r[f]:.1f}"
        T[f"R3.{k}.wrong.range"] = f"{min(r['wrong'])}--{max(r['wrong'])}"
        T[f"R3.{k}.correct.range"] = f"{min(r['correct'])}--{max(r['correct'])}"
    for nm, v in R["averaged"].items():
        T[f"R3.avg.{nm}"], T[f"R3.avg.{nm}.CI"] = _m(v["diff"]), _ci(v["ci95"])
        T[f"R3.avg.{nm}.abs"] = f"{abs(v['diff']):.1f}"
        T[f"R3.avg.{nm}.OS"] = _m(v["one_sided_95"][0])
    s3 = R["sc3"]
    T["SC3.wr"], T["SC3.cr"] = s3["B1F_SC3_counts"]["wrong"], s3["B1F_SC3_counts"]["correct"]
    d = s3["B1F_SC3_vs_ReCheck_P_single"]
    for nm in ("wrong_release", "correct_release", "U2"):
        T[f"SC3.d.{nm}"], T[f"SC3.d.{nm}.CI"] = _m(d[nm]["diff"]), _ci(d[nm]["ci95"])


def main() -> int:
    A.main()
    verdicts()
    fresh_cells()
    placeholders()
    fresh_sensitivity()
    review_checks()
    practitioner()
    wording()
    dev699()
    isolation()
    context_ci_figure()
    utility()
    review2()
    runs3()
    review3()
    adjudication()
    flow()
    (A.ESWA / "paper_tokens_v2.json").write_text(json.dumps(T, indent=1, default=str))
    print("tokens", len(T))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

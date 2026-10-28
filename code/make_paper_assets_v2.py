"""
Tables, figures and number tokens for the revised ESWA manuscript (after the 7 Oct review).
Every number in the paper comes from these analysis files:
  Submission/m1/analysis_v2.json                 development, three judges (analyze_v2.py)
  Submission/m1/replay_same_transcript_<j>.json  same-transcript objection ablation
  Submission/mechanism/context_analysis*.json    controlled context experiment v1 / v2
  Submission/m1/wording_intervention.json        paired wording intervention
  Submission/locked/locked_analysis.json         locked natural test (label-free until labels return)

    python make_paper_assets_v2.py   -> ESWA_Submission/tables/*.tex, figures/*.pdf, paper_tokens_v2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_m1 import SUB  # noqa: E402

ESWA = SUB / "ESWA_Submission"
TAB, FIG = ESWA / "tables", ESWA / "figures"
T = {}
JUDGES = [("qwen3_next", "Q", "Qwen3-Next"), ("grok45", "G", "Grok 4.5"), ("gemma27_vllm", "M", "Gemma 3 27B")]
LABEL = {
    "generic_judge2": "B1 generic judge", "generic_rubric": "B1R aligned rubric", "facts_judge": "B1F precomputed facts",
    "claimwise_judge": "B1C claim-wise", "generic_tool_judge": "B2 optional tools", "forced_tool_judge": "B2F forced tools",
    "generic_second_look": "B2$'$ second look", "factool_judge": "FacTool", "pt_audit": "Pratama--Tseng audit",
    "zandi_judge_tuned": "Zandi TFC/TDC (tuned)", "mermigkis_verifier_tuned": "Mermigkis verifier (tuned)",
    "pbp_judge": "Proof before preference", "exec_A0": "A0 code only", "exec_A1": "A1 objections unchecked",
    "exec_P": "ReCheck 1.4 (P)", "exec_D": "ReCheck-D", "exec_R": "ReCheck-R", "exec_R11": "ReCheck-R11",
    "exec_R_replay": "ReCheck-R (replay)", "exec_R11_replay": "ReCheck-R11 (replay)",
    "exec_R_replay_gemma_reader": "ReCheck-R, Gemma reader", "always_accept": "Always release",
}
JUDGES_LK = [("gptoss120b", "O", "GPT-OSS-120B"), ("qwen3_next", "Q", "Qwen3-Next"), ("gemma27_vllm", "M", "Gemma 3 27B"),
             ("grok45", "G", "Grok 4.5")]
MAIN_ROWS = ["always_accept", "pt_audit", "generic_judge2", "generic_rubric", "facts_judge", "claimwise_judge",
             "generic_tool_judge", "forced_tool_judge", "generic_second_look", "factool_judge", "zandi_judge_tuned",
             "mermigkis_verifier_tuned", "pbp_judge", "exec_A0", "exec_A1", "exec_P", "exec_D", "exec_R", "exec_R11"]


def jl(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def tex_table(path, cols, header, rows, caption, label, size="\\footnotesize", spec=None):
    spec = spec or ("@{}l" + "r" * (cols - 1) + "@{}")
    env = "tabularx" if "X" in spec else "tabular"
    width = "{\\linewidth}" if env == "tabularx" else ""
    out = ["\\begin{table}[!htb]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
           *(["\\setlength{\\tabcolsep}{5pt}"] if path == "tab_fresh_main.tex" else []),
           f"\\begin{{{env}}}{width}{{{spec}}}", "\\toprule", header, "\\midrule"]
    out += rows
    out += ["\\bottomrule", f"\\end{{{env}}}", "\\end{table}"]
    (TAB / path).write_text("\n".join(out) + "\n")


def dev_tables(A):
    J = A["judges"]
    # main: correct released / wrong released per judge
    rows = []
    for m in MAIN_ROWS:
        cells = []
        have = False
        for j, k, _ in JUDGES:
            v = J.get(j, {}).get("methods", {}).get(m)
            if m in ("exec_R", "exec_R11", "exec_D") and (not v or "incomplete" in v):
                v = J.get(j, {}).get("methods", {}).get(m + "_replay")
            if v and "incomplete" not in v:
                cells += [str(v["correct_release"]), str(v["wrong_release"])]
                have = True
            else:
                cells += ["--", "--"]
        if have:
            rows.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
        if m in ("pbp_judge",):
            rows.append("\\midrule")
    n = A["judges"]["qwen3_next"]["methods"]["always_accept"]
    hdr = ("& \\multicolumn{2}{c}{Qwen3-Next} & \\multicolumn{2}{c}{Grok 4.5} & \\multicolumn{2}{c}{Gemma 3 27B} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
           "Verifier & Correct & Wrong & Correct & Wrong & Correct & Wrong \\\\")
    tex_table("tab_dev_main.tex", 7, hdr, rows,
              f"Development set: correct and wrong explanations released, of {n['correct_release']} correct and "
              f"{n['wrong_release']} wrong. ReCheck-D and -R for Grok and ReCheck-R11 for Qwen and Grok replay the saved "
              "first-stage judge responses of ReCheck 1.4; all other entries are live runs.", "tab:devmain")
    # six cells for the judges' key methods
    for j, k, nm in JUDGES:
        rows = []
        for m in ["generic_judge2", "generic_rubric", "facts_judge", "forced_tool_judge", "exec_A1", "exec_P", "exec_D",
                  "exec_R", "exec_R11"]:
            v = J.get(j, {}).get("methods", {}).get(m)
            if m in ("exec_R", "exec_R11", "exec_D") and (not v or "incomplete" in v):
                v = J.get(j, {}).get("methods", {}).get(m + "_replay")
            if not v or "incomplete" in v:
                continue
            se = "--" if v["selective_error"] is None else f"{100 * v['selective_error']:.1f}"
            rows.append(f"{LABEL[m]} & {v['correct_release']} & {v['correct_reject']} & {v['correct_defer']} & "
                        f"{v['wrong_release']} & {v['wrong_reject']} & {v['wrong_defer']} & {se} & "
                        f"{v['calls_per_item']:.2f} & {v['usd_list']:.2f} \\\\")
        hdr = ("& \\multicolumn{3}{c}{Correct} & \\multicolumn{3}{c}{Wrong} & & & \\\\\n"
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
               "Verifier & Rel. & Rej. & Def. & Rel. & Rej. & Def. & Err.\\,\\% & Calls & US\\$ \\\\")
        tex_table(f"tab_six_{k}.tex", 10, hdr, rows,
                  f"Development set, {nm} judge: all six outcome cells, error rate among released explanations, "
                  "model calls per explanation and list cost" + (" (self-hosted: GPU time, reported in the text)" if k == "M" else "") + ".",
                  f"tab:six{k}", size="\\scriptsize")
    # utility
    rows = []
    for m in ["generic_judge2", "facts_judge", "factool_judge", "exec_A1", "exec_P", "exec_D", "exec_R", "exec_R11"]:
        cells = []
        for j, k, _ in JUDGES:
            v = J.get(j, {}).get("methods", {}).get(m)
            if m in ("exec_R", "exec_R11", "exec_D") and (not v or "incomplete" in v):
                v = J.get(j, {}).get("methods", {}).get(m + "_replay")
            cells += ([f"${v['utility'][l]}$" for l in ("2", "5")] if v and "incomplete" not in v else ["--", "--"])
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{2}{c}{Qwen3-Next} & \\multicolumn{2}{c}{Grok 4.5} & \\multicolumn{2}{c}{Gemma 3 27B} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
           "Verifier & $\\lambda=2$ & $\\lambda=5$ & $\\lambda=2$ & $\\lambda=5$ & $\\lambda=2$ & $\\lambda=5$ \\\\")
    tex_table("tab_utility.tex", 7, hdr, rows,
              "Development set: utility $U_\\lambda$ = correct released $-\\ \\lambda \\times$ wrong released.", "tab:utility")
    # tokens
    for j, k, _ in JUDGES:
        for m, v in J.get(j, {}).get("methods", {}).items():
            if "incomplete" in v:
                continue
            for f in ("correct_release", "wrong_release", "correct_reject", "correct_defer", "wrong_reject", "wrong_defer"):
                T[f"{k}.{m}.{f}"] = v[f]
            T[f"{k}.{m}.SE"] = "--" if v["selective_error"] is None else f"{100 * v['selective_error']:.1f}"
            T[f"{k}.{m}.USD"] = f"{v['usd_list']:.2f}"
            T[f"{k}.{m}.CALLS"] = f"{v['calls_per_item']:.2f}"
            for l in ("1", "2", "5", "10"):
                T[f"{k}.{m}.U{l}"] = v["utility"][l]
            g = v.get("withheld_correct_by_group")
            if g:
                for ds, short in (("home_credit", "HC"), ("taiwan", "TW")):
                    T[f"{k}.{m}.GAP.{short}"] = f"${100 * g[ds + '|gap_male_minus_female']:+.1f}$"
            T[f"{k}.{m}.correct_withheld"] = v["correct_reject"] + v["correct_defer"]
            T[f"{k}.{m}.settled"] = sum(v[x] for x in ("correct_release", "correct_reject", "wrong_release", "wrong_reject"))
            if "tool_use_items" in v:
                T[f"{k}.{m}.TOOLS"] = v["tool_use_items"]
            if "threshold" in v:
                T[f"{k}.{m}.THR"] = v["threshold"]
        for m, d in J.get(j, {}).get("subset_comparisons", {}).items():
            T[f"{k}.sub.{m}.n"] = d["n"]
            for who, c in d.items():
                if isinstance(c, dict):
                    T[f"{k}.sub.{m}.{who}.cr"], T[f"{k}.sub.{m}.{who}.wr"] = c["correct_release"], c["wrong_release"]
                    T[f"{k}.sub.{m}.{who}.nc"] = c["correct_release"] + c["correct_reject"] + c["correct_defer"]
                    T[f"{k}.sub.{m}.{who}.nw"] = c["wrong_release"] + c["wrong_reject"] + c["wrong_defer"]
        for m, d in J.get(j, {}).get("paired_ref_minus_X", {}).items():
            for kind in ("wrong_release", "correct_release"):
                T[f"{k}.d.{m}.{kind}"] = f"${100 * d[kind]['ref_minus_X']:+.1f}$"
                T[f"{k}.d.{m}.{kind}.CI"] = f"[${100 * d[kind]['ci95'][0]:+.1f}$, ${100 * d[kind]['ci95'][1]:+.1f}$]"
    rows = []
    for m in ["generic_judge2", "generic_rubric", "facts_judge", "exec_A0", "exec_P", "exec_R", "exec_R11"]:
        cells = []
        for j, k, _ in JUDGES:
            v = J.get(j, {}).get("methods", {}).get(m)
            if m in ("exec_R", "exec_R11", "exec_D") and (not v or "incomplete" in v):
                v = J.get(j, {}).get("methods", {}).get(m + "_replay")
            g = (v or {}).get("withheld_correct_by_group")
            if g:
                for ds, short in (("home_credit", "HC"), ("taiwan", "TW")):
                    T[f"{k}.{m}.GAP.{short}"] = f"${100 * g[ds + '|gap_male_minus_female']:+.1f}$"
                cells += [f"${100 * g['home_credit|gap_male_minus_female']:+.1f}$", f"${100 * g['taiwan|gap_male_minus_female']:+.1f}$"]
            else:
                cells += ["--", "--"]
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    hdr = ("& \\multicolumn{2}{c}{Qwen3-Next} & \\multicolumn{2}{c}{Grok 4.5} & \\multicolumn{2}{c}{Gemma 3 27B} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
           "Verifier & HC & TW & HC & TW & HC & TW \\\\")
    tex_table("tab_access.tex", 7, hdr, rows,
              "Development set: difference in the share of correct explanations withheld, male minus female, in "
              "percentage points (HC Home Credit, TW Taiwan; 9 borrowers per group and dataset, descriptive).", "tab:access")
    if A.get("kwok"):
        T["KWOK.AUROC"] = f"{A['kwok']['auroc_correct_vs_wrong']:.2f}"
        T["KWOK.N"] = A["kwok"]["n"]


def replay_table():
    rows = []
    for j, k, nm in JUDGES[:2]:
        R = jl(SUB / "m1" / f"replay_same_transcript_{j}.json")
        if not R:
            continue
        lab = {"none": "no check (A1 rule)", "plausibility": "plausibility (A2 rule)", "execute": "execute, no reassessment",
               "execute+R": "execute + reassessment (1.4)", "dispute": "dispute-to-defer (D)"}
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{nm}: {R['n_llm_replayed']} explanations reached the judge}}}} \\\\")
        for p in ("none", "plausibility", "execute", "execute+R", "dispute"):
            c = R["all_items"][p]
            rows.append(f"\\quad {lab[p]} & {c['correct_release']} & {c['correct_reject']} & {c['correct_defer']} & "
                        f"{c['wrong_release']} & {c['wrong_reject']} & {c['wrong_defer']} \\\\")
            for f, v in c.items():
                T[f"RP.{k}.{p}.{f}"] = v
    hdr = ("& \\multicolumn{3}{c}{Correct} & \\multicolumn{3}{c}{Wrong} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
           "First-stage response processed by & Rel. & Rej. & Def. & Rel. & Rej. & Def. \\\\")
    tex_table("tab_replay.tex", 7, hdr, rows,
              "Same saved first-stage judge responses, processed by different objection rules (all 299 development "
              "explanations; code-only decisions identical in every row).", "tab:replay", size="\\scriptsize")


def context_tables():
    rows, fig = [], {}
    for tag in ("v1", "v2"):
        C = jl(SUB / "mechanism" / ("context_analysis.json" if tag == "v1" else f"context_analysis_{tag}_exec_R11.json"))
        if not C:
            continue
        T[f"CX{tag}.NBASE"] = C["n_base"]
        T[f"CX{tag}.N"] = C["n_passages"]
        for j, k, nm in JUDGES:
            jr = C["judges"].get(j)
            if not jr:
                continue
            for m in ("generic_rubric", "exec_P", "exec_D", "exec_R", "exec_R11"):
                v = jr["methods"].get(m)
                if not v:
                    continue
                b = v["by_context"]
                for ctx in ("alone", "claims", "filler"):
                    for met in ("escape", "retention", "unresolved"):
                        T[f"CX{tag}.{k}.{m}.{ctx}.{met}"] = f"{100 * b[ctx][met]:.0f}"
                rows.append(f"{tag} & {nm} & {LABEL.get(m, m)} & " + " & ".join(
                    f"{100 * b[c]['escape']:.0f} & {100 * b[c]['retention']:.0f}" for c in ("alone", "claims", "filler")) + " \\\\")
                fig[(tag, k, m)] = b
            for comp, d in jr["interaction"].items():
                for kk, vv in d.items():
                    key = f"CX{tag}.{k}.DID.{comp}.{kk.replace('|', '.')}"
                    T[key] = f"${100 * vv.get('DiD', vv.get('DiD_(X-ReCheckR)')):+.0f}$"
                    T[key + ".abs"] = f"{abs(100 * vv.get('DiD', vv.get('DiD_(X-ReCheckR)'))):.0f}"
                    T[key + ".CI"] = f"[${100 * vv['ci95'][0]:+.0f}$, ${100 * vv['ci95'][1]:+.0f}$]"
    hdr = ("& & & \\multicolumn{2}{c}{Alone} & \\multicolumn{2}{c}{Among claims} & \\multicolumn{2}{c}{Among filler} \\\\\n"
           "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
           "Pack & Judge & Verifier & Esc. & Ret. & Esc. & Ret. & Esc. & Ret. \\\\")
    tex_table("tab_context.tex", 9, hdr, rows,
              "Controlled context experiment: error escape (\\% of wrong passages released) and retention "
              "(\\% of correct passages released) by context. Pack v1 was the prespecified test of the frozen verifiers; "
              "pack v2, on new borrowers, tested the R11 revision.", "tab:context", size="\\scriptsize",
              spec="@{}lll" + "r" * 6 + "@{}")
    return fig


def context_figure(fig):
    if not fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctxs = ["alone", "claims", "filler"]
    judges = [x for x in JUDGES if any((("v1", x[1], m) in fig) for m in ("generic_rubric", "exec_R"))]
    f, ax = plt.subplots(2, len(judges), figsize=(3.2 * len(judges), 5.0), sharey=True, squeeze=False)
    sty = {"generic_rubric": ("#c0392b", "o", "B1R aligned rubric"), "exec_R": ("#1f618d", "s", "ReCheck-R"),
           "exec_R11": ("#117a65", "^", "ReCheck-R11"), "exec_D": ("#7d3c98", "D", "ReCheck-D")}
    for c, (j, k, nm) in enumerate(judges):
        for r, met in enumerate(("escape", "retention")):
            a = ax[r][c]
            for m, (col, mk, lab) in sty.items():
                b = fig.get(("v1", k, m))
                if b:
                    a.plot(range(3), [100 * b[x][met] for x in ctxs], marker=mk, color=col, label=lab)
            a.set_xticks(range(3))
            a.set_xticklabels(["alone", "+claims", "+filler"])
            if r == 0:
                a.set_title(nm, fontsize=10)
            if c == 0:
                a.set_ylabel(("Wrong passages released (%)" if met == "escape" else "Correct passages released (%)"), fontsize=9)
            a.grid(alpha=0.3)
    ax[0][0].legend(fontsize=7, loc="upper left")
    f.tight_layout()
    f.savefig(FIG / "fig_context.pdf")


def wording():
    W = jl(SUB / "m1" / "wording_intervention.json")
    if not W:
        return
    o = W["overall"]
    T["WI.N"], T["WI.NBAD"] = o["n"], o["n_bad"]
    for k, v in o.items():
        if k not in ("n", "n_bad"):
            T["WI." + k.replace("|", ".")] = f"{100 * v:.0f}" if isinstance(v, float) else v


def locked(src="locked_analysis.json", P="LK", tab="locked", title="Locked natural test"):
    L = jl(SUB / "locked" / src)
    if not L:
        return
    # the fresh test ran the frozen checker 1.6; the locked test ran 1.4
    LABEL = dict(globals()["LABEL"], exec_P="ReCheck 1.6 (P)") if P == "FV" else globals()["LABEL"]
    rows = []
    for key, v in L["methods"].items():
        m, j = key.split("|")
        k = {x[0]: x[1] for x in JUDGES_LK}.get(j)
        if not k:
            continue
        T[f"{P}.{k}.{m}.release"], T[f"{P}.{k}.{m}.reject"], T[f"{P}.{k}.{m}.defer"] = v["release"], v["reject"], v["defer"]
        T[f"{P}.{k}.{m}.calls"] = f"{v['calls_per_item']:.2f}"
        T[f"{P}.{k}.{m}.settled"] = v["release"] + v["reject"]
        if m in ("generic_judge2", "generic_rubric", "facts_judge", "forced_tool_judge", "exec_A0", "exec_P", "exec_D",
                 "exec_R", "exec_R11"):
            rows.append((k, f"{dict((x[1], x[2]) for x in JUDGES_LK)[k]} & {LABEL.get(m, m)} & {v['release']} & {v['reject']} & "
                            f"{v['defer']} & {v['calls_per_item']:.2f} \\\\"))
    order = ["generic_judge2", "generic_rubric", "facts_judge", "forced_tool_judge", "exec_A0", "exec_P", "exec_D",
             "exec_R", "exec_R11"]
    rows = [r for _k, r in sorted(rows, key=lambda x: ("OQMG".index(x[0]),
                                                       next((n for n, m in enumerate(order) if LABEL.get(m, m) in x[1]), 99)))]
    # scored narratives (the records file also holds templates and producer failures)
    T[f"{P}.N"] = (L.get("labels") or {}).get("narratives_labelled", L["n_narratives"])
    if "labels" in L:
        lb = L["labels"]
        T[f"{P}.NERR"] = round(lb["narrative_error_rate"] * lb["narratives_labelled"])
        T[f"{P}.NOK"] = lb["narratives_labelled"] - T[f"{P}.NERR"]
        T[f"{P}.ERRPCT"] = f"{100 * lb['narrative_error_rate']:.1f}"
        rows2 = []
        order = ["generic_judge2", "generic_rubric", "facts_judge", "forced_tool_judge", "claimwise_judge", "exec_A0",
                 "exec_P", "exec_D", "exec_R", "exec_R11"]
        for j, k, nm in JUDGES_LK:
            n_before = len(rows2)
            for m in order:
                v = L["methods"].get(f"{m}|{j}")
                if not v or "cells" not in v:
                    continue
                c = v["cells"]
                sh = L["shown"][f"{m}|{j}"]
                u = L["utility"][f"{m}|{j}"]
                for f_ in c:
                    T[f"{P}.{k}.{m}.{f_}"] = c[f_]
                T[f"{P}.{k}.{m}.SHOWNERR"] = f"{100 * sh['error_rate']:.1f}"
                T[f"{P}.{k}.{m}.U2"] = u["2"]
                T[f"{P}.{k}.{m}.U1"] = u["1"]
                rows2.append(f"{nm} & {LABEL.get(m, m)} & {c['correct_release']} & {c['wrong_release']} & "
                             f"{c['correct_reject'] + c['correct_defer']} & {c['wrong_reject'] + c['wrong_defer']} & "
                             f"{100 * sh['error_rate']:.1f} & ${u['1']}$ & ${u['2']}$ \\\\")
                g = L["group_withheld"][f"{m}|{j}"]
                for ds, short in (("taiwan", "TW"), ("home_credit", "HC")):
                    T[f"{P}.{k}.{m}.GAP.{short}"] = f"${100 * g[ds]['gap_male_minus_female']:+.1f}$"
                    T[f"{P}.{k}.{m}.GAP.{short}.CI"] = f"[${100 * g[ds]['ci95'][0]:+.1f}$, ${100 * g[ds]['ci95'][1]:+.1f}$]"
                esc = L["escape_by_type"][f"{m}|{j}"]
                for t_, n_ in esc.items():
                    T[f"{P}.{k}.{m}.ESC.{t_}"] = n_
            if len(rows2) > n_before:
                rows2.append("\\midrule")
        rows2 = rows2[:-1]
        hdr = ("& & \\multicolumn{2}{c}{Released} & \\multicolumn{2}{c}{Withheld} & Shown & & \\\\\n"
               "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
               "Judge & Verifier & Correct & Wrong & Correct & Wrong & err.\\,\\% & $U_1$ & $U_2$ \\\\")
        caption = (f"{title} ({lb['narratives_labelled']} fresh borrowers, {T[P + '.NERR']} wrong narratives): "
                   "explanations released and withheld, the share of texts shown to the lender that are wrong "
                   "(released narrative, otherwise the template), and utility.")
        if tab == "fresh":
            sample_sizes = {v.get("n") for v in L.get("shown", {}).values() if v.get("n")}
            shown_scope = (f"{next(iter(sample_sizes))} template-labelled borrowers"
                           if len(sample_sizes) == 1 else "the template-labelled sample")
            caption = (f"Fresh-test decisions ({lb['narratives_labelled']} narratives, {T[P + '.NERR']} wrong); "
                       f"shown error uses {shown_scope}.")
        tex_table(f"tab_{tab}_main.tex", 9, hdr, rows2,
                  caption, f"tab:{tab}main", size="\\scriptsize",
                  spec="@{}llrrrrrrr@{}")
        for jj, d in L.get("primary", {}).items():
            k = {x[0]: x[1] for x in JUDGES_LK}[jj]
            for kind in ("wrong_release", "correct_release"):
                T[f"{P}.{k}.P.{kind}"] = f"${100 * d[kind]['R_minus_baseline']:+.1f}$"
                T[f"{P}.{k}.P.{kind}.CI"] = f"[${100 * d[kind]['ci95'][0]:+.1f}$, ${100 * d[kind]['ci95'][1]:+.1f}$]"
                T[f"{P}.{k}.P.{kind}.OS"] = f"${100 * d[kind]['one_sided_95'][0]:+.1f}$" if kind == "correct_release" \
                    else f"${100 * d[kind]['one_sided_95'][1]:+.1f}$"
        qa = L.get("label_quality", {})
        if qa:
            T[f"{P}.QA.agree"] = f"{100 * qa['all']['agreement']:.1f}"
            T[f"{P}.QA.kappa"] = f"{qa['all']['kappa']:.2f}"
            T[f"{P}.QA.agree.nar"] = f"{100 * qa['grounded']['agreement']:.1f}"
            T[f"{P}.QA.kappa.nar"] = f"{qa['grounded']['kappa']:.2f}"
            T[f"{P}.QA.dup.n"], T[f"{P}.QA.dup.ok"] = qa["duplicates"]["n"], qa["duplicates"]["consistent"]
            T[f"{P}.QA.adj"] = qa["adjudicated"]
        for f_, (e_, n_) in L.get("errors_by_producer", {}).items():
            T[f"{P}.PROD.{f_}.err"], T[f"{P}.PROD.{f_}.n"] = e_, n_
        sr = jl(SUB / "locked" / "sensitivity_revised_labels.json")
        if sr:
            T[f"{P}.SR.nex"], T[f"{P}.SR.nkept"] = sr["n_excluded"], sr["n_kept"]
            for jj, k in (("qwen3_next", "Q"), ("gemma27_vllm", "M")):
                for kind in ("wrong_release", "correct_release"):
                    T[f"{P}.SR.{k}.{kind}"] = f"${100 * sr[jj][kind]['diff']:+.1f}$"
                    T[f"{P}.SR.{k}.{kind}.CI"] = f"[${100 * sr[jj][kind]['ci95'][0]:+.1f}$, ${100 * sr[jj][kind]['ci95'][1]:+.1f}$]"
        tv = L["template_vs_narrative"]
        for grp, d in tv.items():
            for f_, v_ in d.items():
                T[f"{P}.TV.{grp}.{f_}"] = (f"{100 * v_:.1f}" if f_ in ("error_rate", "conveys_pd", "top_driver")
                                          else (f"{v_:.2f}" if f_ == "mean_usefulness" else v_))
    tex_table(f"tab_{tab}_decisions.tex", 6, "Judge & Verifier & Released & Rejected & Deferred & Calls \\\\", rows,
              f"{title}: decisions of the frozen verifiers on {L['n_narratives']} narratives for fresh "
              "borrowers, before any label was seen.", f"tab:{tab}", size="\\scriptsize", spec="@{}llrrrr@{}")


def main() -> int:
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    A = jl(SUB / "m1" / "analysis_v2.json")
    if A:
        dev_tables(A)
    replay_table()
    fig = context_tables()
    context_figure(fig)
    wording()
    locked()
    adj_src = "fresh_v2_analysis_independent_adjudication.json"
    if (SUB / "locked" / adj_src).exists():
        locked(adj_src, "FV", "fresh", "Fresh natural test")
    else:
        locked("fresh_v2_analysis.json", "FV", "fresh", "Fresh natural test")
    ls = jl(SUB / "m1" / "label_sensitivity.json")
    if ls:
        for j, k, _ in JUDGES:
            for m, v in ls["judges"].get(j, {}).items():
                for st, x in v.items():
                    T[f"LS.{k}.{m}.{st}.cr"], T[f"LS.{k}.{m}.{st}.wr"] = x["correct_released"], x["wrong_released"]
        T["LS.agreed.n"], T["LS.agreed.err"], T["LS.reverted.err"] = ls["n"]["agreed"], ls["errors"]["agreed"], ls["errors"]["reverted"]
    g = jl(SUB / "gpu_cost.json")
    if g:
        T["GPU.HOURS"], T["GPU.USD"] = f"{g['gpu_hours']:.1f}", f"{g['usd']:.2f}"
    (ESWA / "paper_tokens_v2.json").write_text(json.dumps(T, indent=1, default=str))
    print("tokens", len(T))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

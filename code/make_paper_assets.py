"""
Build every results table (LaTeX) and figure (PDF) of the ESWA paper from saved outputs, so no number
in the paper is typed by hand. Run after analyze_phaseA.py (final labels) for both judges.

    python make_paper_assets.py
Writes Submission/ESWA_Submission/tables/*.tex, figures/*.pdf and paper_numbers.json (every number
the text quotes).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_m1 import M1, SUB, final_labels  # noqa: E402

ESWA = SUB / "ESWA_Submission"
TAB, FIG = ESWA / "tables", ESWA / "figures"
CMP = M1 / "comparators"
N = {}

ORDER = [("always_accept", "Always accept"), ("always_reject", "Always reject"),
         ("pt_audit", "Pratama--Tseng audit"), ("zandi_judge", "Zandi et al.\\ judge"),
         ("mermigkis_verifier", "Mermigkis et al.\\ verifier"), ("pbp_judge", "Tapwal et al.\\ PbP"),
         ("risk_gate", "Risk gate (Badshah et al.)"),
         ("ws_cascade_human", "Kiyani et al.\\ cascade, human"),
         ("ws_cascade_grok", "Kiyani et al.\\ cascade, Grok"),
         ("generic_judge2", "B1 generic judge"), ("generic_tool_judge", "B2 generic + tools"),
         ("generic_second_look", "B2$'$ generic + second look"),
         ("contract", "B3 contract verifier"), ("exec_A0", "A0 code layer only"),
         ("exec_PX", "PX no claim execution"), ("exec_A1", "A1 objections unchecked"),
         ("exec_A2", "A2 objections plausibility-checked"), ("exec_P", "\\textbf{ReCheck (proposed)}")]
GROK_ROWS = {"generic_judge2", "generic_tool_judge", "exec_A1", "exec_A2", "exec_P"}


def load_rows(method, judge, tag=""):
    for name in ([f"{method}__{judge}__{tag}"] if tag else []) + [f"{method}__{judge}", f"{method}__none__{tag}",
                                                                   f"{method}__none"]:
        p = CMP / f"{name}.jsonl"
        if p.exists() and "__dry" not in name:
            rows = {}
            for line in open(p):
                r = json.loads(line)
                if r.get("live") is False and "none" not in name:
                    continue
                rows[r["item_id"]] = r
            return rows
    return {}


def frac(k, n):
    return f"{k}/{n}"


def main_table():
    fl = final_labels()
    truth = {i: v for i, (_k, v, _b) in fl.items()}
    out = {}
    lines = []
    for judge, tag, jname in (("qwen3_next", "v1.4", "Qwen3-Next 80B (Bedrock)"), ("grok45", "v1.4", "Grok 4.5 (LinkAPI)")):
        res = {}
        for m, label in ORDER:
            rows = load_rows(m, judge, tag if m.startswith("exec_") else "")
            rows = {i: r for i, r in rows.items() if i in truth}
            if len(rows) < len(truth):
                continue
            err = [i for i in rows if truth[i]]
            ok = [i for i in rows if not truth[i]]
            ea = sum(1 for i in err if rows[i]["accepted"] is True)
            cr = sum(1 for i in ok if rows[i]["accepted"] is True)
            crej = sum(1 for i in ok if rows[i]["accepted"] is False)
            dfr = sum(1 for i in rows if rows[i]["accepted"] is None)
            rel = ea + cr
            calls = sum(len(rows[i].get("calls") or []) for i in rows)
            res[m] = {"label": label, "err_acc": ea, "n_err": len(err), "corr_rel": cr, "n_ok": len(ok),
                      "corr_rej": crej, "deferred": dfr, "sel_err": (ea / rel if rel else None), "calls": calls,
                      "human_reviews": sum(1 for i in rows if rows[i].get("strong_queried")
                                           and m == "ws_cascade_human")}
        out[judge] = res
        rows_out = []
        for m, label in ORDER:
            if m not in res or (judge == "grok45" and m not in GROK_ROWS):
                continue
            r = res[m]
            se = "--" if r["sel_err"] is None else f"{100 * r['sel_err']:.1f}"
            extra = f" ({r['human_reviews']} reviews)" if m == "ws_cascade_human" else ""
            rows_out.append(f"{label}{extra} & {frac(r['err_acc'], r['n_err'])} & {frac(r['corr_rel'], r['n_ok'])} & "
                            f"{frac(r['corr_rej'], r['n_ok'])} & {r['deferred']} & {se} & {r['calls'] / len(truth):.2f} \\\\")
        (TAB / f"tab_main_{'qwen' if judge == 'qwen3_next' else 'grok'}.tex").write_text("\n".join(rows_out) + "\n")
    N["main"] = out
    N["final_labels"] = {"n": len(truth), "errors": sum(truth.values()), "correct": len(truth) - sum(truth.values()),
                         "by_basis": dict(Counter(b for _k, _v, b in fl.values()))}
    return out, truth


def paired_table():
    lines = []
    for judge, tag in (("qwen3_next", "v1.4"), ("grok45", "v1.4")):
        p = M1 / f"phaseA_{judge}{('_' + tag) if tag else ''}_final.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        N.setdefault("paired", {})[judge] = d["paired_P_minus_X_cluster_bootstrap"]
        for m, label in ORDER:
            if m not in d["paired_P_minus_X_cluster_bootstrap"] or m in ("always_accept", "always_reject"):
                continue
            x = d["paired_P_minus_X_cluster_bootstrap"][m]
            rc, ea = x["released_correct"], x["errors_accepted"]
            fmt = lambda v: f"{100 * v['P_minus_X']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"  # noqa: E731
            lines.append(f"{'Qwen3-Next' if judge == 'qwen3_next' else 'Grok 4.5'} & {label} & {fmt(ea)} & {fmt(rc)} \\\\")
        lines.append("\\midrule")
    (TAB / "tab_paired.tex").write_text("\n".join(lines[:-1]) + "\n")


def access_table(truth):
    key = {r["item_id"]: r for r in map(json.loads, open(SUB / "annotation_key_m1.jsonl")) if not r.get("_meta")}
    norm = {"m": "male", "male": "male", "f": "female", "female": "female"}
    lines, acc = [], {}
    for judge, tag in (("qwen3_next", "v1.4"), ("grok45", "v1.4")):
        for m in ("generic_judge2", "generic_tool_judge", "exec_A0", "exec_P"):
            rows = load_rows(m, judge, tag if m.startswith("exec_") else "")
            if not rows:
                continue
            cells = {}
            for ds in ("home_credit", "taiwan"):
                for g in ("female", "male"):
                    its = [i for i in rows if i in truth and not truth[i] and key[i]["dataset"] == ds
                           and norm.get(str(key[i].get("group")).lower()) == g]
                    w = sum(1 for i in its if rows[i]["accepted"] is not True)
                    cells[(ds, g)] = (w, len(its))
            acc[f"{m}|{judge}"] = {f"{d}|{g}": v for (d, g), v in cells.items()}
            if m == "exec_A0" and judge == "grok45":
                continue
            lab = dict(ORDER)[m] + ("" if m == "exec_A0" else f" ({'Qwen3-Next' if judge == 'qwen3_next' else 'Grok 4.5'})")
            row = [lab]
            for ds in ("home_credit", "taiwan"):
                (wf, nf), (wm, nm) = cells[(ds, "female")], cells[(ds, "male")]
                gap = 100 * (wm / nm - wf / nf) if nf and nm else 0
                row += [frac(wf, nf), frac(wm, nm), f"{gap:+.1f}"]
            lines.append(" & ".join(row) + " \\\\")
    (TAB / "tab_access.tex").write_text("\n".join(lines) + "\n")
    N["access"] = acc


def heldout_tables():
    v3 = SUB / "mechanism" / "v3_runs"
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(SUB / "mechanism" / "mechanism_pack_v3_annotator.jsonl")}
    lines, res = [], {}
    for m, judge, label in (("exec_A0", "none", "A0 code layer"),
                            ("generic_judge2", "qwen3_next", "B1 (Qwen3-Next)"),
                            ("generic_tool_judge", "qwen3_next", "B2 (Qwen3-Next)"),
                            ("exec_P", "qwen3_next", "\\textbf{ReCheck (Qwen3-Next)}"),
                            ("generic_judge2", "grok45", "B1 (Grok 4.5)"),
                            ("generic_tool_judge", "grok45", "B2 (Grok 4.5)"),
                            ("exec_P", "grok45", "\\textbf{ReCheck (Grok 4.5)}")):
        p = v3 / f"{m}__{judge}.jsonl"
        if not p.exists():
            continue
        rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}
        if len(rows) < len(items):
            continue
        err = [r for r in rows.values() if r["label"] == "error"]
        ok = [r for r in rows.values() if r["label"] == "correct"]
        prov = [r for r in ok if r["version"] == "provocation"]
        r_ = {"err_acc": sum(1 for r in err if r["accepted"] is True), "n_err": len(err),
              "err_rej": sum(1 for r in err if r["accepted"] is False),
              "ok_acc": sum(1 for r in ok if r["accepted"] is True), "n_ok": len(ok),
              "ok_rej": sum(1 for r in ok if r["accepted"] is False),
              "prov_rej": sum(1 for r in prov if r["accepted"] is False), "n_prov": len(prov),
              "deferred": sum(1 for r in rows.values() if r["accepted"] is None),
              "calls": sum(len(r.get("calls") or []) for r in rows.values())}
        res[f"{m}|{judge}"] = r_
        lines.append(f"{label} & {frac(r_['err_rej'], r_['n_err'])} & {frac(r_['err_acc'], r_['n_err'])} & "
                     f"{frac(r_['ok_acc'], r_['n_ok'])} & {frac(r_['ok_rej'], r_['n_ok'])} & "
                     f"{frac(r_['prov_rej'], r_['n_prov'])} & {r_['deferred']} \\\\")
    (TAB / "tab_heldout_v3.tex").write_text("\n".join(lines) + "\n")
    N["heldout_v3"] = res
    # developer-written packs (code layer): v1 first look, v2 once, v2 after fixes
    mech = SUB / "mechanism"
    rows = []
    for f, label in (("eval_code_layer_v1_firstlook.json", "v1 dev.\\ wording (1.0)"),
                     ("eval_code_layer_v1_firstlook.json", "v1 held-out wording (1.0)"),
                     ("eval_code_layer_v2_once.json", "v2 held-out, once (1.1)"),
                     ("eval_code_layer_v2_DEVELOPMENT.json", "v2 after fixes (1.4)")):
        d = json.load(open(mech / f))["overall"]
        o = d.get("dev") if "dev." in label else d.get("heldout") if "held-out wording" in label else d["v2"]
        rows.append(f"{label} & {o['detection']} & {o['missed_as_support']} & {o['false_alarm']} & "
                    f"{o['true_witness_verified']} & {o['forged_witness_rejected']} \\\\")
        N.setdefault("mech_dev_packs", {})[label] = o
    (TAB / "tab_mechanism.tex").write_text("\n".join(rows) + "\n")


def objection_outcomes():
    out = {}
    for judge, tag in (("qwen3_next", "v1.4"), ("grok45", "v1.4")):
        rows = load_rows("exec_P", judge, tag)
        c = Counter()
        for r in rows.values():
            for w in (r.get("detail") or {}).get("witnesses", []):
                c[(w.get("rule"), w.get("outcome"))] += 1
        out[judge] = {f"{a}|{b}": n for (a, b), n in c.items()}
        out[judge + "_totals"] = dict(Counter(b for (a, b) in c.elements()))
    N["objections"] = out
    lines = []
    outcomes = ["verified", "semantic_plausible", "disproved", "malformed", "unexecutable", "semantic_invalid"]
    for judge in ("qwen3_next", "grok45"):
        t = out[judge + "_totals"]
        lines.append(f"{'Qwen3-Next' if judge == 'qwen3_next' else 'Grok 4.5'} & " +
                     " & ".join(str(t.get(o, 0)) for o in outcomes) + f" & {sum(t.values())} \\\\")
    (TAB / "tab_objections.tex").write_text("\n".join(lines) + "\n")


def annotation_numbers():
    sc = json.load(open(M1 / "annotation_scores.json"))
    N["annotation"] = {"agreement": sc["agreement"], "challenge": sc["challenge"],
                       "integrity": {r: {k: v for k, v in d.items() if k in ("expected_items", "returned_items")}
                                     for r, d in sc["integrity"].items()}}
    N["denominators"] = json.load(open(M1 / "denominators.json"))


def figures(main, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, judge, title in ((axes[0], "qwen3_next", "Judge: Qwen3-Next 80B"), (axes[1], "grok45", "Judge: Grok 4.5")):
        res = main.get(judge, {})
        for m, label in ORDER:
            if m not in res or m in ("always_reject",):
                continue
            r = res[m]
            x = 100 * r["err_acc"] / r["n_err"]
            y = 100 * r["corr_rel"] / r["n_ok"]
            kind = ("P" if m == "exec_P" else "abl" if m.startswith("exec_") else
                    "rel" if m in ("pt_audit", "zandi_judge", "mermigkis_verifier", "pbp_judge", "risk_gate",
                                   "ws_cascade_human", "ws_cascade_grok") else "ctl" if m.startswith("always")
                    else "base")
            st = {"P": dict(marker="*", s=220, c="#C62828", zorder=5), "abl": dict(marker="D", s=40, c="#EF9A9A"),
                  "rel": dict(marker="s", s=46, c="#1565C0"), "base": dict(marker="o", s=52, c="#6D4C41"),
                  "ctl": dict(marker="x", s=46, c="#757575")}[kind]
            ax.scatter(x, y, edgecolors="black" if kind != "ctl" else None, linewidths=0.5, **st)
            short = {"exec_P": "ReCheck", "exec_A0": "A0", "exec_A1": "A1", "exec_A2": "A2", "exec_PX": "PX",
                     "generic_judge2": "B1", "generic_tool_judge": "B2", "generic_second_look": "B2'",
                     "contract": "B3", "pt_audit": "Pratama", "zandi_judge": "Zandi", "mermigkis_verifier": "Mermigkis",
                     "pbp_judge": "PbP", "risk_gate": "Risk gate", "ws_cascade_human": "Cascade (human)",
                     "ws_cascade_grok": "Cascade (Grok)", "always_accept": "accept all"}[m]
            ax.annotate(short, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Wrong explanations accepted (%)")
        ax.grid(alpha=0.3)
        ax.set_xlim(-3, 103)
    axes[0].set_ylabel("Correct explanations released (%)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    # access figure
    acc = N["access"]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    labels, gaps_hc, gaps_tw = [], [], []
    for k in ("generic_judge2|qwen3_next", "generic_tool_judge|qwen3_next", "exec_A0|qwen3_next", "exec_P|qwen3_next",
              "generic_judge2|grok45", "generic_tool_judge|grok45", "exec_P|grok45"):
        if k not in acc:
            continue
        m, j = k.split("|")
        nm = {"generic_judge2": "B1", "generic_tool_judge": "B2", "exec_A0": "A0", "exec_P": "ReCheck"}[m]
        labels.append(nm if m == "exec_A0" else f"{nm}\n{'Qwen' if j == 'qwen3_next' else 'Grok'}")
        for ds, store in (("home_credit", gaps_hc), ("taiwan", gaps_tw)):
            wf, nf = acc[k][f"{ds}|female"]
            wm, nm_ = acc[k][f"{ds}|male"]
            store.append(100 * (wm / nm_ - wf / nf))
    import numpy as np
    xs = np.arange(len(labels))
    ax.bar(xs - 0.18, gaps_hc, 0.36, label="Home Credit", color="#5C6BC0")
    ax.bar(xs + 0.18, gaps_tw, 0.36, label="Taiwan", color="#26A69A")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Withheld correct explanations,\nmale minus female (points)", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_access.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    main_res, truth = main_table()
    paired_table()
    access_table(truth)
    heldout_tables()
    objection_outcomes()
    annotation_numbers()
    figures(main_res, truth)
    (ESWA / "paper_numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print("tables:", sorted(p.name for p in TAB.iterdir()))
    print("figures:", sorted(p.name for p in FIG.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

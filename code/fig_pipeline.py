"""Figure 1 of the ESWA paper: the ReCheck pipeline with the dispute state (vector PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/home/Debz/Hackathon/IDBI_Hackathon/Submission/ESWA_Submission/figures/fig_pipeline.pdf"
fig, ax = plt.subplots(figsize=(10.5, 5.9))
ax.set_xlim(0, 105); ax.set_ylim(-3, 56); ax.axis("off")
C = {"in": "#E8EEF7", "code": "#E3F1E3", "llm": "#FFF1D6", "read": "#E0F2F1", "out": "#F3E5F5", "edge": "#37474F"}


def box(x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=C[kind], ec=C["edge"], lw=0.9))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=8.4, weight="bold")
    ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=6.9, linespacing=1.22)


def arrow(x1, y1, x2, y2, label=None, ly=0.0, lx=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=C["edge"]))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="bottom", fontsize=6.7, style="italic")


# row 1: code layer
box(1, 38, 16, 16, "Inputs", "evidence: facts,\nattributions, PD\n+ LLM explanation", "in")
box(21, 38, 18, 16, "1  Claims", "validate evidence;\nsplit the text into\nclaims (own splitter)", "code")
box(43, 38, 24, 16, "2  Execute claims", "VALUE, DIRECTION, RANK,\nSCALE, TARGET, TEMPORAL,\nCODE, FACT_DRIVER,\nCOVERAGE, COMPLETENESS", "code")
box(72, 44, 14, 10, "Reject", "executed\ncontradiction", "out")
box(90, 44, 14, 10, "Release", "every claim\nsupported", "out")
arrow(17.6, 46, 20.4, 46)
arrow(39.6, 46, 42.4, 46)
arrow(67.6, 50, 71.4, 50, "contradiction", 0.3)
ax.plot([67.6, 97], [40.5, 40.5], color=C["edge"], lw=0.9)
arrow(97, 40.5, 97, 43.6)
ax.text(82, 38.0, "all supported (no model call)", ha="center", fontsize=6.7, style="italic")
# row 2: judge and witnesses
box(43, 19, 24, 15, "3  Judge on residual claims", "every objection is a witness:\nrule, operation, quote,\nevidence ids, claimed value", "llm")
box(72, 19, 14, 15, "4  Execute\nwitnesses", "\nverified | disproved |\nmalformed |\nunexecutable", "code")
box(90, 24, 14, 10, "Reject", "witness\nverified", "out")
arrow(55, 37.4, 55, 34.6, "residual", 0.0, 4.0)
arrow(67.6, 26.5, 71.4, 26.5)
arrow(86.6, 29, 89.4, 29)
# row 3: dispute state
box(43, 1, 24, 15, "5  Dispute?", "objection not confirmed,\nclaim not supported by\nthe judge, or disputed\nomission", "code")
box(72, 1, 14, 15, "6  Independent\nreader", "\nother publisher; sees\nneither verdict; returns\nits own binding,\nexecuted by code", "read")
box(90, 1, 14, 18, "Decide", "no dispute: release\nD: defer\nR: reading supported\n= release; contradicted\n= reject; else defer", "out")
arrow(76, 18.4, 66.0, 16.4, "not confirmed", 0.2, 3.5)
arrow(55, 18.4, 55, 16.6)
ax.text(54, 17.0, "judge statuses", fontsize=6.7, style="italic", ha="right")
arrow(67.6, 8.5, 71.4, 8.5, "dispute (R)", 0.3)
arrow(86.6, 8.5, 89.4, 8.5)
ax.plot([55, 55, 97], [0.4, -1.6, -1.6], color=C["edge"], lw=0.9)
arrow(97, -1.6, 97, 0.4)
ax.text(76, -1.3, "no dispute: release; policy D: defer", fontsize=6.7, style="italic", ha="center", va="bottom")
# legend
for y, k, t in ((28, "code", "deterministic code"), (22.5, "llm", "LLM judge (one call)"),
                (17, "read", "independent reader (disputes only)"), (11.5, "out", "decision")):
    ax.add_patch(FancyBboxPatch((1.2, y), 3.0, 2.2, boxstyle="round,pad=0.1", fc=C[k], ec=C["edge"], lw=0.6))
    ax.text(5.6, y + 1.1, t, fontsize=7.2, va="center")
fig.savefig(OUT, bbox_inches="tight")
print(OUT)

"""Section 5 figures from MEASURED results (numbers5.json, analysis5.json) and real detections on real images.
Palette: Okabe–Ito (CVD-safe, validated)."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).parent
N = json.load(open(HERE / "numbers5.json")); AN = json.load(open(HERE / "analysis5.json"))
BLUE, ORANGE, GREEN, VERM, PURP, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444", "axes.linewidth": 0.6, "xtick.color": "#333", "ytick.color": "#333", "axes.grid": True,
                     "grid.color": "#e5e5e5", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False})
def save(fig, name): fig.savefig(HERE / name, dpi=220, bbox_inches="tight"); plt.close(fig); print("saved", name)
def lab(ax, bars, fmt="%.1f", dy=0.15, size=6.8):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt % b.get_height(), ha="center", va="bottom", fontsize=size, color="#222")

# ---------------------------------------------------------------- Fig 6: the NMS-free deficit and its recovery (A, B, C)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
bases = ["A", "B", "C"]; names = ["A: YOLO26s", "B: YOLO26s-P2", "C: B + VW"]
outs = [("nms", "one-to-many head + NMS", GREY), ("free", "one-to-one head (NMS-free)", ORANGE), ("slr", "one-to-one + SLR (NMS-free)", BLUE)]
def val(b, o, g, m):
    if o == "slr" and b == "C": return N["C"]["slrv"][g]["ap" if m == 0 else "mr"]      # C: head with visibility branch (3 seeds)
    return N[b][o][g][m]
for ax, (m, title, yl) in zip(axes, [(0, "(a) AP on All (higher is better)", (84, 92)), (1, "(b) MR⁻² on All (lower is better)", (44, 56))]):
    x = np.arange(3); w = 0.26
    for k, (o, lbl, col) in enumerate(outs):
        v = [val(b, o, "all", m) for b in bases]
        bars = ax.bar(x + (k - 1) * w, v, w * 0.92, color=col, label=lbl, linewidth=0); lab(ax, bars, "%.2f", 0.08, 6.3)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(*yl); ax.set_title(title, fontsize=9, loc="left"); ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.06))
fig.text(0.5, -0.13, "SLR = Second-Look re-scoring head; on C the head includes the visibility branch (mean of 3 seeds).", ha="center", fontsize=7, color="#555")
save(fig, "fig6_nmsfree_gap.png")

# ---------------------------------------------------------------- Fig 7: per-occlusion-subset results (one-to-one outputs)
rows = [("A", "A", "free"), ("B", "B", "free"), ("C", "C", "free"), ("A + SLR", "A", "slr"), ("B + SLR", "B", "slr"), ("B + SLR-V", "B", "slrv"), ("C + SLR-V", "C", "slrv")]
cols = [GREY, "#bbbbbb", "#dddddd", SKY, BLUE, PURP, GREEN]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
subs = ["light", "heavy", "severe"]; subn = ["Light\n(v ≥ 0.65)", "Heavy\n(0.30 ≤ v < 0.65)", "Severe\n(v < 0.30)"]
for ax, (m, title, yl) in zip(axes, [(0, "(a) AP per visibility subset", (70, 96)), (1, "(b) MR⁻² per visibility subset", (25, 56))]):
    x = np.arange(3); w = 0.115
    for k, (lbl, b, o) in enumerate(rows):
        v = [(N[b][o][g]["ap" if m == 0 else "mr"] if o == "slrv" else N[b][o][g][m]) for g in subs]
        bars = ax.bar(x + (k - 3) * w, v, w * 0.9, color=cols[k], label=lbl, linewidth=0)
        if lbl in ("B", "B + SLR"): lab(ax, bars, "%.1f", 0.2, 5.8)
    ax.set_xticks(x); ax.set_xticklabels(subn, fontsize=7.5); ax.set_ylim(*yl); ax.set_title(title, fontsize=9, loc="left"); ax.grid(axis="x", visible=False)
h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=7, fontsize=6.8, bbox_to_anchor=(0.5, -0.1), handlelength=1.2, columnspacing=1.0)
save(fig, "fig7_subsets.png")

# ---------------------------------------------------------------- Fig 8: what the head does (score shifts, FP composition)
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), gridspec_kw={"wspace": 0.42})
bins = np.array(AN["score_shift_bins"]); ctr = (bins[:-1] + bins[1:]) / 2
ax = axes[0]
for c, lbl, col in [("tp_hi", "true positives", BLUE), ("dup_hi", "duplicates", VERM), ("bg_hi", "background", GREY)]:
    h = np.array(AN["score_shift"][c]["hist"], float); h = h / h.sum()
    ax.plot(ctr, h, color=col, lw=1.6, label=f'{lbl} (n = {AN["score_shift"][c]["n"]:,})')
    ax.axvline(AN["score_shift"][c]["median"], color=col, lw=0.8, ls="--")
ax.set_xlim(-4, 3); ax.set_ylim(-0.01, 0.37); ax.set_xlabel("score change by the head, Δ logit"); ax.set_ylabel("fraction of candidates"); ax.legend(fontsize=6.3, loc="upper left", frameon=True, framealpha=1, edgecolor="none", facecolor="white")
ax.text(0.5, -0.36, "(a) Δ for confident candidates (s ≥ 0.3)", transform=ax.transAxes, ha="center", va="top", fontsize=9)
ax = axes[1]
vb = [">=0.65", "0.3-0.65", "<0.3"]; vn = ["v ≥ 0.65\n(Light)", "0.3 ≤ v < 0.65\n(Heavy)", "v < 0.3\n(Severe)"]
med = [AN["score_shift_by_vis"][k]["median"] for k in vb]; q1 = [AN["score_shift_by_vis"][k]["q25"] for k in vb]; q3 = [AN["score_shift_by_vis"][k]["q75"] for k in vb]
bars = ax.bar(range(3), med, 0.55, color=[SKY, BLUE, "#004c7a"], linewidth=0)
ax.errorbar(range(3), med, yerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)], fmt="none", ecolor="#333", elinewidth=0.8, capsize=3)
lab(ax, bars, "%.2f", 0.02, 7); ax.set_xticks(range(3)); ax.set_xticklabels(vn, fontsize=7); ax.set_ylabel("Δ logit (median, IQR)"); ax.grid(axis="x", visible=False)
ax.text(0.5, -0.36, "(b) true positives, by visibility", transform=ax.transAxes, ha="center", va="top", fontsize=9); ax.set_ylim(-0.6, 1.4)
ax = axes[2]
cats = [("duplicate", "duplicate of a detected person", VERM), ("spanning_two", "box spanning two people", ORANGE), ("near_ignore", "inside / near ignore region", GREY), ("background", "other background", "#cccccc")]
outs3 = [("free", "one-to-one"), ("slr", "+ SLR"), ("nms", "NMS")]
bottom = np.zeros(3)
for key, lbl, col in cats:
    v = np.array([N["errors"][o]["FP_pct"][key] for o, _ in outs3]); ax.bar(range(3), v, 0.55, bottom=bottom, color=col, label=lbl, linewidth=0.5, edgecolor="white")
    for i in range(3):
        if v[i] >= 8: ax.text(i, bottom[i] + v[i] / 2, f"{v[i]:.0f}%", ha="center", va="center", fontsize=6.5, color="white" if col in (VERM, GREY) else "#222")
    bottom += v
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in outs3], fontsize=7.5); ax.set_ylabel("share of FP at 1 FPPI (%)"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
ax.legend(fontsize=6.3, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1); ax.text(0.5, -0.36, "(c) composition of the false positives", transform=ax.transAxes, ha="center", va="top", fontsize=9)
save(fig, "fig8_head_behaviour.png")

# ---------------------------------------------------------------- Fig 9: MR-FPPI curves + crowd density
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), gridspec_kw={"wspace": 0.38})
for ax, key, title in [(axes[0], "curve_mr", "(a) All"), (axes[1], "curve_mr_severe", "(b) Severe (v < 0.30)")]:
    for o, lbl, col in [("nms", "one-to-many + NMS", GREY), ("free", "one-to-one (NMS-free)", ORANGE), ("slr", "one-to-one + SLR", BLUE)]:
        ax.plot(AN[o]["curve_fppi"], AN[o][key], color=col, lw=1.6, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, 1e0); ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.64, 0.8, 1.0]); ax.set_yticklabels(["0.20", "0.30", "0.40", "0.50", "0.64", "0.80", "1.00"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("false positives per image"); ax.set_ylabel("miss rate"); ax.text(0.5, -0.33, title, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.axvspan(1e-2, 1e0, color="#f4f4f4", zorder=0)
axes[0].legend(fontsize=6.5, loc="lower left")
ax = axes[2]; d = N["density"]; x = np.arange(4); w = 0.26
for k, (o, lbl, col) in enumerate([("nms", "one-to-many + NMS", GREY), ("free", "one-to-one", ORANGE), ("slr", "one-to-one + SLR", BLUE)]):
    v = [100 * r["recall"] for r in d[o]]; bars = ax.bar(x + (k - 1) * w, v, w * 0.92, color=col, label=lbl, linewidth=0); lab(ax, bars, "%.0f", 0.3, 6)
ax.set_xticks(x); ax.set_xticklabels([f'{r["range"][0]}–{r["range"][1] - 1}' if r["range"][1] < 400 else f'≥ {r["range"][0]}' for r in d["slr"]], fontsize=7.5)
ax.set_xlabel("annotated persons per image"); ax.set_ylabel("recall at 1 FPPI (%)"); ax.set_ylim(40, 100); ax.grid(axis="x", visible=False)
ax.text(0.5, -0.33, "(c) recall at 1 FPPI vs. crowd size", transform=ax.transAxes, ha="center", va="top", fontsize=9); ax.legend(fontsize=6.3, loc="upper right", ncol=1)
save(fig, "fig9_curves_density.png")
print("done charts")

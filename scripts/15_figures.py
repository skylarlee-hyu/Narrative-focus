"""Figures 1-5 of the manuscript, from the tables written by scripts 02, 03 and 14."""
import _setup  # noqa: F401
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from finsent import config as C

out = C.FIGURES
H = [1, 3, 5, 10, 30]
LEADER, FIRM, TECH, NONTECH, GOLD, GRAY, INK = "#3E5A76", "#B5654A", "#B5654A", "#5B8A72", "#C8974A", "#9AA5B1", "#2B2F36"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#4A4F57", "axes.labelcolor": INK,
                     "xtick.color": "#5A606A", "ytick.color": "#5A606A", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.9,
                     "savefig.dpi": 150})
LABEL = {"ORDINARY": "Ordinary sample", "FULL": "Full sample"}
c1 = pd.read_csv(C.TABLES / "supp_S1_pairwise_contrasts.csv")
c2 = pd.read_csv(C.TABLES / "appendix_C1_strategy_returns.csv").rename(columns={"t_excess": "t_ex", "p_excess": "p_ex"})
f1 = pd.read_csv(C.TABLES / "figure1_mean_bhar_paths.csv").rename(columns={"Sample": "sample"})
f1 = f1.rename(columns={f"bhar_t{h}": f"excess_ret_t{h}" for h in [1, 3, 5, 10, 30]})


def star(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def sel(sample, comp, firm, cond):
    return c1[(c1.Sample == sample) & (c1.Comparison == comp) & (c1.Firm == firm) & (c1.Cond == cond)]


def mark(ax, xs, ys, ps, color, below=True):
    for x, y, p in zip(xs, ys, ps):
        s = star(p)
        if s:
            ax.annotate(s, (x, y), textcoords="offset points", xytext=(0, -14 if below else 7), ha="center",
                        color=color, fontsize=10, fontweight="bold")


# Figure 1: mean BHAR paths, 2 x 2
fig, axes = plt.subplots(2, 2, figsize=(10.48, 7.17), sharey="row")
for r, ft in enumerate(["Tech", "Non-Tech"]):
    for c, smp in enumerate(["ORDINARY", "FULL"]):
        ax = axes[r, c]
        for fr, col in [("Leader", LEADER), ("Firm", FIRM)]:
            for tone, ls in [("Positive", "-"), ("Negative", "--")]:
                row = f1[(f1["sample"] == smp) & (f1.ftype == ft) & (f1.frame == fr) & (f1.tone == tone)].iloc[0]
                ax.plot(range(5), [row[f"excess_ret_t{h}"] for h in H], ls, color=col, lw=2.2, marker="o", ms=4.5)
        ax.axhline(0, color="#3A3F46", lw=0.9); ax.margins(y=0.12)
        ax.set_xticks(range(5)); ax.set_xticklabels(H)
        ax.set_title(f"{'Technology' if ft == 'Tech' else 'Non-technology'} · {LABEL[smp]}", fontweight="bold", color=INK)
        if c == 0: ax.set_ylabel("Mean BHAR")
        if r == 1: ax.set_xlabel("Horizon (trading days)")
handles = [plt.Line2D([], [], color=LEADER, lw=2.5), plt.Line2D([], [], color=FIRM, lw=2.5),
           plt.Line2D([], [], color=INK, lw=2.5), plt.Line2D([], [], color=INK, lw=2.5, ls="--")]
fig.legend(handles, ["Leader-centered", "Firm-centered", "Positive news", "Negative news"], loc="upper center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out / "figure1_bhar_paths.png"); plt.close(fig)

# Figure 3: Leader - Firm, technology firms, ordinary vs full sample, by valence
fig, axes = plt.subplots(1, 2, figsize=(10.67, 4.55), sharey=True)
for ax, tone in zip(axes, ["Positive", "Negative"]):
    for smp, col, mk, lab in [("ORDINARY", LEADER, "o", "Ordinary sample"), ("FULL", GRAY, "s", "Full sample")]:
        d = sel(smp, "Leader vs Firm", "Tech", tone)
        ax.plot(H, d.MeanDiff, color=col, lw=2.5, marker=mk, ms=7, label=lab)
        if smp == "ORDINARY":
            mark(ax, H, d.MeanDiff, d.p, col)
    ax.axhline(0, color="#3A3F46", lw=0.9); ax.margins(y=0.12); ax.set_xticks(H)
    ax.set_xlabel("Horizon (trading days)"); ax.set_title(f"{tone} news", fontweight="bold", color=INK)
axes[0].set_ylabel("Leader − Firm  mean BHAR")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False); fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out / "figure3_ordinary_vs_full_sample.png"); plt.close(fig)

# Figure 2: Leader - Firm by valence, ordinary sample, technology vs non-technology
fig, axes = plt.subplots(1, 2, figsize=(10.67, 4.55), sharey=True)
for ax, tone in zip(axes, ["Positive", "Negative"]):
    for ft, col, mk in [("Tech", TECH, "o"), ("Non-Tech", NONTECH, "s")]:
        d = sel("ORDINARY", "Leader vs Firm", ft, tone)
        ax.plot(H, d.MeanDiff, color=col, lw=2.5, marker=mk, ms=7, label={"Tech": "Technology", "Non-Tech": "Non-technology"}[ft])
        mark(ax, H, d.MeanDiff, d.p, col, below=(ft == "Tech"))
    ax.axhline(0, color="#3A3F46", lw=0.9); ax.margins(y=0.12); ax.set_xticks(H); ax.set_xlabel("Horizon (trading days)")
    ax.set_title(f"{tone} news", fontweight="bold", color=INK)
axes[0].set_ylabel("Leader − Firm  mean BHAR")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False); fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out / "figure2_leader_minus_firm_by_valence.png"); plt.close(fig)

# Figure 4: strategy returns, ordinary vs full sample
fig, axes = plt.subplots(1, 2, figsize=(10.67, 4.70), sharey=True)
styles = [("Firm", "Tech", FIRM, "-", "o"), ("Firm", "Non-Tech", NONTECH, "-", "o"),
          ("Leader", "Tech", LEADER, "--", "s"), ("Leader", "Non-Tech", GOLD, "--", "s")]
for ax, smp in zip(axes, ["ORDINARY", "FULL"]):
    for fr, ft, col, ls, mk in styles:
        d = c2[(c2.Sample == smp) & (c2.Framing == fr) & (c2.Firm == ft)]
        ax.plot(H, d.Excess, ls, color=col, lw=2.5, marker=mk, ms=6, label=f"{fr}-centered · {'Technology' if ft == 'Tech' else 'Non-technology'}")
        mark(ax, H, d.Excess, d.p_ex, col, below=(fr == "Leader"))
    ax.axhline(0, color="#3A3F46", lw=0.9); ax.margins(y=0.12); ax.set_xticks(H); ax.set_xlabel("Holding horizon (trading days)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.set_title(LABEL[smp], fontweight="bold", color=INK)
axes[0].set_ylabel("Simulated strategy return (market-adjusted)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False); fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(out / "figure4_strategy_returns.png"); plt.close(fig)
# Figure 5: dose-response, ordinary sample, valence rule of Section 2.2
b = pd.read_csv(C.TABLES / "figure5_dose_response_bins.csv")
sl = pd.read_csv(C.TABLES / "appendix_C2_dose_response_slopes.csv")
POS, NEG = INK, GOLD
fig, axes = plt.subplots(2, 5, figsize=(14.4, 6.9), sharey="row")
for r, ft in enumerate(["Tech", "Non-Tech"]):
    for c, h in enumerate(H):
        ax = axes[r, c]
        for k, (tone, ls, col) in enumerate([("Positive", "-", POS), ("Negative", "--", NEG)]):
            d = b[(b.ftype == ft) & (b.tone == tone) & (b.horizon == h)]
            ax.plot(d["bin"], d["mean"], ls, color=col, marker="o", ms=4, lw=1.8, label=f"{tone} news")
            ax.fill_between(d["bin"], d["mean"] - 1.96 * d["sem"], d["mean"] + 1.96 * d["sem"], color=col, alpha=.12)
            s_ = sl[(sl.ftype == ft) & (sl.tone == tone) & (sl.horizon == f"t+{h}")].iloc[0]
            big = s_.p < .01
            ax.text(.03, .05 + .09 * k, f"{tone[:3].lower()}. slope {100 * s_.slope:.2f}{star(s_.p)}", transform=ax.transAxes,
                    color=col, fontsize=9 if big else 8, fontweight="bold" if big else "normal")
        ax.axhline(0, color="#3A3F46", lw=0.8); ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        if r == 0: ax.set_title(f"t+{h}", fontweight="bold", color=INK)
        if c == 0: ax.set_ylabel(f"{'Technology' if ft == 'Tech' else 'Non-technology'}\nMean BHAR")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False)
fig.supxlabel("Leader article share of the firm-day"); fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out / "figure5_dose_response.png"); plt.close(fig)
print("figures written")

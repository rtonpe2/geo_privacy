"""Build every paper figure from results/*.csv so figures and text share one run.

Usage: python scripts/make_figures.py results/ paper/
Writes vector PDFs (used by main.tex) plus PNG previews.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots validated for CVD separation; low-contrast slots get direct
# labels + distinct markers so identity never depends on color alone.
SERIES = {
    "spatial_only":    dict(label="Spatial only",       color="#2a78d6", marker="o"),
    "cuisine_landuse": dict(label="Cuisine → land use", color="#eb6834", marker="s"),
    "cuisine_units":   dict(label="Cuisine → units",    color="#1baf7a", marker="^"),
    "cuisine_year":    dict(label="Cuisine → year built", color="#eda100", marker="D"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # bundled with matplotlib, so output matches on any machine
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
EPS_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def eps_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(EPS_TICKS)
    ax.set_xticklabels([f"{e:g}" for e in EPS_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("ε (ft$^{-1}$, log scale)")


def wilson(p, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_single_query(res, out):
    df = pd.read_csv(res / "single_query.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for cfg, st in SERIES.items():
        s = df[df.config == cfg].sort_values("epsilon")
        ax.plot(s.epsilon, s.reid_rate * 100, color=st["color"], marker=st["marker"], label=st["label"])
    # Direct labels: the land-use line separates; the other three overlap, so one label covers them.
    last = df[df.epsilon == df.epsilon.max()].set_index("config").reid_rate * 100
    ax.annotate("Cuisine → land use", (0.05, last["cuisine_landuse"]), xytext=(7, 0),
                textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ax.annotate("Spatial only, units,\nyear built (overlap)", (0.05, last["spatial_only"]), xytext=(7, -4),
                textcoords="offset points", fontsize=7.5, color=INK, va="center")
    eps_axis(ax)
    ax.set_xlim(0.0008, 0.13)
    ax.set_ylabel("Re-identification rate (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=False)
    save(fig, out, "fig_single_query")


def fig_multi_record(res, out, eps=0.01):
    single = pd.read_csv(res / "single_query.csv")
    multi = pd.read_csv(res / "multi_record.csv")
    n_visits = int(pd.read_csv(res / "data_summary.csv", index_col=0).loc["test_visits", "value"])
    multi = multi[multi.epsilon == eps].copy()
    multi["bucket"] = multi.n_obs.where(multi.n_obs <= 4, 5)
    multi["hits"] = multi.reid_rate * multi.n_restaurants
    pooled = multi.groupby(["config", "bucket"]).agg(hits=("hits", "sum"), n=("n_restaurants", "sum")).reset_index()
    pooled["rate"] = pooled.hits / pooled.n

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    offsets = {"spatial_only": -0.06, "cuisine_landuse": 0.06}
    for cfg in ["spatial_only", "cuisine_landuse"]:
        st = SERIES[cfg]
        p1 = single[(single.config == cfg) & (single.epsilon == eps)].reid_rate.iloc[0]
        xs, ps, ns = [1], [p1], [n_visits]
        sub = pooled[pooled.config == cfg].sort_values("bucket")
        xs += sub.bucket.tolist(); ps += sub.rate.tolist(); ns += sub.n.tolist()
        xs = np.array(xs) + offsets[cfg]; ps = np.array(ps); ns = np.array(ns)
        lo, hi = wilson(ps, ns)
        ax.errorbar(xs, ps * 100, yerr=[(ps - lo) * 100, (hi - ps) * 100], color=st["color"],
                    marker=st["marker"], capsize=2.5, elinewidth=1, label=st["label"])
    counts = pooled[pooled.config == "spatial_only"].sort_values("bucket").n.tolist()
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([f"1\n({n_visits:,} visits)"] +
                       [f"{b}\n({c:,})" for b, c in zip(["2", "3", "4", "5–8"], counts)])
    ax.set_xlabel("Linked inspections per restaurant (records attacked)")
    ax.set_ylabel("Re-identification rate (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_multi_record")


def fig_utility(res, out):
    sc = pd.read_csv(res / "utility_score.csv").sort_values("epsilon")
    cu = pd.read_csv(res / "utility_cuisine.csv").sort_values("epsilon")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    a = axes[0]
    a.plot(sc.epsilon, sc.rmse, color="#2a78d6", marker="o", label="RMSE")
    a.plot(sc.epsilon, sc.mae, color="#eb6834", marker="s", label="MAE")
    a.set_ylabel("Error in tract mean score (points)")
    a.set_title("Inspection score distortion", loc="left")
    a = axes[1]
    a.plot(cu.epsilon, cu.tvd_mean, color="#2a78d6", marker="o", label="Mean TVD")
    a.plot(cu.epsilon, cu.tvd_median, color="#eb6834", marker="s", label="Median TVD")
    a.set_ylabel("Total variation distance")
    a.set_title("Cuisine composition distortion", loc="left")
    for a in axes:
        eps_axis(a)
        a.set_ylim(0, None)
        a.legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "fig_utility")


def fig_tradeoff(res, out):
    sq = pd.read_csv(res / "single_query.csv")
    sc = pd.read_csv(res / "utility_score.csv").set_index("epsilon")
    cu = pd.read_csv(res / "utility_cuisine.csv").set_index("epsilon")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for a, (util, col, xlabel) in zip(axes, [(sc, "rmse", "Inspection score distortion (RMSE, points)"),
                                             (cu, "tvd_mean", "Cuisine composition distortion (mean TVD)")]):
        for cfg in ["spatial_only", "cuisine_landuse"]:
            st = SERIES[cfg]
            s = sq[sq.config == cfg].sort_values("epsilon")
            x = util.loc[s.epsilon, col].values
            a.plot(x, s.reid_rate * 100, color=st["color"], marker=st["marker"], label=st["label"])
            if cfg == "cuisine_landuse":
                for e, xi, yi in zip(s.epsilon, x, s.reid_rate * 100):
                    off = (2, 11) if e == 0.002 else (5, 4)
                    a.annotate(f"ε={e:g}", (xi, yi), xytext=off, textcoords="offset points",
                               fontsize=7, color=MUTED)
        a.set_xlabel(xlabel)
        a.set_ylim(0, None)
    axes[0].set_ylabel("Re-identification rate (%)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig_tradeoff")


if __name__ == "__main__":
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    fig_single_query(res, out)
    fig_multi_record(res, out)
    fig_utility(res, out)
    fig_tradeoff(res, out)
    print("figures written to", out)

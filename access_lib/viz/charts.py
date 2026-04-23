"""
viz/charts.py — Statistical charts. All labels in English.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
except ImportError as e:
    raise ImportError(f"Install matplotlib: {e}")

from .style import (
    NAVY, BLUE, BLUE_SOFT, ORANGE, TEAL, RED_ACC, GREEN_ACC,
    GRAY_10, GRAY_30, GRAY_60, GRAY_80, WHITE,
    SPEC_STYLES, GROUP_COLORS, MODE_COLORS,
    CMAP_ACCESS, CMAP_HEAT,
    FIG_CHART, FIG_WIDE, FIG_SQUARE,
    label_bars, savefig,
)

_MODE_LABEL = {"car": "Car", "walk": "Walking", "pt": "Public Transit"}


# ─── Lorenz curve ─────────────────────────────────────────────────────────────
def lorenz_curve(
    acc_arrays: Dict[str, np.ndarray],
    demand: Optional[np.ndarray] = None,
    title: str = "Lorenz Curves — Accessibility Inequality",
    figsize: Tuple = FIG_SQUARE,
    colors: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    colors = colors or [BLUE, ORANGE, TEAL, RED_ACC, GREEN_ACC, BLUE_SOFT]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0,1],[0,1], color=GRAY_60, lw=1.2, ls="--", label="Perfect equality", zorder=1)
    ax.fill_between([0,1],[0,1],0, color=GRAY_10, alpha=0.4, zorder=0)

    # Pre-compute all Gini values to place annotations without overlaps
    gini_vals = {}
    curve_data = {}
    for label, arr in acc_arrays.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a) & (a >= 0)]
        if demand is not None:
            d = np.asarray(demand, dtype=float)[:len(a)]
            idx = np.argsort(a)
            a_s, d_s = a[idx], d[idx]
            cum_d = np.cumsum(d_s) / d_s.sum()
            cum_a = np.cumsum(a_s * d_s) / (a_s * d_s).sum() if (a_s*d_s).sum() > 0 else np.zeros_like(cum_d)
            x = np.concatenate([[0], cum_d])
            y = np.concatenate([[0], cum_a])
        else:
            a_s = np.sort(a)
            x = np.linspace(0, 1, len(a_s)+1)
            y = np.concatenate([[0], np.cumsum(a_s)/a_s.sum()]) if a_s.sum() > 0 else np.zeros(len(a_s)+1)
        gv = float(1 - 2*np.trapz(y, x))
        gini_vals[label] = gv
        curve_data[label] = (x, y)

    for i, (label, (x, y)) in enumerate(curve_data.items()):
        color = colors[i % len(colors)]
        gv = gini_vals[label]
        ax.plot(x, y, color=color, lw=2.0, label=f"{label}  (G={gv:.3f})", zorder=i+2)

    # ── Gini values: left column, stacked vertically ──
    for i, (label, gv) in enumerate(gini_vals.items()):
        color = colors[i % len(colors)]
        ax.text(0.03, 0.73 - i * 0.065, f"G = {gv:.3f}",
                transform=ax.transAxes, fontsize=10, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                          edgecolor=color, alpha=0.92, lw=1.5),
                zorder=20)

    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Cumulative population share (least → most accessible)", fontsize=10)
    ax.set_ylabel("Cumulative accessibility share", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, edgecolor=GRAY_30)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


# ─── Scenario bar chart ────────────────────────────────────────────────────────
def scenario_bars(
    summary_df: pd.DataFrame,
    metric: str = "mean",
    ci_col: Optional[str] = None,
    title: str = "Scenario Comparison",
    figsize: Tuple = FIG_CHART,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    df = summary_df.copy()
    n  = len(df)
    scenarios = df["scenario"].tolist()
    baseline_val = float(df.iloc[0][metric])

    colors = []
    for _, row in df.iterrows():
        v = float(row[metric])
        if row["scenario"] == scenarios[0]: colors.append(GRAY_60)
        elif v > baseline_val + 1e-9:       colors.append(GREEN_ACC)
        elif v < baseline_val - 1e-9:       colors.append(RED_ACC)
        else:                               colors.append(BLUE_SOFT)

    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios":[3,1], "hspace":0.08},
    )
    x = np.arange(n)
    bars = ax_main.bar(x, df[metric], color=colors, edgecolor=WHITE, lw=0.5, width=0.65, zorder=2)
    label_bars(ax_main, bars, fmt="{:.5f}", fontsize=7.5)
    ax_main.axhline(baseline_val, color=GRAY_60, lw=1.2, ls=":", label=f"Baseline ({baseline_val:.5f})")
    if ci_col and ci_col in df.columns:
        ax_main.errorbar(x, df[metric], yerr=df[ci_col]*1.96, fmt="none",
                         color=NAVY, lw=1.2, capsize=4, capthick=1, zorder=3)
    ax_main.set_xticks(x); ax_main.set_xticklabels([])
    ax_main.set_ylabel(_metric_label(metric), fontsize=10)
    ax_main.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax_main.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_main.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))

    deltas = [float(df.iloc[i][metric]) - baseline_val for i in range(n)]
    d_colors = [GREEN_ACC if d > 0 else (RED_ACC if d < 0 else GRAY_60) for d in deltas]
    ax_delta.bar(x, deltas, color=d_colors, edgecolor=WHITE, lw=0.5, width=0.65, zorder=2)
    ax_delta.axhline(0, color=GRAY_60, lw=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(scenarios, rotation=22, ha="right", fontsize=9)
    ax_delta.set_ylabel("Δ vs baseline", fontsize=10)
    ax_delta.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.5f"))
    for xi, d in zip(x, deltas):
        if abs(d) > 1e-9:
            ax_delta.text(xi, d + np.sign(d)*abs(d)*0.05, f"{d:+.5f}",
                          ha="center", va="bottom" if d>0 else "top",
                          fontsize=7, color=d_colors[x.tolist().index(xi)])
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


# ─── Sensitivity heatmap (single mode) ────────────────────────────────────────
def sensitivity_heatmap(
    grid_df: pd.DataFrame, metric_col: str = "gini",
    beta_col: str = "beta", radius_col: str = "radius_min",
    title: str = "E2SFCA Sensitivity",
    cmap: str = CMAP_HEAT, figsize: Tuple = FIG_SQUARE,
    highlight: Optional[Tuple[float,float]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    pivot = grid_df.pivot(index=beta_col, columns=radius_col, values=metric_col)
    pivot = pivot.sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                   vmin=pivot.values.min(), vmax=pivot.values.max())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i,j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color=WHITE if v > pivot.values.mean() else NAVY)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{r} min" for r in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"β={b:.1f}" for b in pivot.index], fontsize=8)
    ax.set_xlabel("Catchment radius", fontsize=10)
    ax.set_ylabel("Decay parameter β", fontsize=10)
    ax.set_title(f"{title}\nMetric: {metric_col}", fontsize=11, fontweight="bold",
                 color=NAVY, pad=10)
    if highlight:
        b_list = list(pivot.index)[::-1]
        r_list = list(pivot.columns)
        try:
            ri = b_list.index(highlight[0])
            ci = r_list.index(highlight[1])
            from matplotlib.patches import Rectangle as _Rect
            rect = _Rect((ci-0.5, ri-0.5), 1, 1,
                         linewidth=3.0, edgecolor="#FFD700",
                         facecolor="none", zorder=10)
            ax.add_patch(rect)
        except ValueError: pass
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label(metric_col, fontsize=9)
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


# ─── Sensitivity heatmap — ALL modes (3 columns) ──────────────────────────────
def sensitivity_heatmap_multimode(
    df_all: pd.DataFrame,
    metric_col: str = "gini",
    highlight_car:  Optional[Tuple[float,float]] = None,
    highlight_walk: Optional[Tuple[float,float]] = None,
    highlight_pt:   Optional[Tuple[float,float]] = None,
    figsize: Tuple = (18, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    highlights = {"car": highlight_car, "walk": highlight_walk, "pt": highlight_pt}
    modes = ["car", "walk", "pt"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"wspace":0.3})
    for ax, mode in zip(axes, modes):
        sub = df_all[df_all["mode"]==mode]
        if sub.empty: ax.set_visible(False); continue
        pivot = sub.pivot(index="beta", columns="radius_min", values=metric_col)
        pivot = pivot.sort_index(ascending=False)
        im = ax.imshow(pivot.values, aspect="auto", cmap=CMAP_HEAT,
                       vmin=pivot.values.min(), vmax=pivot.values.max())
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i,j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                        color=WHITE if v > pivot.values.mean() else NAVY)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{r} min" for r in pivot.columns], fontsize=7.5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"β={b:.1f}" for b in pivot.index], fontsize=7.5)
        ax.set_xlabel("Catchment radius", fontsize=9)
        ax.set_ylabel("Decay β", fontsize=9)
        color = MODE_COLORS.get(mode, NAVY)
        ax.set_title(f"{_MODE_LABEL.get(mode, mode)}\n({metric_col})",
                     fontsize=10, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax, shrink=0.85).set_label(metric_col, fontsize=8)
        hl = highlights.get(mode)
        if hl:
            b_list = list(pivot.index)[::-1]
            r_list = list(pivot.columns)
            try:
                ri = b_list.index(hl[0]); ci = r_list.index(hl[1])
                from matplotlib.patches import Rectangle as _Rect
                rect = _Rect((ci-0.5, ri-0.5), 1, 1,
                             linewidth=3.0, edgecolor="#FFD700",
                             facecolor="none", zorder=10)
                ax.add_patch(rect)
            except ValueError: pass
    fig.suptitle(f"E2SFCA Sensitivity Analysis — All Transport Modes\n(metric: {metric_col})",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.03)
    if save_path: savefig(fig, save_path)
    return fig


# ─── Monte Carlo violin ────────────────────────────────────────────────────────
def mc_violin(
    mc_results: Dict[str, np.ndarray],
    baseline_label: str = "S1_telemedicine",
    title: str = "Monte Carlo Uncertainty — Mean E2SFCA",
    figsize: Tuple = FIG_WIDE,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    scenarios = list(mc_results.keys())
    data = [mc_results[s] for s in scenarios]
    # Filter out arrays with all NaN
    valid = [(s, d) for s, d in zip(scenarios, data) if np.isfinite(d).any()]
    if not valid: return plt.figure()
    scenarios, data = zip(*valid)
    colors = [GRAY_60 if s == baseline_label else BLUE for s in scenarios]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, positions=range(len(scenarios)), showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_edgecolor(NAVY); pc.set_alpha(0.7)
    parts["cmedians"].set_color(WHITE); parts["cmedians"].set_linewidth(2.0)

    ax.boxplot(data, positions=range(len(scenarios)), widths=0.08, patch_artist=True,
               boxprops=dict(facecolor=WHITE, color=NAVY, lw=0.8),
               medianprops=dict(color=ORANGE, lw=1.5),
               whiskerprops=dict(color=GRAY_60, lw=0.8),
               capprops=dict(color=GRAY_60, lw=0.8),
               flierprops=dict(marker=".", ms=3, color=GRAY_60, alpha=0.5))

    for i, (s, d) in enumerate(zip(scenarios, data)):
        d_fin = d[np.isfinite(d)]
        if len(d_fin) == 0: continue
        lo, hi = np.percentile(d_fin, [2.5, 97.5])
        ax.text(i, hi+(hi-lo)*0.06, f"95% CI\n[{lo:.4f},{hi:.4f}]",
                ha="center", va="bottom", fontsize=6.5, color=GRAY_80)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Mean E2SFCA index", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=10)
    handles = [mpatches.Patch(facecolor=GRAY_60, edgecolor=NAVY, label="Baseline"),
               mpatches.Patch(facecolor=BLUE,    edgecolor=NAVY, label="Scenario"),
               Line2D([0],[0], color=ORANGE, lw=1.5, label="Median")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


# ─── Mode contribution ─────────────────────────────────────────────────────────
def mode_contribution(
    acc_by_mode: Dict[str, np.ndarray],
    mode_weights: Dict[str, float],
    demand: Optional[np.ndarray] = None,
    title: str = "Transport Mode Contribution to Composite E2SFCA",
    figsize: Tuple = (10, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    modes = list(acc_by_mode.keys())
    totals = {}
    for mode, arr in acc_by_mode.items():
        w = mode_weights.get(mode, 0)
        a = np.asarray(arr, dtype=float); a = a[np.isfinite(a)]
        totals[mode] = (float(np.average(a, weights=demand[:len(a)]) if demand is not None else np.mean(a))) * w
    total_sum = sum(totals.values())
    shares = {m: v/total_sum*100 for m,v in totals.items()} if total_sum > 0 else {m: 0 for m in totals}
    colors = [MODE_COLORS.get(m, BLUE) for m in modes]
    labels_en = [_MODE_LABEL.get(m, m) for m in modes]

    fig, (ax_abs, ax_pie) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace":0.35})
    x = np.arange(len(modes))
    bars = ax_abs.bar(x, [totals[m] for m in modes], color=colors, edgecolor=WHITE, lw=0.5)
    ax_abs.set_xticks(x); ax_abs.set_xticklabels(labels_en, fontsize=9)
    ax_abs.set_ylabel("Weighted contribution (w × μ E2SFCA)", fontsize=9)
    ax_abs.set_title("Absolute contribution by mode", fontsize=10, fontweight="bold", color=NAVY)
    label_bars(ax_abs, bars, fmt="{:.5f}", fontsize=8)

    share_vals = [shares[m] for m in modes]
    ax_pie.pie(
        share_vals,
        labels=[f"{_MODE_LABEL.get(m,m)}\n{v:.1f}%" for m,v in zip(modes,share_vals)],
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor=WHITE, lw=1.5),
        textprops=dict(fontsize=9, color=NAVY),
    )
    ax_pie.set_title("Share in composite index", fontsize=10, fontweight="bold", color=NAVY)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


# ─── OD histograms ─────────────────────────────────────────────────────────────
def od_histograms(
    od_matrices: Dict[str, np.ndarray],
    radii: Dict[str, float],
    title: str = "Travel Time Distributions (OD matrices)",
    figsize: Tuple = FIG_WIDE,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    modes = list(od_matrices.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=figsize, gridspec_kw={"wspace":0.1})
    if len(modes)==1: axes=[axes]
    for ax, mode in zip(axes, modes):
        OD = od_matrices[mode]; color = MODE_COLORS.get(mode, BLUE)
        flat = OD[np.isfinite(OD)].flatten(); flat = flat[flat > 0]
        if len(flat) == 0: ax.set_visible(False); continue
        flat = np.minimum(flat, np.percentile(flat, 99)) / 60
        ax.hist(flat, bins=50, color=color, alpha=0.75, edgecolor=WHITE, lw=0.3)
        radius = radii.get(mode)
        if radius:
            r_m = radius/60
            ax.axvline(r_m, color=NAVY, lw=1.8, ls="--", label=f"Radius = {r_m:.0f} min")
            pct = float((flat <= r_m).mean()*100)
            ax.text(r_m+0.3, ax.get_ylim()[1]*0.88, f"{pct:.0f}%\nwithin\nradius",
                    fontsize=7.5, color=NAVY, va="top",
                    bbox=dict(facecolor=WHITE, edgecolor=GRAY_30, pad=3, alpha=0.85))
        ax.set_xlabel("Travel time (min)", fontsize=9)
        if mode==modes[0]: ax.set_ylabel("OD pairs count", fontsize=9)
        ax.set_title(_MODE_LABEL.get(mode, mode), fontsize=10, fontweight="bold", color=color)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


# ─── Norm compliance ───────────────────────────────────────────────────────────
def norm_compliance(
    settlements: pd.DataFrame,
    acc_col: str = "acc_baseline",
    norm_threshold: Optional[float] = None,
    norm_label: str = "Accessibility threshold",
    title: str = "Settlements: Regulatory Compliance",
    figsize: Tuple = FIG_CHART,
    normalize: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    df = settlements.copy().sort_values(acc_col, ascending=True)
    if norm_threshold is None:
        norm_threshold = float(df[acc_col].quantile(0.2))

    # ── Normalize to 0–100 scale for readability ──
    raw_vals = df[acc_col].values
    if normalize and raw_vals.max() > 0:
        scale = 100.0 / raw_vals.max()
        plot_vals = raw_vals * scale
        plot_threshold = norm_threshold * scale
        x_label = "Normalized accessibility index (0–100)"
        fmt_val = lambda v: f"{v:.1f}"
    else:
        plot_vals = raw_vals
        plot_threshold = norm_threshold
        x_label = "E2SFCA index"
        fmt_val = lambda v: f"{v:.4f}"

    colors = [RED_ACC if v < norm_threshold else BLUE for v in raw_vals]
    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1], len(df)*0.38)))
    name_col = "name" if "name" in df.columns else df.index
    names = df[name_col] if "name" in df.columns else df.index
    bars = ax.barh(names, plot_vals, color=colors, edgecolor=WHITE, lw=0.4, height=0.7)
    ax.axvline(plot_threshold, color=ORANGE, lw=2.0, ls="--",
               label=f"{norm_label} ({fmt_val(plot_threshold)})")
    for bar, pv, rv in zip(bars, plot_vals, raw_vals):
        ax.text(pv + plot_vals.max()*0.01,
                bar.get_y()+bar.get_height()/2,
                fmt_val(pv),
                va="center", fontsize=8.5, color=NAVY)
    n_below = int((raw_vals < norm_threshold).sum())
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(f"{title}\n{n_below}/{len(df)} settlements below threshold",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    if save_path: savefig(fig, save_path)
    return fig


def _metric_label(m: str) -> str:
    return {"mean":"Mean E2SFCA","gini":"Gini coefficient","p25":"25th percentile",
            "p50":"Median","p75":"75th percentile","delta":"Δ vs baseline"}.get(m, m)

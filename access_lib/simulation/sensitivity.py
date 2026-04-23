"""
simulation/sensitivity.py — Parameter sensitivity analysis.

Audit fix v1.1:
  ANAL-1: run_grid() now accepts a mode parameter and the notebook
          calls it for car, walk, and pt separately.
          plot_heatmaps() and plot_curves() updated to show mode label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import E2SFCAParams, DecayType
from ..core.engine import e2sfca
from ..core.aggregation import gini


@dataclass
class SensitivityAnalysis:
    """
    Grid search over E2SFCA parameter space.
    ANAL-1: now supports per-mode analysis.

    Usage
    -----
    sa = SensitivityAnalysis()
    df_car  = sa.run_grid(OD_car,  population, supply, mode="car")
    df_walk = sa.run_grid(OD_walk, population, supply, mode="walk")
    df_pt   = sa.run_grid(OD_pt,   population, supply, mode="pt")
    sa.plot_heatmaps_multimode({"car": df_car, "walk": df_walk, "pt": df_pt})
    """
    beta_vals:   List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0]
    )
    radius_vals: List[float] = field(
        default_factory=lambda: [1200.0, 1800.0, 2700.0, 3600.0, 5400.0]
    )

    def run_grid(
        self,
        OD:         np.ndarray,
        population: np.ndarray,
        supply:     np.ndarray,
        params:     Optional[E2SFCAParams] = None,
        mode:       str = "car",
    ) -> pd.DataFrame:
        """
        Compute E2SFCA metrics across (beta, radius) grid for one mode.
        ANAL-1: `mode` column added to output so multi-mode results can be merged.
        """
        from scipy.stats import spearmanr

        params = params or E2SFCAParams()
        ref_radius = {"car": 5400.0, "walk": 2700.0, "pt": 3600.0}.get(mode, 3600.0)
        A_ref = e2sfca(OD, population, supply, 1.0, ref_radius, params)

        rows = []
        for radius in self.radius_vals:
            for beta in self.beta_vals:
                A = e2sfca(OD, population, supply, beta, radius, params)
                pos = A[A > 0]
                rho = float(spearmanr(A, A_ref).statistic) if len(pos) > 0 else float("nan")
                rows.append({
                    "mode":        mode,
                    "radius_min":  round(radius / 60),
                    "beta":        beta,
                    "mean":        round(float(pos.mean()), 5) if len(pos) else 0,
                    "gini":        round(gini(A), 4),
                    "nonzero_pct": round(float((A > 0).mean() * 100), 1),
                    "spearman_rho": round(rho, 3),
                })

        return pd.DataFrame(rows)

    def run_all_modes(
        self,
        OD_matrices: Dict[str, np.ndarray],
        population:  np.ndarray,
        supply:      np.ndarray,
        params:      Optional[E2SFCAParams] = None,
    ) -> pd.DataFrame:
        """ANAL-1: convenience method — runs grid for all modes and concatenates."""
        frames = []
        for mode, OD in OD_matrices.items():
            frames.append(self.run_grid(OD, population, supply, params, mode=mode))
        return pd.concat(frames, ignore_index=True)

    # ── Single-mode plots ────────────────────────────────────────────────────

    def plot_heatmaps(
        self,
        df:        pd.DataFrame,
        save_path: Optional["Path"] = None,
        figsize:   Tuple[int, int] = (16, 5),
    ) -> "plt.Figure":
        import matplotlib.pyplot as plt
        import seaborn as sns

        mode = df["mode"].iloc[0] if "mode" in df.columns else "car"
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(
            f"Sensitivity: β × Catchment Radius — {mode.upper()} mode",
            fontsize=13, fontweight="bold",
        )
        for ax, metric, label, cmap in [
            (axes[0], "mean",          "Mean Accessibility",      "RdYlGn"),
            (axes[1], "gini",          "Gini Coefficient",        "RdYlGn_r"),
            (axes[2], "spearman_rho",  "Spearman ρ vs reference", "RdYlGn"),
        ]:
            pivot = df.pivot(index="beta", columns="radius_min", values=metric)
            sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".3f",
                        linewidths=0.4, cbar_kws={"shrink": 0.8})
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Catchment (min)")
            ax.set_ylabel("β")
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    # ── ANAL-1: Multi-mode combined plot ────────────────────────────────────

    def plot_heatmaps_multimode(
        self,
        dfs:       Dict[str, pd.DataFrame],
        save_path: Optional["Path"] = None,
    ) -> "plt.Figure":
        """
        ANAL-1: 3×3 grid — rows = modes (car, walk, pt),
        columns = metrics (mean, gini, spearman_rho).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        modes   = list(dfs.keys())
        metrics = [("mean", "Mean Acc", "RdYlGn"),
                   ("gini", "Gini",     "RdYlGn_r"),
                   ("spearman_rho", "Spearman ρ", "RdYlGn")]

        fig, axes = plt.subplots(len(modes), 3, figsize=(16, 4 * len(modes)))
        if len(modes) == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle("Sensitivity Analysis — All Transport Modes",
                     fontsize=13, fontweight="bold")

        for r, mode in enumerate(modes):
            df = dfs[mode]
            for c, (metric, label, cmap) in enumerate(metrics):
                ax = axes[r][c]
                pivot = df.pivot(index="beta", columns="radius_min", values=metric)
                sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".3f",
                            linewidths=0.4, cbar_kws={"shrink": 0.8})
                ax.set_title(f"{mode.upper()} — {label}", fontweight="bold")
                ax.set_xlabel("Catchment (min)")
                ax.set_ylabel("β")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_curves(
        self,
        df:        pd.DataFrame,
        save_path: Optional["Path"] = None,
        figsize:   Tuple[int, int] = (16, 4),
    ) -> "plt.Figure":
        import matplotlib.pyplot as plt

        mode = df["mode"].iloc[0] if "mode" in df.columns else "car"
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"Sensitivity: Line Plots — {mode.upper()} Mode",
                     fontsize=12, fontweight="bold")

        beta_fixed   = df[df["radius_min"] == 60].sort_values("beta")
        radius_fixed = df[df["beta"] == 1.0].sort_values("radius_min")

        ax = axes[0]
        ax.plot(beta_fixed["beta"], beta_fixed["mean"], "o-",
                color="#2980b9", lw=2, ms=7, label="Mean acc")
        ax2 = ax.twinx()
        ax2.plot(beta_fixed["beta"], beta_fixed["spearman_rho"], "s--",
                 color="#e67e22", lw=1.5, ms=6, label="Spearman ρ")
        ax2.set_ylabel("Spearman ρ", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax.set_xlabel(f"β  (radius = 60 min, {mode})")
        ax.set_ylabel("Mean accessibility")
        ax.set_title("β-sensitivity", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[1]
        ax.plot(radius_fixed["radius_min"], radius_fixed["mean"], "o-",
                color="#2980b9", lw=2, ms=7, label="Mean acc")
        ax2 = ax.twinx()
        ax2.plot(radius_fixed["radius_min"], radius_fixed["spearman_rho"], "s--",
                 color="#e67e22", lw=1.5, ms=6, label="Spearman ρ")
        ax2.set_ylabel("Spearman ρ", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax.set_xlabel(f"Catchment radius (min), β=1.0, {mode}")
        ax.set_ylabel("Mean accessibility")
        ax.set_title("Radius-sensitivity", fontweight="bold")

        ax = axes[2]
        ax.plot(beta_fixed["beta"], beta_fixed["gini"], "o-",
                color="#e74c3c", lw=2, ms=7)
        ax.set_xlabel(f"β  (radius = 60 min, {mode})")
        ax.set_ylabel("Gini coefficient")
        ax.set_title("Equity: Gini vs β", fontweight="bold")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

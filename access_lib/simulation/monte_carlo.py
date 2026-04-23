"""
simulation/monte_carlo.py — Monte Carlo uncertainty quantification.

Audit fix v1.1:
  BUG-4: scenarios dict is built once (not re-instantiated with different
         strategies per run). S3 uses the same instance as Step 10.
  BUG-1/2: composite_accessibility() is called with ext_demand and
         specializations forwarded from ScenarioResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

from ..config import (
    E2SFCAParams, ModeWeights,
    MC_N_ITER, MC_SEED,
    MC_CAP_STD, MC_POP_STD,
    MC_OD_STD_SUMMER, MC_OD_STD_WINTER,
    SEASONAL,
)
from ..core.engine import composite_accessibility
from ..scenarios.base import BaseScenario


@dataclass
class NoiseParams:
    cap_std:  float = MC_CAP_STD
    pop_std:  float = MC_POP_STD
    od_std:   float = MC_OD_STD_SUMMER

    @classmethod
    def for_season(cls, season: str) -> "NoiseParams":
        od_std = MC_OD_STD_WINTER if season in ("winter", "autumn") else MC_OD_STD_SUMMER
        return cls(od_std=od_std)


@dataclass
class MCResult:
    scenario_name: str
    values:        np.ndarray

    @property
    def mean(self) -> float: return float(self.values.mean())
    @property
    def std(self) -> float:  return float(self.values.std())
    @property
    def ci_lo(self) -> float: return float(np.percentile(self.values, 2.5))
    @property
    def ci_hi(self) -> float: return float(np.percentile(self.values, 97.5))

    def summary_row(self, baseline_mean: float) -> dict:
        return {
            "scenario": self.scenario_name,
            "mean":     round(self.mean, 5),
            "std":      round(self.std, 5),
            "ci_lo":    round(self.ci_lo, 5),
            "ci_hi":    round(self.ci_hi, 5),
            "delta":    round(self.mean - baseline_mean, 5),
        }


class MonteCarlo:
    """
    Monte Carlo wrapper for policy scenario comparison.

    BUG-4 fix: scenarios are passed in as pre-built instances (same objects
    used in the standalone scenario cells). Do NOT re-instantiate scenarios
    here with different parameters.
    """

    def __init__(
        self,
        params:       E2SFCAParams,
        mode_weights: ModeWeights,
        scenarios:    List[BaseScenario],
        n_iter:       int = MC_N_ITER,
        seed:         int = MC_SEED,
    ):
        self.params       = params
        self.mode_weights = mode_weights
        self.scenarios    = scenarios
        self.n_iter       = n_iter
        self.seed         = seed

    def run(
        self,
        baseline_od:      Dict[str, np.ndarray],
        baseline_supply:  np.ndarray,
        demand:           np.ndarray,
        season:           str = "summer",
        specializations:  Optional[np.ndarray] = None,
        **scenario_kwargs,
    ) -> Dict[str, "MCResult"]:
        rng   = np.random.default_rng(self.seed)
        noise = NoiseParams.for_season(season)

        # Pre-build scenario inputs ONCE (deterministic)
        prebuilt = {}
        print("Pre-building scenario inputs...")
        for sc in self.scenarios:
            try:
                kwargs = dict(scenario_kwargs)
                if specializations is not None:
                    kwargs["specializations"] = specializations
                result = sc.build_inputs(
                    baseline_od=baseline_od,
                    baseline_supply=baseline_supply,
                    demand=demand,
                    **kwargs,
                )
                prebuilt[sc.name] = result
                print(f"  ✓ {sc.name}")
            except Exception as e:
                print(f"  ✗ {sc.name}: {e}")

        store: Dict[str, List[float]] = {s.name: [] for s in self.scenarios}

        for i in tqdm(range(self.n_iter), desc="MC"):
            noisy_supply = self._noisy_supply(baseline_supply, rng, noise, season)
            noisy_demand = self._noisy_demand(demand, rng, noise)
            noisy_od     = self._noisy_od(baseline_od, rng, noise)

            for sc in self.scenarios:
                result = prebuilt.get(sc.name)
                if result is None:
                    store[sc.name].append(float("nan"))
                    continue
                try:
                    eff_od    = result.effective_od(noisy_od)
                    # FIX: 50th-facility — apply noise to base slice only
                    _raw_sup  = result.effective_supply(noisy_supply)
                    eff_sup   = self._merge_noisy_supply(noisy_supply, _raw_sup)
                    # BUG-2/1 fix: forward specializations and ext_demand
                    eff_specs = (
                        result.effective_specializations(specializations)
                        if specializations is not None else None
                    )
                    A = composite_accessibility(
                        eff_od, noisy_demand, eff_sup,
                        self.params, self.mode_weights.as_dict(),
                        ext_demand=result.ext_demand,
                        specializations=eff_specs,
                    )
                    store[sc.name].append(float(np.average(A, weights=noisy_demand)))
                except Exception as e:
                    store[sc.name].append(float("nan"))
                    if i == 0:
                        print(f"  ⚠ {sc.name!r} iter 0: {e}")

        return {
            name: MCResult(name, np.array(vals, dtype=np.float64))
            for name, vals in store.items()
        }

    @staticmethod
    def summary(results: Dict[str, "MCResult"]) -> pd.DataFrame:
        names = list(results.keys())
        baseline_mean = results[names[0]].mean if names else 0.0
        rows = [r.summary_row(baseline_mean) for r in results.values()]
        df = pd.DataFrame(rows)
        print(f"\n  {'Scenario':<24}{'Mean':>9}{'Std':>8}{'95% CI':>22}{'Δ vs S0':>10}")
        print("  " + "─" * 75)
        for _, row in df.iterrows():
            print(f"  {row['scenario']:<24}{row['mean']:>9.4f}{row['std']:>8.4f}"
                  f"  [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]{row['delta']:>10.4f}")
        return df

    @staticmethod
    def _noisy_supply(supply, rng, noise, season) -> np.ndarray:
        season_speed = SEASONAL.get(season, {}).get("speed", 1.0)
        cap_noise    = rng.lognormal(0.0, noise.cap_std, size=len(supply))
        return np.maximum(supply * cap_noise * season_speed, 1.0).astype(np.float32)

    @staticmethod
    def _merge_noisy_supply(noisy_baseline: np.ndarray,
                             result_supply:   np.ndarray) -> np.ndarray:
        """
        FIX: when a scenario appended a new facility (len=50 vs baseline 49),
        apply noise to the baseline portion and keep the new facility fixed.
        This avoids shape-mismatch errors and makes MC semantically correct:
        we test uncertainty in EXISTING facilities; the new one is the scenario.
        """
        n_base = len(noisy_baseline)
        if len(result_supply) == n_base:
            return noisy_baseline          # no new facility — use noisy baseline
        # Scenario added facilities: apply noise to base, keep extras fixed
        merged = result_supply.copy().astype(np.float32)
        merged[:n_base] = noisy_baseline   # overwrite baseline portion with noise
        return merged                      # new-facility portion unchanged

    @staticmethod
    def _noisy_demand(demand, rng, noise) -> np.ndarray:
        pop_noise = rng.normal(1.0, noise.pop_std, size=len(demand))
        return np.maximum(demand * pop_noise, 0.1).astype(np.float32)

    @staticmethod
    def _noisy_od(od, rng, noise) -> Dict[str, np.ndarray]:
        result = {}
        for mode, mat in od.items():
            mult  = float(rng.lognormal(0.0, noise.od_std))
            noisy = mat * mult
            noisy[~np.isfinite(mat)] = np.inf
            result[mode] = noisy.astype(np.float32)
        return result

    @staticmethod
    def plot_distributions(
        results:   Dict[str, "MCResult"],
        save_path: Optional["Path"] = None,
        figsize:   Tuple[int, int] = (13, 6),
    ) -> "plt.Figure":
        import matplotlib.pyplot as plt
        names  = list(results.keys())
        data   = [results[n].values for n in names]
        colors = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6","#1abc9c","#e67e22"]
        fig, ax = plt.subplots(figsize=figsize)
        parts = ax.violinplot(data, positions=range(len(names)), showmedians=True)
        for pc, c in zip(parts["bodies"], colors[:len(names)]):
            pc.set_facecolor(c); pc.set_alpha(0.75)
        parts["cmedians"].set_color("#111"); parts["cmedians"].set_linewidth(2)
        baseline_mean = results[names[0]].mean
        ax.axhline(baseline_mean, color="#c0392b", ls="--", lw=1.5,
                   label=f"S0 mean = {baseline_mean:.4f}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_"," ") for n in names], rotation=18, ha="right")
        ax.set_ylabel("Weighted mean E2SFCA accessibility")
        ax.set_title(f"Monte Carlo distributions  (n={len(data[0])} iterations)",
                     fontweight="bold")
        ax.legend()
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

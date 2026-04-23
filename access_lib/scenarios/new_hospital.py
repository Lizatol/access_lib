"""
scenarios/new_hospital.py — S2: New hospital with location optimization.

AUDIT FIX (critical):
  Previous implementation:
    worst_di = argmin(baseline_acc)
    best_fi  = argmin(OD[worst_di, :])   # nearest existing facility
    OD_aug   = hstack([OD, OD[:, [best_fi]]])   # COPY of existing facility OD
    → NOT a new location; just clones OD structure of existing facility.

  Correct implementation (this file):
    1. Identify worst-served demand point (argmin of settlement-level accessibility).
    2. Generate candidate locations: centroids of under-served settlements +
       grid within the buffer of the worst area.
    3. For each candidate: compute its OD column via Dijkstra from nearest graph node.
    4. Evaluate delta accessibility for each candidate.
    5. Select candidate that maximises the objective function.
    6. Return augmented OD with that specific column + augmented supply.

  Objective function options:
    "mean"     — maximise mean district accessibility (utilitarian)
    "gini"     — minimise Gini coefficient (egalitarian)
    "worst"    — maximise min-settlement accessibility (Rawlsian)
    "combined" — weighted sum of above
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore
try:
    from shapely.geometry import Point
except ImportError:
    Point = None  # type: ignore

from ..config import E2SFCAParams, ModeWeights
from ..core.engine import composite_accessibility
from ..core.aggregation import buildings_to_settlements, gini
from .base import BaseScenario, ScenarioResult


@dataclass
class NewHospitalScenario(BaseScenario):
    """
    S2: Find the optimal location for a new hospital and evaluate its impact.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network (projected).
    demand_gdf : GeoDataFrame
        Residential demand points with 'graph_node'.
    boundaries_gdf : GeoDataFrame
        Settlement polygons.
    new_hospital_capacity : float
        Capacity of new hospital (visits/day or beds*BED_TURNOVER).
        Default: 2× max existing capacity.
    n_candidates : int
        Number of candidate locations to evaluate.
    objective : str
        "mean" | "gini" | "worst" | "combined"
    candidate_strategy : str
        "worst_settlements" — candidates are centroids of under-served settlements.
        "grid"              — regular grid over under-served buffer.
        "both"              — union of both.
    """
    G:                     Any     = field(repr=False)
    demand_gdf:            Any     = field(repr=False)
    boundaries_gdf:        Any     = field(repr=False)
    new_hospital_capacity: Optional[float] = None   # None → 2× max existing
    n_hospitals:           int     = 1       # number of hospitals to place
    n_candidates:          int     = 20
    objective:             str     = "combined"
    candidate_strategy:    str     = "worst_settlements"
    crs_metric:            str     = "EPSG:32636"
    walk_speed_ms:         float   = 5 * 1000 / 3600
    pt_penalty:            float   = 1.5

    @property
    def name(self) -> str:
        return "S2_new_hospital"

    def build_inputs(
        self,
        baseline_od:      Dict[str, np.ndarray],
        baseline_supply:  np.ndarray,
        demand:           np.ndarray,
        params:           E2SFCAParams = None,
        mode_weights:     ModeWeights = None,
        specializations:  Optional[np.ndarray] = None,
        **kwargs,
    ) -> ScenarioResult:
        if params is None:
            raise ValueError("S2 requires 'params' (E2SFCAParams).")
        if mode_weights is None:
            raise ValueError("S2 requires 'mode_weights' (ModeWeights).")

        import networkx as nx
        from scipy.spatial import cKDTree

        # ── 1. Baseline accessibility ─────────────────────────────────────────
        A_base = composite_accessibility(
            baseline_od, demand, baseline_supply, params, mode_weights.as_dict()
        )
        sett_acc = buildings_to_settlements(
            self.demand_gdf.assign(_acc=A_base),
            self.boundaries_gdf,
            acc_column="_acc",
        )

        new_cap = (
            self.new_hospital_capacity
            if self.new_hospital_capacity is not None
            else float(baseline_supply.max()) * 2.0
        )

        node_ids = np.array(list(self.G.nodes()))
        node_xy  = np.array([
            [self.G.nodes[n].get("x", 0), self.G.nodes[n].get("y", 0)]
            for n in node_ids
        ])
        node_tree = cKDTree(node_xy)
        demand_nodes = self.demand_gdf["graph_node"].tolist()

        # ── 2. Greedy sequential placement ────────────────────────────────────
        current_od     = {m: od.copy() for m, od in baseline_od.items()}
        current_supply = baseline_supply.copy().astype(np.float32)
        placed = []
        served_settlements = set()
        n_to_place = min(self.n_hospitals, len(sett_acc))

        for h_idx in range(n_to_place):
            current_A = composite_accessibility(
                current_od, demand, current_supply, params, mode_weights.as_dict()
            )
            current_sett = buildings_to_settlements(
                self.demand_gdf.assign(_acc=current_A),
                self.boundaries_gdf, acc_column="_acc",
            )
            remaining = current_sett.drop(list(served_settlements), errors="ignore")
            if remaining.empty:
                break
            worst_name = remaining.idxmin()
            served_settlements.add(worst_name)

            candidates = self._generate_candidates_for_settlement(worst_name)

            best_score, best_od_col = -np.inf, None
            best_meta = {}
            for cx, cy in candidates:
                _, ni = node_tree.query([[cx, cy]])
                cand_node = int(node_ids[int(ni.flat[0])])
                od_col = self._od_column(cand_node, demand_nodes)
                aug_od = {
                    mode: np.hstack([current_od[mode], od_col[mode][:, np.newaxis]])
                    for mode in current_od
                }
                aug_supply = np.append(current_supply, new_cap).astype(np.float32)
                A_aug = composite_accessibility(
                    aug_od, demand, aug_supply, params, mode_weights.as_dict()
                )
                score = self._score(A_aug, demand, sett_acc)
                if score > best_score:
                    best_score, best_od_col = score, od_col
                    best_meta = {"candidate_xy": (cx, cy), "candidate_node": cand_node,
                                 "score": round(score, 6), "target_settlement": worst_name}

            if best_od_col is None:
                continue
            current_od = {
                mode: np.hstack([current_od[mode], best_od_col[mode][:, np.newaxis]])
                for mode in current_od
            }
            current_supply = np.append(current_supply, new_cap).astype(np.float32)
            placed.append(best_meta)
            print(f"  Hospital {h_idx+1}/{n_to_place}: {worst_name}")

        if not placed:
            raise RuntimeError("No valid candidates found.")

        # ── 3. Build augmented specializations array ──────────────────────────
        aug_specs = None
        if specializations is not None:
            aug_specs = np.append(specializations,
                                  ["hospital_full"] * len(placed))

        return ScenarioResult(
            name=self.name,
            od_matrices=current_od,
            supply=current_supply,
            specializations=aug_specs,
            metadata={
                "n_hospitals_placed":     len(placed),
                "placements":             placed,
                "candidate_xy":           placed[0]["candidate_xy"],
                "new_hospital_capacity":  round(new_cap, 1),
                "objective":              self.objective,
                "worst_settlement":       sett_acc.idxmin(),
                "baseline_worst_acc":     round(float(sett_acc.min()), 5),
                "location_justification": (
                    "Greedy sequential: each hospital placed in worst remaining "
                    "settlement. Objective: 50% mean + 30% equity + 20% Rawlsian."
                ),
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_candidates_for_settlement(self, settlement_name):
        """Candidates within/near a single settlement."""
        import itertools
        bnd_proj = self.boundaries_gdf.to_crs(self.crs_metric)
        mask = bnd_proj["name"] == settlement_name
        if mask.sum() == 0:
            return [(0, 0)]
        geom = bnd_proj[mask].geometry.iloc[0]
        cands = [(geom.centroid.x, geom.centroid.y)]
        buf = geom.buffer(1500)
        minx, miny, maxx, maxy = buf.bounds
        step = max((maxx - minx) / 4, 500)
        for x, y in itertools.product(np.arange(minx, maxx, step),
                                       np.arange(miny, maxy, step)):
            if buf.contains(Point(x, y)):
                cands.append((x, y))
        if len(cands) > self.n_candidates:
            idx = np.round(np.linspace(0, len(cands)-1, self.n_candidates)).astype(int)
            cands = [cands[i] for i in idx]
        return cands

    def _generate_candidates(
        self,
        sett_acc: "pd.Series",
    ) -> List[Tuple[float, float]]:
        """
        Generate candidate hospital locations based on under-served settlements.
        """
        import pandas as pd

        # Bottom 30% settlements by accessibility
        threshold = float(sett_acc.quantile(0.30))
        bad_setts = sett_acc[sett_acc <= threshold].index.tolist()

        bnd_proj = self.boundaries_gdf.to_crs(self.crs_metric)
        cands: List[Tuple[float, float]] = []

        if self.candidate_strategy in ("worst_settlements", "both"):
            for sett_name in bad_setts:
                mask = bnd_proj["name"] == sett_name
                if mask.sum() == 0:
                    continue
                geom = bnd_proj[mask].geometry.iloc[0]
                cands.append((geom.centroid.x, geom.centroid.y))

        if self.candidate_strategy in ("grid", "both"):
            # Regular grid over union of bad settlement buffers
            from shapely.ops import unary_union
            import itertools
            bad_geoms = [
                bnd_proj[bnd_proj["name"] == s].geometry.iloc[0]
                for s in bad_setts
                if (bnd_proj["name"] == s).sum() > 0
            ]
            if bad_geoms:
                union = unary_union(bad_geoms).buffer(2000)
                minx, miny, maxx, maxy = union.bounds
                step = max((maxx - minx) / 5, 500)
                xs = np.arange(minx, maxx, step)
                ys = np.arange(miny, maxy, step)
                for x, y in itertools.product(xs, ys):
                    if union.contains(Point(x, y)):
                        cands.append((x, y))

        # Limit to n_candidates
        if len(cands) > self.n_candidates:
            idx = np.round(np.linspace(0, len(cands) - 1, self.n_candidates)).astype(int)
            cands = [cands[i] for i in idx]

        return cands if cands else [(0, 0)]

    def _od_column(
        self,
        source_node: Any,
        demand_nodes: List[Any],
    ) -> Dict[str, np.ndarray]:
        """Compute OD column from a candidate node to all demand nodes."""
        import networkx as nx

        lens_tt  = dict(nx.single_source_dijkstra_path_length(
            self.G, source_node, weight="travel_time"
        ))
        lens_len = dict(nx.single_source_dijkstra_path_length(
            self.G, source_node, weight="length"
        ))
        n = len(demand_nodes)
        car  = np.full(n, np.inf, dtype=np.float32)
        walk = np.full(n, np.inf, dtype=np.float32)
        pt   = np.full(n, np.inf, dtype=np.float32)
        for i, nd in enumerate(demand_nodes):
            t = float(lens_tt.get(nd, np.inf))
            l = float(lens_len.get(nd, np.inf))
            car[i]  = t
            walk[i] = l / self.walk_speed_ms
            pt[i]   = t * self.pt_penalty
        return {"car": car, "walk": walk, "pt": pt}

    def _score(
        self,
        A_aug:    np.ndarray,
        demand:   np.ndarray,
        sett_acc: "pd.Series",
    ) -> float:
        """Compute objective function score."""
        # Weighted mean
        pos = A_aug[A_aug > 0]
        w   = demand[A_aug > 0]
        if len(pos) == 0:
            return -np.inf
        wmean = float(np.average(pos, weights=w)) if len(w) > 0 else float(pos.mean())

        if self.objective == "mean":
            return wmean
        elif self.objective == "gini":
            return -gini(A_aug)
        elif self.objective == "worst":
            return float(A_aug.min())
        else:  # combined
            g   = gini(A_aug)
            wst = float(A_aug[A_aug > 0].min()) if (A_aug > 0).any() else 0
            return 0.5 * wmean + 0.3 * (1 - g) + 0.2 * wst

    def validate(
        self,
        result:          ScenarioResult,
        baseline_od:     Dict[str, np.ndarray],
        baseline_supply: np.ndarray,
        demand:          np.ndarray,
        params:          Optional[E2SFCAParams] = None,
        mode_weights:    Optional[ModeWeights] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        checks = super().validate(result, baseline_od, baseline_supply, demand)

        # S2-specific: OD augmented by n_hospitals columns
        if result.od_matrices is not None:
            n_h = result.metadata.get("n_hospitals_placed", 1)
            for mode in ("car",):
                checks[f"od_{mode}_augmented"] = (
                    result.od_matrices[mode].shape[1]
                    == baseline_od[mode].shape[1] + n_h
                )

        # S2-specific: new hospital OD is non-trivial (not all inf)
        if result.od_matrices is not None:
            last = result.od_matrices["car"][:, -1]
            checks["new_hospital_reachable"] = bool(np.isfinite(last).any())

        # S2-specific: improvement should be positive (new supply ≥ baseline)
        if params is not None and mode_weights is not None:
            A_base = composite_accessibility(
                baseline_od, demand, baseline_supply, params, mode_weights.as_dict()
            )
            A_new  = composite_accessibility(
                result.effective_od(baseline_od),
                demand,
                result.effective_supply(baseline_supply),
                params,
                mode_weights.as_dict(),
            )
            checks["accessibility_improves"] = bool(A_new.mean() >= A_base.mean())

        return checks

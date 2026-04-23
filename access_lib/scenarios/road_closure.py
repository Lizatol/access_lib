"""
scenarios/road_closure.py — S3: Road closure via edge removal in graph.

AUDIT FIX (critical):
  Previous implementation:
    flat = OD.ravel().copy()
    idx  = rng.choice(len(flat), fraction, replace=False)
    flat[idx] *= rng.uniform(2, 8)   # random disruption
    → RANDOM deformation of OD. No specific edge. Not route-aware.

  Correct implementation (this file):
    1. Select closure edges (by polygon, by edge list, or by betweenness).
    2. Remove those edges from the graph (set weight=inf).
    3. Recompute OD for the MODIFIED graph.
    4. Identify IMPACTED demand points: those whose shortest path changed.
    5. Return modified OD only for impacted buildings (delta map is localised).

  Key principle:
    • Road closure is EDGE-AWARE and ROUTE-AWARE.
    • Only buildings whose actual shortest path passed through the closed edge
      are affected — not all buildings in the target settlement.
    • The delta map shows only impacted buildings, not a district-wide average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore

from ..core.graph import (
    remove_edges,
    find_edges_in_polygon,
    find_critical_edges,
    affected_demand_nodes,
    compute_od_column,
)
from .base import BaseScenario, ScenarioResult


@dataclass
class RoadClosureScenario(BaseScenario):
    """
    S3: Close a set of road edges and recompute OD for affected buildings.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph (projected).
    demand_gdf : GeoDataFrame
        Residential demand points with 'graph_node'.
    facilities_gdf : GeoDataFrame
        Healthcare facilities with 'graph_node'.
    closure_edges : list of (u, v) tuples, optional
        Explicit edges to close. If None, uses automatic selection.
    closure_polygon : geometry, optional
        Close all edges whose midpoint falls within this polygon.
    closure_strategy : str
        "polygon"     — close edges in polygon (requires closure_polygon).
        "critical"    — close top-k betweenness-critical edges.
        "explicit"    — use closure_edges list (requires closure_edges).
    n_critical : int
        Number of critical edges to close (for "critical" strategy).
    impact_threshold_s : float
        Minimum change in travel time (seconds) to count a building as impacted.
    """
    G:                    Any            = field(repr=False)
    demand_gdf:           Any            = field(repr=False)
    facilities_gdf:       Any            = field(repr=False)
    closure_edges:        Optional[List[Tuple[Any, Any]]] = None
    closure_polygon:      Optional[Any]  = None   # shapely geometry
    closure_strategy:     str            = "critical"
    n_critical:           int            = 5
    impact_threshold_s:   float          = 120.0  # 2-min change counts as impacted
    walk_speed_ms:        float          = 5 * 1000 / 3600
    pt_penalty:           float          = 1.5
    crs_metric:           str            = "EPSG:32636"

    @property
    def name(self) -> str:
        return "S3_road_closure"

    def build_inputs(
        self,
        baseline_od:      Dict[str, np.ndarray],
        baseline_supply:  np.ndarray,
        demand:           np.ndarray,
        **kwargs,
    ) -> ScenarioResult:
        """
        1. Select edges to close.
        2. Build modified graph.
        3. Пересчитываем OD только для учреждений рядом с закрытием (быстро).
        4. Identify impacted buildings.
        """
        import networkx as nx
        import numpy as np
        from scipy.spatial import cKDTree

        # ── 1. Выбираем рёбра ────────────────────────────────────────────────
        edges = self._select_edges()
        if not edges:
            raise ValueError("No closure edges found. Check closure_strategy.")

        # ── 2. Модифицированный граф ─────────────────────────────────────────
        G2 = remove_edges(self.G, edges)

        # ── 3. Пересчёт OD только для учреждений рядом с закрытием ──────────
        # Вместо пересчёта всех 49 учреждений — только тех, что ≤10 км от рёбер.
        closed_nodes = set()
        for u, v in edges:
            closed_nodes.update([u, v])

        # Координаты закрытых узлов
        closed_xy = np.array([
            [self.G.nodes[n].get("x", 0), self.G.nodes[n].get("y", 0)]
            for n in closed_nodes if n in self.G.nodes
        ], dtype=np.float64)

        fac_proj = self.facilities_gdf.to_crs(self.crs_metric)
        fac_geom = fac_proj.geometry
        if not (fac_geom.geom_type == "Point").all():
            fac_geom = fac_geom.centroid
        fac_xy = np.column_stack([fac_geom.x.values, fac_geom.y.values])

        # Расширенный радиус: 20 км — критичные рёбра влияют на большую зону
        RADIUS_M = 20_000
        if len(closed_xy) > 0:
            _dists, _ = cKDTree(closed_xy).query(fac_xy)
            affected_mask = _dists < RADIUS_M
        else:
            affected_mask = np.ones(len(fac_xy), dtype=bool)

        affected_idx = np.where(affected_mask)[0]
        print(f"  S3: {len(edges)} рёбер → пересчёт OD "
              f"для {len(affected_idx)}/{len(fac_xy)} учреждений")

        demand_nodes   = self.demand_gdf["graph_node"].tolist()
        facility_nodes = self.facilities_gdf["graph_node"].tolist()
        n_dem = len(demand_nodes)

        od2_car  = baseline_od["car"].copy()
        od2_walk = baseline_od["walk"].copy()
        od2_pt   = baseline_od["pt"].copy()

        for j in affected_idx:
            fac_node = facility_nodes[j]
            cols = compute_od_column(
                G2, fac_node, demand_nodes,
                walk_speed_ms=self.walk_speed_ms,
                pt_penalty=self.pt_penalty,
            )
            od2_car[:, j]  = cols["car"]
            od2_walk[:, j] = cols["walk"]
            od2_pt[:, j]   = cols["pt"]

        new_od = {"car": od2_car, "walk": od2_walk, "pt": od2_pt}

        # ── 4. Задетые здания ─────────────────────────────────────────────────
        impacted = affected_demand_nodes(
            baseline_od["car"], od2_car,
            threshold_s=self.impact_threshold_s,
        )
        n_impacted   = int(impacted.sum())
        pct_impacted = n_impacted / max(n_dem, 1) * 100
        print(f"  Задетые здания: {n_impacted:,} ({pct_impacted:.1f}%)")

        return ScenarioResult(
            name=self.name,
            od_matrices=new_od,
            supply=None,
            metadata={
                "closed_edges":       edges,
                "n_closed_edges":     len(edges),
                "closure_strategy":   self.closure_strategy,
                "n_impacted":         n_impacted,
                "pct_impacted":       round(pct_impacted, 2),
                "impacted_mask":      impacted,
                "impact_threshold_s": self.impact_threshold_s,
                "n_fac_recomputed":   len(affected_idx),
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_edges(self) -> List[Tuple[Any, Any]]:
        """Select edges based on strategy."""
        if self.closure_strategy == "explicit":
            if not self.closure_edges:
                raise ValueError("closure_edges required for explicit strategy.")
            return self.closure_edges

        if self.closure_strategy == "polygon":
            if self.closure_polygon is None:
                raise ValueError("closure_polygon required for polygon strategy.")
            return find_edges_in_polygon(self.G, self.closure_polygon)

        if self.closure_strategy == "critical":
            demand_nodes   = self.demand_gdf["graph_node"].tolist()
            facility_nodes = self.facilities_gdf["graph_node"].tolist()
            return find_critical_edges(
                self.G, demand_nodes, facility_nodes, top_k=self.n_critical
            )

        raise ValueError(f"Unknown closure_strategy: {self.closure_strategy!r}")

    def validate(
        self,
        result:          ScenarioResult,
        baseline_od:     Dict[str, np.ndarray],
        baseline_supply: np.ndarray,
        demand:          np.ndarray,
        params:          Any = None,
        mode_weights:    Any = None,
        **kwargs,
    ) -> Dict[str, bool]:
        checks = super().validate(result, baseline_od, baseline_supply, demand)

        # S3-specific: accessibility must not improve (road closure = disruption)
        if params is not None and mode_weights is not None:
            from ..core.engine import composite_accessibility
            A_base = composite_accessibility(
                baseline_od, demand, baseline_supply, params,
                mode_weights.as_dict() if hasattr(mode_weights, "as_dict") else mode_weights
            )
            A_new  = composite_accessibility(
                result.effective_od(baseline_od), demand,
                result.effective_supply(baseline_supply), params,
                mode_weights.as_dict() if hasattr(mode_weights, "as_dict") else mode_weights
            )
            checks["accessibility_does_not_improve"] = bool(A_new.mean() <= A_base.mean() * 1.01)

        # S3-specific: at least some buildings must be impacted
        meta = result.metadata
        if "n_impacted" in meta:
            checks["has_impacted_buildings"] = meta["n_impacted"] > 0

        # S3-specific: impacted buildings must be a proper subset (not all)
        if "pct_impacted" in meta:
            checks["partial_impact"] = 0 < meta["pct_impacted"] < 100

        return checks

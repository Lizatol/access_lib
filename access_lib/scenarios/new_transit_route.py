"""
scenarios/new_transit_route.py — S5: New public transit route.

GAP-2 fix: implements the missing "добавление маршрутов транспорта" scenario.

Design principles:
  • Only PT OD matrix is modified (car and walk unchanged).
  • Only buildings within walk distance of NEW stops are affected.
  • Effect is localised: delta map shows impacted buildings, not district average.
  • Multiple route segments supported (a route = ordered list of stops).

How it works:
  1. Define new stops along the route (list of (lon, lat) tuples).
  2. For each demand building within walking radius of any new stop:
       new_pt_time[i,j] = walk_to_new_stop[i] + headway_pen
                        + inter_stop_bus_time(new_stop → fac_stop)
                        + walk_from_fac_stop[j]
  3. Keep the MINIMUM of baseline_pt and new_pt per (i,j) cell.
     Buildings not near new stops keep baseline values unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None
    Point = None

from .base import BaseScenario, ScenarioResult
from ..config import WALK_SPEED_MS, PT_PENALTY


@dataclass
class NewTransitRouteScenario(BaseScenario):
    """
    S5: Add a new public transit route and evaluate its effect on PT accessibility.

    Parameters
    ----------
    new_stops : list of (lon, lat) tuples
        Coordinates of new bus/tram stops along the route.
    demand_gdf : GeoDataFrame
        Residential buildings with geometry.
    facilities_gdf : GeoDataFrame
        Healthcare facilities with geometry.
    bus_stops_gdf : GeoDataFrame
        Existing bus stops (used to compute leg-3 inter-stop times).
    walk_radius_m : float
        Max walking distance to new stop to benefit from the route (default 600 m).
    headway_s : float
        Mean headway of the new route in seconds (default 900 s = 15 min).
    bus_speed_ms : float
        Bus speed on the new route in m/s.
    crs_metric : str
        Projected CRS for distance calculations.
    """
    new_stops:         List[Tuple[float, float]]  # (lon, lat) in WGS84
    demand_gdf:        Any = field(repr=False)
    facilities_gdf:    Any = field(repr=False)
    bus_stops_gdf:     Any = field(repr=False)
    walk_radius_m:     float = 600.0
    headway_s:         float = 900.0              # 15-min headway
    bus_speed_ms:      float = 25 * 1000 / 3600   # 25 km/h
    crs_metric:        str   = "EPSG:32636"
    impact_threshold_s: float = 60.0              # 1-min improvement counts

    @property
    def name(self) -> str:
        return "S5_new_transit_route"

    def build_inputs(
        self,
        baseline_od:      Dict[str, np.ndarray],
        baseline_supply:  np.ndarray,
        demand:           np.ndarray,
        **kwargs,
    ) -> ScenarioResult:
        """
        Recompute PT OD for buildings near new stops.
        Car and walk OD are returned unchanged.
        """
        from scipy.spatial import cKDTree

        n_dem = len(demand)
        n_fac = len(baseline_supply)

        # ── Project everything to metric CRS ─────────────────────────────────
        dem_proj  = self.demand_gdf.to_crs(self.crs_metric)
        fac_proj  = self.facilities_gdf.to_crs(self.crs_metric)

        dem_xy  = np.column_stack([
            dem_proj.geometry.centroid.x, dem_proj.geometry.centroid.y
        ])
        fac_xy  = np.column_stack([
            fac_proj.geometry.centroid.x, fac_proj.geometry.centroid.y
        ])

        # Project new stops
        new_stops_proj = self._project_stops(self.new_stops, self.crs_metric)
        new_stop_tree  = cKDTree(new_stops_proj)

        # Existing stops for leg-3 proxy
        if self.bus_stops_gdf is not None and len(self.bus_stops_gdf) > 0:
            stop_proj = self.bus_stops_gdf.to_crs(self.crs_metric)
            all_stop_xy = np.column_stack([
                stop_proj.geometry.centroid.x, stop_proj.geometry.centroid.y
            ])
            # Merge new stops into the stop network
            all_stop_xy = np.vstack([all_stop_xy, new_stops_proj])
        else:
            all_stop_xy = new_stops_proj

        all_stop_tree = cKDTree(all_stop_xy)

        # ── Identify impacted demand buildings ───────────────────────────────
        walk_dists, nearest_new_idx = new_stop_tree.query(dem_xy, k=1)
        impacted_mask = walk_dists <= self.walk_radius_m
        impacted_idx  = np.where(impacted_mask)[0]

        print(f"  S5: {len(self.new_stops)} new stops → "
              f"{impacted_mask.sum():,} buildings within {self.walk_radius_m:.0f} m")

        if impacted_mask.sum() == 0:
            print("  ⚠  No buildings impacted — check stop coordinates and walk_radius_m")
            return ScenarioResult(
                name=self.name,
                metadata={"n_impacted": 0, "pct_impacted": 0.0,
                          "impacted_mask": impacted_mask},
            )

        # ── Compute improved PT times for impacted buildings ──────────────────
        # Leg 1: walk to nearest new stop
        leg1 = walk_dists / WALK_SPEED_MS                  # (n_dem,)
        # Leg 2: headway penalty
        hw_pen = 0.5 * self.headway_s
        # Leg 3: bus to the stop nearest each facility
        _, fac_stop_idx = all_stop_tree.query(fac_xy, k=1)
        fac_stop_xy_arr = all_stop_xy[fac_stop_idx]        # (n_fac, 2)
        new_stop_xy_imp = new_stops_proj[nearest_new_idx[impacted_idx]]  # (n_imp, 2)
        inter_dist = np.linalg.norm(
            new_stop_xy_imp[:, np.newaxis, :] - fac_stop_xy_arr[np.newaxis, :, :],
            axis=2,
        )                                                  # (n_imp, n_fac)
        leg3 = inter_dist / self.bus_speed_ms
        # Leg 4: walk from fac stop to facility
        leg4, _ = cKDTree(fac_xy).query(fac_stop_xy_arr)
        leg4 = (leg4 / WALK_SPEED_MS).astype(np.float32)  # (n_fac,)

        new_pt_imp = (
            leg1[impacted_idx, np.newaxis]  # (n_imp, 1)
            + hw_pen
            + leg3                          # (n_imp, n_fac)
            + leg4[np.newaxis, :]           # (1, n_fac)
        ).astype(np.float32)

        # ── Build modified PT OD matrix ───────────────────────────────────────
        od_pt_new = baseline_od["pt"].copy()
        od_pt_new[np.ix_(impacted_idx, np.arange(n_fac))] = np.minimum(
            od_pt_new[np.ix_(impacted_idx, np.arange(n_fac))],
            new_pt_imp,
        )

        # Impacted: buildings where PT time actually improved by threshold
        delta_pt = baseline_od["pt"][impacted_idx, :].min(axis=1) - new_pt_imp.min(axis=1)
        truly_impacted_local = delta_pt >= self.impact_threshold_s
        truly_impacted_mask  = impacted_mask.copy()
        truly_impacted_mask[impacted_idx] = truly_impacted_local

        n_truly  = int(truly_impacted_mask.sum())
        pct_true = n_truly / max(n_dem, 1) * 100

        print(f"  Truly improved (Δ ≥ {self.impact_threshold_s:.0f}s): "
              f"{n_truly:,} ({pct_true:.1f}%)")

        return ScenarioResult(
            name=self.name,
            od_matrices={
                "car":  baseline_od["car"],   # unchanged
                "walk": baseline_od["walk"],  # unchanged
                "pt":   od_pt_new,
            },
            supply=None,   # supply unchanged
            metadata={
                "n_new_stops":      len(self.new_stops),
                "walk_radius_m":    self.walk_radius_m,
                "headway_s":        self.headway_s,
                "n_near_stops":     int(impacted_mask.sum()),
                "n_impacted":       n_truly,
                "pct_impacted":     round(pct_true, 2),
                "impacted_mask":    truly_impacted_mask,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _project_stops(
        self,
        stops_wgs84: List[Tuple[float, float]],
        crs_metric:  str,
    ) -> np.ndarray:
        """Convert (lon, lat) WGS84 list to metric CRS numpy array."""
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)
        xs, ys = transformer.transform(
            [s[0] for s in stops_wgs84],
            [s[1] for s in stops_wgs84],
        )
        return np.column_stack([xs, ys])

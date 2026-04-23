"""
core/demand.py — Population demand estimation utilities.

FIX-5: DEMAND BIAS — buildings without floor data default to 1 floor,
       causing unrealistic population distribution (large footprint, low pop).

Fixes:
  1. estimate_floors(): assigns floors from OSM attribute → building type →
     settlement density percentile → district median (four-level fallback).
  2. compute_demand(): distributes settlement population to buildings
     proportionally to estimated floor area.
  3. Full logging: reports how many buildings use each fallback level.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore


# ─── Building-type floor defaults ────────────────────────────────────────────
# Based on Russian SP 54.13330.2022 and OSM data patterns for Leningrad Oblast.
FLOORS_BY_TYPE: Dict[str, float] = {
    "apartments":    5.0,
    "residential":   2.5,
    "house":         1.5,
    "detached":      1.5,
    "semidetached_house": 2.0,
    "terrace":       2.0,
    "bungalow":      1.0,
    "cabin":         1.0,
    "dormitory":     4.0,
    "yes":           2.0,   # generic residential
    "":              2.0,   # missing tag — urban area default
}

FLOORS_DEFAULT_URBAN:  float = 5.0
FLOORS_DEFAULT_RURAL:  float = 1.5
FLOORS_DEFAULT_GLOBAL: float = 2.0


def estimate_floors(
    buildings:          "pd.DataFrame",
    floor_col:          Optional[str] = None,
    building_type_col:  Optional[str] = "building",
    area_col:           Optional[str] = None,
    settlement_col:     Optional[str] = "_settlement",
    verbose:            bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    FIX-5: Four-level floor estimation with full fallback logging.

    Level 1 (best):  OSM 'building:levels' attribute.
    Level 2:         Building type lookup (FLOORS_BY_TYPE).
    Level 3:         Settlement-level median floors from level-1 buildings.
    Level 4 (last):  District median floors.

    Returns
    -------
    floors : np.ndarray  — estimated floors per building
    stats  : dict        — {level_name: n_buildings_using_that_level}
    """
    n = len(buildings)
    floors = np.full(n, np.nan, dtype=float)
    stats: Dict[str, int] = {
        "osm_attribute": 0,
        "building_type": 0,
        "settlement_median": 0,
        "district_median": 0,
    }

    # ── Level 1: OSM floor attribute ─────────────────────────────────────────
    floor_cols_to_try = [floor_col] if floor_col else []
    floor_cols_to_try += ["building:levels", "levels", "floors",
                          "building_levels", "etagi"]
    for fc in floor_cols_to_try:
        if fc and fc in buildings.columns:
            parsed = pd.to_numeric(buildings[fc], errors="coerce")
            valid = parsed.notna() & (parsed >= 1) & (parsed <= 60)
            floors[valid.values] = parsed[valid].values
            stats["osm_attribute"] = int(valid.sum())
            break

    missing = np.isnan(floors)

    # ── Level 2: building type ────────────────────────────────────────────────
    if building_type_col and building_type_col in buildings.columns and missing.any():
        btype = buildings[building_type_col].fillna("").str.lower()
        for idx in np.where(missing)[0]:
            t = btype.iloc[idx]
            if t in FLOORS_BY_TYPE:
                floors[idx] = FLOORS_BY_TYPE[t]
                stats["building_type"] += 1
        missing = np.isnan(floors)

    # ── Level 3: settlement median (from level-1 buildings) ──────────────────
    if settlement_col and settlement_col in buildings.columns and missing.any():
        sett = buildings[settlement_col]
        known = ~np.isnan(floors)
        sett_medians = (
            pd.Series(floors[known], index=buildings.index[known])
            .groupby(sett[known])
            .median()
        )
        for idx in np.where(missing)[0]:
            s = sett.iloc[idx]
            if s in sett_medians.index:
                floors[idx] = sett_medians[s]
                stats["settlement_median"] += 1
        missing = np.isnan(floors)

    # ── Level 4: district median ──────────────────────────────────────────────
    if missing.any():
        known_floors = floors[~np.isnan(floors)]
        district_med = float(np.median(known_floors)) if len(known_floors) > 0 else FLOORS_DEFAULT_GLOBAL
        floors[missing] = district_med
        stats["district_median"] = int(missing.sum())

    # Clip to [1, 60]
    floors = np.clip(floors, 1.0, 60.0)

    if verbose:
        print(f"  Floor estimation (n={n:,}):")
        for k, v in stats.items():
            pct = v / max(n, 1) * 100
            print(f"    {k:<22}  {v:>7,}  ({pct:5.1f}%)")
        print(f"    district median used:  {floors[np.isnan(np.full(n, np.nan))].mean():.1f} floors")

    return floors.astype(np.float32), stats


def compute_demand(
    buildings:       "gpd.GeoDataFrame",
    boundaries:      "gpd.GeoDataFrame",
    pop_col:         str = "population_2025",
    name_col:        str = "name",
    floor_col:       Optional[str] = None,
    building_type_col: Optional[str] = "building",
    min_demand:      float = 0.01,
    verbose:         bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    FIX-5: Distribute settlement population to buildings proportional to floor area.

    Formula per settlement s:
        demand_i = population_s × (floor_area_i / Σ floor_area in s)

    where floor_area_i = footprint_area_i × estimated_floors_i.

    Buildings not matched to any settlement receive:
        leftover_population / n_unmatched

    Returns
    -------
    demand  : (n,) float32 — population per building
    stats   : dict with summary counts
    """
    n = len(buildings)
    demand = np.zeros(n, dtype=np.float64)
    stats: Dict[str, int] = {"matched": 0, "unmatched": 0}

    # ── 1. Estimate floors ───────────────────────────────────────────────────
    floors, floor_stats = estimate_floors(
        buildings,
        floor_col=floor_col,
        building_type_col=building_type_col,
        verbose=verbose,
    )

    # ── 2. Floor area ─────────────────────────────────────────────────────────
    footprint = buildings.geometry.area.values.astype(np.float64)
    floor_area = footprint * floors.astype(np.float64)

    # ── 3. Load settlement population ────────────────────────────────────────
    bnd = boundaries.copy()
    if pop_col not in bnd.columns:
        raise KeyError(
            f"Column '{pop_col}' not found in boundaries. "
            f"Available: {list(bnd.columns)}"
        )
    bnd[pop_col] = pd.to_numeric(bnd[pop_col], errors="coerce").fillna(0)
    total_pop = float(bnd[pop_col].sum())

    if verbose:
        print(f"\n  Population from '{pop_col}': {total_pop:,.0f} total")

    # ── 4. Spatial join: building → settlement ────────────────────────────────
    bld_crs = buildings.to_crs(bnd.crs)
    pts = gpd.GeoDataFrame(
        {"_fa": floor_area},
        geometry=gpd.points_from_xy(
            bld_crs.geometry.centroid.x,
            bld_crs.geometry.centroid.y,
        ),
        crs=bld_crs.crs,
    )
    joined = gpd.sjoin(pts, bnd[[name_col, pop_col, "geometry"]],
                       how="left", predicate="within")

    # ── 5. Distribute population per settlement ───────────────────────────────
    for sett_name, grp in joined.groupby(name_col):
        total_fa = grp["_fa"].sum()
        if total_fa <= 0:
            continue
        pop_s = float(grp[pop_col].iloc[0])
        demand[grp.index] = pop_s * grp["_fa"].values / total_fa

    matched = joined[name_col].notna().sum()
    unmatched_idx = np.where(demand == 0)[0]
    stats["matched"]   = int(matched)
    stats["unmatched"] = len(unmatched_idx)

    # ── 6. Distribute leftover to unmatched buildings ─────────────────────────
    if stats["unmatched"] > 0:
        leftover = max(0.0, total_pop - demand.sum())
        if leftover > 0:
            demand[unmatched_idx] = leftover / stats["unmatched"]
        if verbose:
            print(f"  Unmatched buildings: {stats['unmatched']:,} "
                  f"→ {leftover / max(stats['unmatched'], 1):.2f} pop each (leftover)")

    # ── 7. Clip and validate ──────────────────────────────────────────────────
    demand = np.clip(demand, min_demand, None)
    demand = demand.astype(np.float32)

    _total_assigned = demand.sum()
    if verbose:
        print(f"\n  Demand stats:")
        print(f"    Total assigned: {_total_assigned:,.0f}  "
              f"(census: {total_pop:,.0f}, diff: {abs(_total_assigned-total_pop):.0f})")
        print(f"    Mean per building: {demand.mean():.2f}")
        print(f"    p50: {float(np.median(demand)):.2f}  "
              f"p95: {float(np.percentile(demand, 95)):.2f}")

    return demand, {**stats, **floor_stats}

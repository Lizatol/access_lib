"""
core/aggregation.py — Building → Settlement aggregation.

Audit fixes:
  • One canonical aggregation procedure used by all scenarios.
  • Primary output: settlement-level (weighted average by demand).
  • Secondary output: building-level detail preserved.
  • All scenario deltas are computed at the SAME aggregation level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
try:
    import geopandas as gpd
except ImportError:
    gpd = None
try:
    import duckdb as _duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False
from typing import Dict, Optional


# ─── Settlement-level aggregation ────────────────────────────────────────────

def buildings_to_settlements(
    buildings:   gpd.GeoDataFrame,    # building-level demand points
    boundaries:  gpd.GeoDataFrame,    # settlement polygons with 'name' column
    acc_column:  str,                 # accessibility column in buildings
    demand_col:  str = "demand",      # demand weight column (uniform if missing)
    name_col:    str = "name",        # settlement name column in boundaries
) -> pd.Series:
    """
    Aggregate building-level accessibility to settlement level.
    Returns Series indexed by settlement name.

    Method: weighted mean (weight = demand, or uniform if demand column absent).
    This is the CANONICAL aggregation for all scenario comparisons.
    """
    bldg_proj = buildings.to_crs(boundaries.crs)

    # Use demand column if present, otherwise uniform weight of 1
    _has_demand = demand_col in bldg_proj.columns
    _cols = [acc_column, "geometry"] + ([demand_col] if _has_demand else [])

    joined = gpd.sjoin(
        bldg_proj[_cols],
        boundaries[[name_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # FIX Pandas FutureWarning: select only non-key columns before groupby.apply
    _agg_cols = [acc_column] + ([demand_col] if _has_demand else [])
    _joined_agg = joined[[name_col] + _agg_cols]

    def _weighted_mean(g):
        vals = g[acc_column].fillna(0).values
        wts  = (g[demand_col].fillna(1).clip(lower=0.01).values
                if _has_demand else np.ones(len(g)))
        return float(np.average(vals, weights=wts))

    return (
        _joined_agg.groupby(name_col)
        .apply(_weighted_mean, include_groups=False)
        .rename(acc_column)
    )


def buildings_to_settlements_multi(
    buildings:   gpd.GeoDataFrame,
    boundaries:  gpd.GeoDataFrame,
    acc_columns: list[str],
    demand_col:  str = "demand",
    name_col:    str = "name",
) -> pd.DataFrame:
    """
    Aggregate multiple accessibility columns at once.
    Returns DataFrame with columns = acc_columns, index = settlement names.
    """
    bldg_proj = buildings.to_crs(boundaries.crs)
    joined = gpd.sjoin(
        bldg_proj[acc_columns + [demand_col, "geometry"]],
        boundaries[[name_col, "geometry"]],
        how="left",
        predicate="within",
    )
    def _wagg(g):
        wts = g[demand_col].fillna(1).clip(lower=0.01).values
        row = {}
        for col in acc_columns:
            vals = g[col].fillna(0).values
            row[col] = float(np.average(vals, weights=wts))
        return pd.Series(row)

    _multi_cols = [name_col] + acc_columns + [demand_col]
    _joined_sub = joined[_multi_cols]
    return _joined_sub.groupby(name_col).apply(_wagg, include_groups=False)


# ─── Settlement GeoDataFrame enrichment ──────────────────────────────────────

def enrich_settlements(
    boundaries:   gpd.GeoDataFrame,
    buildings:    gpd.GeoDataFrame,
    acc_arrays:   Dict[str, np.ndarray],   # {"scenario_name": acc_array, ...}
    demand_col:   str = "demand",
    name_col:     str = "name",
) -> gpd.GeoDataFrame:
    """
    Add accessibility columns and delta columns to settlements GeoDataFrame.

    Returns a new GeoDataFrame; never mutates boundaries in-place.

    All acc_arrays must correspond to the same demand points in buildings
    (same row order). Deltas are computed relative to the first key ("baseline").
    """
    result = boundaries.copy()
    cols = list(acc_arrays.keys())
    baseline_col = cols[0]

    # Attach acc arrays to buildings temporarily
    bldgs_tmp = buildings.copy()
    for name, arr in acc_arrays.items():
        bldgs_tmp[f"_acc_{name}"] = arr

    # Aggregate each
    for name in cols:
        sett_acc = buildings_to_settlements(
            bldgs_tmp, boundaries,
            acc_column=f"_acc_{name}",
            demand_col=demand_col,
            name_col=name_col,
        )
        result = result.merge(
            sett_acc.rename(f"acc_{name}").reset_index(),
            on=name_col, how="left",
        )

    # Compute deltas vs baseline
    for name in cols[1:]:
        result[f"delta_{name}"] = result[f"acc_{name}"] - result[f"acc_{baseline_col}"]

    return result



# ─── DuckDB-accelerated aggregation (optional fast path) ─────────────────────

def buildings_to_settlements_fast(
    buildings:   "gpd.GeoDataFrame",
    boundaries:  "gpd.GeoDataFrame",
    acc_column:  str,
    demand_col:  str = "demand",
    name_col:    str = "name",
) -> "pd.Series":
    """
    Fast path for building → settlement aggregation using DuckDB in-memory.

    Falls back to the standard geopandas-based function if DuckDB is not
    installed or the spatial extension is unavailable.

    Performance: typically 5-20× faster than sjoin-groupby for >50k rows.
    """
    if not _HAS_DUCKDB:
        return buildings_to_settlements(buildings, boundaries,
                                        acc_column, demand_col, name_col)
    try:
        con = _duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        bld = buildings.to_crs(boundaries.crs).copy()
        bld["_cx"] = bld.geometry.centroid.x
        bld["_cy"] = bld.geometry.centroid.y
        _has_demand = demand_col in bld.columns
        bld["_w"] = bld[demand_col].fillna(1).clip(lower=0.01) if _has_demand else 1.0
        bld["_v"] = bld[acc_column].fillna(0)

        bnd = boundaries.copy()
        bnd["_wkt"] = bnd.geometry.to_wkt()

        bld_df = bld[["_cx","_cy","_v","_w"]].reset_index(drop=True)
        bnd_df = bnd[[name_col,"_wkt"]].reset_index(drop=True)

        con.register("bld", bld_df)
        con.register("bnd", bnd_df)

        result = con.execute(f"""
            SELECT b.{name_col},
                   SUM(p._v * p._w) / NULLIF(SUM(p._w), 0) AS acc
            FROM bld p
            JOIN bnd b
              ON ST_Within(ST_Point(p._cx, p._cy), ST_GeomFromText(b._wkt))
            GROUP BY b.{name_col}
        """).df()
        con.close()
        return result.set_index(name_col)["acc"].rename(acc_column)
    except Exception:
        # DuckDB spatial extension may not be available in all environments
        return buildings_to_settlements(buildings, boundaries,
                                        acc_column, demand_col, name_col)


# ─── Gini coefficient ─────────────────────────────────────────────────────────

def gini(values: np.ndarray) -> float:
    """
    Gini coefficient for an accessibility array.
    0 = perfect equality, 1 = perfect inequality.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= 0)]
    if len(v) < 2:
        return float("nan")
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n)


# ─── Summary statistics ───────────────────────────────────────────────────────

def scenario_summary(
    acc_arrays:  Dict[str, np.ndarray],
    demand:      np.ndarray,
) -> pd.DataFrame:
    """
    Compute summary statistics for all scenario accessibility arrays.

    Columns: scenario, mean, std, p05, p25, p50, p75, p95, gini, n_zero
    All comparisons use DEMAND-WEIGHTED mean for consistency.
    """
    rows = []
    baseline_col = list(acc_arrays.keys())[0]
    baseline_wmean = _weighted_mean(acc_arrays[baseline_col], demand)

    for name, arr in acc_arrays.items():
        pos = arr[arr > 0]
        wmean = _weighted_mean(arr, demand)
        rows.append({
            "scenario":  name,
            "mean":      round(float(wmean), 5),
            "std":       round(float(pos.std()) if len(pos) > 0 else 0, 5),
            "p05":       round(float(np.percentile(pos, 5))  if len(pos) else 0, 5),
            "p25":       round(float(np.percentile(pos, 25)) if len(pos) else 0, 5),
            "p50":       round(float(np.percentile(pos, 50)) if len(pos) else 0, 5),
            "p75":       round(float(np.percentile(pos, 75)) if len(pos) else 0, 5),
            "p95":       round(float(np.percentile(pos, 95)) if len(pos) else 0, 5),
            "gini":      round(gini(arr), 4),
            "n_zero":    int((arr == 0).sum()),
            "delta":     round(float(wmean - baseline_wmean), 5),
        })
    return pd.DataFrame(rows)


def _weighted_mean(arr: np.ndarray, weights: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(a[mask], weights=w[mask]))

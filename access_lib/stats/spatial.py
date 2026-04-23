"""
stats/spatial.py — Spatial statistics: Moran's I, LISA, Spearman, Gini.

Self-contained module: no global state, no side effects.
All functions return DataFrames or enriched GeoDataFrames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore

from ..core.aggregation import gini   # re-export via stats too


# ─── Moran's I ───────────────────────────────────────────────────────────────

def morans_i(
    gdf:          gpd.GeoDataFrame,
    acc_columns:  List[str],
    weight_types: List[str] = ("Queen", "KNN-3"),
    permutations: int = 999,
) -> pd.DataFrame:
    """
    Compute Global Moran's I for each accessibility column × weight scheme.

    Returns DataFrame with columns:
        index, weights, I, EI, p, pattern
    """
    try:
        from libpysal.weights import Queen, KNN
        from esda.moran import Moran
    except ImportError:
        raise ImportError(
            "libpysal and esda are required for spatial statistics. "
            "Install: pip install libpysal esda"
        )

    # Drop NaN rows before building weights (critical for correctness)
    valid_cols = [c for c in acc_columns if c in gdf.columns and gdf[c].std() > 1e-6]
    if not valid_cols:
        print("⚠  No valid (non-constant) acc columns for Moran's I")
        return pd.DataFrame()

    gdf_c = gdf.dropna(subset=valid_cols).copy()
    n_dropped = len(gdf) - len(gdf_c)
    if n_dropped:
        print(f"  ▸ Dropped {n_dropped} NaN rows before Moran's I")
    if len(gdf_c) < 4:
        print("⚠  Too few rows for Moran's I")
        return pd.DataFrame()

    wbuilders = {
        "Queen": lambda g: Queen.from_dataframe(g, silence_warnings=True),
        "KNN-3": lambda g: KNN.from_dataframe(
            g, k=min(3, len(g) - 1), silence_warnings=True
        ),
    }

    rows = []
    print("▷ Global Moran's I")
    print(f"  {'Column':<22} {'W':<8} {'I':>7} {'E[I]':>7} {'p':>7}  pattern")
    print("  " + "─" * 60)

    for col in valid_cols:
        y = gdf_c[col].values.astype(np.float64)
        for wname in weight_types:
            if wname not in wbuilders:
                continue
            try:
                w = wbuilders[wname](gdf_c)
                w.transform = "R"
                mi = Moran(y, w, permutations=permutations)
                pat = (
                    "clustered"  if mi.I > mi.EI and mi.p_sim < 0.05 else
                    "dispersed"  if mi.I < mi.EI and mi.p_sim < 0.05 else
                    "random"
                )
                print(
                    f"  {col:<22} {wname:<8} "
                    f"{mi.I:>7.3f} {mi.EI:>7.3f} {mi.p_sim:>7.3f}  {pat}"
                )
                rows.append({
                    "index": col, "weights": wname,
                    "I": round(mi.I, 4), "EI": round(mi.EI, 4),
                    "p": round(mi.p_sim, 4), "pattern": pat,
                })
            except Exception as e:
                print(f"  {col:<22} {wname:<8} ERROR: {e}")

    return pd.DataFrame(rows)


# ─── Local Moran's I (LISA) ───────────────────────────────────────────────────

def lisa(
    gdf:          gpd.GeoDataFrame,
    col:          str,
    permutations: int = 999,
    p_threshold:  float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Compute Local Moran's I (LISA) for one accessibility column.

    Returns enriched GeoDataFrame with column 'lisa':
        "HH" — high surrounded by high (hot spot)
        "LL" — low surrounded by low  (cold spot)
        "HL" — high surrounded by low (outlier)
        "LH" — low surrounded by high (outlier)
        "ns" — not significant
    """
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran_Local
    except ImportError:
        raise ImportError("libpysal and esda are required for LISA.")

    gdf_c = gdf.dropna(subset=[col]).copy()
    if len(gdf_c) < 4:
        print(f"⚠  Too few rows for LISA on {col!r}")
        return gdf

    w = Queen.from_dataframe(gdf_c, silence_warnings=True)
    w.transform = "R"

    y = gdf_c[col].values.astype(np.float64)
    lm = Moran_Local(y, w, permutations=permutations)

    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    labels = np.where(
        lm.p_sim < p_threshold,
        np.vectorize(quad_map.get)(lm.q, "ns"),
        "ns",
    )
    gdf_c["lisa"] = labels

    hh = (labels == "HH").sum()
    ll = (labels == "LL").sum()
    sig = (labels != "ns").sum()
    print(f"  LISA [{col}]: significant={sig}  HH={hh}  LL={ll}")

    # Merge back to original index
    result = gdf.copy()
    result["lisa"] = "ns"
    result.loc[gdf_c.index, "lisa"] = gdf_c["lisa"]
    return result


# ─── Spearman correlations ────────────────────────────────────────────────────

def spearman_matrix(
    gdf:     gpd.GeoDataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Pairwise Spearman rank correlation between accessibility columns.
    Useful for checking scenario similarity.
    """
    from scipy.stats import spearmanr

    valid = [c for c in columns if c in gdf.columns]
    n = len(valid)
    mat = np.ones((n, n))
    for i, c1 in enumerate(valid):
        for j, c2 in enumerate(valid):
            if i < j:
                v = gdf[[c1, c2]].dropna()
                if len(v) > 3:
                    r, _ = spearmanr(v[c1], v[c2])
                    mat[i, j] = mat[j, i] = round(float(r), 3)
    return pd.DataFrame(mat, index=valid, columns=valid)


# ─── Inequality analysis ──────────────────────────────────────────────────────

def inequality_report(
    acc_arrays:  Dict[str, np.ndarray],
    demand:      np.ndarray,
    low_access_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Gini + low-access zone analysis for all scenario arrays.

    low_access_threshold: demand-weighted percentile (default: p25 of baseline).
    """
    baseline_arr = list(acc_arrays.values())[0]
    if low_access_threshold is None:
        low_access_threshold = float(np.percentile(
            baseline_arr[baseline_arr > 0], 25
        ))

    rows = []
    baseline_gini = gini(baseline_arr)

    for name, arr in acc_arrays.items():
        pos = arr[arr > 0]
        g = gini(arr)
        low_mask = arr < low_access_threshold
        low_pop  = float(demand[low_mask].sum())
        total_pop = float(demand.sum())
        rows.append({
            "scenario":       name,
            "gini":           round(g, 4),
            "delta_gini":     round(g - baseline_gini, 4),
            "low_access_pct": round(low_mask.mean() * 100, 1),
            "low_access_pop": round(low_pop, 0),
            "low_access_pop_share": round(
                low_pop / total_pop * 100 if total_pop > 0 else 0, 1
            ),
            "p10":  round(float(np.percentile(pos, 10)) if len(pos) else 0, 4),
            "p90":  round(float(np.percentile(pos, 90)) if len(pos) else 0, 4),
            "p90_p10_ratio": round(
                np.percentile(pos, 90) / max(np.percentile(pos, 10), 1e-6)
                if len(pos) > 0 else float("nan"), 2
            ),
        })

    df = pd.DataFrame(rows)
    print("\n  Inequality report:")
    print(df.to_string(index=False))
    return df

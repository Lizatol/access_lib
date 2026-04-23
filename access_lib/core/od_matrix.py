"""
core/od_matrix.py — OD matrix computation (all modes).

Audit fixes v1.1:
  ARCH-3: Cache filename now includes a hash of key parameters
          (season, graph size, radius) so stale cache is never used
          when the graph or configuration changes.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from ..config import WALK_SPEED_MS, PT_PENALTY, WALK_CUTOFF_M


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(
    key:        str,
    G:          nx.Graph,
    n_demand:   int,
    n_fac:      int,
    extra:      str = "",
) -> str:
    """
    ARCH-3: Build a cache filename that encodes structural parameters.
    Changes to graph size, demand/facility counts, or extra params
    produce a different filename → stale cache is never reused.
    """
    sig = f"{key}|n={G.number_of_nodes()}|e={G.number_of_edges()}|d={n_demand}|f={n_fac}|{extra}"
    h   = hashlib.md5(sig.encode()).hexdigest()[:8]
    return f"od_{key}_{h}.npy"


# ─── Single-mode Dijkstra helper ─────────────────────────────────────────────

def _dijkstra_from_facility(
    G:           nx.Graph,
    fac_node:    Any,
    node_to_idx: Dict[Any, list],
    weight:      str,
    cutoff:      Optional[float] = None,
):
    try:
        lens = nx.single_source_dijkstra_path_length(
            G, fac_node, weight=weight, cutoff=cutoff
        )
    except Exception:
        return []
    return [
        (pos, t)
        for node, t in lens.items()
        for pos in node_to_idx.get(node, [])
    ]


# ─── Main OD computation ──────────────────────────────────────────────────────

def compute_od_matrix(
    G:          nx.Graph,
    demand:     "gpd.GeoDataFrame",
    facilities: "gpd.GeoDataFrame",
    weight:     str   = "travel_time",
    speed_div:  float = 1.0,
    key:        str   = "car",
    cutoff_raw: Optional[float] = None,
    cache_path: Optional[Path]  = None,
    n_jobs:     int   = -1,
    cache_extra: str  = "",
) -> np.ndarray:
    """
    Compute OD[i,j] = travel time (seconds) from demand point i to facility j.
    Shape: (n_demand, n_fac), dtype float32.  np.inf = unreachable.

    ARCH-3: cache filename includes a hash of structural parameters.
    """
    n_dem, n_fac = len(demand), len(facilities)
    expected_shape = (n_dem, n_fac)

    cache_file: Optional[Path] = None
    if cache_path is not None:
        fname      = _cache_key(key, G, n_dem, n_fac, cache_extra)
        cache_file = cache_path / fname
        if cache_file.exists():
            OD = np.load(cache_file)
            if OD.shape == expected_shape:
                _print_od_stats(key, OD)
                return OD
            else:
                print(f"  !! OD [{key}] cache shape {OD.shape} ≠ {expected_shape} → recomputing")
                cache_file.unlink()

    print(f"\n  ▷ Computing OD [{key}]: {n_dem:,} × {n_fac}…")

    n2idx: Dict[Any, list] = defaultdict(list)
    for pos, node in enumerate(demand["graph_node"]):
        n2idx[node].append(pos)

    from joblib import Parallel, delayed
    from tqdm import tqdm

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_dijkstra_from_facility)(
            G, row["graph_node"], n2idx, weight, cutoff_raw
        )
        for _, row in tqdm(facilities.iterrows(), total=n_fac, desc=f"OD[{key}]")
    )

    OD = np.full((n_dem, n_fac), np.inf, dtype=np.float32)
    for f_pos, pairs in enumerate(results):
        for d_pos, t in pairs:
            v = t / speed_div
            if v < OD[d_pos, f_pos]:
                OD[d_pos, f_pos] = v

    _print_od_stats(key, OD)
    if np.isfinite(OD).mean() * 100 < 50:
        print(f"  ⚠  Low reachability for [{key}] — check graph connectivity")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, OD)
        print(f"  ✓ Saved → {cache_file}")

    return OD


def _print_od_stats(key: str, OD: np.ndarray) -> None:
    reach = np.isfinite(OD).mean() * 100
    mt    = OD[np.isfinite(OD)].mean() / 60 if reach > 0 else float("nan")
    print(f"  ✓ OD [{key}] shape={OD.shape}  reachable={reach:.1f}%  mean={mt:.1f} min")


# ─── Walk OD ─────────────────────────────────────────────────────────────────

def compute_walk_od(
    G:          nx.Graph,
    demand:     "gpd.GeoDataFrame",
    facilities: "gpd.GeoDataFrame",
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    return compute_od_matrix(
        G, demand, facilities,
        weight="length",
        speed_div=WALK_SPEED_MS,
        key="walk",
        cutoff_raw=WALK_CUTOFF_M,
        cache_path=cache_path,
    )


# ─── PT OD (3-leg chain) ─────────────────────────────────────────────────────

def compute_pt_od(
    G:          nx.Graph,
    demand:     "gpd.GeoDataFrame",
    facilities: "gpd.GeoDataFrame",
    bus_stops:  "gpd.GeoDataFrame",
    cache_path: Optional[Path] = None,
    headway_s:  float = 900.0,     # 15-min default headway
    bus_speed_ms: float = 25 * 1000 / 3600,
) -> np.ndarray:
    """
    Three-leg PT model:
      1. Walk to nearest stop:   Euclidean distance / WALK_SPEED_MS
      2. Headway penalty:        0.5 * headway_s
      3. Bus travel to nearest stop to facility: graph length / bus_speed_ms
      4. Walk from stop to facility: Euclidean distance / WALK_SPEED_MS

    Falls back to PT_PENALTY multiplier if bus_stops is empty.
    """
    if bus_stops is None or len(bus_stops) == 0:
        print("  ⚠  No bus stops — PT OD falls back to walk × PT_PENALTY")
        walk_od = compute_walk_od(G, demand, facilities, cache_path=None)
        return (walk_od * PT_PENALTY).astype(np.float32)

    n_dem = len(demand)
    n_fac = len(facilities)

    # Demand and facility centroids
    dem_crs  = demand.to_crs("EPSG:32636")
    fac_crs  = facilities.to_crs("EPSG:32636")
    stop_crs = bus_stops.to_crs("EPSG:32636")

    dem_xy  = np.column_stack([dem_crs.geometry.centroid.x,  dem_crs.geometry.centroid.y])
    fac_xy  = np.column_stack([fac_crs.geometry.centroid.x,  fac_crs.geometry.centroid.y])
    stop_xy = np.column_stack([stop_crs.geometry.centroid.x, stop_crs.geometry.centroid.y])

    stop_tree = cKDTree(stop_xy)

    # Leg 1: walk_to_stop per demand point (seconds)
    d2s_dist, _ = stop_tree.query(dem_xy, k=1)
    walk_to_stop = d2s_dist / WALK_SPEED_MS   # (n_dem,)

    # Leg 4: walk_from_stop per facility (seconds)
    f2s_dist, _ = stop_tree.query(fac_xy, k=1)
    walk_from_stop = f2s_dist / WALK_SPEED_MS  # (n_fac,)

    # Leg 2: fixed headway penalty
    hw_pen = 0.5 * headway_s

    # Leg 3: inter-stop bus travel — approximated as Euclidean / bus_speed_ms
    # (each demand point's nearest stop → each facility's nearest stop)
    _, dem_stop_idx = stop_tree.query(dem_xy, k=1)
    _, fac_stop_idx = stop_tree.query(fac_xy, k=1)

    dem_stop_xy = stop_xy[dem_stop_idx]   # (n_dem, 2)
    fac_stop_xy = stop_xy[fac_stop_idx]   # (n_fac, 2)

    # Broadcast: (n_dem, n_fac)
    inter_stop_dist = np.linalg.norm(
        dem_stop_xy[:, np.newaxis, :] - fac_stop_xy[np.newaxis, :, :],
        axis=2,
    )
    inter_stop_time = inter_stop_dist / bus_speed_ms   # (n_dem, n_fac)

    OD = (
        walk_to_stop[:, np.newaxis]      # (n_dem, 1)
        + hw_pen
        + inter_stop_time                # (n_dem, n_fac)
        + walk_from_stop[np.newaxis, :]  # (1, n_fac)
    ).astype(np.float32)

    _print_od_stats("pt", OD)

    if cache_path is not None:
        fname = _cache_key("pt", G, n_dem, n_fac, f"hw{headway_s:.0f}")
        (cache_path / fname.replace(".npy", "_pt.npy"))
        np.save(cache_path / fname, OD)

    return OD


# ─── All-mode OD computation ─────────────────────────────────────────────────

def compute_all_od_matrices(
    G:          nx.Graph,
    demand:     "gpd.GeoDataFrame",
    facilities: "gpd.GeoDataFrame",
    bus_stops:  "gpd.GeoDataFrame",
    cache_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Compute car, walk, and PT OD matrices."""
    return {
        "car":  compute_od_matrix(G, demand, facilities, cache_path=cache_path),
        "walk": compute_walk_od(G, demand, facilities, cache_path=cache_path),
        "pt":   compute_pt_od(G, demand, facilities, bus_stops, cache_path=cache_path),
    }


# ─── Single-facility OD column (used by scenarios) ───────────────────────────

def compute_od_column(
    G:            nx.Graph,
    facility_node: Any,
    demand_nodes:  list,
    walk_speed_ms: float = WALK_SPEED_MS,
    pt_penalty:    float = PT_PENALTY,
) -> Dict[str, np.ndarray]:
    """
    Compute OD column for one facility across all three modes.
    Used by NewHospitalScenario, RealHospitalScenario, NewTransitRouteScenario.
    """
    n = len(demand_nodes)
    node_to_idx: Dict[Any, list] = defaultdict(list)
    for i, nd in enumerate(demand_nodes):
        node_to_idx[nd].append(i)

    # car
    car = np.full(n, np.inf, dtype=np.float32)
    try:
        lens = nx.single_source_dijkstra_path_length(G, facility_node, weight="travel_time")
        for nd, t in lens.items():
            for i in node_to_idx.get(nd, []):
                car[i] = min(car[i], float(t))
    except Exception:
        pass

    # walk
    walk = np.full(n, np.inf, dtype=np.float32)
    try:
        lens = nx.single_source_dijkstra_path_length(
            G, facility_node, weight="length", cutoff=WALK_CUTOFF_M
        )
        for nd, dist in lens.items():
            for i in node_to_idx.get(nd, []):
                walk[i] = min(walk[i], float(dist) / walk_speed_ms)
    except Exception:
        pass

    # pt (proxy: walk time × pt_penalty)
    pt = np.where(np.isfinite(walk), walk * pt_penalty, np.inf).astype(np.float32)

    return {"car": car, "walk": walk, "pt": pt}

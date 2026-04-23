"""
core/graph.py — Road network graph construction and manipulation.

Audit fixes:
  • Graph construction is a pure function (no global G).
  • edge_remove() enables proper road closure — modifies a copy, not the original.
  • snap_to_node() is a utility used by all scenarios.
  • build_graph() is cacheable via joblib.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from ..config import ROAD_SPEEDS, WALK_SPEED_MS

# Optional OSMnx import (not required if graph is loaded from cache)
try:
    import osmnx as ox
    _HAS_OSMNX = True
except ImportError:
    _HAS_OSMNX = False


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph(
    boundary_polygon,
    crs_metric: str = "EPSG:32636",
    network_type: str = "drive",
    cache_path: Optional[Path] = None,
) -> nx.MultiDiGraph:
    """
    Download or load a projected road network graph.

    Returns a MultiDiGraph with edge attributes:
      length        — metres
      travel_time   — seconds (at ROAD_SPEEDS)
      highway       — original OSM tag

    Caches the graph as GraphML so subsequent calls avoid re-downloading.
    """
    if not _HAS_OSMNX:
        raise ImportError("osmnx is required for graph construction.")

    cache_file = (cache_path / "road_graph.graphml") if cache_path else None
    if cache_file and cache_file.exists():
        print(f"  ▸ Loading graph from cache: {cache_file}")
        G = ox.load_graphml(cache_file)
        return G

    print(f"  ▸ Downloading road network ({network_type})…")
    G = ox.graph_from_polygon(boundary_polygon, network_type=network_type)

    # Project to metric CRS
    G = ox.project_graph(G, to_crs=crs_metric)

    # Assign travel_time from ROAD_SPEEDS
    G = _assign_travel_times(G)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(G, cache_file)
        print(f"  ✓ Graph saved → {cache_file}")

    return G


def _assign_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Assign travel_time (seconds) to each edge based on highway type and ROAD_SPEEDS.
    Overwrites any existing travel_time attribute.
    """
    for u, v, k, data in G.edges(data=True, keys=True):
        hw   = data.get("highway", "unclassified")
        hw   = hw[0] if isinstance(hw, list) else hw
        spd  = ROAD_SPEEDS.get(hw, 40.0)
        length_m = float(data.get("length", 1.0))
        G[u][v][k]["travel_time"] = length_m / (spd * 1000 / 3600)
    return G


# ─── Graph surgery for road closure ──────────────────────────────────────────

def remove_edges(
    G: nx.MultiDiGraph,
    edges: list[Tuple[Any, Any]],
) -> nx.MultiDiGraph:
    """
    Return a copy of G with the specified edges removed (weight set to ∞).

    Audit fix: road closure must be edge-aware.
    We set weight=inf rather than deleting, so the graph topology is preserved
    and path-existence checks still work — only routing changes.

    edges: list of (u, v) tuples (all parallel keys between u,v are removed).
    """
    G2 = G.copy()
    for u, v in edges:
        if G2.has_edge(u, v):
            for k in list(G2[u][v].keys()):
                G2[u][v][k]["travel_time"] = float("inf")
                G2[u][v][k]["length"]      = float("inf")
    return G2


def find_edges_in_polygon(G: nx.MultiDiGraph, polygon) -> list[Tuple[Any, Any]]:
    """
    Find all edges whose midpoint falls inside the given polygon.
    Used to identify roads that pass through an area (e.g. a settlement polygon).
    Returns list of (u, v) tuples.
    """
    from shapely.geometry import Point
    node_pos = dict(G.nodes(data=True))
    affected = []
    for u, v, data in G.edges(data=True):
        x_u = node_pos[u].get("x", 0)
        y_u = node_pos[u].get("y", 0)
        x_v = node_pos[v].get("x", 0)
        y_v = node_pos[v].get("y", 0)
        mid = Point((x_u + x_v) / 2, (y_u + y_v) / 2)
        if polygon.contains(mid):
            affected.append((u, v))
    return affected


def find_critical_edges(
    G:              nx.MultiDiGraph,
    demand_nodes:   list,
    facility_nodes: list,
    top_k:          int = 5,
) -> list[Tuple[Any, Any]]:
    """
    Быстрая эвристика критичных рёбер — через степень узлов (degree centrality).

    edge_betweenness_centrality на 168k узлах считается часами.
    Вместо этого: находим узлы-«мосты» по высокой степени на путях
    между спросом и учреждениями, берём рёбра между ними.

    Метод: строим «скелет» — узлы с самым высоким in+out degree в
    окрестности учреждений, берём рёбра между ними.
    """
    from collections import Counter

    # Собираем узлы в радиусе 2 хопов от учреждений
    relevant: set = set(facility_nodes)
    for fac in facility_nodes:
        if fac in G:
            relevant.update(G.predecessors(fac))
            relevant.update(G.successors(fac))
            for nb in list(G.successors(fac)):
                relevant.update(G.successors(nb))

    # Степень узлов (in + out) — высокая степень = потенциальный узкий проход
    degree_counter: Counter = Counter()
    for n in relevant:
        if n in G:
            degree_counter[n] = G.in_degree(n) + G.out_degree(n)

    # Топ-узлы по степени
    top_nodes = [n for n, _ in degree_counter.most_common(top_k * 4)]

    # Рёбра между топ-узлами
    edges: list[Tuple[Any, Any]] = []
    seen: set = set()
    for u in top_nodes:
        for v in G.successors(u):
            if v in set(top_nodes) and (u, v) not in seen:
                edges.append((u, v))
                seen.add((u, v))
                if len(edges) >= top_k:
                    return edges

    # Fallback: просто рёбра от самого загруженного узла
    if not edges and top_nodes:
        u = top_nodes[0]
        for v in list(G.successors(u))[:top_k]:
            edges.append((u, v))

    return edges[:top_k]


# ─── Node utilities ───────────────────────────────────────────────────────────

def snap_to_node(
    G: nx.MultiDiGraph,
    lon: float,
    lat: float,
    crs_metric: str = "EPSG:32636",
    crs_input:  str = "EPSG:4326",
) -> Any:
    """
    Snap geographic coordinates to the nearest graph node.
    Returns the node id.
    """
    import pyproj
    transformer = pyproj.Transformer.from_crs(crs_input, crs_metric, always_xy=True)
    x, y = transformer.transform(lon, lat)

    node_ids  = np.array(list(G.nodes()))
    node_xy   = np.array([[G.nodes[n].get("x", 0), G.nodes[n].get("y", 0)]
                           for n in node_ids])
    _, idx    = cKDTree(node_xy).query([[x, y]])
    return int(node_ids[int(idx)])


def compute_od_column(
    G:              nx.MultiDiGraph,
    source_node:    Any,
    demand_nodes:   list,
    weight:         str = "travel_time",
    walk_speed_ms:  float = WALK_SPEED_MS,
    pt_penalty:     float = 1.5,
) -> Dict[str, np.ndarray]:
    """
    Compute OD columns (one column per mode) from a single facility node
    to all demand nodes.

    Returns dict: {"car": array, "walk": array, "pt": array}
    Each array has shape (n_demand,), dtype float32, inf=unreachable.
    """
    lens_time = dict(nx.single_source_dijkstra_path_length(
        G, source_node, weight="travel_time"
    ))
    lens_len  = dict(nx.single_source_dijkstra_path_length(
        G, source_node, weight="length"
    ))

    n = len(demand_nodes)
    car  = np.full(n, np.inf, dtype=np.float32)
    walk = np.full(n, np.inf, dtype=np.float32)
    pt   = np.full(n, np.inf, dtype=np.float32)

    for i, node in enumerate(demand_nodes):
        t  = lens_time.get(node, np.inf)
        l  = lens_len.get(node, np.inf)
        car[i]  = float(t)
        walk[i] = float(l) / walk_speed_ms
        pt[i]   = float(t) * pt_penalty

    return {"car": car, "walk": walk, "pt": pt}


def affected_demand_nodes(
    OD_before:    np.ndarray,
    OD_after:     np.ndarray,
    threshold_s:  float = 60.0,
) -> np.ndarray:
    """
    Return boolean mask of demand points whose travel time changed by
    more than threshold_s seconds in any direction after an OD modification.

    Used by road closure to identify truly impacted buildings
    rather than showing global delta maps.
    """
    delta = np.abs(OD_after - OD_before)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    return delta.max(axis=1) > threshold_s

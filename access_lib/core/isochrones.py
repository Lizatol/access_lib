"""
core/isochrones.py — Network-based изохроны по дорожному графу.

Не круги — реальные зоны достижимости по дорогам.
Аудит: "нужны isochrones по графу, а не buffer/circle"
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import geopandas as gpd
    import networkx as nx
    from shapely.geometry import MultiPolygon, Point
    from shapely.ops import unary_union
except ImportError:
    gpd = nx = None  # type: ignore


def build_isochrone(
    G,                          # nx.MultiDiGraph (projected, с travel_time)
    center_node:    Any,        # node id (facility node)
    time_limit_s:   float,      # радиус в секундах (3600 = 60 мин)
    buffer_m:       float = 50, # буфер вокруг рёбер (метры)
    crs_metric:     str = "EPSG:32636",
) -> "shapely.geometry.Polygon":
    """
    Строит изохрону как union рёбер графа в пределах time_limit_s + buffer.

    Алгоритм:
      1. Dijkstra от center_node с cutoff=time_limit_s
      2. Берём все рёбра где хотя бы один конец достижим
      3. Буферизуем и объединяем в полигон
    """
    # Все узлы в пределах time_limit_s
    reachable = nx.single_source_dijkstra_path_length(
        G, center_node, cutoff=time_limit_s, weight="travel_time"
    )

    # Координаты достижимых узлов
    node_data = dict(G.nodes(data=True))
    points = [
        Point(node_data[n]["x"], node_data[n]["y"])
        for n in reachable
        if n in node_data
    ]
    if not points:
        # fallback: точечный буфер
        cx = node_data[center_node]["x"]
        cy = node_data[center_node]["y"]
        return Point(cx, cy).buffer(buffer_m)

    # Берём рёбра графа между достижимыми узлами
    reachable_set = set(reachable.keys())
    edge_geoms = []
    for u, v, data in G.edges(data=True):
        if u in reachable_set or v in reachable_set:
            xu = node_data.get(u, {}).get("x", 0)
            yu = node_data.get(u, {}).get("y", 0)
            xv = node_data.get(v, {}).get("x", 0)
            yv = node_data.get(v, {}).get("y", 0)
            from shapely.geometry import LineString
            edge_geoms.append(LineString([(xu, yu), (xv, yv)]))

    if not edge_geoms:
        return unary_union([p.buffer(buffer_m) for p in points])

    # Буферизуем рёбра и объединяем
    buffered = [g.buffer(buffer_m) for g in edge_geoms]
    polygon  = unary_union(buffered)
    return polygon


def build_isochrones_for_facilities(
    G,
    facilities_gdf,             # GeoDataFrame с graph_node
    time_limits_min: List[float] = [20, 40, 60],
    buffer_m:        float = 50,
    crs_metric:      str = "EPSG:32636",
) -> "gpd.GeoDataFrame":
    """
    Строит изохроны для всех учреждений и всех временных порогов.
    Возвращает GeoDataFrame с колонками: facility_idx, time_min, geometry, fullname.
    """
    rows = []
    for idx, row in facilities_gdf.iterrows():
        node = row.get("graph_node")
        if node is None or node not in G.nodes:
            continue
        name = row.get("fullname", row.get("name", f"fac_{idx}"))
        spec = row.get("specialization", "unknown")
        for t_min in time_limits_min:
            t_s = t_min * 60
            try:
                iso = build_isochrone(G, node, t_s, buffer_m, crs_metric)
            except Exception:
                continue
            rows.append({
                "facility_idx": idx,
                "fullname":     name,
                "specialization": spec,
                "time_min":     t_min,
                "geometry":     iso,
            })

    if not rows:
        return gpd.GeoDataFrame(columns=["facility_idx","fullname","time_min","geometry"],
                                 crs=crs_metric)
    return gpd.GeoDataFrame(rows, crs=crs_metric)


def classify_buildings_by_isochrone(
    buildings_gdf,
    isochrone_polygon,          # один полигон изохроны
    time_label:  str = "60min",
) -> "gpd.GeoDataFrame":
    """
    Добавляет булеву колонку f"within_{time_label}" в buildings_gdf.
    Здания внутри изохроны = доступные, снаружи = недоступные.
    """
    gdf = buildings_gdf.copy()
    # Используем центроид для полигонов
    geom = gdf.geometry
    if not (geom.geom_type == "Point").all():
        geom = geom.centroid

    # Исправляем невалидную геометрию перед within
    _iso = isochrone_polygon.buffer(0) if not isochrone_polygon.is_valid else isochrone_polygon
    gdf[f"within_{time_label}"] = geom.within(_iso)
    return gdf


def unserved_population(
    buildings_gdf,
    threshold_col:   str   = "access_total",
    threshold_val:   float = None,
    within_col:      str   = None,         # альтернатива: булева колонка "within_60min"
    demand_col:      str   = "demand",
) -> Dict[str, float]:
    """
    Считает количество и долю зданий/населения вне зоны доступности.

    Два режима:
      1. threshold_col/threshold_val — здания с низким индексом E2SFCA
      2. within_col — здания вне изохроны

    Возвращает dict с ключами: n_buildings, pct_buildings, demand, pct_demand.
    """
    if within_col and within_col in buildings_gdf.columns:
        mask_out = ~buildings_gdf[within_col]
    elif threshold_col in buildings_gdf.columns:
        if threshold_val is None:
            threshold_val = float(
                buildings_gdf[threshold_col].quantile(0.25)
            )
        mask_out = buildings_gdf[threshold_col] < threshold_val
    else:
        raise ValueError(f"Ни '{threshold_col}' ни '{within_col}' не найдены в GDF.")

    n_total  = len(buildings_gdf)
    n_out    = int(mask_out.sum())

    if demand_col in buildings_gdf.columns:
        dem_total = float(buildings_gdf[demand_col].sum())
        dem_out   = float(buildings_gdf.loc[mask_out, demand_col].sum())
    else:
        dem_total = float(n_total)
        dem_out   = float(n_out)

    return {
        "n_buildings":    n_out,
        "pct_buildings":  round(n_out / max(n_total, 1) * 100, 2),
        "demand":         round(dem_out, 0),
        "pct_demand":     round(dem_out / max(dem_total, 1) * 100, 2),
        "threshold":      threshold_val,
        "mode":           within_col or threshold_col,
    }

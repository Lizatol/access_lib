"""
viz/interactive.py — Интерактивные карты Folium.

1. Изохроны учреждений с подсветкой зданий внутри/снаружи
2. Карта недоступных зданий (unserved population)
3. Карта по специализациям
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import folium
    from folium.plugins import MarkerCluster
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore


# Цвета специализаций
SPEC_COLORS = {
    "hospital_full":          "#e74c3c",
    "outpatient_specialized": "#e67e22",
    "primary_therapist":      "#27ae60",
    "primary_basic":          "#8e44ad",
    "pediatric":              "#3498db",
    "maternal":               "#e91e8c",
    "unknown":                "#95a5a6",
}


def _check_folium():
    if not _HAS_FOLIUM:
        raise ImportError("folium не установлен: pip install folium")


def isochrone_map(
    G,
    facilities_gdf,
    buildings_gdf,
    time_limit_min:  float = 60.0,
    buffer_m:        float = 50,
    crs_metric:      str   = "EPSG:32636",
    crs_wgs:         str   = "EPSG:4326",
    max_buildings:   int   = 5000,   # лимит точек для производительности
) -> "folium.Map":
    """
    Интерактивная карта: изохрона по графу + здания (зелёные/красные).

    Клик по учреждению → показывает его зону и подсвечивает здания.
    """
    _check_folium()
    from accessibility_lib.core.isochrones import build_isochrone

    # Центр карты
    fac_wgs = facilities_gdf.to_crs(crs_wgs)
    center  = [fac_wgs.geometry.y.mean(), fac_wgs.geometry.x.mean()]

    m = folium.Map(location=center, zoom_start=11,
                   tiles="CartoDB positron")

    bld_wgs = buildings_gdf.to_crs(crs_wgs)
    # Downsample если зданий очень много
    if len(bld_wgs) > max_buildings:
        bld_wgs = bld_wgs.sample(max_buildings, random_state=42)

    geom_bld = bld_wgs.geometry
    if not (geom_bld.geom_type == "Point").all():
        geom_bld = geom_bld.centroid

    # Изохрона от первого учреждения как пример (потом можно перебирать)
    for _, fac_row in fac_wgs.iterrows():
        node  = fac_row.get("graph_node")
        name  = fac_row.get("fullname", fac_row.get("name", "?"))
        spec  = fac_row.get("specialization", "unknown")
        color = SPEC_COLORS.get(spec, "#333333")

        if node is None or node not in G.nodes:
            # Просто маркер
            folium.CircleMarker(
                location=[fac_row.geometry.y, fac_row.geometry.x],
                radius=8, color=color, fill=True, fill_opacity=0.9,
                popup=folium.Popup(f"<b>{name}</b><br>{spec}", max_width=200),
            ).add_to(m)
            continue

        # Строим изохрону
        try:
            iso = build_isochrone(G, node, time_limit_min * 60,
                                  buffer_m, crs_metric)
            iso_wgs = gpd.GeoDataFrame(geometry=[iso], crs=crs_metric).to_crs(crs_wgs)
            iso_geom_wgs = iso_wgs.geometry.iloc[0]

            folium.GeoJson(
                iso_geom_wgs.__geo_interface__,
                style_function=lambda x, c=color: {
                    "fillColor": c, "color": c,
                    "fillOpacity": 0.15, "weight": 2,
                },
                tooltip=f"{name} — {time_limit_min:.0f} мин",
            ).add_to(m)

            # Здания внутри/снаружи
            _iso_proj = iso
            _pts_proj = gpd.GeoDataFrame(
                geometry=buildings_gdf.to_crs(crs_metric).geometry
                         if (buildings_gdf.geometry.geom_type == "Point").all()
                         else buildings_gdf.to_crs(crs_metric).geometry.centroid,
                crs=crs_metric,
            )
            _within = _pts_proj.geometry.within(_iso_proj.buffer(0))
            _inside  = bld_wgs[_within.values[:len(bld_wgs)]]
            _outside = bld_wgs[~_within.values[:len(bld_wgs)]]

            # Зелёные (доступные)
            _g_inside = _inside.geometry if (_inside.geometry.geom_type == "Point").all() \
                        else _inside.geometry.centroid
            for _pt in _g_inside:
                folium.CircleMarker(
                    [_pt.y, _pt.x], radius=2,
                    color="green", fill=True, fill_opacity=0.5, weight=0,
                ).add_to(m)

            # Красные (недоступные)
            _g_outside = _outside.geometry if (_outside.geometry.geom_type == "Point").all() \
                         else _outside.geometry.centroid
            for _pt in _g_outside:
                folium.CircleMarker(
                    [_pt.y, _pt.x], radius=2,
                    color="red", fill=True, fill_opacity=0.4, weight=0,
                ).add_to(m)

        except Exception as _e:
            pass

        # Маркер учреждения
        folium.CircleMarker(
            location=[fac_row.geometry.y, fac_row.geometry.x],
            radius=8, color=color, fill=True, fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{name}</b><br>{spec}<br>Изохрона: {time_limit_min:.0f} мин",
                max_width=250,
            ),
        ).add_to(m)

    return m


def unserved_map(
    buildings_gdf,
    threshold_col: str   = "acc_combined",
    threshold_val: float = None,
    demand_col:    str   = "demand",
    crs_wgs:       str   = "EPSG:4326",
    max_points:    int   = 8000,
) -> "folium.Map":
    """
    Карта недоступных зданий (красные = unserved, серые = served).
    """
    _check_folium()

    gdf_wgs = buildings_gdf.to_crs(crs_wgs)

    if threshold_val is None and threshold_col in gdf_wgs.columns:
        threshold_val = float(gdf_wgs[threshold_col].quantile(0.25))

    if threshold_col in gdf_wgs.columns:
        mask_out = gdf_wgs[threshold_col] < threshold_val
    else:
        mask_out = pd.Series(False, index=gdf_wgs.index)

    n_out = mask_out.sum()
    dem_out = gdf_wgs.loc[mask_out, demand_col].sum() if demand_col in gdf_wgs.columns else n_out

    # Центр
    geom_c = gdf_wgs.geometry.centroid if (gdf_wgs.geometry.geom_type != "Point").any() \
             else gdf_wgs.geometry
    center = [geom_c.y.mean(), geom_c.x.mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Downsample
    if len(gdf_wgs) > max_points:
        gdf_wgs = gdf_wgs.sample(max_points, random_state=42)
        mask_out = mask_out.loc[gdf_wgs.index]

    geom = gdf_wgs.geometry
    if not (geom.geom_type == "Point").all():
        geom = geom.centroid

    for i, (pt, is_out) in enumerate(zip(geom, mask_out)):
        folium.CircleMarker(
            [pt.y, pt.x], radius=2,
            color="red" if is_out else "#aaaaaa",
            fill=True, fill_opacity=0.6 if is_out else 0.3,
            weight=0,
        ).add_to(m)

    # Легенда
    legend = f"""
    <div style="position:fixed; bottom:30px; left:30px; background:white;
                padding:10px; border:1px solid #ccc; font-size:13px; z-index:9999">
      <b>Недоступные здания</b><br>
      <span style="color:red">●</span> Unserved: {n_out:,} зд. ({n_out/max(len(buildings_gdf),1)*100:.1f}%)<br>
      <span style="color:#aaa">●</span> Served<br>
      Порог: {threshold_val:.3f}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


def specialization_map(
    facilities_gdf,
    boundaries_gdf,
    settlement_summary_df,      # DataFrame из settlement_layer_summary()
    crs_wgs: str = "EPSG:4326",
) -> "folium.Map":
    """
    Интерактивная карта по специализациям:
    хлороплет поселений по каждой группе + маркеры учреждений.
    """
    _check_folium()
    import branca.colormap as cm

    bnd_wgs = boundaries_gdf.merge(
        settlement_summary_df, left_on="name", right_on="settlement_name", how="left"
    ).to_crs(crs_wgs)
    fac_wgs = facilities_gdf.to_crs(crs_wgs)

    center = [bnd_wgs.geometry.centroid.y.mean(), bnd_wgs.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    acc_cols = [c for c in settlement_summary_df.columns if c.startswith("access_")]
    colors_seq = ["YlOrRd", "Blues", "Greens", "Purples"]

    for i, col in enumerate(acc_cols):
        if col not in bnd_wgs.columns:
            continue
        vals = bnd_wgs[col].dropna()
        if len(vals) == 0:
            continue

        cmap = cm.LinearColormap(
            ["#ffffff", colors_seq[i % len(colors_seq)][-2:]
             if False else ["white", "#e74c3c", "#c0392b"][: 2]],
            vmin=float(vals.min()), vmax=float(vals.max()),
        )

        layer = folium.FeatureGroup(name=col.replace("access_", "").capitalize(),
                                     show=(i == 0))
        for _, row in bnd_wgs.iterrows():
            val = row.get(col)
            if val is None or (hasattr(val, "__class__") and str(type(val)) == "<class 'float'>"
                               and np.isnan(val)):
                fill = "#cccccc"
            else:
                fill = cmap(float(val))
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, f=fill: {
                    "fillColor": f, "color": "#555", "weight": 1, "fillOpacity": 0.6
                },
                tooltip=f"{row.get('name','?')}: {col}={val:.3f}" if val else row.get("name","?"),
            ).add_to(layer)
        layer.add_to(m)

    # Маркеры учреждений
    fac_layer = folium.FeatureGroup(name="Учреждения", show=True)
    for _, row in fac_wgs.iterrows():
        spec  = row.get("specialization", "unknown")
        color = SPEC_COLORS.get(spec, "#333")
        name  = row.get("fullname", row.get("name", "?"))
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=7, color=color, fill=True, fill_opacity=0.9,
            popup=folium.Popup(f"<b>{name}</b><br>{spec}", max_width=200),
        ).add_to(fac_layer)
    fac_layer.add_to(m)

    folium.LayerControl().add_to(m)
    return m

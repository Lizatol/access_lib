"""
viz/interactive_map.py — Folium interactive HTML map.

Features:
  - Building-level accessibility (coloured dots, toggle layers)
  - Settlement choropleth with click → full stats popup
  - Facility markers with popup (name, type, capacity)
  - Scenario layer switcher (S1–S4 accessibility)
  - New facility star marker
  - Closed road highlight
  - Fullscreen, measure tool, scalebar
  - Legend for accessibility quintiles
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

try:
    import geopandas as gpd
    import folium
    from folium.plugins import Fullscreen, MeasureControl, Search
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False


_QUINTILE_COLORS = ["#C62828","#E8622A","#F9C74F","#90BE6D","#2D6A4F"]
_SPEC_COLORS = {
    "hospital_full":          "#C62828",
    "outpatient_specialized": "#E8622A",
    "primary_therapist":      "#1A7A4A",
    "primary_basic":          "#5C2D91",
    "pediatric":              "#1A5EA8",
    "maternal":               "#AD1457",
    "unknown":                "#7A7A7A",
}
_SPEC_LABELS = {
    "hospital_full":          "Hospital (full)",
    "outpatient_specialized": "Outpatient specialist",
    "primary_therapist":      "Primary care (therapist)",
    "primary_basic":          "Primary care (FAP/basic)",
    "pediatric":              "Paediatric",
    "maternal":               "Maternal",
    "unknown":                "Other",
}


def _quintile_color(val: float, breaks: np.ndarray) -> str:
    if val <= 0 or len(breaks) < 2:
        return "#C8C8C8"
    for i, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
        if lo <= val <= hi:
            return _QUINTILE_COLORS[min(i, len(_QUINTILE_COLORS)-1)]
    return _QUINTILE_COLORS[-1]


def _classify_breaks(vals, n=5):
    has = vals > 0
    if has.sum() == 0:
        return np.zeros(n+1)
    breaks = np.percentile(vals[has], np.linspace(0, 100, n+1))
    return np.unique(breaks)


def build_interactive_map(
    buildings_gdf:    gpd.GeoDataFrame,
    settlements_gdf:  gpd.GeoDataFrame,
    facilities_gdf:   gpd.GeoDataFrame,
    acc_col:          str = "acc_combined",
    scenario_cols:    Optional[Dict[str, str]] = None,
    new_facility_gdf: Optional[gpd.GeoDataFrame] = None,
    closed_roads_gdf: Optional[gpd.GeoDataFrame] = None,
    max_buildings:    int = 8000,
    save_path:        Optional[Path] = None,
) -> "folium.Map":
    """
    Build a fully interactive Folium map.

    Parameters
    ----------
    buildings_gdf    : GeoDataFrame with acc_col and optional scenario columns.
    settlements_gdf  : GeoDataFrame with settlement polygons; must have 'name' and acc_col.
    facilities_gdf   : GeoDataFrame with 'specialization', 'fullname'/'name', 'effective_capacity'.
    acc_col          : Column name for baseline accessibility.
    scenario_cols    : {display_name: column_name} dict for scenario layers.
    max_buildings    : Downsample buildings for performance.
    save_path        : If provided, saves as HTML.
    """
    if not _HAS_FOLIUM:
        raise ImportError("Install folium: pip install folium")

    CRS_WGS = "EPSG:4326"
    bldg    = buildings_gdf.to_crs(CRS_WGS).copy()
    sett    = settlements_gdf.to_crs(CRS_WGS).copy()
    fac     = facilities_gdf.to_crs(CRS_WGS).copy()

    # Centroid for buildings
    geom = bldg.geometry
    if not (geom.geom_type == "Point").all():
        bldg["_cx"] = geom.centroid.x
        bldg["_cy"] = geom.centroid.y
    else:
        bldg["_cx"] = geom.x
        bldg["_cy"] = geom.y

    # Downsample
    if len(bldg) > max_buildings:
        bldg = bldg.sample(max_buildings, random_state=42)

    # Compute quintile breaks from full dataset (use passed values if available)
    all_vals = buildings_gdf[acc_col].fillna(0).values
    breaks = _classify_breaks(all_vals)

    # Map centre
    center_lat = float(sett.geometry.centroid.y.mean())
    center_lon = float(sett.geometry.centroid.x.mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # ── 1. SETTLEMENT CHOROPLETH ───────────────────────────────────────────────
    if acc_col in sett.columns:
        sett_vals = sett[acc_col].dropna()
        vmin, vmax = float(sett_vals.quantile(0.05)), float(sett_vals.quantile(0.95))

        def _sett_style(feature):
            val = feature["properties"].get(acc_col, 0) or 0
            pct = (val - vmin) / max(vmax - vmin, 1e-9)
            pct = max(0, min(1, pct))
            # Green-Yellow-Red scale
            r = int(255 * (1 - pct))
            g = int(200 * pct)
            b = 0
            fill = f"#{r:02x}{g:02x}{b:02x}"
            return {"fillColor": fill, "color": "#333333", "weight": 1.2,
                    "fillOpacity": 0.55}

        def _sett_highlight(feature):
            return {"weight": 3, "color": "#0D2137", "fillOpacity": 0.75}

        # Build tooltip + popup content
        sett_copy = sett.copy()
        # Add all scenario column values to popup
        all_cols = [acc_col] + (list(scenario_cols.values()) if scenario_cols else [])
        for col in all_cols:
            if col not in sett_copy.columns:
                sett_copy[col] = float("nan")

        folium.GeoJson(
            sett_copy.__geo_interface__,
            name="Settlements (baseline accessibility)",
            style_function=_sett_style,
            highlight_function=_sett_highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["Settlement:"],
                localize=True,
            ),
            popup=folium.GeoJsonPopup(
                fields=["name", acc_col] + ([c for c in (scenario_cols.values() if scenario_cols else [])
                                             if c in sett_copy.columns]),
                aliases=["Settlement", "Baseline E2SFCA"] + (
                    list(scenario_cols.keys()) if scenario_cols else []
                ),
                localize=True,
                max_width=300,
            ),
        ).add_to(m)

    # ── 2. BUILDING ACCESSIBILITY LAYER ───────────────────────────────────────
    bldg_layer = folium.FeatureGroup(name=f"Buildings — {acc_col}", show=True)
    for _, row in bldg.iterrows():
        val = float(row.get(acc_col, 0) or 0)
        color = _quintile_color(val, breaks)
        acc_str = f"{val:.5f}" if val > 0 else "no access"

        popup_html = f"""
        <div style='font-family:sans-serif; font-size:12px; width:200px'>
          <b>Building</b><br>
          <b>Baseline E2SFCA:</b> {acc_str}<br>
        """
        if scenario_cols:
            for sc_name, sc_col in scenario_cols.items():
                sc_val = float(row.get(sc_col, 0) or 0)
                delta  = sc_val - val
                sign   = "+" if delta >= 0 else ""
                popup_html += f"  <b>{sc_name}:</b> {sc_val:.5f} ({sign}{delta:.5f})<br>"
        popup_html += "</div>"

        folium.CircleMarker(
            location=[row["_cy"], row["_cx"]],
            radius=3,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.75, weight=0,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(bldg_layer)
    bldg_layer.add_to(m)

    # ── 3. SCENARIO BUILDING LAYERS ───────────────────────────────────────────
    if scenario_cols:
        for sc_name, sc_col in scenario_cols.items():
            if sc_col not in bldg.columns:
                continue
            sc_breaks = _classify_breaks(buildings_gdf[sc_col].fillna(0).values)
            sc_layer  = folium.FeatureGroup(name=f"Buildings — {sc_name}", show=False)
            for _, row in bldg.iterrows():
                val   = float(row.get(sc_col, 0) or 0)
                color = _quintile_color(val, sc_breaks)
                folium.CircleMarker(
                    location=[row["_cy"], row["_cx"]],
                    radius=3,
                    color=color, fill=True, fill_color=color,
                    fill_opacity=0.75, weight=0,
                    popup=folium.Popup(f"{sc_name}: {val:.5f}", max_width=150),
                ).add_to(sc_layer)
            sc_layer.add_to(m)

    # ── 4. FACILITIES ─────────────────────────────────────────────────────────
    fac_layer = folium.FeatureGroup(name="Healthcare facilities", show=True)
    for _, row in fac.iterrows():
        spec   = str(row.get("specialization", "unknown"))
        color  = _SPEC_COLORS.get(spec, "#333333")
        name   = str(row.get("fullname", row.get("name", "?")))
        cap    = row.get("effective_capacity", "?")
        cap_s  = f"{cap:.0f} visits/day" if isinstance(cap, (int, float)) and not np.isnan(float(cap)) else "?"

        popup_html = f"""
        <div style='font-family:sans-serif; font-size:12px; width:220px'>
          <b style='color:{color}'>{name}</b><br>
          <b>Type:</b> {_SPEC_LABELS.get(spec, spec)}<br>
          <b>Capacity:</b> {cap_s}<br>
        </div>"""

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=8, color=color, fill=True, fill_color=color,
            fill_opacity=0.9, weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{name} — {_SPEC_LABELS.get(spec, spec)}",
        ).add_to(fac_layer)
    fac_layer.add_to(m)

    # ── 5. NEW FACILITY MARKER ─────────────────────────────────────────────────
    if new_facility_gdf is not None and len(new_facility_gdf):
        nf    = new_facility_gdf.to_crs(CRS_WGS)
        nf_lg = folium.FeatureGroup(name="New facility (S2)", show=True)
        for _, row in nf.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=folium.Popup(f"<b>New hospital</b><br>S2 optimal location", max_width=200),
                tooltip="New hospital (S2)",
                icon=folium.Icon(color="orange", icon="plus-sign", prefix="glyphicon"),
            ).add_to(nf_lg)
        nf_lg.add_to(m)

    # ── 6. CLOSED ROADS ────────────────────────────────────────────────────────
    if closed_roads_gdf is not None and len(closed_roads_gdf):
        cr    = closed_roads_gdf.to_crs(CRS_WGS)
        cr_lg = folium.FeatureGroup(name="Closed roads (S3)", show=True)
        folium.GeoJson(
            cr.__geo_interface__,
            name="Closed roads",
            style_function=lambda x: {"color":"#C62828","weight":4,"opacity":0.9},
            tooltip="Road closure (S3)",
        ).add_to(cr_lg)
        cr_lg.add_to(m)

    # ── 7. LEGEND ─────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:12px 16px; border:1px solid #ccc;
                border-radius:8px; font-family:sans-serif; font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
      <b>Accessibility quintiles</b><br>
      <span style="background:#C62828;display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px"></span>Q1 — Very low<br>
      <span style="background:#E8622A;display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px"></span>Q2 — Low<br>
      <span style="background:#F9C74F;display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px"></span>Q3 — Medium<br>
      <span style="background:#90BE6D;display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px"></span>Q4 — High<br>
      <span style="background:#2D6A4F;display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px"></span>Q5 — Very high<br>
      <span style="background:#C8C8C8;display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:6px"></span>No access<br>
      <hr style="margin:6px 0">
      <b>Facilities</b><br>
      <span style="color:#C62828">●</span> Hospital &nbsp;
      <span style="color:#E8622A">●</span> Specialist<br>
      <span style="color:#1A7A4A">●</span> Primary care &nbsp;
      <span style="color:#1A5EA8">●</span> Paediatric
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── 8. PLUGINS ────────────────────────────────────────────────────────────
    Fullscreen().add_to(m)
    MeasureControl(position="topright", primary_length_unit="meters").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        print(f"Interactive map saved: {save_path}")

    return m

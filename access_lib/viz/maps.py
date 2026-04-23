"""
viz/maps.py — Publication-quality cartographic visualisations. English only.

Key fixes vs previous version:
 - Buildings rendered as SCATTER (centroid dots) at district scale → always visible
 - Buildings rendered as polygon fills at settlement zoom scale
 - North arrow on right side; legend on left
 - specialization_maps accepts explicit spec_groups dict (no relative import)
 - scenario_before_after: auto-zoom focus_settlements
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
except ImportError as e:
    raise ImportError(f"Install required packages: {e}")

from .style import (
    SPEC_STYLES, GROUP_COLORS, MODE_COLORS,
    CMAP_ACCESS, CMAP_DELTA, NAVY, GRAY_30, GRAY_60, GRAY_80, WHITE,
    ORANGE, RED_ACC, GREEN_ACC, BLUE,
    FIG_FULL, FIG_HALF_H, FIG_WIDE, FIG_PANEL4,
    add_scalebar, add_north_arrow, add_stats_box,
    add_colorbar, apply_map_style, savefig,
)

CRS_PLOT = "EPSG:3857"
_QUINTILE_COLORS = ["#C62828","#E8622A","#F9C74F","#90BE6D","#2D6A4F"]


def _add_basemap(ax, style="light"):
    try:
        import contextily as cx
        src = cx.providers.CartoDB.Positron if style=="light" else cx.providers.CartoDB.DarkMatter
        cx.add_basemap(ax, crs=CRS_PLOT, source=src, zoom="auto", alpha=0.32)
    except Exception:
        pass

def _gini(v):
    v = np.sort(v[np.isfinite(v)&(v>=0)])
    if len(v)<2 or v.sum()==0: return float("nan")
    n=len(v); return float((2*((np.arange(1,n+1)*v).sum())/(n*v.sum()))-(n+1)/n)

def _classify_quintiles(vals, n=5):
    has = vals > 0
    labels = pd.Series(0, index=vals.index)
    if has.sum()==0: return labels, np.zeros(n+1)
    breaks = np.percentile(vals[has], np.linspace(0,100,n+1))
    breaks = np.unique(breaks)
    for i,(lo,hi) in enumerate(zip(breaks[:-1],breaks[1:]),1):
        labels[(vals>=lo)&(vals<=hi)&has] = i
    return labels, breaks

def _scatter_quintile(ax, bldg_m, acc_col, n=5, s=4, alpha=0.85):
    """Plot building centroids as scatter coloured by quintile. Always visible."""
    vals = bldg_m[acc_col].fillna(0)
    labels, breaks = _classify_quintiles(vals, n)
    geom = bldg_m.geometry
    if not (geom.geom_type=="Point").all():
        pts = geom.centroid
    else:
        pts = geom

    no_acc = bldg_m[labels==0]
    if len(no_acc):
        ax.scatter(pts[labels==0].x, pts[labels==0].y,
                   c="#C8C8C8", s=s, alpha=0.45, linewidths=0, zorder=2)

    patches = [mpatches.Patch(color="#C8C8C8", label="No access")] if (labels==0).any() else []
    for i,color in enumerate(_QUINTILE_COLORS[:n], 1):
        sub_pts = pts[labels==i]
        if len(sub_pts):
            ax.scatter(sub_pts.x, sub_pts.y,
                       c=color, s=s, alpha=alpha, linewidths=0, zorder=3)
        if len(breaks)>i:
            patches.append(mpatches.Patch(color=color, label=f"Q{i}: {breaks[i-1]:.3f}–{breaks[i]:.3f}"))
    return patches, breaks

def _poly_quintile(ax, bldg_m, acc_col, n=5, alpha=0.88, breaks_override=None):
    """Plot building polygons coloured by quintile. Use for zoomed views."""
    vals = bldg_m[acc_col].fillna(0)
    if breaks_override is not None:
        labels = pd.Series(0, index=vals.index)
        has = vals>0
        for i,(lo,hi) in enumerate(zip(breaks_override[:-1],breaks_override[1:]),1):
            labels[(vals>=lo)&(vals<=hi)&has] = i
        breaks = breaks_override
    else:
        labels, breaks = _classify_quintiles(vals, n)
    bldg_m = bldg_m.copy(); bldg_m["_q"]=labels
    no_acc = bldg_m[bldg_m["_q"]==0]
    if len(no_acc): no_acc.plot(ax=ax, color="#C8C8C8", linewidth=0, alpha=0.5, zorder=2)
    for i,color in enumerate(_QUINTILE_COLORS[:n],1):
        sub = bldg_m[bldg_m["_q"]==i]
        if len(sub): sub.plot(ax=ax, color=color, linewidth=0, alpha=alpha, zorder=3)
    return labels, breaks


# ─── 1. PROVISION MAP (building scatter, district scale) ──────────────────────
def provision_map(
    buildings: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame, acc_col: str = "acc_combined",
    title: str = "Healthcare Accessibility — Building Level",
    subtitle: str = "Accessibility quintiles  |  ▲ = facilities",
    n_quintiles: int = 5, figsize: Tuple = FIG_FULL,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    bldg_m = buildings.to_crs(CRS_PLOT)
    fac_m  = facilities.to_crs(CRS_PLOT)
    bnd_m  = boundaries.to_crs(CRS_PLOT)

    fig, ax = plt.subplots(figsize=figsize)
    bnd_m.plot(ax=ax, facecolor="none", edgecolor=GRAY_60, linewidth=0.9, zorder=1)
    patches, _ = _scatter_quintile(ax, bldg_m, acc_col, n_quintiles, s=5)
    _add_basemap(ax)

    for spec,(color,lbl_en,_) in SPEC_STYLES.items():
        sub = fac_m[fac_m["specialization"]==spec]
        if len(sub):
            ax.scatter(sub.geometry.x, sub.geometry.y, c=color, s=80,
                       marker="^", zorder=7, edgecolors=WHITE, linewidths=0.8)
            patches.append(mpatches.Patch(color=color, label=f"▲ {lbl_en}"))

    for _,row in bnd_m.iterrows():
        if row.geometry and not row.geometry.is_empty:
            c = row.geometry.centroid
            raw_name = str(row.get("name",""))
            short = raw_name.replace(' СП','').replace(' ГП','')
            ax.text(c.x, c.y, short, fontsize=7.5, ha="center", va="center",
                    color=WHITE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=NAVY, alpha=0.65, edgecolor="none"),
                    zorder=6)

    # Legend LEFT, scalebar+north RIGHT
    ax.legend(handles=patches, title="Accessibility quintiles / Facilities",
              loc="lower left", fontsize=7, title_fontsize=8,
              framealpha=0.93, edgecolor=GRAY_30)
    add_scalebar(ax, side="right"); add_north_arrow(ax, x=0.96, y=0.12)
    fig.text(0.5,0.98, title, ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)
    fig.text(0.5,0.957, subtitle, ha="center", va="top", fontsize=9.5, color=GRAY_60)
    apply_map_style(ax)
    plt.tight_layout(rect=[0,0,1,0.955])
    if save_path: savefig(fig, save_path)
    return fig


# ─── 2. SETTLEMENT CHOROPLETH ──────────────────────────────────────────────────
def settlement_choropleth(
    settlements: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame,
    acc_col: str = "acc_baseline",
    title: str = "Baseline Accessibility — Settlement Level",
    subtitle: str = "",
    figsize: Tuple = FIG_FULL,
    stats_dict: Optional[Dict[str,str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    sett_m = settlements.to_crs(CRS_PLOT)
    fac_m  = facilities.to_crs(CRS_PLOT)
    vals   = sett_m[acc_col].dropna()
    vmin,vmax = float(vals.quantile(0.03)), float(vals.quantile(0.97))

    fig, ax = plt.subplots(figsize=figsize)
    sett_m.plot(column=acc_col, ax=ax, cmap=CMAP_ACCESS, vmin=vmin, vmax=vmax,
                edgecolor=NAVY, linewidth=0.8, alpha=0.88,
                legend=False, missing_kwds={"color":"#E0E0E0"}, zorder=2)
    _add_basemap(ax)

    fac_handles = []
    for spec,(color,lbl_en,_) in SPEC_STYLES.items():
        sub = fac_m[fac_m["specialization"]==spec]
        if len(sub):
            ax.scatter(sub.geometry.x, sub.geometry.y, c=color, s=70, marker="^",
                       zorder=5, edgecolors=WHITE, linewidths=0.7, label=lbl_en)
            fac_handles.append(mpatches.Patch(color=color, label=f"▲ {lbl_en}"))

    for _,row in sett_m.iterrows():
        if row.geometry and not row.geometry.is_empty:
            c = row.geometry.centroid
            val = row.get(acc_col, float("nan"))
            if pd.notna(val):
                raw_name = str(row.get('name',''))
                short = raw_name.replace(' СП','').replace(' ГП','')
                ax.text(c.x, c.y, f"{short}\n{val:.4f}",
                        fontsize=8, ha="center", va="center",
                        color=WHITE, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor=NAVY, alpha=0.72, edgecolor="none"),
                        zorder=6)

    add_colorbar(fig, ax, CMAP_ACCESS, vmin, vmax, label="E2SFCA accessibility index")
    # Legend LEFT, scalebar RIGHT
    if fac_handles:
        ax.legend(handles=fac_handles, title="Facility type",
                  loc="lower left", fontsize=8, title_fontsize=9,
                  framealpha=0.9, edgecolor=GRAY_30)
    if stats_dict: add_stats_box(ax, stats_dict, loc="upper left")
    add_scalebar(ax, side="right"); add_north_arrow(ax, x=0.96, y=0.12)
    fig.text(0.5,0.98, title, ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.5,0.957, subtitle, ha="center", va="top", fontsize=9.5, color=GRAY_60)
    apply_map_style(ax)
    plt.tight_layout(rect=[0,0,1,0.955])
    if save_path: savefig(fig, save_path)
    return fig


# ─── 3. DELTA MAP ──────────────────────────────────────────────────────────────
def delta_map(
    settlements: gpd.GeoDataFrame, delta_col: str,
    scenario_label: str, baseline_label: str = "S0 Baseline",
    figsize: Tuple = FIG_HALF_H,
    impacted_buildings: Optional[gpd.GeoDataFrame] = None,
    facilities: Optional[gpd.GeoDataFrame] = None,
    new_facility: Optional[gpd.GeoDataFrame] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    sett_m = settlements.to_crs(CRS_PLOT)
    deltas = sett_m[delta_col].dropna()
    abs_max = max(abs(deltas.quantile(0.05)), abs(deltas.quantile(0.95)), 1e-9)

    fig, ax = plt.subplots(figsize=figsize)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    sett_m.plot(column=delta_col, ax=ax, cmap=CMAP_DELTA, norm=norm,
                edgecolor="#555", linewidth=0.5, alpha=0.88,
                legend=False, missing_kwds={"color":"#E8E8E8"}, zorder=2)
    _add_basemap(ax)

    if impacted_buildings is not None and len(impacted_buildings):
        imp_m = impacted_buildings.to_crs(CRS_PLOT)
        pts = imp_m.geometry if (imp_m.geometry.geom_type=="Point").all() else imp_m.geometry.centroid
        ax.scatter(pts.x, pts.y, c="#C62828", s=5, alpha=0.5, zorder=4,
                   label="Impacted buildings")

    if facilities is not None:
        fac_m = facilities.to_crs(CRS_PLOT)
        for spec,(color,*_) in SPEC_STYLES.items():
            sub = fac_m[fac_m["specialization"]==spec]
            if len(sub): ax.scatter(sub.geometry.x, sub.geometry.y, c=color, s=50,
                                    marker="^", zorder=5, edgecolors=WHITE, lw=0.6)

    if new_facility is not None and len(new_facility):
        nf = new_facility.to_crs(CRS_PLOT)
        ax.scatter(nf.geometry.x, nf.geometry.y, c="#FFD700", s=200,
                   marker="*", zorder=8, edgecolors=NAVY, lw=1.0, label="New facility")

    for _,row in sett_m.iterrows():
        val = row.get(delta_col, float("nan"))
        if pd.notna(val) and abs(val) > abs_max*0.05:
            c = row.geometry.centroid
            ax.text(c.x, c.y, f"{val:+.3f}", fontsize=7, ha="center", va="center",
                    color="#003060" if val>0 else "#6B0000", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=WHITE)], zorder=6)

    add_colorbar(fig, ax, CMAP_DELTA, -abs_max, abs_max,
                 label=f"Δ E2SFCA  ({scenario_label} − {baseline_label})", diverging=True)
    add_scalebar(ax); add_north_arrow(ax, x=0.96, y=0.12)
    n_imp = int((sett_m[delta_col]>abs_max*0.01).sum())
    n_wrs = int((sett_m[delta_col]<-abs_max*0.01).sum())
    add_stats_box(ax, {"Improved":str(n_imp),"Worsened":str(n_wrs),
                       "Max Δ":f"{float(deltas.max()):+.4f}",
                       "Min Δ":f"{float(deltas.min()):+.4f}"}, loc="upper left", title="Statistics")
    handles,_ = ax.get_legend_handles_labels()
    if handles: ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.text(0.5,0.98, f"Accessibility Change: {scenario_label}", ha="center", va="top",
             fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.5,0.957, f"Δ = {scenario_label} − {baseline_label}", ha="center", va="top",
             fontsize=9, color=GRAY_60)
    apply_map_style(ax)
    plt.tight_layout(rect=[0,0,1,0.955])
    if save_path: savefig(fig, save_path)
    return fig


# ─── 4. SCENARIO PANEL ─────────────────────────────────────────────────────────
def scenario_panel(
    settlements: gpd.GeoDataFrame, acc_cols: Dict[str,str],
    figsize=None, ncols=3, shared_scale=True,
    save_path: Optional[Path]=None,
) -> plt.Figure:
    n=len(acc_cols); ncols=min(ncols,n); nrows=(n+ncols-1)//ncols
    figsize=figsize or (7*ncols, 7*nrows)
    sett_m=settlements.to_crs(CRS_PLOT)
    if shared_scale:
        all_v=np.concatenate([sett_m[col].dropna().values for col in acc_cols.values() if col in sett_m.columns])
        vmin,vmax=float(np.percentile(all_v,3)),float(np.percentile(all_v,97))
    fig,axes=plt.subplots(nrows,ncols,figsize=figsize,gridspec_kw={"hspace":0.05,"wspace":0.02})
    axes_flat=np.array(axes).ravel() if n>1 else [axes]
    for ax,(panel_title,col) in zip(axes_flat,acc_cols.items()):
        if col not in sett_m.columns: ax.set_visible(False); continue
        _vmin=vmin if shared_scale else float(sett_m[col].quantile(0.03))
        _vmax=vmax if shared_scale else float(sett_m[col].quantile(0.97))
        sett_m.plot(column=col,ax=ax,cmap=CMAP_ACCESS,vmin=_vmin,vmax=_vmax,
                    edgecolor="#555",lw=0.4,alpha=0.9,legend=False)
        _add_basemap(ax)
        ax.set_title(panel_title,fontsize=10,fontweight="bold",color=NAVY,pad=6)
        apply_map_style(ax)
        cv=sett_m[col].dropna()
        ax.text(0.02,0.03,f"μ={cv.mean():.4f}  G={_gini(cv.values):.2f}",
                transform=ax.transAxes,fontsize=7,color=GRAY_60,va="bottom")
    for ax in axes_flat[n:]: ax.set_visible(False)
    if shared_scale:
        sm=ScalarMappable(norm=Normalize(vmin=vmin,vmax=vmax),cmap=CMAP_ACCESS); sm.set_array([])
        cbar=fig.colorbar(sm,ax=axes_flat[:n],orientation="vertical",shrink=0.5,pad=0.02)
        cbar.set_label("E2SFCA index",fontsize=9,color=GRAY_80); cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Scenario Comparison — Settlement-Level Accessibility",
                 fontsize=13,fontweight="bold",color=NAVY,y=1.02)
    if save_path: savefig(fig,save_path)
    return fig


# ─── 5. FACILITIES MAP ─────────────────────────────────────────────────────────
def facilities_map(
    facilities: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame,
    roads=None, figsize=FIG_FULL, save_path=None,
) -> plt.Figure:
    fac_m=facilities.to_crs(CRS_PLOT); bnd_m=boundaries.to_crs(CRS_PLOT)
    fig,ax=plt.subplots(figsize=figsize)
    bnd_m.plot(ax=ax,facecolor="#F0F4F8",edgecolor=GRAY_60,lw=1.0,alpha=0.5,zorder=1)
    if roads is not None: roads.to_crs(CRS_PLOT).plot(ax=ax,color="#C8D6E5",lw=0.5,alpha=0.6,zorder=2)
    _add_basemap(ax)
    has_cap="effective_capacity" in fac_m.columns
    patches=[]
    for spec,(color,lbl_en,_) in SPEC_STYLES.items():
        sub=fac_m[fac_m["specialization"]==spec]
        if not len(sub): continue
        sizes=((sub["effective_capacity"].clip(lower=1)/fac_m["effective_capacity"].max()*200+40)
               if has_cap else 80)
        ax.scatter(sub.geometry.x,sub.geometry.y,c=color,s=sizes,marker="^",
                   zorder=5,edgecolors=WHITE,lw=0.8,alpha=0.92)
        patches.append(mpatches.Patch(color=color,label=f"{lbl_en} (n={len(sub)})"))
    ax.legend(handles=patches,title="Specialisation",loc="lower left",fontsize=8,
              title_fontsize=9,framealpha=0.93,edgecolor=GRAY_30)
    add_scalebar(ax); add_north_arrow(ax,x=0.96,y=0.12)
    fig.text(0.5,0.98,"Healthcare Facilities by Specialisation",ha="center",va="top",
             fontsize=14,fontweight="bold",color=NAVY)
    apply_map_style(ax); plt.tight_layout(rect=[0,0,1,0.96])
    if save_path: savefig(fig,save_path)
    return fig


# ─── 6. MODE COMPARISON (scatter at district scale) ───────────────────────────
def mode_comparison(
    buildings: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame,
    acc_car_col="acc_car", acc_walk_col="acc_walk",
    acc_pt_col="acc_pt", acc_comp_col="acc_combined",
    figsize=(20,6), save_path=None,
) -> plt.Figure:
    bldg_m=buildings.to_crs(CRS_PLOT); bnd_m=boundaries.to_crs(CRS_PLOT)
    cols={f"Car\n(w=40%)":acc_car_col, f"Walking\n(w=20%)":acc_walk_col,
          f"Public Transit\n(w=40%)":acc_pt_col, f"Composite\n(E2SFCA)":acc_comp_col}
    cols={k:v for k,v in cols.items() if v in bldg_m.columns}
    all_pos=np.concatenate([bldg_m[c].fillna(0).values for c in cols.values()])
    all_pos=all_pos[all_pos>0]
    vmin=float(np.percentile(all_pos,3)) if len(all_pos) else 0
    vmax=float(np.percentile(all_pos,97)) if len(all_pos) else 1

    fig,axes=plt.subplots(1,len(cols),figsize=figsize,gridspec_kw={"wspace":0.03})
    if len(cols)==1: axes=[axes]
    hdr_colors=[MODE_COLORS.get("car",NAVY), MODE_COLORS.get("walk",NAVY),
                MODE_COLORS.get("pt",NAVY), NAVY]
    geom=bldg_m.geometry
    pts=geom.centroid if not (geom.geom_type=="Point").all() else geom

    for i,(ax,(title,col)) in enumerate(zip(axes,cols.items())):
        bnd_m.plot(ax=ax,facecolor="none",edgecolor=GRAY_60,lw=0.5,zorder=1)
        vals=bldg_m[col].fillna(0).values
        # Grey for zero-access
        mask0=vals==0
        if mask0.any():
            ax.scatter(pts.x.values[mask0],pts.y.values[mask0],c="#DCDCDC",
                       s=2,alpha=0.5,linewidths=0,zorder=2)
        # Colour non-zero by continuous scale
        mask_pos=vals>0
        if mask_pos.any():
            norm=plt.Normalize(vmin=vmin,vmax=vmax)
            cmap=plt.get_cmap(CMAP_ACCESS)
            col_vals=cmap(norm(vals[mask_pos]))
            ax.scatter(pts.x.values[mask_pos],pts.y.values[mask_pos],
                       c=col_vals,s=3,alpha=0.88,linewidths=0,zorder=3)
        _add_basemap(ax)
        ax.set_title(title,fontsize=10,fontweight="bold",color=hdr_colors[i%4],pad=8)
        apply_map_style(ax)
        pct=float(mask_pos.mean()*100)
        ax.text(0.03,0.03,f"{pct:.0f}% buildings with access",
                transform=ax.transAxes,fontsize=7.5,color=GRAY_60,va="bottom")

    sm=ScalarMappable(norm=Normalize(vmin=vmin,vmax=vmax),cmap=CMAP_ACCESS); sm.set_array([])
    cbar=fig.colorbar(sm,ax=axes,orientation="horizontal",shrink=0.5,pad=0.04,aspect=40)
    cbar.set_label("E2SFCA index (shared scale)",fontsize=9,color=GRAY_80)
    fig.suptitle("Multimodal Decomposition — E2SFCA by Transport Mode",
                 fontsize=13,fontweight="bold",color=NAVY,y=1.03)
    if save_path: savefig(fig,save_path)
    return fig


# ─── 7. SPECIALISATION MAPS ────────────────────────────────────────────────────
def specialization_maps(
    settlements: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame,
    spec_groups: Optional[Dict] = None,  # {title: (col, [spec_list])}
    figsize=FIG_PANEL4, save_path=None,
) -> plt.Figure:
    # Default spec_groups if not provided (avoids relative import)
    if spec_groups is None:
        spec_groups = {
            "Primary care":       ("access_primary",     ["primary_therapist","primary_basic"]),
            "Paediatrics":        ("access_pediatric",   ["pediatric"]),
            "Specialist outpat.": ("access_specialized", ["outpatient_specialized","maternal"]),
            "Hospital":           ("access_hospital",    ["hospital_full"]),
        }
    sett_m=settlements.to_crs(CRS_PLOT); fac_m=facilities.to_crs(CRS_PLOT)
    group_keys=list(spec_groups.keys())
    # Shared scale
    all_v=np.concatenate([sett_m[col].dropna().values for col,_ in spec_groups.values()
                          if col in sett_m.columns])
    all_v=all_v[all_v>0]
    vmin=float(np.percentile(all_v,3)) if len(all_v) else 0
    vmax=float(np.percentile(all_v,97)) if len(all_v) else 1
    grp_color_keys=["primary","pediatric","specialized","hospital"]

    fig,axes=plt.subplots(2,2,figsize=figsize,gridspec_kw={"hspace":0.04,"wspace":0.04})
    axes_flat=axes.ravel()
    for ax,(title,(col,spec_list)) in zip(axes_flat,spec_groups.items()):
        if col not in sett_m.columns:
            sett_m.plot(ax=ax,color="#E8E8E8",edgecolor="#999",lw=0.4)
            ax.set_title(f"{title}\n(column '{col}' not found)",fontsize=9,color=RED_ACC,pad=6)
            apply_map_style(ax); continue
        sett_m.plot(column=col,ax=ax,cmap=CMAP_ACCESS,vmin=vmin,vmax=vmax,
                    edgecolor="#666",lw=0.4,alpha=0.9,legend=False,
                    missing_kwds={"color":"#E0E0E0"})
        _add_basemap(ax)
        sub=fac_m[fac_m["specialization"].isin(spec_list)]
        gkey=grp_color_keys[list(spec_groups.keys()).index(title)%4]
        if len(sub): ax.scatter(sub.geometry.x,sub.geometry.y,
                                c=GROUP_COLORS.get(gkey,NAVY),s=70,marker="^",
                                zorder=5,edgecolors=WHITE,lw=0.7)
        ax.set_title(title,fontsize=10,fontweight="bold",color=GROUP_COLORS.get(gkey,NAVY),pad=6)
        apply_map_style(ax)
        ax.text(0.03,0.04,f"μ={sett_m[col].mean():.4f}",
                transform=ax.transAxes,fontsize=8,color=GRAY_60)
    sm=ScalarMappable(norm=Normalize(vmin=vmin,vmax=vmax),cmap=CMAP_ACCESS); sm.set_array([])
    cbar=fig.colorbar(sm,ax=axes_flat,orientation="vertical",shrink=0.6,pad=0.02)
    cbar.set_label("E2SFCA index (shared scale)",fontsize=9,color=GRAY_80)
    fig.suptitle("Specialisation Decomposition — Four E2SFCA Components",
                 fontsize=13,fontweight="bold",color=NAVY,y=1.02)
    if save_path: savefig(fig,save_path)
    return fig


# ─── 8. SETTLEMENT ZOOM (one figure PER settlement) ──────────────────────────
def settlement_zoom(
    buildings: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame, settlement_names: List[str],
    acc_col: str = "acc_combined", n_quintiles: int = 5,
    buffer_m: float = 400.0, figsize=None, save_path=None,
) -> List[plt.Figure]:
    """Returns a list of figures — one per settlement. Each has independent breaks."""
    bldg_m=buildings.to_crs(CRS_PLOT); bnd_m=boundaries.to_crs(CRS_PLOT); fac_m=facilities.to_crs(CRS_PLOT)
    from shapely.geometry import box as _sbox
    figures = []

    for idx, name in enumerate(settlement_names):
        single_size = figsize or (14, 14)
        fig, ax = plt.subplots(figsize=single_size)

        sett_row=bnd_m[bnd_m["name"]==name]
        if sett_row.empty:
            ax.set_title(f"'{name}' not found",fontsize=12,color=RED_ACC)
            ax.set_axis_off(); figures.append(fig); continue

        geom=sett_row.geometry.iloc[0]
        minx,miny,maxx,maxy=geom.bounds
        padx=max((maxx-minx)*0.15, buffer_m)
        pady=max((maxy-miny)*0.15, buffer_m)
        xlim=(minx-padx, maxx+padx); ylim=(miny-pady, maxy+pady)
        view_box=_sbox(xlim[0],ylim[0],xlim[1],ylim[1])

        clip_idx=bldg_m.geometry.intersects(view_box)
        bldg_clip=bldg_m[clip_idx].copy()

        if len(bldg_clip)==0:
            ax.set_title(f"{name}\n(no buildings)",fontsize=12,color=RED_ACC)
            ax.set_axis_off(); figures.append(fig); continue

        vals_clip=bldg_clip[acc_col].fillna(0)
        labels,breaks=_classify_quintiles(vals_clip, n_quintiles)
        bldg_clip["_q"]=labels

        # Set extent FIRST so contextily fetches correct tiles for this zoom
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        _add_basemap(ax)

        # ╔══════════════════════════════════════════════════════════════╗
        # ║  BUILDING APPEARANCE IN SETTLEMENT ZOOM                      ║
        # ║  Edit edgecolor / linewidth / alpha below to adjust look.    ║
        # ║  File: viz/maps.py, function settlement_zoom()              ║
        # ╚══════════════════════════════════════════════════════════════╝
        no_acc=bldg_clip[bldg_clip["_q"]==0]
        if len(no_acc): no_acc.plot(ax=ax,color="#CCCCCC",edgecolor="#AAAAAA",
                                     linewidth=0.5,alpha=0.8,zorder=3)
        for qi,color in enumerate(_QUINTILE_COLORS[:n_quintiles],1):
            sub=bldg_clip[bldg_clip["_q"]==qi]
            if len(sub): sub.plot(ax=ax,color=color,edgecolor=color,
                                  linewidth=0.6,alpha=0.95,zorder=4+qi)

        sett_row.plot(ax=ax,facecolor="none",edgecolor=NAVY,lw=2.0,
                      linestyle="--",alpha=0.8,zorder=10)

        fac_clip=fac_m[fac_m.geometry.intersects(view_box)]
        for spec,(color,lbl_en,_) in SPEC_STYLES.items():
            sub=fac_clip[fac_clip["specialization"]==spec]
            if len(sub): ax.scatter(sub.geometry.x,sub.geometry.y,
                                    c=color,s=160,marker="^",zorder=12,
                                    edgecolors=WHITE,lw=1.2)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        apply_map_style(ax)
        add_scalebar(ax); add_north_arrow(ax,x=0.96,y=0.12)

        acc_pos=vals_clip[vals_clip>0]
        mu=float(acc_pos.mean()) if len(acc_pos) else 0
        n_zero=int((vals_clip==0).sum()); n_bldg=len(bldg_clip)
        ax.set_title(f"{name}\n\u03bc={mu:.4f}  |  zero-access: {n_zero}/{n_bldg} "
                     f"({n_zero/max(n_bldg,1)*100:.0f}%)",
                     fontsize=14,fontweight="bold",color=NAVY,pad=10)

        patches=[mpatches.Patch(color="#CCCCCC",label="No access")] if n_zero>0 else []
        for qi,color in enumerate(_QUINTILE_COLORS[:n_quintiles],1):
            if len(breaks)>qi:
                patches.append(mpatches.Patch(color=color,
                    label=f"Q{qi}: {breaks[qi-1]:.4f}\u2013{breaks[qi]:.4f}"))
        ax.legend(handles=patches,title="Quintile (local breaks)",
                  loc="lower left",fontsize=10,title_fontsize=11,
                  framealpha=0.95,edgecolor=GRAY_30)

        plt.tight_layout()
        if save_path:
            stem = Path(save_path).stem
            parent = Path(save_path).parent
            sfx = Path(save_path).suffix
            savefig(fig, parent / f"{stem}_{idx+1}{sfx}")
        figures.append(fig)

    return figures


# ─── 9. BEFORE / AFTER ─────────────────────────────────────────────────────────
def scenario_before_after(
    buildings: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    acc_baseline_col: str, acc_scenario_col: str,
    scenario_label: str,
    focus_settlements: Optional[List[str]] = None,
    new_facility_pt: Optional[gpd.GeoDataFrame] = None,
    closed_roads: Optional[gpd.GeoDataFrame] = None,
    road_length_km: Optional[float] = None,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
    n_quintiles: int = 5, figsize: Tuple = (18,8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    bldg_m=buildings.to_crs(CRS_PLOT); bnd_m=boundaries.to_crs(CRS_PLOT); fac_m=facilities.to_crs(CRS_PLOT)
    vals_base=bldg_m[acc_baseline_col].fillna(0)
    _,breaks=_classify_quintiles(vals_base,n_quintiles)

    # Compute zoom extent from focus_settlements
    xlim=ylim=None
    if focus_settlements:
        sel=bnd_m[bnd_m["name"].isin(focus_settlements)]
        if not sel.empty:
            minx,miny,maxx,maxy=sel.total_bounds
            pad=max((maxx-minx)*0.15, 800)
            xlim=(minx-pad, maxx+pad); ylim=(miny-pad, maxy+pad)

    geom=bldg_m.geometry
    pts=geom.centroid if not (geom.geom_type=="Point").all() else geom

    def _panel(ax, col, title, cr_gdf=None):
        # ── Layer order: basemap → roads → polygons → boundary → facilities ──
        _add_basemap(ax)
        # Show road network if available and closed_roads present
        if roads_gdf is not None and closed_roads is not None:
            try:
                roads_p = roads_gdf.to_crs(CRS_PLOT)
                roads_p.plot(ax=ax, color="#A0B0C0", lw=0.5, alpha=0.45, zorder=1)
            except Exception: pass
        vals=bldg_m[col].fillna(0).values
        labels=np.zeros(len(vals),dtype=int)
        has=vals>0
        if len(breaks)>1:
            for i,(lo,hi) in enumerate(zip(breaks[:-1],breaks[1:]),1):
                labels[(vals>=lo)&(vals<=hi)&has]=i
        # Building polygons — always visible above basemap (zorder 3-4)
        mask0=labels==0
        if mask0.any():
            bldg_m[mask0].plot(ax=ax,color="#BBBBBB",linewidth=0,alpha=0.65,zorder=3)
        for qi,color in enumerate(_QUINTILE_COLORS[:n_quintiles],1):
            mask_q=labels==qi
            if mask_q.any():
                bldg_m[mask_q].plot(ax=ax,color=color,linewidth=0,alpha=0.90,zorder=4)
        # Settlement boundary — thin dashed on top of buildings
        bnd_m.plot(ax=ax,facecolor="none",edgecolor=NAVY,lw=0.7,
                   linestyle="--",alpha=0.55,zorder=5)
        # Facilities
        for spec,(color,*_) in SPEC_STYLES.items():
            sub=fac_m[fac_m["specialization"]==spec]
            if len(sub): ax.scatter(sub.geometry.x,sub.geometry.y,
                                    c=color,s=60,marker="^",zorder=7,
                                    edgecolors=WHITE,lw=0.7)
        # Closed road overlay (right panel only)
        if cr_gdf is not None and len(cr_gdf):
            cr_p=cr_gdf.to_crs(CRS_PLOT)
            cr_p.plot(ax=ax,color=RED_ACC,lw=4.5,alpha=0.95,zorder=8,
                      label="Closed road segment")
            try:
                cr_m=cr_gdf.to_crs("EPSG:32636")
                _total_m = cr_m.geometry.length.sum()
                _mid=cr_p.unary_union.centroid
                ax.annotate(
                    f"Road closure\n{_total_m/1000:.1f} km total",
                    (_mid.x,_mid.y), xytext=(16,16),
                    textcoords="offset points",
                    fontsize=9,ha="left",color=RED_ACC,fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35",facecolor=WHITE,
                              edgecolor=RED_ACC,alpha=0.92,lw=1.5),
                    arrowprops=dict(arrowstyle="-|>",color=RED_ACC,lw=1.2),
                    zorder=9,
                )
            except Exception: pass
        if xlim: ax.set_xlim(xlim); ax.set_ylim(ylim)
        apply_map_style(ax); add_scalebar(ax, side="right")
        ax.set_title(title,fontsize=11,fontweight="bold",color=NAVY,pad=8)
        pos=bldg_m[col][bldg_m[col]>0]; mu=float(pos.mean()) if len(pos) else 0
        ax.text(0.03,0.03,f"μ={mu:.5f}  Gini={_gini(bldg_m[col].values):.3f}",
                transform=ax.transAxes,fontsize=8,color=GRAY_60,va="bottom")

    fig,(ax_b,ax_a)=plt.subplots(1,2,figsize=figsize,gridspec_kw={"wspace":0.04})
    _panel(ax_b,acc_baseline_col,"Baseline (S0)")
    _panel(ax_a,acc_scenario_col,scenario_label,cr_gdf=closed_roads)

    if new_facility_pt is not None and len(new_facility_pt):
        nf=new_facility_pt.to_crs(CRS_PLOT)
        ax_a.scatter(nf.geometry.x,nf.geometry.y,c="#FFD700",s=260,marker="*",
                     zorder=9,edgecolors=NAVY,lw=1.2,label="New facility")
        ax_a.legend(loc="lower right",fontsize=8.5,framealpha=0.9)

    # closed_roads rendered inside _panel for correct layer order

    patches=[mpatches.Patch(color="#C8C8C8",label="No access")]
    patches+=[mpatches.Patch(color=c,label=f"Q{i+1}") for i,c in enumerate(_QUINTILE_COLORS[:n_quintiles])]
    fig.legend(handles=patches,title="Quintile (baseline breaks)",
               loc="lower center",ncol=n_quintiles+1,fontsize=8.5,
               title_fontsize=9,framealpha=0.93,bbox_to_anchor=(0.5,-0.02))

    dmu=float((bldg_m[acc_scenario_col]-bldg_m[acc_baseline_col]).mean()) if acc_scenario_col in bldg_m.columns else 0
    fig.suptitle(f"Before / After — {scenario_label}   (Δμ = {dmu:+.5f})",
                 fontsize=13,fontweight="bold",color=NAVY,y=1.01)
    plt.tight_layout(rect=[0,0.06,1,1])
    if save_path: savefig(fig,save_path)
    return fig

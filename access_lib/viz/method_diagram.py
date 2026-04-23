"""
viz/method_diagram.py — Methodology illustrations for E2SFCA.
Clean layout: white boxes, coloured borders, large text, no overlaps.
Font sizes scaled 1.5× for presentation/print readability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except ImportError as e:
    raise ImportError(f"Install matplotlib: {e}")

from .style import (
    NAVY, BLUE, BLUE_SOFT, ORANGE, TEAL, RED_ACC, GREEN_ACC,
    GRAY_30, GRAY_60, GRAY_80, WHITE,
    MODE_COLORS, GROUP_COLORS, savefig,
)


def plot_e2sfca_schema(
    mode_weights: dict = None,
    figsize: Tuple = (26, 34),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    if mode_weights is None:
        mode_weights = {"car": 0.40, "walk": 0.20, "pt": 0.40}

    fig = plt.figure(figsize=figsize, facecolor=WHITE)
    ax  = fig.add_axes([0.02, 0.01, 0.96, 0.98])
    ax.set_xlim(0, 22); ax.set_ylim(0, 28)
    ax.set_axis_off(); ax.set_facecolor(WHITE)

    # fs_* = font sizes, all ×1.5 vs previous version
    FS_BOX   = 18    # was 12
    FS_BOX_S = 16    # was 11 (smaller boxes)
    FS_SUB   = 15    # was 10 (subtitle inside box)
    FS_HDR   = 16    # was 11 (section headers)
    FS_TITLE = 25    # was 17
    FS_STITLE= 16    # was 11

    def _box(cx, cy, w, h, t1, t2="", bc=NAVY, fs=FS_BOX):
        ax.add_patch(FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.25",
            facecolor=WHITE, edgecolor=bc, linewidth=2.5, zorder=3))
        if t2:
            ax.text(cx, cy+0.28, t1, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#111", zorder=4)
            ax.text(cx, cy-0.32, t2, ha="center", va="center",
                    fontsize=fs-3, color="#555", zorder=4)
        else:
            ax.text(cx, cy, t1, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#111", zorder=4)

    def _arr(x1,y1,x2,y2,c=GRAY_60):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>",color=c,lw=2.0,mutation_scale=18),zorder=2)
    def _ln(x1,y1,x2,y2,c=GRAY_30):
        ax.plot([x1,x2],[y1,y2],color=c,lw=1.0,alpha=0.5,zorder=1)
    def _hdr(cx,cy,t,fs=FS_HDR):
        ax.text(cx,cy,t,ha="center",va="center",fontsize=fs,color=GRAY_80,fontweight="bold",zorder=4)

    # Title
    ax.text(11,27.2,"Multimodal E2SFCA with Specialisation Decomposition",
            ha="center",fontsize=FS_TITLE,fontweight="bold",color=NAVY)
    ax.text(11,26.6,"Vsevolozhsky District  \u00b7  ITMO University 2026",
            ha="center",fontsize=FS_STITLE,color=GRAY_60,style="italic")

    # Row 1: Input
    _box(5.5,25.5,8,1.2,"Demand: residential buildings",
         "Population \u221d footprint \u00d7 floors (per settlement)",bc=BLUE,fs=FS_BOX)
    _box(16.5,25.5,8,1.2,"Supply: healthcare facilities",
         "Capacity = doctors \u00d7 25 patients/day",bc=RED_ACC,fs=FS_BOX)

    # Row 2: Graph
    _arr(5.5,24.9,11,23.8); _arr(16.5,24.9,11,23.8)
    _box(11,23.2,11,1.1,"Road network graph (OSM)",
         "Dijkstra shortest paths \u2192 travel-time matrices",bc="#2C3E50",fs=FS_BOX)

    # Row 3: OD matrices
    _arr(11,22.65,11,22.1)
    _hdr(11,21.9,"\u2193  Three OD matrices (multimodality axis)  \u2193")
    mx=[3.7,11,18.3]
    mi=[("Car",f"w={mode_weights['car']:.0%}","60 min | \u03b2=1.5",MODE_COLORS["car"]),
        ("Walking",f"w={mode_weights['walk']:.0%}","30 min | \u03b2=1.0",MODE_COLORS["walk"]),
        ("Public Transit",f"w={mode_weights['pt']:.0%}","90 min | \u03b2=1.2",MODE_COLORS["pt"])]
    for x,(ml,mw,md,mc) in zip(mx,mi):
        _arr(11,21.7,x,21.2)
        _box(x,20.6,6,1.2,f"OD \u2014 {ml} ({mw})",f"radius={md}",bc=mc,fs=FS_BOX_S)

    # Row 4: Decay
    for x in mx:
        _arr(x,20.0,x,19.4)
        _box(x,18.75,6,1.2,"Gaussian decay W(t,\u03b2,r)",
             "W = exp(\u2212(t/\u03c3)\u00b2)   \u03c3 = r\u00b70.5/\u221a\u03b2",bc="#2C3E50",fs=FS_BOX_S)

    # Row 5: Specialisation
    _hdr(11,17.85,"\u2193  E2SFCA per specialisation (4 groups)  \u2193")
    for x in mx: _arr(x,18.15,x,17.65)
    sx=[2.8,8,13.5,19.2]
    si=[("Primary care","therapist+FAP",GROUP_COLORS["primary"]),
        ("Paediatrics","children's clinics",GROUP_COLORS["pediatric"]),
        ("Specialist","outpatient+maternal",GROUP_COLORS["specialized"]),
        ("Hospital","inpatient (full)",GROUP_COLORS["hospital"])]
    for x in mx:
        for s in sx: _ln(x,17.65,s,17.05)
    for s,(sl,ss,sc) in zip(sx,si):
        _box(s,16.4,4.5,1.2,sl,ss,bc=sc,fs=FS_BOX_S)

    # M/M/1
    _box(19.2,14.8,4,0.9,"M/M/1 queue correction",
         "\u03c1<0.95 | \u03b1=0.001",bc=GROUP_COLORS["hospital"],fs=FS_SUB)
    _arr(19.2,15.8,19.2,15.3)

    _hdr(11,15.3,"\u2190  4 independent E2SFCA sub-indices  \u2192")

    # Row 6: Weighted sum
    for s in sx: _arr(s,15.8,s,14.5)
    for s in sx: _ln(s,14.5,11,14.1)
    _box(11,13.5,17,1.2,"Weighted mode summation per building",
         f"A = {mode_weights['car']:.0%}\u00b7A_car + {mode_weights['walk']:.0%}\u00b7A_walk + {mode_weights['pt']:.0%}\u00b7A_PT",
         bc=NAVY,fs=FS_BOX+1)

    # Row 7: Telemedicine + composite
    _arr(11,12.9,13,12.0)
    _box(4,11.4,6.5,1.2,"Telemedicine",
         "distance-independent | A_tele=V/D",bc=TEAL,fs=FS_BOX_S)
    _arr(7.3,11.4,9.5,11.4)
    _box(14.5,11.4,11,1.2,"Composite index A\u1d62 per building",
         "A_total = A_composite + A_tele",bc="#1A3A5C",fs=FS_BOX+1)

    # Row 8: Outputs
    _arr(14.5,10.8,14.5,10.0)
    _hdr(14.5,9.7,"\u2193  Outputs  \u2193")
    ox=[3,8.5,14,19.5]
    oi=[("Building-level\naccessibility",BLUE),("Settlement-level\nchoropleth",TEAL),
        ("Scenario \u0394-maps\n(before/after)",ORANGE),("Gini \u00b7 Lorenz\n\u00b7 Moran I",RED_ACC)]
    for o,(ol,oc) in zip(ox,oi):
        _ln(14.5,9.5,o,9.1)
        _box(o,8.4,4.5,1.2,ol,bc=oc,fs=FS_BOX_S)

    # Row 9: Scenarios
    _hdr(11,7.2,"SCENARIO ANALYSIS",fs=FS_BOX)
    scx=[3,8.5,14,19.5]
    sci=[("S1: Telemedicine\nboost",TEAL),("S2: 3 new\nhospitals",GREEN_ACC),
         ("S3: Road closure\nstress test",RED_ACC),("S4: Real facility\nopening",ORANGE)]
    for s,(sl,sc) in zip(scx,sci):
        _box(s,6.2,4.5,1.2,sl,bc=sc,fs=FS_BOX_S)
        _ln(s,6.85,s,7.05)

    # Legend
    ax.text(11,4.6,
            "E2SFCA = Enhanced Two-Step Floating Catchment Area   |   "
            "M/M/1 = single-server queuing model   |   OD = Origin\u2013Destination",
            ha="center",fontsize=FS_SUB,color=GRAY_60,style="italic",
            bbox=dict(boxstyle="round,pad=0.5",facecolor="#F8F9FA",
                      edgecolor=GRAY_30,alpha=0.9))

    if save_path: savefig(fig, save_path)
    return fig


def plot_decay_curves(radius_s=1800.0, betas=None, figsize=(12,5), save_path=None):
    if betas is None: betas=[0.5,1.0,1.5,2.0,3.0]
    t=np.linspace(0,radius_s*1.2,500); r_m=radius_s/60
    fig,(ax_g,ax_e)=plt.subplots(1,2,figsize=figsize,sharey=True,gridspec_kw={"wspace":0.08})
    cols=plt.cm.viridis(np.linspace(0.15,0.9,len(betas)))
    for b,c in zip(betas,cols):
        s=radius_s*0.5/(b**0.5); wg=np.exp(-(t/s)**2); we=np.exp(-b*t/radius_s)
        lw=2.5 if b==1.5 else 1.5; ls="-" if b==1.5 else "--"
        sfx=" \u2190 active" if b==1.5 else ""
        ax_g.plot(t/60,wg,color=c,lw=lw,ls=ls,label=f"\u03b2={b:.1f}{sfx}")
        ax_e.plot(t/60,we,color=c,lw=lw,ls=ls,label=f"\u03b2={b:.1f}{sfx}")
    for ax in (ax_g,ax_e):
        ax.axvline(r_m,color=RED_ACC,lw=1.5,ls=":",label=f"radius={r_m:.0f} min")
        ax.axhline(0,color=GRAY_30,lw=0.8); ax.set_xlim(0,r_m*1.2); ax.set_ylim(-0.02,1.05)
        ax.set_xlabel("Travel time (min)"); ax.legend(fontsize=8)
    ax_g.set_ylabel("Weight W(t)"); ax_g.set_title("Gaussian",fontweight="bold",color=NAVY)
    ax_e.set_title("Exponential",fontweight="bold",color=NAVY)
    fig.suptitle(f"Decay Functions (radius={r_m:.0f} min)",fontsize=12,fontweight="bold",color=NAVY,y=1.02)
    plt.tight_layout()
    if save_path: savefig(fig,save_path)
    return fig


def plot_mm1_effect(figsize=(11,5), save_path=None):
    from .style import BLUE_SOFT
    a=0.001; rho=np.linspace(0.01,0.94,200)
    caps=[50.,150.,300.]; cc=[BLUE,TEAL,ORANGE]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=figsize,gridspec_kw={"wspace":0.3})
    for cap,c in zip(caps,cc):
        mu=cap/86400.; W=rho/(mu*np.maximum(1-rho,1e-4)); W=np.minimum(W,86400.)
        ax1.plot(rho,np.exp(-a*W),color=c,lw=2,label=f"cap={cap:.0f}")
    ax1.axhline(1,color=GRAY_60,lw=0.8,ls="--"); ax1.set_xlim(0,1); ax1.set_ylim(0,1.05)
    ax1.set_xlabel("\u03c1"); ax1.set_ylabel("Effective/nominal"); ax1.legend(fontsize=8)
    ax1.set_title("Capacity reduction",fontweight="bold",color=NAVY)
    cr=np.linspace(1,500,200)
    for r,c in zip([0.3,0.5,0.7,0.85,0.95],[BLUE_SOFT,TEAL,ORANGE,RED_ACC,NAVY]):
        mu=cr/86400.; W=r/(mu*np.maximum(1-r,1e-4)); W=np.minimum(W,86400.)
        ax2.plot(cr,cr*np.exp(-a*W),color=c,lw=2,label=f"\u03c1={r:.2f}")
    ax2.plot(cr,cr,color=GRAY_30,lw=1,ls=":",label="No correction")
    ax2.set_xlabel("Nominal capacity"); ax2.set_ylabel("Effective capacity"); ax2.legend(fontsize=8)
    ax2.set_title("Effective vs nominal",fontweight="bold",color=NAVY)
    fig.suptitle("M/M/1 Queue Correction",fontsize=12,fontweight="bold",color=NAVY,y=1.02)
    plt.tight_layout()
    if save_path: savefig(fig,save_path)
    return fig

"""
viz/style.py — ITMO-branded design system for all visualisations.

ITMO Official colours:
  #2150A2  Bluetiful (primary)
  #EC1946  Pink Flamingo (accent)
  #7F00FF  Violet
  #00BFFF  Capri
  #9FE82D  Green Lizard
  #FFD700  Honey Yellow
  #FF6347  Tart Orange

Font: Golos Text (with fallbacks)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib as mpl

# ─── ITMO Brand Palette ──────────────────────────────────────────────────────
ITMO_BLUE   = "#2150A2"
ITMO_RED    = "#EC1946"
ITMO_VIOLET = "#7F00FF"
ITMO_CAPRI  = "#00BFFF"
ITMO_GREEN  = "#9FE82D"
ITMO_YELLOW = "#FFD700"
ITMO_ORANGE = "#FF6347"

# ─── Semantic aliases ─────────────────────────────────────────────────────────
NAVY      = "#0D2137"
BLUE      = ITMO_BLUE
BLUE_SOFT = "#5E8CD4"
ORANGE    = ITMO_ORANGE
TEAL      = "#008B8B"
RED_ACC   = ITMO_RED
GREEN_ACC = "#2D8A4E"

GRAY_10   = "#F5F6F8"
GRAY_30   = "#C4C9D2"
GRAY_60   = "#6B7280"
GRAY_80   = "#374151"
WHITE     = "#FFFFFF"

# ─── Facility styles: (colour, English_label, Russian_label) ──────────────────
#     maps.py unpacks as: for spec, (color, lbl_en, _) in SPEC_STYLES.items()
SPEC_STYLES: Dict[str, Tuple[str, str, str]] = {
    "hospital_full":          (ITMO_RED,    "Hospital (full)",       "Стационар"),
    "outpatient_specialized": (ITMO_ORANGE, "Outpatient specialist", "Специализированный"),
    "primary_therapist":      ("#1A7A4A",   "Primary care (GP)",     "ПМП: терапевт"),
    "primary_basic":          (ITMO_VIOLET, "Primary care (FAP)",    "ПМП: ФАП"),
    "pediatric":              (ITMO_BLUE,   "Paediatric",            "Педиатрический"),
    "maternal":               ("#AD1457",   "Maternal",              "Материнство"),
    "unknown":                (GRAY_60,     "Other",                 "Прочее"),
}

GROUP_COLORS: Dict[str, str] = {
    "primary":     "#1A7A4A",
    "pediatric":   ITMO_BLUE,
    "specialized": ITMO_ORANGE,
    "hospital":    ITMO_RED,
}

MODE_COLORS: Dict[str, str] = {
    "car":  ITMO_RED,
    "walk": "#2D8A4E",
    "pt":   ITMO_BLUE,
}

MODE_LABELS: Dict[str, str] = {
    "car": "Car", "walk": "Walking", "pt": "Public Transit",
}

# ─── Colormaps ────────────────────────────────────────────────────────────────
CMAP_ACCESS = "YlGnBu"      # yellow→green→blue (intuitive: warm=low, cool=high)
CMAP_DELTA  = "RdBu_r"      # red=worse, blue=better
CMAP_HEAT   = "YlOrRd"      # for sensitivity heatmaps

# ─── Figure sizes ─────────────────────────────────────────────────────────────
FIG_FULL    = (14, 12)
FIG_HALF_H  = (10, 8)
FIG_WIDE    = (16, 5)
FIG_CHART   = (12, 8)
FIG_SQUARE  = (9, 9)
FIG_PANEL4  = (16, 14)

# ─── Font stack (Golos → fallbacks) ──────────────────────────────────────────
_FONT_STACK = "Golos Text, Inter, Segoe UI, Helvetica, DejaVu Sans, sans-serif"


def setup_style() -> None:
    """Apply ITMO-branded matplotlib style globally."""
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    _FONT_STACK.split(", "),
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.edgecolor":     GRAY_30,
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "grid.color":         GRAY_30,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   GRAY_30,
        "figure.facecolor":   WHITE,
        "axes.facecolor":     WHITE,
        "savefig.dpi":        250,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
    })


# ─── Map helpers ─────────────────────────────────────────────────────────────

def add_scalebar(ax, length_m: int = 5000, crs: str = "EPSG:3857",
                 side: str = "right") -> None:
    """Add a simple scalebar to the map. side='right' places it bottom-right."""
    xl = ax.get_xlim(); yl = ax.get_ylim()
    xspan = xl[1] - xl[0]; yspan = yl[1] - yl[0]
    if side == "right":
        xs = xl[1] - 0.05 * xspan - length_m
    else:
        xs = xl[0] + 0.05 * xspan
    ys = yl[0] + 0.04 * yspan
    ax.plot([xs, xs + length_m], [ys, ys],
            color=NAVY, lw=3, solid_capstyle="butt", zorder=50)
    ax.plot([xs, xs], [ys, ys + yspan*0.007], color=NAVY, lw=2, zorder=50)
    ax.plot([xs+length_m, xs+length_m], [ys, ys+yspan*0.007], color=NAVY, lw=2, zorder=50)
    label = f"{length_m//1000} km" if length_m >= 1000 else f"{length_m} m"
    ax.text(xs + length_m/2, ys + yspan*0.016, label,
            ha="center", va="bottom", fontsize=8, color=NAVY, zorder=50)


def add_north_arrow(ax, x: float = 0.97, y: float = 0.12, size: float = 0.04) -> None:
    import matplotlib.patheffects as pe
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - size * 1.1),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", fontsize=14, fontweight="bold", color=NAVY, zorder=50,
        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.8),
        path_effects=[pe.withStroke(linewidth=2, foreground=WHITE)],
    )


def add_stats_box(ax, stats: Dict[str, str],
                  loc: str = "upper left", title: str = "") -> None:
    lines = []
    if title:
        lines.append(title)
        lines.append("─" * len(title))
    for k, v in stats.items():
        lines.append(f"{k}: {v}")
    text = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.5", facecolor=WHITE,
                 edgecolor=GRAY_30, alpha=0.93)
    x, y, ha, va = {
        "upper left":  (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left":  (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }.get(loc, (0.02, 0.98, "left", "top"))
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8, color=GRAY_80, ha=ha, va=va,
            bbox=props, family="monospace", zorder=50)


def add_colorbar(fig, ax, cmap, vmin, vmax,
                 label: str = "E2SFCA", diverging: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if diverging \
           else Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(label, fontsize=9, color=GRAY_80)
    cbar.ax.tick_params(labelsize=8)


def apply_map_style(ax) -> None:
    ax.set_axis_off()


def label_bars(ax, rects, fmt="{:.3f}", fontsize=8) -> None:
    """Label bar chart rectangles with their values."""
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=NAVY)


def savefig(fig, path, dpi: int = 250) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  ✓ Saved: {path.name}")

"""
core/facilities.py — Facility processing: cleaning, specialization, capacity.

Audit fix:
  • Self-contained class with no external globals.
  • Capacity formula is deterministic and documented.
  • Returns a copy — never mutates the original GeoDataFrame.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore  # geopandas optional at import time

from ..config import (
    SPEC_MAP,
    PATIENTS_PER_DOCTOR,
    BED_TURNOVER,
    MIN_CAPACITY,
    SEASONAL,
)


class FacilityProcessor:
    """
    Cleans, classifies, and computes effective capacity for healthcare facilities.

    Usage:
        fp = FacilityProcessor(raw_gdf)
        processed = fp.clean().assign_specialization().compute_capacity("summer").result
    """

    # Extended OSM healthcare -> specialization mapping.
    # SORTED LONGEST-FIRST in assign_specialization() to prevent substring collisions.
    SPEC_MAP = SPEC_MAP

    PATIENTS_PER_DOCTOR = PATIENTS_PER_DOCTOR

    def __init__(self, fac: gpd.GeoDataFrame):
        self._fac = fac.copy()

    # ── Fluent API ────────────────────────────────────────────────────────────

    def clean(self) -> "FacilityProcessor":
        """Coerce numeric columns; fill NaNs with 0."""
        for col in ("doctors", "beds", "visits_per_shift", "season_load_factor"):
            if col in self._fac.columns:
                self._fac[col] = (
                    self._fac[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                )
        return self

    def assign_specialization(self) -> "FacilityProcessor":
        """
        Map OSM tags to specialization categories using SPEC_MAP.
        Longest keywords are matched first to avoid substring collisions
        ('polyclinic' must match before 'clinic').
        """
        if "specialization" not in self._fac.columns:
            self._fac["specialization"] = "unknown"

        ordered = sorted(self.SPEC_MAP.items(), key=lambda x: -len(x[0]))

        for src_col in ("type", "amenity", "healthcare"):
            if src_col not in self._fac.columns:
                continue
            for keyword, spec in ordered:
                mask = self._fac[src_col].str.lower().str.contains(
                    keyword, na=False, regex=False
                )
                not_set = self._fac["specialization"].isin(["unknown", None, ""])
                self._fac.loc[mask & not_set, "specialization"] = spec

        n_unk = (self._fac["specialization"] == "unknown").sum()
        if n_unk:
            print(f"  [{n_unk} facilities unmatched → 'unknown'; check SPEC_MAP]")
        return self

    def compute_capacity(self, season: str = "summer") -> "FacilityProcessor":
        """
        Compute effective_capacity per facility.

        Formula (literature-aligned):
          primary / outpatient:  doctors * PATIENTS_PER_DOCTOR
                                 (fallback: visits_per_shift if doctors=0)
          hospital_full:         beds * BED_TURNOVER
                                 + doctors * PATIENTS_PER_DOCTOR / 2
          maternal / pediatric:  same as primary, with bed supplement

        Season load factor applied last. Capacity is floored at MIN_CAPACITY.
        """
        has = lambda c: c in self._fac.columns

        load = (
            self._fac["season_load_factor"]
            if has("season_load_factor")
            else pd.Series(1.0, index=self._fac.index)
        )
        season_multiplier = SEASONAL.get(season, {}).get("speed", 1.0)

        caps: list[float] = []
        for idx, row in self._fac.iterrows():
            spec  = row.get("specialization", "unknown")
            bl    = float(load.get(idx, 1.0)) * season_multiplier
            docs  = float(row.get("doctors", 0)          if has("doctors")          else 0)
            beds  = float(row.get("beds", 0)             if has("beds")             else 0)
            vps   = float(row.get("visits_per_shift", 0) if has("visits_per_shift") else 0)

            if spec == "hospital_full":
                cap = beds * BED_TURNOVER + docs * self.PATIENTS_PER_DOCTOR * 0.5
            elif spec in ("maternal", "pediatric"):
                cap = docs * self.PATIENTS_PER_DOCTOR + beds * BED_TURNOVER * 0.5
            else:
                # primary / outpatient: doctor-based, visits fallback
                cap = docs * self.PATIENTS_PER_DOCTOR if docs > 0 else vps

            caps.append(max(float(cap) * bl, MIN_CAPACITY))

        self._fac["effective_capacity"] = caps
        return self

    @property
    def result(self) -> gpd.GeoDataFrame:
        """Return the processed GeoDataFrame."""
        return self._fac.copy()

    # ── Convenience ──────────────────────────────────────────────────────────

    def full_pipeline(self, season: str = "summer") -> gpd.GeoDataFrame:
        return self.clean().assign_specialization().compute_capacity(season).result

    def summary(self) -> pd.DataFrame:
        """Capacity summary by specialization."""
        return (
            self._fac.groupby("specialization")["effective_capacity"]
            .agg(["count", "sum", "mean", "min", "max"])
            .round(1)
        )


def make_virtual_facility(
    capacity:       float,
    specialization: str,
    crs:            str,
    label:          str = "virtual",
) -> gpd.GeoDataFrame:
    """
    Create a single-row GeoDataFrame representing a virtual (non-geographic) facility.

    Used by the Telemedicine scenario to inject a virtual primary-care supply
    directly into the E2SFCA supply side, instead of post-processing the result.

    The geometry is set to None (POINT EMPTY) because distance decay does not
    apply — the facility is reachable from all demand points with zero travel time.
    """
    from shapely.geometry import Point
    return gpd.GeoDataFrame(
        {
            "fullname":           [label],
            "specialization":     [specialization],
            "effective_capacity": [float(capacity)],
            "doctors":            [0],
            "beds":               [0],
            "visits_per_shift":   [float(capacity)],
            "is_virtual":         [True],
        },
        geometry=[Point(0, 0)],   # placeholder; OD column will be zeros
        crs=crs,
    )

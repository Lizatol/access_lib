"""
accessibility_lib/config.py — Central configuration.

CALIBRATION v7:
    radius_car_s  = 3600 s (60 min) — OD car median=51.9 min → captures 52%
    radius_walk_s = 1800 s (30 min) — OD walk median=45 min
    radius_pt_s   = 5400 s (90 min) — OD PT median=80.9 min → captures 60%
    beta_car  = 1.5  — moderate Gaussian decay
    beta_walk = 1.0  — gentle
    beta_pt   = 1.2  — softer for transit users
    PATIENTS_PER_DOCTOR = 25 — WHO Europe standard
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple


class DecayType(str, Enum):
    GAUSSIAN    = "gaussian"
    EXPONENTIAL = "exponential"
    LINEAR      = "linear"
    POWER       = "power"


# ─── Specialisation types ─────────────────────────────────────────────────────
SPEC_TYPES = (
    "hospital_full",
    "outpatient_specialized",
    "pediatric",
    "primary_therapist",
    "primary_basic",
    "maternal",
    "unknown",
)

SPEC_MAP: Dict[str, str] = {
    "midwife station":   "primary_basic",
    "childrens clinic":  "pediatric",
    "children's clinic": "pediatric",
    "polyclinic":        "outpatient_specialized",
    "specialist":        "outpatient_specialized",
    "outpatient":        "outpatient_specialized",
    "pediatric":         "pediatric",
    "maternity":         "maternal",
    "midwifery":         "maternal",
    "hospital":          "hospital_full",
    "clinic":            "primary_therapist",
    "doctors":           "primary_therapist",
    "fms":               "primary_basic",
    "FMS":               "primary_basic",
}

PRIMARY_CARE_SPECS = frozenset({"primary_therapist", "primary_basic"})

SPEC_GROUPS = {
    "primary":      ["primary_therapist", "primary_basic"],
    "pediatric":    ["pediatric"],
    "specialized":  ["outpatient_specialized", "maternal"],
    "hospital":     ["hospital_full"],
}


@dataclass(frozen=True)
class E2SFCAParams:
    """
    E2SFCA parameters — calibrated to actual OD matrix distributions.

    radius_car_s  = 3600 s (60 min) — OD car p50=51.9 min, captures 52%
    radius_walk_s = 1800 s (30 min) — OD walk p50=45.3 min
    radius_pt_s   = 5400 s (90 min) — OD PT p50=80.9 min, captures 60%
    """
    beta_car:    float = 1.5
    beta_walk:   float = 1.0
    beta_pt:     float = 1.2

    radius_car_s:  float = 3600.0   # 60 min
    radius_walk_s: float = 1800.0   # 30 min
    radius_pt_s:   float = 5400.0   # 90 min

    decay_type:    DecayType = DecayType.GAUSSIAN
    gaussian_sigma_factor: float = 0.5   # σ = radius * factor / √β

    apply_mm1:  bool  = True
    rho_max:    float = 0.95       # M/M/1: max utilisation
    alpha_q:    float = 0.001      # M/M/1: queue sensitivity
    w_max_s:    float = 86400.0    # M/M/1: max wait (seconds)

    nearest_k:  int   = 5
    normalise:  bool  = False


@dataclass
class ModeWeights:
    car:  float = 0.40
    walk: float = 0.20
    pt:   float = 0.40

    def as_dict(self) -> Dict[str, float]:
        return {"car": self.car, "walk": self.walk, "pt": self.pt}


# ─── Transport constants ─────────────────────────────────────────────────────
WALK_SPEED_MS: float    = 5 * 1000 / 3600   # 5 km/h → m/s
PT_PENALTY:    float    = 1.5                # waiting/transfer penalty
PT_BUS_SPEED_KMH: float = 25.0
WALK_CUTOFF_M: float    = 6_000.0            # max walking distance (m)

ROAD_SPEEDS: Dict[str, float] = {
    "motorway": 110.0, "motorway_link": 90.0,
    "trunk": 90.0,     "trunk_link": 70.0,
    "primary": 70.0,   "primary_link": 60.0,
    "secondary": 60.0, "secondary_link": 50.0,
    "tertiary": 50.0,  "tertiary_link": 40.0,
    "residential": 30.0, "living_street": 10.0,
    "unclassified": 40.0, "service": 20.0,
    "track": 15.0,     "path": 5.0,
    "cycleway": 15.0,  "footway": 5.0,
}

# ─── Facility capacity ───────────────────────────────────────────────────────
PATIENTS_PER_DOCTOR: int   = 25    # WHO Europe standard
BED_TURNOVER:        float = 0.3
MIN_CAPACITY:        float = 1.0

# ─── Seasonal adjustments ────────────────────────────────────────────────────
SEASONAL: Dict[str, Dict[str, float]] = {
    "summer": {"demand": 0.85, "speed": 1.10},
    "autumn": {"demand": 1.10, "speed": 0.95},
    "winter": {"demand": 1.20, "speed": 0.80},
    "spring": {"demand": 0.95, "speed": 1.00},
}

# ─── Monte Carlo ─────────────────────────────────────────────────────────────
MC_N_ITER:   int   = 500
MC_SEED:     int   = 42
MC_CAP_STD:  float = 0.10
MC_POP_STD:  float = 0.05
MC_OD_STD_SUMMER: float = 0.03
MC_OD_STD_WINTER: float = 0.15


@dataclass
class Config:
    root:          Path = field(default_factory=lambda: Path("."))
    active_season: str  = "summer"
    e2sfca:        E2SFCAParams = field(default_factory=E2SFCAParams)
    mode_weights:  ModeWeights  = field(default_factory=ModeWeights)
    mc_n_iter:     int = MC_N_ITER
    mc_seed:       int = MC_SEED

    @property
    def data_path(self) -> Path:
        return self.root

    @property
    def cache_path(self) -> Path:
        p = self.root / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_path(self) -> Path:
        p = self.root / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figures_path(self) -> Path:
        p = self.results_path / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def tables_path(self) -> Path:
        p = self.results_path / "tables"
        p.mkdir(parents=True, exist_ok=True)
        return p

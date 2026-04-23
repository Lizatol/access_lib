"""
scenarios/telemedicine.py — S1: Telemedicine as distance-independent primary care.

FIX-2 CRITICAL: Telemedicine redesigned to be truly distance-independent.

BEFORE (broken):
  - Virtual facility added to OD matrices with travel_time = 0
  - Went through decay_w() → W_virtual = 1.0 for all buildings
  - Went through nearest_k filter → virtual facility ALWAYS wins slot #1
    displacing the 3rd (or kth) real nearest facility
  - Result: NEGATIVE net effect in facility-rich areas (displaced real supply)
  - ext_demand with negative values caused denominator explosions

AFTER (correct):
  - TelemedicineScenario.build_inputs() returns UNMODIFIED OD and supply
  - It computes tele_capacity and returns it in metadata
  - The notebook (or calling code) uses add_telemedicine_contribution():
      A_s1 = add_telemedicine_contribution(A_baseline, tele_capacity, total_demand)
  - Result: UNIFORM positive delta = tele_capacity/total_demand for ALL buildings
  - Physically correct: remote consultation is equally accessible from any home

Usage pattern in notebook:
  result_s1 = s1.build_inputs(OD_MATRICES, supply, demand, specializations=specs)
  A_physical = composite_accessibility(OD_MATRICES, demand, supply, ...)
  A_s1 = add_telemedicine_contribution(
      A_physical,
      tele_capacity=result_s1.metadata["tele_capacity"],
      total_demand=result_s1.metadata["total_demand"],
  )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

import numpy as np

from ..config import PRIMARY_CARE_SPECS
from .base import BaseScenario, ScenarioResult

TELE_SPEC = "primary_therapist"


@dataclass
class TelemedicineScenario(BaseScenario):
    """
    S1: Distance-independent telemedicine boost to primary care accessibility.

    Parameters
    ----------
    tele_capacity_fraction : float
        virtual_cap = tele_capacity_fraction × total_primary_care_capacity.
        Represents the fraction of primary care that can be delivered remotely.
    adoption_rate : float
        Fraction of the population with effective access to telemedicine.
        Stored in metadata for reporting; does NOT reduce physical OD.
    scope_specs : frozenset
        Specializations in scope (only primary care).
    """
    tele_capacity_fraction: float = 0.30
    adoption_rate:          float = 0.30
    scope_specs:            FrozenSet[str] = PRIMARY_CARE_SPECS

    @property
    def name(self) -> str:
        return "S1_telemedicine"

    def build_inputs(
        self,
        baseline_od:      Dict[str, np.ndarray],
        baseline_supply:  np.ndarray,
        demand:           np.ndarray,
        specializations:  Optional[np.ndarray] = None,
        **kwargs,
    ) -> ScenarioResult:
        """
        FIX-2: Returns UNMODIFIED OD and supply.

        The telemedicine contribution is computed OUTSIDE the OD pipeline
        using add_telemedicine_contribution() from core.engine.

        Returns ScenarioResult with:
          - od_matrices:     None → use baseline OD unchanged
          - supply:          None → use baseline supply unchanged
          - specializations: None → use baseline specs unchanged
          - ext_demand:      None → no demand modification
          - metadata:        {tele_capacity, total_demand, adoption_rate, ...}
            → caller uses these to invoke add_telemedicine_contribution()
        """
        # Compute virtual capacity from primary care facilities
        if specializations is not None:
            primary_mask = np.isin(specializations, list(self.scope_specs))
        else:
            primary_mask = np.ones(len(baseline_supply), dtype=bool)

        primary_cap  = float(baseline_supply[primary_mask].sum())
        virtual_cap  = max(primary_cap * self.tele_capacity_fraction, 0.0)
        total_demand = float(demand.sum())

        return ScenarioResult(
            name=self.name,
            od_matrices=None,       # FIX-2: do NOT modify OD
            supply=None,            # FIX-2: do NOT modify supply
            specializations=None,   # FIX-2: do NOT add virtual spec
            ext_demand=None,        # FIX-2: do NOT modify demand denominators
            metadata={
                "tele_capacity":             round(virtual_cap, 2),
                "primary_capacity_baseline": round(primary_cap, 2),
                "tele_capacity_fraction":    self.tele_capacity_fraction,
                "adoption_rate":             self.adoption_rate,
                "total_demand":              round(total_demand, 2),
                "R_tele":                    round(virtual_cap / max(total_demand, 1), 8),
                "scope_specs":               list(self.scope_specs),
                "n_primary_facilities":      int(primary_mask.sum()),
                "implementation":            "additive_uniform",
                "usage": (
                    "from accessibility_lib.core.engine import add_telemedicine_contribution\n"
                    "A_s1 = add_telemedicine_contribution(\n"
                    "    A_physical, result_s1.metadata['tele_capacity'],\n"
                    "    result_s1.metadata['total_demand'])"
                ),
            },
        )

    def compute_contribution(
        self,
        A_physical:           np.ndarray,
        total_demand:         Optional[float] = None,
        digital_divide_factor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convenience wrapper: compute and return A_physical + A_tele.
        Requires A_physical and optionally total_demand override.
        """
        from ..core.engine import add_telemedicine_contribution
        _td = total_demand if total_demand is not None else float(np.sum(np.ones(len(A_physical))))
        # We can't know total_demand here without demand array — caller should pass it.
        # This method is a convenience; primary pattern uses metadata from build_inputs().
        raise NotImplementedError(
            "Call add_telemedicine_contribution(A_physical, tele_capacity, total_demand) "
            "using values from result.metadata['tele_capacity'] and result.metadata['total_demand']."
        )

    def validate(
        self,
        result:          ScenarioResult,
        baseline_od:     Dict[str, np.ndarray],
        baseline_supply: np.ndarray,
        demand:          np.ndarray,
        **kwargs,
    ) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}

        # FIX-2: OD and supply must be UNCHANGED
        checks["od_unchanged"] = result.od_matrices is None
        checks["supply_unchanged"] = result.supply is None
        checks["ext_demand_none"] = result.ext_demand is None

        # Metadata must carry capacity and demand
        meta = result.metadata
        checks["tele_capacity_positive"] = meta.get("tele_capacity", 0) > 0
        checks["total_demand_positive"]  = meta.get("total_demand", 0) > 0
        checks["R_tele_computable"] = (
            meta.get("tele_capacity", 0) > 0 and
            meta.get("total_demand", 0) > 0
        )

        # Verify uniform addend is correctly computable
        if checks["R_tele_computable"]:
            R = meta["tele_capacity"] / meta["total_demand"]
            checks["R_tele_positive"]   = R > 0
            checks["R_tele_reasonable"] = R < 1.0  # sanity: < 100% of per-person demand

        return checks

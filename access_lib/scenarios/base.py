"""
scenarios/base.py — BaseScenario ABC.

Audit fix v1.1:
  • ScenarioResult gains a `specializations` field (BUG-2 fix).
    Telemedicine uses this to extend the specializations array with
    'primary_therapist' for the virtual facility, so composite_accessibility()
    knows which layer the virtual facility belongs to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class ScenarioResult:
    """
    Output of one scenario run.

    Only the fields that the scenario actually changes are non-None.
    The caller computes composite_accessibility using the original params.
    """
    name: str

    # Modified OD matrices.  None = use baseline OD unchanged.
    od_matrices:     Optional[Dict[str, np.ndarray]] = None

    # Modified supply (capacity) array.  None = use baseline supply.
    supply:          Optional[np.ndarray] = None

    # BUG-1 fix: extra demand array (n_fac,) forwarded to e2sfca().
    ext_demand:      Optional[np.ndarray] = None

    # BUG-2 fix: augmented specializations array (n_fac_augmented,).
    # None = use baseline specializations unchanged.
    specializations: Optional[np.ndarray] = None

    metadata:        dict = field(default_factory=dict)

    def effective_od(
        self,
        baseline_od: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if self.od_matrices is None:
            return baseline_od
        result = dict(baseline_od)
        result.update(self.od_matrices)
        return result

    def effective_supply(self, baseline_supply: np.ndarray) -> np.ndarray:
        return self.supply if self.supply is not None else baseline_supply

    def effective_specializations(
        self,
        baseline_specializations: np.ndarray,
    ) -> np.ndarray:
        """BUG-2 fix: return augmented specializations, or baseline if unchanged."""
        return (
            self.specializations
            if self.specializations is not None
            else baseline_specializations
        )


class BaseScenario(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build_inputs(
        self,
        baseline_od:     Dict[str, np.ndarray],
        baseline_supply: np.ndarray,
        demand:          np.ndarray,
        **kwargs,
    ) -> ScenarioResult: ...

    def validate(
        self,
        result:          ScenarioResult,
        baseline_od:     Dict[str, np.ndarray],
        baseline_supply: np.ndarray,
        demand:          np.ndarray,
    ) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        eff_supply = result.effective_supply(baseline_supply)
        checks["supply_positive"] = bool((eff_supply > 0).all())
        for mode, od in result.effective_od(baseline_od).items():
            checks[f"od_{mode}_non_negative"] = bool(
                (od[np.isfinite(od)] >= 0).all()
            )
        return checks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

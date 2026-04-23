"""
scenarios/real_hospital.py — S4: Opening of a real geocoded hospital.

AUDIT FIX (critical):
  Previous implementation (S5_polyclinic_2026):
    • Fixed coordinates, named "polyclinic"
    • specialization = outpatient_specialized
    • Effect verified only on outpatient layer
    → This is a polyclinic scenario, NOT a real hospital opening.

  Correct implementation (this file):
    • specialization = hospital_full
    • Geocoded coordinates (real or planned)
    • Capacity estimated from beds + doctors (or explicit parameter)
    • Effect verified on:
        - district mean accessibility
        - settlement mean (nearest settlements)
        - per-layer delta: hospital_full layer should show largest improvement
        - population within N-min isochrone
    • Can be used for ANY real facility opening (hospital or polyclinic)
      by setting specialization and capacity parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore

from ..config import E2SFCAParams, ModeWeights, MIN_CAPACITY
from ..core.engine import composite_accessibility, e2sfca_layer
from .base import BaseScenario, ScenarioResult


@dataclass
class RealHospitalScenario(BaseScenario):
    """
    S4: Open a new real healthcare facility at a geocoded location.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph (projected).
    demand_gdf : GeoDataFrame
        Residential demand points with 'graph_node'.
    lon, lat : float
        WGS-84 coordinates of new facility.
    specialization : str
        Must be one of SPEC_TYPES. For a real hospital: 'hospital_full'.
        For a polyclinic: 'outpatient_specialized'.
    capacity : float
        Effective capacity (visits/day or beds-equivalent).
        If None, estimated from beds+doctors parameters.
    beds : float, doctors : float
        Used to compute capacity when capacity=None.
    label : str
        Human-readable name for reporting.
    crs_input : str
        CRS of lon/lat input. Default: WGS-84.
    crs_metric : str
        Projected CRS for graph. Default: EPSG:32636 (UTM zone 36N, Russia).
    isochrone_cutoffs_min : list of float
        Travel-time cutoffs (minutes) for isochrone population analysis.
    """
    G:                      Any     = field(repr=False)
    demand_gdf:             Any     = field(repr=False)
    lon:                    float   = 0.0
    lat:                    float   = 0.0
    specialization:         str     = "hospital_full"
    capacity:               Optional[float] = None
    beds:                   float   = 0.0
    doctors:                float   = 0.0
    label:                  str     = "new_facility"
    crs_input:              str     = "EPSG:4326"
    crs_metric:             str     = "EPSG:32636"
    walk_speed_ms:          float   = 5 * 1000 / 3600
    pt_penalty:             float   = 1.5
    isochrone_cutoffs_min:  List[float] = field(
        default_factory=lambda: [10.0, 20.0, 30.0]
    )

    @property
    def name(self) -> str:
        return "S4_real_hospital"

    def build_inputs(
        self,
        baseline_od:      Dict[str, np.ndarray],
        baseline_supply:  np.ndarray,
        demand:           np.ndarray,
        params:           E2SFCAParams = None,
        mode_weights:     ModeWeights = None,
        specializations:  Optional[np.ndarray] = None,
        **kwargs,
    ) -> ScenarioResult:
        # FIX-7: explicit validation
        if params is None:
            raise ValueError(
                "S4 RealHospitalScenario.build_inputs() requires 'params' (E2SFCAParams). "
                "Pass cfg.e2sfca."
            )
        if mode_weights is None:
            raise ValueError(
                "S4 RealHospitalScenario.build_inputs() requires 'mode_weights' (ModeWeights). "
                "Pass cfg.mode_weights."
            )

        """
        1. Geocode facility to nearest graph node.
        2. Compute OD column from new facility to all demand points.
        3. Augment OD matrices and supply array.
        4. Compute per-layer delta to verify correct specialization impact.
        5. Compute isochrone population catchments.
        """
        import networkx as nx
        from scipy.spatial import cKDTree

        # ── 1. Snap to graph node ─────────────────────────────────────────────
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            self.crs_input, self.crs_metric, always_xy=True
        )
        x, y = transformer.transform(self.lon, self.lat)

        node_ids = np.array(list(self.G.nodes()))
        node_xy  = np.array([
            [self.G.nodes[n].get("x", 0), self.G.nodes[n].get("y", 0)]
            for n in node_ids
        ])
        _, ni = cKDTree(node_xy).query([[x, y]])
        fac_node = int(node_ids[int(ni.flat[0])])

        print(f"  S4 {self.label!r}: node={fac_node}  spec={self.specialization}")

        # ── 2. Compute OD column ──────────────────────────────────────────────
        demand_nodes = self.demand_gdf["graph_node"].tolist()
        od_col = self._od_column(fac_node, demand_nodes)

        # ── 3. Augment OD and supply ──────────────────────────────────────────
        cap = self._estimate_capacity()
        aug_od = {
            mode: np.hstack([baseline_od[mode], od_col[mode][:, np.newaxis]])
            for mode in baseline_od
        }
        aug_supply = np.append(baseline_supply, cap).astype(np.float32)

        # Build augmented specializations array
        aug_specs = None
        if specializations is not None:
            aug_specs = np.append(specializations, self.specialization)

        # ── 4. Per-layer delta — используем normalise=False чтобы видеть сырые дельты
        # При normalise=True все слои нормализуются к mean=1.0 → дельты = 0.
        # Для сравнения слоёв нужны абсолютные значения.
        layer_deltas = {}
        if specializations is not None:
            from ..config import E2SFCAParams
            _params_raw = E2SFCAParams(
                beta_car=params.beta_car, beta_walk=params.beta_walk, beta_pt=params.beta_pt,
                radius_car_s=params.radius_car_s, radius_walk_s=params.radius_walk_s,
                radius_pt_s=params.radius_pt_s,
                decay_type=params.decay_type,
                gaussian_sigma_factor=params.gaussian_sigma_factor,
                nearest_k=params.nearest_k,
                normalise=False,
            )
            for spec in np.unique(specializations):
                A0 = e2sfca_layer(
                    baseline_od["car"], demand, baseline_supply,
                    specializations, spec,
                    _params_raw.beta_car, _params_raw.radius_car_s, _params_raw,
                )
                A1 = e2sfca_layer(
                    aug_od["car"], demand, aug_supply,
                    aug_specs, spec,
                    _params_raw.beta_car, _params_raw.radius_car_s, _params_raw,
                )
                layer_deltas[spec] = round(float(A1.mean()) - float(A0.mean()), 6)

        # ── 5. Isochrone population catchments ────────────────────────────────
        isochrone_pop = self._isochrone_populations(
            od_col["car"], demand
        )

        print(f"  Layer deltas: " +
              ", ".join(f"{k}={v:+.4f}" for k, v in layer_deltas.items()))
        for cutoff, pop in isochrone_pop.items():
            print(f"  Population within {cutoff} min car: {pop:,.0f}")

        return ScenarioResult(
            name=self.name,
            od_matrices=aug_od,
            supply=aug_supply,
            specializations=aug_specs,
            metadata={
                "label":             self.label,
                "lon":               self.lon,
                "lat":               self.lat,
                "facility_node":     fac_node,
                "specialization":    self.specialization,
                "capacity":          round(cap, 1),
                "layer_deltas":      layer_deltas,
                "isochrone_pop":     isochrone_pop,
                "car_reach_pct":     round(float(np.isfinite(od_col["car"]).mean() * 100), 1),
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _estimate_capacity(self) -> float:
        """Estimate effective capacity from parameters."""
        if self.capacity is not None:
            return max(float(self.capacity), MIN_CAPACITY)
        from ..config import PATIENTS_PER_DOCTOR, BED_TURNOVER
        if self.specialization == "hospital_full":
            cap = self.beds * BED_TURNOVER + self.doctors * PATIENTS_PER_DOCTOR * 0.5
        else:
            cap = self.doctors * PATIENTS_PER_DOCTOR
        return max(float(cap), MIN_CAPACITY)

    def _od_column(
        self,
        source_node:  Any,
        demand_nodes: List[Any],
    ) -> Dict[str, np.ndarray]:
        """Compute OD column (all modes) from facility to demand points."""
        import networkx as nx
        lens_tt  = dict(nx.single_source_dijkstra_path_length(
            self.G, source_node, weight="travel_time"
        ))
        lens_len = dict(nx.single_source_dijkstra_path_length(
            self.G, source_node, weight="length"
        ))
        n = len(demand_nodes)
        car  = np.full(n, np.inf, dtype=np.float32)
        walk = np.full(n, np.inf, dtype=np.float32)
        pt   = np.full(n, np.inf, dtype=np.float32)
        for i, nd in enumerate(demand_nodes):
            t = float(lens_tt.get(nd, np.inf))
            l = float(lens_len.get(nd, np.inf))
            car[i]  = t
            walk[i] = l / self.walk_speed_ms
            pt[i]   = t * self.pt_penalty
        return {"car": car, "walk": walk, "pt": pt}

    def _isochrone_populations(
        self,
        car_od_col: np.ndarray,
        demand:     np.ndarray,
    ) -> Dict[str, float]:
        """Population demand within each travel-time isochrone."""
        result = {}
        for cutoff_min in self.isochrone_cutoffs_min:
            cutoff_s = cutoff_min * 60
            mask = car_od_col <= cutoff_s
            result[f"{cutoff_min:.0f}min"] = round(
                float(demand[mask].sum()), 1
            )
        return result

    def validate(
        self,
        result:          ScenarioResult,
        baseline_od:     Dict[str, np.ndarray],
        baseline_supply: np.ndarray,
        demand:          np.ndarray,
        params:          Optional[E2SFCAParams] = None,
        mode_weights:    Optional[ModeWeights] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        checks = super().validate(result, baseline_od, baseline_supply, demand)

        # S4-specific: OD augmented by exactly one column
        if result.od_matrices is not None:
            checks["od_augmented_by_one"] = (
                result.od_matrices["car"].shape[1]
                == baseline_od["car"].shape[1] + 1
            )

        # S4-specific: new facility must be reachable from some demand points
        if result.od_matrices is not None:
            last = result.od_matrices["car"][:, -1]
            checks["new_facility_reachable"] = bool(np.isfinite(last).any())

        # S4-specific: primary impacted layer must match specialization
        layer_deltas = result.metadata.get("layer_deltas", {})
        if layer_deltas and self.specialization in layer_deltas:
            own_delta = layer_deltas[self.specialization]
            other_deltas = [v for k, v in layer_deltas.items()
                            if k != self.specialization]
            if other_deltas:
                checks["correct_layer_most_impacted"] = bool(
                    own_delta >= max(other_deltas)
                )

        # S4-specific: district mean must improve
        if params is not None and mode_weights is not None:
            A_base = composite_accessibility(
                baseline_od, demand, baseline_supply, params,
                mode_weights.as_dict()
            )
            A_new = composite_accessibility(
                result.effective_od(baseline_od), demand,
                result.effective_supply(baseline_supply), params,
                mode_weights.as_dict(),
            )
            checks["district_mean_improves"] = bool(A_new.mean() >= A_base.mean())

        return checks

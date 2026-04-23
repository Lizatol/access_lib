"""
core/engine.py — Deterministic E2SFCA computation engine.

CRITICAL FIXES applied in this version:
  FIX-1  SPECIALIZATION CONSISTENCY
         _e2sfca_by_spec() now applies apply_mm1 UNIFORMLY to ALL specializations.
         Previously gated on spec == 'hospital_full' — broke cross-spec comparability.

  FIX-2  TELEMEDICINE (new: add_telemedicine_contribution)
         Telemedicine must NOT go through OD + decay — it is distance-independent.
         A_tele = virtual_cap / total_demand  (uniform scalar, added AFTER E2SFCA).

  FIX-3  MULTIMODAL NORMALIZATION (normalize_modes flag in composite_accessibility)
         Each mode normalized to mean=1 before weighting → prevents car dominance.

  FIX-4  SCALE OUTPUT (scale_accessibility utility)
         Exposes raw and mean-scaled values without altering internal formula.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Dict, Optional

import numpy as np

from ..config import E2SFCAParams, DecayType


# ─── Decay functions ──────────────────────────────────────────────────────────

def decay_gaussian(t: np.ndarray, radius: float,
                   sigma_factor: float = 0.5,
                   beta: float = 1.0) -> np.ndarray:
    """Gaussian decay. beta>1 → steeper. sigma = radius*sigma_factor/sqrt(beta)."""
    sigma = radius * sigma_factor / (beta ** 0.5)
    return np.exp(-(t / sigma) ** 2).astype(np.float32)


def decay_exponential(t: np.ndarray, beta: float, radius: float) -> np.ndarray:
    return np.exp(-beta * t / radius).astype(np.float32)


def decay_w(t: np.ndarray, beta: float, radius: float,
            params: E2SFCAParams) -> np.ndarray:
    if params.decay_type is DecayType.GAUSSIAN:
        return decay_gaussian(t, radius, params.gaussian_sigma_factor, beta=beta)
    return decay_exponential(t, beta, radius)


# ─── Nearest-k filter ────────────────────────────────────────────────────────

def filter_nearest_k(OD: np.ndarray, k: int) -> np.ndarray:
    n_dem, n_fac = OD.shape
    if n_fac <= k:
        return OD.copy()
    out = np.full_like(OD, np.inf)
    k_eff = min(k, n_fac)
    sentinel = np.finfo(np.float32).max
    safe = np.where(np.isfinite(OD), OD, sentinel)
    idx_k = np.argpartition(safe, k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(n_dem)[:, None]
    out[rows, idx_k] = OD[rows, idx_k]
    return out


# ─── M/M/1 queue adjustment ──────────────────────────────────────────────────

def mm1_capacity_adjust(capacity: np.ndarray,
                         demand_weighted: np.ndarray,
                         params: E2SFCAParams) -> np.ndarray:
    rho = np.minimum(demand_weighted / np.maximum(capacity, 1.0), params.rho_max)
    mu  = np.maximum(capacity, 1.0) / 86400.0
    W   = np.minimum(rho / (mu * np.maximum(1.0 - rho, 1e-4)), params.w_max_s)
    return capacity * np.exp(-params.alpha_q * W)


# ─── Core E2SFCA ─────────────────────────────────────────────────────────────

def e2sfca(
    OD:         np.ndarray,
    population: np.ndarray,
    supply:     np.ndarray,
    beta:       float,
    radius:     float,
    params:     E2SFCAParams,
    apply_mm1:  bool = False,
    ext_demand: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised E2SFCA (Luo & Qi 2009). apply_mm1 is a plain bool — no spec gating."""
    W = np.where(OD <= radius, decay_w(OD, beta, radius, params), 0.0)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0).astype(np.float32)

    demand_w = W.T @ population.astype(np.float32)
    if ext_demand is not None:
        demand_w = demand_w + np.asarray(ext_demand, dtype=np.float32)

    supply_eff = supply.astype(np.float32).copy()
    if apply_mm1:
        supply_eff = mm1_capacity_adjust(supply_eff, demand_w, params)

    floor = max(params.nearest_k * 0.001, 1e-6)
    R = supply_eff / np.maximum(demand_w, floor)
    A = W @ R

    if params.normalise:
        pos = A[A > 0]
        if len(pos) > 0:
            A = A / pos.mean()
    return A.astype(np.float32)


# ─── Layer-specific E2SFCA ───────────────────────────────────────────────────

def e2sfca_layer(
    OD:              np.ndarray,
    population:      np.ndarray,
    supply:          np.ndarray,
    specializations: np.ndarray,
    layer:           str,
    beta:            float,
    radius:          float,
    params:          E2SFCAParams,
    apply_mm1:       bool = False,
    ext_demand:      Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = specializations == layer
    if mask.sum() == 0:
        return np.zeros(len(population), dtype=np.float32)
    ext_layer = ext_demand[mask] if ext_demand is not None else None
    return e2sfca(OD[:, mask], population, supply[mask],
                  beta, radius, params, apply_mm1=apply_mm1, ext_demand=ext_layer)


# ─── Per-specialization E2SFCA ────────────────────────────────────────────────

def _e2sfca_by_spec(
    OD:              np.ndarray,
    population:      np.ndarray,
    supply:          np.ndarray,
    specializations: np.ndarray,
    beta:            float,
    radius:          float,
    params_raw:      E2SFCAParams,
    apply_mm1:       bool = False,
    ext_demand:      Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FIX-1: apply_mm1 is now UNIFORM across ALL specializations.

    BEFORE (broken): apply_mm1=(apply_mm1 and spec == 'hospital_full')
      → MM1 applied only to hospital, not primary care / outpatient.
      → Cross-spec comparisons invalid (different pipeline).

    AFTER (correct): apply_mm1=apply_mm1
      → Same pipeline for all specs. Differences come from DATA only
        (capacity size, demand catchment), not logic branching.
    """
    A = np.zeros(len(population), dtype=np.float32)
    for spec in np.unique(specializations):
        mask = specializations == spec
        ext_spec = ext_demand[mask] if ext_demand is not None else None
        A_spec = e2sfca(
            OD[:, mask], population, supply[mask],
            beta, radius, params_raw,
            apply_mm1=apply_mm1,   # FIX-1: uniform — no per-spec gating
            ext_demand=ext_spec,
        )
        A += A_spec
    return A


# ─── Composite multimodal accessibility ──────────────────────────────────────

def composite_accessibility(
    OD_matrices:     Dict[str, np.ndarray],
    population:      np.ndarray,
    supply:          np.ndarray,
    params:          E2SFCAParams,
    mode_weights:    Dict[str, float],
    apply_mm1:       bool = False,
    ext_demand:      Optional[np.ndarray] = None,
    specializations: Optional[np.ndarray] = None,
    normalize_modes: bool = False,
) -> np.ndarray:
    """
    Weighted sum of mode-specific E2SFCA.

    FIX-1: apply_mm1 forwarded uniformly (no mode or spec gating).

    FIX-3 (normalize_modes):
        When True, each mode array is normalized to mean=1 over positive
        values BEFORE weighting. Prevents car (100% reach) from dominating
        walk (3% reach) in the composite.

        composite = Σ_m  weight_m * (A_m / mean_pos(A_m))

        Use False (default) for scenario delta computation and research.
        Use True for visualization and policy maps.
    """
    params_raw = _dc_replace(params, normalise=False)
    mode_cfg = [
        ("car",  params.beta_car,  params.radius_car_s),
        ("walk", params.beta_walk, params.radius_walk_s),
        ("pt",   params.beta_pt,   params.radius_pt_s),
    ]
    A = np.zeros(len(population), dtype=np.float32)

    for mode, beta, radius in mode_cfg:
        if mode not in OD_matrices:
            continue
        OD = OD_matrices[mode]
        if params.nearest_k > 0:
            OD = filter_nearest_k(OD, params.nearest_k)

        if specializations is not None:
            A_mode = _e2sfca_by_spec(
                OD, population, supply, specializations,
                beta, radius, params_raw,
                apply_mm1=apply_mm1,    # FIX-1
                ext_demand=ext_demand,
            )
        else:
            A_mode = e2sfca(
                OD, population, supply, beta, radius, params_raw,
                apply_mm1=apply_mm1,
                ext_demand=ext_demand,
            )

        # FIX-3: per-mode normalization before weighting
        if normalize_modes:
            pos_m = A_mode[A_mode > 0]
            if len(pos_m) > 0:
                A_mode = A_mode / pos_m.mean()

        A += mode_weights.get(mode, 0.0) * A_mode

    if params.normalise:
        pos = A[A > 0]
        if len(pos) > 0:
            A = A / pos.mean()
    return A.astype(np.float32)


# ─── FIX-2: Telemedicine contribution (distance-independent) ─────────────────

def add_telemedicine_contribution(
    A_physical:    np.ndarray,
    tele_capacity: float,
    total_demand:  float,
    tele_weight:   float = 1.0,
    digital_divide_factor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FIX-2 CRITICAL: Correct telemedicine — distance-independent.

    Telemedicine must NOT go through OD matrices or decay functions.
    Every building has equal access to remote primary care regardless
    of physical distance to a clinic.

    Formula:
        A_tele_i = (tele_capacity / total_demand) * tele_weight * divide_i
        A_total_i = A_physical_i + A_tele_i

    Parameters
    ----------
    A_physical : (n,) float32
        Physical E2SFCA from composite_accessibility(), no telemedicine.
    tele_capacity : float
        Virtual capacity (doctor-equivalents). Typically:
            primary_cap * tele_capacity_fraction
    total_demand : float
        sum(demand) — population across all demand points.
    tele_weight : float
        Scaling weight. Default 1.0.
    digital_divide_factor : (n,) float32 or None
        Per-building factor in [0, 1]. 1.0 = full access, 0.5 = halved.
        None → uniform 1.0 (no digital divide penalty).

    Returns
    -------
    A_total : (n,) float32
    """
    if total_demand <= 0:
        raise ValueError(f"total_demand must be positive, got {total_demand}")
    if tele_capacity < 0:
        raise ValueError(f"tele_capacity must be >= 0, got {tele_capacity}")

    R_tele = float(tele_capacity) / float(total_demand) * float(tele_weight)

    if digital_divide_factor is not None:
        dd = np.asarray(digital_divide_factor, dtype=np.float32)
        if dd.shape != A_physical.shape:
            raise ValueError(
                f"digital_divide_factor shape {dd.shape} != A_physical {A_physical.shape}"
            )
        tele_arr = (R_tele * dd).astype(np.float32)
    else:
        tele_arr = np.full(len(A_physical), R_tele, dtype=np.float32)

    return (np.asarray(A_physical, dtype=np.float32) + tele_arr).astype(np.float32)


# ─── FIX-4: Scale output ─────────────────────────────────────────────────────

def scale_accessibility(A: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    FIX-4: Post-processing scale. Does NOT modify raw E2SFCA.

    Use this for visualization and committee reports ONLY.
    Always store and use raw values for scenario deltas and research.

    method = "mean":
        A_scaled = A / mean(A[A > 0])   → district mean ≈ 1.0

    method = "percentile_rank":
        A_scaled[i] = rank(A_i) / n_positive   → uniform [0, 1]
        Robust to outliers; ideal for choropleth maps.
    """
    A = np.asarray(A, dtype=np.float64)
    pos_mask = A > 0

    if method == "mean":
        pos = A[pos_mask]
        if len(pos) == 0:
            return A.astype(np.float32)
        return (A / pos.mean()).astype(np.float32)

    elif method == "percentile_rank":
        result = np.zeros_like(A, dtype=np.float32)
        pos_idx = np.where(pos_mask)[0]
        if len(pos_idx) == 0:
            return result
        ranks = np.argsort(np.argsort(A[pos_idx])) + 1
        result[pos_idx] = ranks.astype(np.float32) / len(pos_idx)
        return result

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'percentile_rank'.")


# ─── Per-mode and per-type breakdowns ────────────────────────────────────────

def accessibility_by_mode(
    OD_matrices:     Dict[str, np.ndarray],
    population:      np.ndarray,
    supply:          np.ndarray,
    params:          E2SFCAParams,
    apply_mm1:       bool = False,
    specializations: Optional[np.ndarray] = None,
    ext_demand:      Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Per-transport-mode E2SFCA. FIX-1: apply_mm1 uniform."""
    params_raw = _dc_replace(params, normalise=False)
    mode_cfg = [
        ("car",  params.beta_car,  params.radius_car_s),
        ("walk", params.beta_walk, params.radius_walk_s),
        ("pt",   params.beta_pt,   params.radius_pt_s),
    ]
    result: Dict[str, np.ndarray] = {}
    for mode, beta, radius in mode_cfg:
        if mode not in OD_matrices:
            continue
        OD = OD_matrices[mode]
        if params.nearest_k > 0:
            OD = filter_nearest_k(OD, params.nearest_k)
        if specializations is not None:
            A = _e2sfca_by_spec(OD, population, supply, specializations,
                                beta, radius, params_raw,
                                apply_mm1=apply_mm1, ext_demand=ext_demand)
        else:
            A = e2sfca(OD, population, supply, beta, radius, params_raw,
                       apply_mm1=apply_mm1, ext_demand=ext_demand)
        if params.normalise:
            pos = A[A > 0]
            if len(pos) > 0:
                A = A / pos.mean()
        result[mode] = A.astype(np.float32)
    return result


def accessibility_by_type(
    OD_matrices:     Dict[str, np.ndarray],
    population:      np.ndarray,
    supply:          np.ndarray,
    specializations: np.ndarray,
    params:          E2SFCAParams,
    mode_weights:    Dict[str, float],
    apply_mm1:       bool = False,
) -> Dict[str, np.ndarray]:
    """Per-facility-type composite E2SFCA. FIX-1: apply_mm1 uniform."""
    result: Dict[str, np.ndarray] = {}
    for spec in np.unique(specializations):
        mask = specializations == spec
        od_spec = {m: OD_matrices[m][:, mask] for m in OD_matrices}
        result[spec] = composite_accessibility(
            od_spec, population, supply[mask],
            params, mode_weights,
            apply_mm1=apply_mm1,   # FIX-1: uniform
        )
    return result

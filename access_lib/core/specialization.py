"""
core/specialization.py — Декомпозиция E2SFCA по специализациям.

Считает отдельный индекс доступности для каждой группы учреждений.
Результат: колонки access_primary, access_pediatric, access_specialized, access_hospital
в GeoDataFrame зданий и агрегация по поселениям.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None  # type: ignore


# Группировка специализаций в 4 ключевые категории ВКР
SPEC_GROUPS: Dict[str, List[str]] = {
    "primary":    ["primary_therapist", "primary_basic"],
    "pediatric":  ["pediatric"],
    "specialized":["outpatient_specialized", "maternal"],
    "hospital":   ["hospital_full"],
}


def compute_layer_accessibility(
    OD:              np.ndarray,          # (n_demand, n_fac)
    population:      np.ndarray,          # (n_demand,)
    supply:          np.ndarray,          # (n_fac,)
    specializations: np.ndarray,          # (n_fac,) строки
    params,                               # E2SFCAParams
    beta:            float,
    radius:          float,
) -> Dict[str, np.ndarray]:
    """
    Возвращает dict {group_name: accessibility_array} для каждой группы.

    Использует e2sfca_layer из engine — тот же алгоритм, только на подмножестве facilities.
    """
    from .engine import e2sfca_layer

    result: Dict[str, np.ndarray] = {}
    for group, specs in SPEC_GROUPS.items():
        # Маска по всем специализациям группы
        mask = np.isin(specializations, specs)
        if mask.sum() == 0:
            result[group] = np.zeros(len(population), dtype=np.float32)
            continue
        # Считаем по слою
        A = np.zeros(len(population), dtype=np.float32)
        for spec in specs:
            A_spec = e2sfca_layer(
                OD, population, supply, specializations,
                spec, beta, radius, params,
            )
            A += A_spec
        result[group] = A
    return result


def add_layer_columns(
    buildings_gdf,                        # GeoDataFrame зданий
    OD_matrices:     Dict[str, np.ndarray],
    population:      np.ndarray,
    supply:          np.ndarray,
    specializations: np.ndarray,
    params,
    mode_weights:    Dict[str, float],
) -> "gpd.GeoDataFrame":
    """
    Добавляет колонки access_primary, access_pediatric, access_specialized, access_hospital
    в GeoDataFrame зданий. Агрегирует по модам с теми же весами что composite.
    """
    gdf = buildings_gdf.copy()

    mode_cfg = [
        ("car",  params.beta_car,  params.radius_car_s),
        ("walk", params.beta_walk, params.radius_walk_s),
        ("pt",   params.beta_pt,   params.radius_pt_s),
    ]

    # Инициализируем нулями
    for group in SPEC_GROUPS:
        gdf[f"access_{group}"] = 0.0

    for mode, beta, radius in mode_cfg:
        if mode not in OD_matrices:
            continue
        w = mode_weights.get(mode, 0)
        if w == 0:
            continue
        layer_acc = compute_layer_accessibility(
            OD_matrices[mode], population, supply,
            specializations, params, beta, radius,
        )
        for group, arr in layer_acc.items():
            gdf[f"access_{group}"] = (
                gdf[f"access_{group}"].values + w * arr
            ).astype(np.float32)

    return gdf


def settlement_layer_summary(
    buildings_gdf,      # GeoDataFrame с access_* колонками и graph_node
    boundaries_gdf,     # GeoDataFrame поселений
    demand_col:  str = "demand",
    name_col:    str = "name",
) -> pd.DataFrame:
    """
    Агрегирует access_primary/pediatric/specialized/hospital по поселениям.
    Возвращает DataFrame: settlement_name × 4 индекса + population.
    """
    acc_cols = [f"access_{g}" for g in SPEC_GROUPS if f"access_{g}" in buildings_gdf.columns]
    if not acc_cols:
        raise ValueError("Нет колонок access_*. Сначала запусти add_layer_columns().")

    bldg_proj = buildings_gdf.to_crs(boundaries_gdf.crs)
    _has_dem  = demand_col in bldg_proj.columns

    joined = gpd.sjoin(
        bldg_proj[acc_cols + (["demand", "geometry"] if _has_dem else ["geometry"])],
        boundaries_gdf[[name_col, "geometry"]],
        how="left", predicate="within",
    )

    def _wagg(g):
        wts = g[demand_col].fillna(1).clip(lower=0.01).values if _has_dem else None
        row = {}
        for col in acc_cols:
            v = g[col].fillna(0).values
            row[col] = float(np.average(v, weights=wts) if wts is not None else v.mean())
        if _has_dem:
            row["population_demand"] = float(g[demand_col].sum())
        return pd.Series(row)

    summary = joined.groupby(name_col).apply(_wagg).reset_index()
    summary.rename(columns={name_col: "settlement_name"}, inplace=True)
    return summary

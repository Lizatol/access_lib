"""stats — Moran's I, LISA, Gini, Spearman, inequality analysis."""

from .spatial import morans_i, lisa, spearman_matrix, inequality_report  # noqa
from ..core.aggregation import gini  # re-export                          # noqa

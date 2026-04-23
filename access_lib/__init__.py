"""
Healthcare Spatial Accessibility Library — Vsevolozhsky District
================================================================
Multimodal Gaussian E2SFCA + 4 Policy Scenarios + Monte Carlo

Architecture (per audit):
  core/        — E2SFCA engine, facility processing, graph, OD, aggregation
  scenarios/   — 4 scenario classes (policy-only, no param mixing)
  simulation/  — Monte Carlo wrapper (wraps scenarios, not mixed in)
  stats/       — Moran I, LISA, Gini, Spearman
  viz/         — Settlement maps, delta maps, scenario panels
  tests/       — Correctness checks per scenario
"""
from .config import Config, DecayType  # noqa: F401

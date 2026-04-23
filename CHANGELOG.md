# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-04-24

### Added
- Multimodal E2SFCA engine with Gaussian decay (car, walking, public transit)
- Specialisation decomposition: primary care, paediatrics, specialist, inpatient
- M/M/1 queue correction for hospital capacity
- Four policy scenarios: telemedicine, new hospitals (×3), road closure, real facility
- Greedy sequential hospital placement algorithm targeting worst-served settlements
- Monte Carlo simulation for parameter uncertainty
- Sensitivity analysis for decay parameter (β) and catchment radius (r)
- Inequality metrics: Gini, Lorenz curves, Moran's I
- Publication-ready cartographic visualisations
- Interactive Folium map export
- Sample data for Vsevolozhsky District (19 settlements, 49 facilities)
- Complete walkthrough notebook

"""core — E2SFCA engine, facilities, graph, OD matrix, aggregation."""

from .engine      import e2sfca, composite_accessibility, e2sfca_layer  # noqa
from .facilities  import FacilityProcessor, make_virtual_facility        # noqa
from .aggregation import (                                                # noqa
    buildings_to_settlements,
    enrich_settlements,
    scenario_summary,
    gini,
)
from .specialization import (           # noqa
    compute_layer_accessibility,
    add_layer_columns,
    settlement_layer_summary,
    SPEC_GROUPS,
)
from .isochrones import (               # noqa
    build_isochrone,
    build_isochrones_for_facilities,
    classify_buildings_by_isochrone,
    unserved_population,
)

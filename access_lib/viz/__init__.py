"""viz — Визуализация пространственной доступности.

Модули:
  style          — Дизайн-система: цвета, размеры, setup_style()
  maps           — Картографические визуализации (provision_map, delta_map, ...)
  charts         — Статистические графики (lorenz, scenario_bars, ...)
  method_diagram — Методологические иллюстрации (schema, decay, mm1)
  interactive    — Интерактивные Folium-карты

Стандартный импорт в ноутбуке:
    from accessibility_lib.viz.style import setup_style
    from accessibility_lib.viz import maps, charts, method_diagram
    setup_style()
"""

from .style import setup_style                              # noqa
from . import maps, charts, method_diagram, interactive    # noqa

__all__ = ["setup_style", "maps", "charts", "method_diagram", "interactive"]

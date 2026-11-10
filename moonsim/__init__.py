"""moonsim: a deterministic, seedable crypto market sandbox.

The engine is pure Python + numpy and has no UI dependencies::

    from moonsim import SimConfig, run, presets
    result = run(presets.get("pump_and_dump").config)
    print(result.summary())
"""

from .config import SimConfig, from_dict, from_flat, override, to_dict, to_flat, validate
from .engine import FLOW_GROUPS, ConservationError, Event, SimResult, Simulation, run, run_many

__all__ = [
    "SimConfig",
    "Simulation",
    "SimResult",
    "Event",
    "ConservationError",
    "FLOW_GROUPS",
    "run",
    "run_many",
    "from_dict",
    "from_flat",
    "override",
    "to_dict",
    "to_flat",
    "validate",
]
__version__ = "2.0.0"

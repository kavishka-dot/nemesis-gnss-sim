"""
nemesis_sim — Geodetic-accuracy GPS L1 C/A signal simulator.

Public API
----------
NEMESISSimulator  : top-level simulation class
AttackConfig      : attack configuration dataclass
SVObs             : per-satellite observable dataclass
save_int16        : save IQ as interleaved int16
save_cf32         : save IQ as float32 complex
load_int16        : load int16 IQ file
load_cf32         : load float32 IQ file
"""

from .simulator import NEMESISSimulator
from .attacks import AttackConfig, apply_attack, build_attack
from .observations import SVObs, compute_observations
from .io import save_int16, save_cf32, load_int16, load_cf32
from .almanac import ALMANAC, SVEphemeris
from .rinex import load_rinex, select_closest, rinex_summary
from . import constants

__version__ = "0.1.0"
__author__  = "Kavishka Gihan"

__all__ = [
    "NEMESISSimulator",
    "AttackConfig",
    "SVObs",
    "compute_observations",
    "apply_attack",
    "build_attack",
    "save_int16",
    "save_cf32",
    "load_int16",
    "load_cf32",
    "ALMANAC",
    "SVEphemeris",
    "load_rinex",
    "select_closest",
    "rinex_summary",
    "constants",
]


def _welcome() -> None:
    """Print a short welcome line on import (Colab / notebook use)."""
    import sys
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        # Interactive session or Jupyter
        print(f"nemesis-sim {__version__} — GPS L1 C/A Simulator | "
              f"github.com/kavishka-dot/nemesis-gnss-sim")

_welcome()

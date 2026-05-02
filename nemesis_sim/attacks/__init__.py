"""
NEMESIS spoofing attack models.

Three attack classes matching the NEMESIS paper taxonomy:
  - MeaconingAttack  : uniform capture-and-rebroadcast delay
  - SlowDriftAttack  : per-SV linear pseudorange ramp
  - AdversarialAttack: full synthetic constellation at false position
"""

from __future__ import annotations

import numpy as np

from ..observations import SVObs
from .adversarial import AdversarialAttack
from .base import AttackConfig, BaseAttack
from .meaconing import MeaconingAttack
from .slow_drift import SlowDriftAttack

_REGISTRY: dict[str, type[BaseAttack]] = {
    "meaconing":  MeaconingAttack,
    "slow_drift": SlowDriftAttack,
    "adversarial": AdversarialAttack,
}


def build_attack(cfg: AttackConfig) -> BaseAttack | None:
    """
    Factory: return the appropriate attack instance for cfg.attack_type.
    Returns None for attack_type == 'none'.
    """
    if cfg.attack_type == "none":
        return None
    cls = _REGISTRY.get(cfg.attack_type)
    if cls is None:
        raise ValueError(
            f"Unknown attack type '{cfg.attack_type}'. "
            f"Valid options: {list(_REGISTRY)}"
        )
    return cls(cfg)


def apply_attack(
    obs: list[SVObs],
    cfg: AttackConfig,
    gps_tow: float,
    user_ecef: np.ndarray,
    user_vel: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    doy: float,
) -> list[SVObs]:
    """
    Convenience function: apply attack described by cfg to obs.
    Returns obs unchanged if cfg.attack_type == 'none'.
    """
    attack = build_attack(cfg)
    if attack is None:
        return obs
    return attack.apply(obs, gps_tow, user_ecef, user_vel, lat_deg, lon_deg, alt_m, doy)


__all__ = [
    "AttackConfig",
    "BaseAttack",
    "MeaconingAttack",
    "SlowDriftAttack",
    "AdversarialAttack",
    "build_attack",
    "apply_attack",
]

"""
Attack model base class and configuration dataclass.

All attack classes subclass BaseAttack and implement apply().
This makes adding a new attack class a single-file addition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..observations import SVObs


@dataclass
class AttackConfig:
    """
    Unified configuration for all NEMESIS attack classes.

    Parameters
    ----------
    attack_type : one of 'none', 'meaconing', 'slow_drift', 'adversarial'

    Meaconing
    ---------
    meaconing_delay_s  : capture-and-rebroadcast delay (seconds)
    meaconing_power_db : spoofer power advantage over truth (dB), default 3

    Slow Drift
    ----------
    drift_rate_m_s : uniform per-SV pseudorange drift rate (m/s)
    drift_per_sv   : optional dict {prn: rate_m_s} for per-SV override
    drift_start_s  : GPS TOW at which drift begins (seconds)

    Adversarial
    -----------
    false_lat, false_lon, false_alt : target false geodetic position
    spoof_power_db : spoofer power advantage over authentic signal (dB)
    """

    attack_type: str = "none"

    # Meaconing
    meaconing_delay_s:  float = 0.0
    meaconing_power_db: float = 3.0

    # Slow drift
    drift_rate_m_s:  float = 1.0
    drift_per_sv:    Optional[dict[int, float]] = None
    drift_start_s:   float = 0.0

    # Adversarial
    false_lat:       float = 0.0
    false_lon:       float = 0.0
    false_alt:       float = 0.0
    spoof_power_db:  float = 6.0


class BaseAttack(ABC):
    """Abstract base class for NEMESIS spoofing attack models."""

    def __init__(self, cfg: AttackConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def apply(
        self,
        obs: list[SVObs],
        gps_tow: float,
        user_ecef: np.ndarray,
        user_vel: np.ndarray,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        doy: float,
    ) -> list[SVObs]:
        """
        Apply attack to a list of true observations.

        Parameters
        ----------
        obs      : clearsky observations from compute_observations()
        gps_tow  : current GPS time of week
        user_ecef, user_vel : receiver state
        lat_deg, lon_deg, alt_m : receiver geodetic position
        doy      : day of year

        Returns
        -------
        Attacked list of SVObs (pseudoranges / Doppler modified).
        The returned list is always a new list of new SVObs instances;
        the input is never mutated.
        """
        ...

    @staticmethod
    def _copy_sv(sv: SVObs, **overrides) -> SVObs:
        """Return a shallow copy of sv with optional field overrides."""
        d = sv.__dict__.copy()
        d.update(overrides)
        return SVObs(**d)

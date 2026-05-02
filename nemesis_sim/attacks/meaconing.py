"""
Meaconing attack model.

A meaconer captures authentic GPS signals and rebroadcasts them with
a uniform time delay Δτ. The victim receiver tracks the delayed signals
and computes a position biased toward the direction of the meaconing
transmitter.

Key signatures vs clearsky:
  - All pseudoranges uniformly offset by c·Δτ
  - Doppler shifts UNCHANGED (replay at original rate)
  - The Doppler-pseudorange inconsistency is the primary detection feature
    exploited by NEMESIS.

Reference: NEMESIS paper Section III-A.
"""

from __future__ import annotations

import numpy as np

from ..constants import C
from ..observations import SVObs
from .base import BaseAttack


class MeaconingAttack(BaseAttack):
    """Uniform capture-and-rebroadcast delay across all SVs."""

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
        Apply meaconing delay to all pseudoranges.

        Doppler is left unchanged — this is the key observable inconsistency.
        A receiver's tracking loop sees Doppler consistent with the true
        satellite position but pseudoranges consistent with a delayed signal.
        """
        delta_rho = C * self.cfg.meaconing_delay_s
        return [
            self._copy_sv(sv, pseudorange_m=sv.pseudorange_m + delta_rho)
            for sv in obs
        ]

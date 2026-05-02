"""
Slow drift attack model.

Each satellite's pseudorange is corrupted with a time-linear bias:
  b_i(t) = ṙ_i · (t − t_start)   for t ≥ t_start

The corresponding Doppler is also shifted by Δf_i = −ṙ_i · f_L1 / c
so that the signal remains internally self-consistent (a naive receiver
would accept it). The Doppler-to-pseudorange-rate ratio reveals the
attack to NEMESIS.

Reference: NEMESIS paper Section III-B.
"""

from __future__ import annotations

import numpy as np

from ..constants import L1_FREQ, C
from ..observations import SVObs
from .base import BaseAttack


class SlowDriftAttack(BaseAttack):
    """Per-SV linear pseudorange ramp with consistent Doppler perturbation."""

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
        Apply slow drift to pseudoranges and Doppler.

        The drift has not started yet if gps_tow < drift_start_s.
        """
        dt = max(0.0, gps_tow - self.cfg.drift_start_s)
        attacked: list[SVObs] = []

        for sv in obs:
            rate = (
                self.cfg.drift_per_sv.get(sv.prn, self.cfg.drift_rate_m_s)
                if self.cfg.drift_per_sv
                else self.cfg.drift_rate_m_s
            )
            bias_m = rate * dt
            # Doppler shift consistent with pseudorange drift rate
            delta_dop = -rate * L1_FREQ / C

            attacked.append(
                self._copy_sv(
                    sv,
                    pseudorange_m=sv.pseudorange_m + bias_m,
                    doppler_hz=sv.doppler_hz + delta_dop,
                )
            )

        return attacked

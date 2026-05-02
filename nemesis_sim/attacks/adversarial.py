"""
Adversarial (synthesised constellation) spoofing attack.

The spoofer generates a complete fake GPS constellation whose pseudoranges
and Dopplers are consistent with the victim being at a false geodetic position
(false_lat, false_lon, false_alt).

The victim receiver, if deceived, will compute the false position.

Key signatures vs clearsky:
  - Per-SV pseudorange deltas that are geometrically consistent with a
    different position (not a simple uniform offset)
  - Doppler deltas consistent with LOS vectors from a different location

Reference: NEMESIS paper Section III-C.
"""

from __future__ import annotations

import numpy as np

from ..observations import SVObs, compute_observations
from ..propagator.transforms import lla_to_ecef
from .base import BaseAttack


class AdversarialAttack(BaseAttack):
    """
    Full synthetic constellation at a false geodetic position.

    For each SV visible to the true receiver, replaces the true pseudorange
    and Doppler with values synthesised for the false position.
    SVs that are not visible from the false position retain their true values
    (the spoofer cannot fake what it cannot compute).
    """

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
        Replace pseudoranges/Dopplers with values for the false position.
        """
        # Compute observations as seen from the false position
        false_ecef = lla_to_ecef(self.cfg.false_lat, self.cfg.false_lon, self.cfg.false_alt)
        spoofed = compute_observations(
            user_ecef=false_ecef,
            user_vel=np.zeros(3),   # spoofer synthesises for a stationary false target
            lat_deg=self.cfg.false_lat,
            lon_deg=self.cfg.false_lon,
            alt_m=self.cfg.false_alt,
            gps_tow=gps_tow,
            doy=doy,
            el_mask_deg=0.0,       # include all SVs even below horizon
        )
        spoof_map: dict[int, SVObs] = {s.prn: s for s in spoofed}

        attacked: list[SVObs] = []
        for sv in obs:
            if sv.prn in spoof_map:
                s = spoof_map[sv.prn]
                attacked.append(
                    self._copy_sv(
                        sv,
                        pseudorange_m=s.pseudorange_m,
                        doppler_hz=s.doppler_hz,
                    )
                )
            else:
                attacked.append(sv)

        return attacked

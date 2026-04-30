"""
GPS satellite clock correction model.

Implements IS-GPS-200 §20.3.3.3.3:
  dT_sv = af0 + af1·(t − toc) + af2·(t − toc)² + ΔTrel − TGD

where ΔTrel = F·e·√A·sin(Ek) is the relativistic periodic correction.
"""

from __future__ import annotations

import numpy as np

from ..constants import MU, F_REL, HALF_WEEK
from ..almanac import SVEphemeris


def sv_clock_correction(eph: SVEphemeris, gps_tow: float) -> float:
    """
    Satellite clock correction at GPS time-of-week.

    Returns the total clock error dT_sv in seconds.
    A positive value means the satellite clock is ahead of GPS time,
    which shortens the apparent pseudorange.

    The L1-only user should subtract TGD (already included here).

    Parameters
    ----------
    eph     : SVEphemeris broadcast ephemeris
    gps_tow : GPS time of week (seconds)

    Returns
    -------
    dT_sv : clock correction (seconds)
    """
    tc = gps_tow - eph.toc
    if tc >  HALF_WEEK: tc -= 2 * HALF_WEEK
    if tc < -HALF_WEEK: tc += 2 * HALF_WEEK

    # Approximate eccentric anomaly for relativistic term
    A  = eph.sqrtA ** 2
    n0 = np.sqrt(MU / A ** 3)
    Ek = _approx_eccentric_anomaly(eph.M0 + n0 * tc, eph.e)

    # Relativistic periodic correction (IS-GPS-200 Eq. 4)
    delta_rel = F_REL * eph.e * eph.sqrtA * np.sin(Ek)

    dtsv = eph.af0 + eph.af1 * tc + eph.af2 * tc ** 2 + delta_rel

    # Remove L1 group delay for L1-only users
    dtsv -= eph.TGD

    return float(dtsv)


def _approx_eccentric_anomaly(M: float, e: float, n_iter: int = 8) -> float:
    """Quick Kepler solver for clock correction (lower precision ok here)."""
    E = M
    for _ in range(n_iter):
        E = M + e * np.sin(E)
    return E

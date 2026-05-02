"""
IS-GPS-200 §20.3.3.4.3 Keplerian orbit propagator.

Computes satellite ECEF position and velocity from broadcast ephemeris,
including all 6 second-harmonic corrections and the corrected RAAN rate.
"""

from __future__ import annotations

import numpy as np

from ..almanac import SVEphemeris
from ..constants import HALF_WEEK, MU, OMEGA_E

# Maximum Kepler iteration tolerance
_KEPLER_TOL: float = 1e-14
_KEPLER_MAX_ITER: int = 12


def _normalize_time(t: float, toe: float) -> float:
    """Compute time-from-epoch with GPS week half-week rollover."""
    tk = t - toe
    if tk >  HALF_WEEK: tk -= 2 * HALF_WEEK
    if tk < -HALF_WEEK: tk += 2 * HALF_WEEK
    return tk


def eccentric_anomaly(M: float, e: float) -> float:
    """
    Solve Kepler's equation M = E − e·sin(E) by Newton iteration.

    Parameters
    ----------
    M : mean anomaly (rad)
    e : eccentricity

    Returns
    -------
    E : eccentric anomaly (rad)
    """
    E = M
    for _ in range(_KEPLER_MAX_ITER):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < _KEPLER_TOL:
            break
    return E


def sv_position_velocity(
    eph: SVEphemeris, gps_tow: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute satellite ECEF position and velocity at GPS time-of-week.

    Implements IS-GPS-200 §20.3.3.4.3 in full, including:
      - Corrected mean motion
      - Kepler's equation (Newton iteration to 1e-14 rad)
      - True anomaly from eccentric anomaly
      - All 6 harmonic corrections (Cuc, Cus, Crc, Crs, Cic, Cis)
      - Corrected RAAN with Earth rotation rate subtracted

    Parameters
    ----------
    eph     : SVEphemeris broadcast ephemeris
    gps_tow : GPS time of week at signal reception (seconds)

    Returns
    -------
    pos : ECEF position  (3,) metres
    vel : ECEF velocity  (3,) metres/second
    """
    A = eph.sqrtA ** 2
    n0 = np.sqrt(MU / A ** 3)            # Computed mean motion (rad/s)

    tk = _normalize_time(gps_tow, eph.toe)

    n  = n0 + eph.deltaN                  # Corrected mean motion
    Mk = eph.M0 + n * tk                  # Mean anomaly

    # ── Kepler's equation ────────────────────────────────────
    Ek = eccentric_anomaly(Mk, eph.e)

    # ── True anomaly ─────────────────────────────────────────
    sin_vk = np.sqrt(1.0 - eph.e ** 2) * np.sin(Ek) / (1.0 - eph.e * np.cos(Ek))
    cos_vk = (np.cos(Ek) - eph.e)                    / (1.0 - eph.e * np.cos(Ek))
    vk = np.arctan2(sin_vk, cos_vk)

    # ── Argument of latitude + harmonic corrections ──────────
    phi_k  = vk + eph.omega
    sin2ph = np.sin(2.0 * phi_k)
    cos2ph = np.cos(2.0 * phi_k)

    delta_u = eph.Cus * sin2ph + eph.Cuc * cos2ph   # arg-of-lat correction (rad)
    delta_r = eph.Crs * sin2ph + eph.Crc * cos2ph   # radius correction (m)
    delta_i = eph.Cis * sin2ph + eph.Cic * cos2ph   # inclination correction (rad)

    uk = phi_k + delta_u
    rk = A * (1.0 - eph.e * np.cos(Ek)) + delta_r
    ik = eph.i0 + eph.iDot * tk + delta_i

    # ── In-plane position ────────────────────────────────────
    xk_prime = rk * np.cos(uk)
    yk_prime = rk * np.sin(uk)

    # ── Corrected longitude of ascending node ────────────────
    Omega_k = (
        eph.Omega0
        + (eph.OmegaDot - OMEGA_E) * tk
        - OMEGA_E * eph.toe
    )

    cos_Ok = np.cos(Omega_k)
    sin_Ok = np.sin(Omega_k)
    cos_ik = np.cos(ik)
    sin_ik = np.sin(ik)

    # ── ECEF position ────────────────────────────────────────
    x = xk_prime * cos_Ok - yk_prime * cos_ik * sin_Ok
    y = xk_prime * sin_Ok + yk_prime * cos_ik * cos_Ok
    z = yk_prime * sin_ik

    # ── ECEF velocity ────────────────────────────────────────
    Edot   = n / (1.0 - eph.e * np.cos(Ek))
    vdot   = Edot * np.sqrt(1.0 - eph.e ** 2) / (1.0 - eph.e * np.cos(Ek))

    rdot   = A * eph.e * np.sin(Ek) * Edot + 2.0 * vdot * (eph.Crs * cos2ph - eph.Crc * sin2ph)
    udot   = vdot * (1.0 + 2.0 * (eph.Cus * cos2ph - eph.Cuc * sin2ph))
    idot_  = eph.iDot + 2.0 * vdot * (eph.Cis * cos2ph - eph.Cic * sin2ph)
    Omega_dot = eph.OmegaDot - OMEGA_E

    xp_dot = rdot * np.cos(uk) - rk * udot * np.sin(uk)
    yp_dot = rdot * np.sin(uk) + rk * udot * np.cos(uk)

    vx = (
        xp_dot * cos_Ok
        - yp_dot * cos_ik * sin_Ok
        + yk_prime * sin_ik * sin_Ok * idot_
        - (xk_prime * sin_Ok + yk_prime * cos_ik * cos_Ok) * Omega_dot
    )
    vy = (
        xp_dot * sin_Ok
        + yp_dot * cos_ik * cos_Ok
        - yk_prime * sin_ik * cos_Ok * idot_
        + (xk_prime * cos_Ok - yk_prime * cos_ik * sin_Ok) * Omega_dot
    )
    vz = yp_dot * sin_ik + yk_prime * cos_ik * idot_

    return np.array([x, y, z]), np.array([vx, vy, vz])

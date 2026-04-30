"""
Klobuchar single-frequency ionospheric delay model.

Implements IS-GPS-200 §20.3.3.5.2.5 exactly.
Broadcast α/β coefficients embedded from GPS week 2238.

Reference: Klobuchar, J.A. (1987). Ionospheric Time-Delay Algorithm for
Single-Frequency GPS Users. IEEE Transactions on Aerospace and Electronic
Systems, AES-23(3), 325–331.
"""

from __future__ import annotations

import numpy as np

from ..constants import PI, C

# GPS week 2238 broadcast ionospheric correction coefficients
# These are representative real values from the GPS control segment
ALPHA: np.ndarray = np.array([7.4506e-9,  1.4901e-8, -5.9605e-8, -1.1921e-7])  # seconds
BETA: np.ndarray  = np.array([9.0112e4,   1.6384e5,  -6.5536e4,  -5.2429e5])   # seconds


def ionospheric_delay_s(
    lat_deg: float,
    lon_deg: float,
    el_deg: float,
    az_deg: float,
    gps_tow: float,
    alpha: np.ndarray = ALPHA,
    beta: np.ndarray  = BETA,
) -> float:
    """
    Klobuchar ionospheric group delay on L1 (seconds).

    Parameters
    ----------
    lat_deg : receiver geodetic latitude (degrees)
    lon_deg : receiver longitude (degrees)
    el_deg  : satellite elevation angle (degrees)
    az_deg  : satellite azimuth angle (degrees)
    gps_tow : GPS time of week (seconds)
    alpha   : ionospheric α coefficients (4,)  default GPS week 2238
    beta    : ionospheric β coefficients (4,)  default GPS week 2238

    Returns
    -------
    T_iono : L1 group delay (seconds, always ≥ 5 ns)
    """
    # Convert to semi-circles (IS-GPS-200 convention)
    el    = el_deg  / 180.0
    phi_u = lat_deg / 180.0
    lam_u = lon_deg / 180.0
    az    = np.radians(az_deg)

    # Earth-centred angle to sub-ionospheric point (semi-circles)
    psi = 0.0137 / (el + 0.11) - 0.022

    # Sub-ionospheric point geodetic latitude (semi-circles)
    phi_i = phi_u + psi * np.cos(az)
    phi_i = float(np.clip(phi_i, -0.416, 0.416))

    # Sub-ionospheric point longitude (semi-circles)
    lam_i = lam_u + psi * np.sin(az) / np.cos(phi_i * PI)

    # Geomagnetic latitude of sub-ionospheric point (semi-circles)
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * PI)

    # Local time at sub-ionospheric point (seconds)
    t = (4.32e4 * lam_i + gps_tow) % 86400.0

    # Obliquity factor
    F = 1.0 + 16.0 * (0.53 - el) ** 3

    # Amplitude of ionospheric delay
    AMP = float(np.polyval(alpha[::-1], phi_m))
    AMP = max(AMP, 0.0)

    # Period of ionospheric delay
    PER = float(np.polyval(beta[::-1], phi_m))
    PER = max(PER, 72000.0)

    # Phase argument
    X = 2.0 * PI * (t - 50400.0) / PER

    if abs(X) < 1.57:
        T_iono = F * (5.0e-9 + AMP * (1.0 - X**2 / 2.0 + X**4 / 24.0))
    else:
        T_iono = F * 5.0e-9

    return float(T_iono)


def ionospheric_delay_m(
    lat_deg: float,
    lon_deg: float,
    el_deg: float,
    az_deg: float,
    gps_tow: float,
) -> float:
    """Klobuchar L1 delay in metres (convenience wrapper)."""
    return C * ionospheric_delay_s(lat_deg, lon_deg, el_deg, az_deg, gps_tow)

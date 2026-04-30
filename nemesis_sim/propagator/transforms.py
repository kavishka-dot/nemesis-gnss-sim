"""
Coordinate transforms: ECEF ↔ LLA, ECEF → ENU, Sagnac rotation.

All angles in degrees at the public API; radians used internally.
"""

from __future__ import annotations

import numpy as np

from ..constants import WGS84_A, WGS84_E2, WGS84_B, OMEGA_E, PI


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """
    Geodetic (WGS-84) → ECEF XYZ.

    Parameters
    ----------
    lat_deg : geodetic latitude (degrees)
    lon_deg : longitude (degrees)
    alt_m   : height above ellipsoid (metres)

    Returns
    -------
    np.ndarray shape (3,)  [x, y, z] in metres
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array(
        [
            (N + alt_m) * np.cos(lat) * np.cos(lon),
            (N + alt_m) * np.cos(lat) * np.sin(lon),
            (N * (1.0 - WGS84_E2) + alt_m) * np.sin(lat),
        ]
    )


def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    ECEF XYZ → geodetic (WGS-84) using Bowring's iterative method.

    Returns
    -------
    (lat_deg, lon_deg, alt_m)
    """
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    if abs(lat) < np.radians(80.0):
        alt = p / np.cos(lat) - N
    else:
        alt = z / np.sin(lat) - N * (1.0 - WGS84_E2)
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def ecef_to_enu(delta_ecef: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Rotate an ECEF difference vector to local East-North-Up.

    Parameters
    ----------
    delta_ecef : ECEF difference vector (3,)
    lat_deg    : observer geodetic latitude (degrees)
    lon_deg    : observer longitude (degrees)

    Returns
    -------
    np.ndarray (3,)  [East, North, Up]
    """
    la = np.radians(lat_deg)
    lo = np.radians(lon_deg)
    R = np.array(
        [
            [-np.sin(lo),               np.cos(lo),              0.0         ],
            [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la) ],
            [ np.cos(la) * np.cos(lo),  np.cos(la) * np.sin(lo), np.sin(la) ],
        ]
    )
    return R @ delta_ecef


def elevation_azimuth(
    user_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    lat_deg: float,
    lon_deg: float,
) -> tuple[float, float]:
    """
    Elevation and azimuth from user to satellite.

    Returns
    -------
    (elevation_deg, azimuth_deg)  elevation ∈ (−90, 90), azimuth ∈ [0, 360)
    """
    diff = sat_ecef - user_ecef
    enu = ecef_to_enu(diff, lat_deg, lon_deg)
    e, n, u = enu
    h = np.sqrt(e**2 + n**2)
    el = float(np.degrees(np.arctan2(u, h)))
    az = float(np.degrees(np.arctan2(e, n)) % 360.0)
    return el, az


def rotate_ecef_sagnac(pos: np.ndarray, transit_time_s: float) -> np.ndarray:
    """
    Rotate an ECEF position vector by Earth's rotation during signal transit.

    This corrects for the Sagnac effect: during the ~67 ms transit time
    the Earth rotates, shifting the receiver frame relative to the
    inertial frame in which the satellite position is computed.

    Parameters
    ----------
    pos            : ECEF position vector at transmit time (3,)
    transit_time_s : signal travel time (seconds)

    Returns
    -------
    Rotated ECEF position (3,)
    """
    theta = OMEGA_E * transit_time_s
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [ cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [  0.0,   0.0,  1.0],
        ]
    )
    return R @ pos


def sagnac_correction(sat_ecef: np.ndarray, user_ecef: np.ndarray) -> float:
    """
    Scalar Sagnac range correction (metres).

    ΔSagnac = (Ω_E / c) · (x_sv · y_u − y_sv · x_u)

    This is the closed-form equivalent of the ECEF rotation approach.
    Both methods agree to < 1 mm.
    """
    from ..constants import OMEGA_E, C
    return (OMEGA_E / C) * (sat_ecef[0] * user_ecef[1] - sat_ecef[1] * user_ecef[0])

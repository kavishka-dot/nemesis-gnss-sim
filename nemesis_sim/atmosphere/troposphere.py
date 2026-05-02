"""
Tropospheric delay model: Neill Mapping Function + GPT standard atmosphere.

Implements:
  - GPT (Global Pressure and Temperature) standard atmosphere
    Latitude- and season-dependent P, T, e without external data files.
  - Saastamoinen Zenith Hydrostatic Delay (ZHD)
  - Askne-Nordius Zenith Wet Delay (ZWD)
  - Neill Mapping Function (NMF) for hydrostatic and wet components

Reference: Neill, A.E. (1996). Global mapping functions for the atmosphere
delay at radio wavelengths. Journal of Geophysical Research, 101(B2), 3227–3246.
"""

from __future__ import annotations

import numpy as np

from ..constants import PI, C

# GPT standard atmosphere at reference latitudes (5° grid)
_GPT_LAT = np.radians([15.0, 30.0, 45.0, 60.0, 75.0])
_GPT_P0  = np.array([1013.25, 1017.25, 1015.75, 1011.75, 1013.00])  # hPa
_GPT_T0  = np.array([ 299.65,  294.15,  283.15,  272.15,  263.65])  # K
_GPT_E0  = np.array([  26.31,   21.79,   11.66,    6.78,    4.11])  # hPa

# Seasonal amplitude coefficients
_GPT_AMP_T = 15.0   # K
_GPT_AMP_E =  9.0   # hPa


def gpt_met(lat_deg: float, alt_m: float, doy: float) -> tuple[float, float, float]:
    """
    GPT standard atmosphere at receiver location.

    Parameters
    ----------
    lat_deg : geodetic latitude (degrees)
    alt_m   : height above ellipsoid (metres)
    doy     : day of year (1–365)

    Returns
    -------
    (P_hPa, T_K, e_hPa) : pressure, temperature, partial water vapour pressure
    """
    lat_r = abs(np.radians(lat_deg))
    lat_r = float(np.clip(lat_r, _GPT_LAT[0], _GPT_LAT[-1]))

    P0 = float(np.interp(lat_r, _GPT_LAT, _GPT_P0))
    T0 = float(np.interp(lat_r, _GPT_LAT, _GPT_T0))
    e0 = float(np.interp(lat_r, _GPT_LAT, _GPT_E0))

    # Annual cycle (Northern Hemisphere minimum at DOY 28, Southern at DOY 211)
    doy_min = 28.0 if lat_deg >= 0.0 else 211.0
    cos_term = np.cos(2.0 * PI * (doy - doy_min) / 365.25)

    T0 -= _GPT_AMP_T * cos_term
    e0 -= _GPT_AMP_E * cos_term

    # Height correction (standard lapse rate 6.5 K/km)
    lapse = 6.5e-3  # K/m
    T = T0 - lapse * alt_m
    T = max(T, 50.0)  # guard against extreme altitudes

    # Pressure: hypsometric equation
    P = P0 * (T / T0) ** (9.80665 / (287.054 * lapse))

    # Water vapour: approximate scaling
    e = e0 * (T / T0) ** (9.80665 * 0.6077 / (287.054 * lapse))

    return float(P), float(T), float(e)


def _nmf(el_rad: float, a: float, b: float, c: float) -> float:
    """Neill mapping function continued-fraction form."""
    sin_e = np.sin(el_rad)
    num = 1.0 + a / (1.0 + b / (1.0 + c))
    den = sin_e + a / (sin_e + b / (sin_e + c))
    return num / den


def tropospheric_delay_s(
    lat_deg: float,
    alt_m: float,
    el_deg: float,
    doy: float = 180.0,
) -> float:
    """
    Total tropospheric slant delay (seconds) using NMF + GPT.

    Parameters
    ----------
    lat_deg : receiver geodetic latitude (degrees)
    alt_m   : receiver height above ellipsoid (metres)
    el_deg  : satellite elevation angle (degrees)
    doy     : day of year (default 180 = boreal summer)

    Returns
    -------
    delay_s : tropospheric delay (seconds)
    """
    el_rad = np.radians(max(el_deg, 3.0))   # clip at 3° to avoid singularity
    P, T, e = gpt_met(lat_deg, alt_m, doy)

    # ── Zenith delays ────────────────────────────────────────
    # Saastamoinen ZHD (metres)
    ZHD = 2.2768e-3 * P / (
        1.0 - 2.66e-3 * np.cos(2.0 * np.radians(lat_deg)) - 2.8e-7 * alt_m
    )

    # Askne-Nordius ZWD approximation (metres)
    ZWD = 0.061 * e / T

    # ── Neill mapping function coefficients ──────────────────
    la = abs(lat_deg)

    # Hydrostatic
    a_h = 1.2769934e-3 + 2.787169e-7 * la - 1.62557e-8 * la**2 - 2.2795e-10 * la**3
    b_h = 2.99490e-3
    c_h = 0.062610e0

    # Wet
    a_w = 5.8021897e-4 + 2.347453e-6 * la - 1.5818e-7 * la**2
    b_w = 1.4275268e-3
    c_w = 4.3472961e-2

    MFH = _nmf(el_rad, a_h, b_h, c_h)
    MFW = _nmf(el_rad, a_w, b_w, c_w)

    delay_m = ZHD * MFH + ZWD * MFW
    return delay_m / C


def tropospheric_delay_m(
    lat_deg: float, alt_m: float, el_deg: float, doy: float = 180.0
) -> float:
    """Tropospheric slant delay in metres (convenience wrapper)."""
    return C * tropospheric_delay_s(lat_deg, alt_m, el_deg, doy)

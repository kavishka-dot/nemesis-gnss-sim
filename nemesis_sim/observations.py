"""
Full IS-GPS-200 pseudorange measurement model.

Computes all observables (pseudorange, Doppler, iono, tropo, clock, Sagnac)
for all visible satellites at a given receiver position and time.

Pseudorange model:
  ρ = r_geo + Δ_sagnac − c·dT_sv + Δ_iono + Δ_tropo
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C, L1_FREQ, OMEGA_E
from .almanac import SVEphemeris, ALMANAC
from .propagator.kepler import sv_position_velocity
from .propagator.clock import sv_clock_correction
from .propagator.transforms import elevation_azimuth, rotate_ecef_sagnac, sagnac_correction
from .atmosphere.klobuchar import ionospheric_delay_m
from .atmosphere.troposphere import tropospheric_delay_m


@dataclass
class SVObs:
    """All observables for one satellite at one measurement epoch."""

    prn: int

    # Geometry
    el_deg: float          # Elevation angle (degrees)
    az_deg: float          # Azimuth angle (degrees, 0–360)
    pos_ecef: np.ndarray   # SV ECEF position after Sagnac correction (m)
    vel_ecef: np.ndarray   # SV ECEF velocity (m/s)

    # Range observables
    range_true_m: float    # Geometric range (m)
    pseudorange_m: float   # Full corrected pseudorange (m)
    doppler_hz: float      # L1 Doppler shift (Hz)

    # Error budget terms (all in metres, positive = extra delay)
    clock_err_m: float     # SV clock correction (m)  — sign: positive shortens ρ
    iono_m: float          # Ionospheric delay (m)
    tropo_m: float         # Tropospheric delay (m)
    sagnac_m: float        # Sagnac correction (m)
    rel_m: float           # Relativistic term (m, absorbed in clock_err_m)


def compute_observations(
    user_ecef: np.ndarray,
    user_vel: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    gps_tow: float,
    doy: float = 180.0,
    el_mask_deg: float = 5.0,
    almanac: list[SVEphemeris] | None = None,
) -> list[SVObs]:
    """
    Compute full pseudoranges and ancillary observables for all visible SVs.

    Uses iterative signal-transit-time correction (3 iterations) to account
    for the satellite's motion during the ~67 ms signal travel time.

    Parameters
    ----------
    user_ecef   : receiver ECEF position (3,) metres
    user_vel    : receiver ECEF velocity (3,) m/s
    lat_deg     : receiver geodetic latitude (degrees)
    lon_deg     : receiver longitude (degrees)
    alt_m       : receiver height above ellipsoid (metres)
    gps_tow     : GPS time of week at reception (seconds)
    doy         : day of year for troposphere seasonality (1–365)
    el_mask_deg : elevation mask angle (degrees)
    almanac     : list of SVEphemeris; defaults to embedded ALMANAC

    Returns
    -------
    List of SVObs sorted by descending elevation, only visible SVs included.
    """
    if almanac is None:
        almanac = ALMANAC

    results: list[SVObs] = []

    for eph in almanac:
        if eph.health != 0:
            continue

        # ── Iterative transit-time correction ────────────────
        geo = 2.0e7   # initial guess: ~20 000 km
        pos_sagnac = np.zeros(3)
        vel_tx = np.zeros(3)

        for _ in range(3):
            t_tx = gps_tow - geo / C
            pos_tx, vel_tx = sv_position_velocity(eph, t_tx)
            # Apply Sagnac (Earth rotation during transit)
            pos_sagnac = rotate_ecef_sagnac(pos_tx, geo / C)
            geo = float(np.linalg.norm(pos_sagnac - user_ecef))

        # ── Geometry ─────────────────────────────────────────
        el, az = elevation_azimuth(user_ecef, pos_sagnac, lat_deg, lon_deg)
        if el < el_mask_deg:
            continue

        # ── Sagnac correction ─────────────────────────────────
        delta_sagnac = sagnac_correction(pos_sagnac, user_ecef)

        # ── SV clock ─────────────────────────────────────────
        dtsv = sv_clock_correction(eph, gps_tow - geo / C)
        clock_m = -C * dtsv   # positive dtsv → SV ahead → shorter ρ

        # ── Ionosphere ───────────────────────────────────────
        iono_m = ionospheric_delay_m(lat_deg, lon_deg, el, az, gps_tow)

        # ── Troposphere ──────────────────────────────────────
        tropo_m = tropospheric_delay_m(lat_deg, alt_m, el, doy=doy)

        # ── Full pseudorange ─────────────────────────────────
        # ρ = r_geo + Δ_sagnac + clock_correction + Δ_iono + Δ_tropo
        pseudo = geo + delta_sagnac + clock_m + iono_m + tropo_m

        # ── Doppler ──────────────────────────────────────────
        los = (pos_sagnac - user_ecef) / geo
        rel_vel = vel_tx - user_vel
        rdot = float(np.dot(rel_vel, los))
        doppler = -rdot * L1_FREQ / C

        results.append(
            SVObs(
                prn=eph.prn,
                el_deg=el,
                az_deg=az,
                pos_ecef=pos_sagnac,
                vel_ecef=vel_tx,
                range_true_m=geo,
                pseudorange_m=pseudo,
                doppler_hz=doppler,
                clock_err_m=clock_m,
                iono_m=iono_m,
                tropo_m=tropo_m,
                sagnac_m=delta_sagnac,
                rel_m=0.0,
            )
        )

    results.sort(key=lambda s: -s.el_deg)
    return results

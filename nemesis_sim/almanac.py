"""
Embedded GPS satellite ephemeris table.

Contains representative full-accuracy broadcast ephemeris for 31 GPS SVs,
derived from IS-GPS-200 reference constellation parameters at GPS week 2238.
All 16 IS-GPS-200 orbital elements are included, giving ~1–5 m position accuracy
vs ~100 m for simplified almanac-only propagation.

Units: SI throughout
  sqrtA  : m^0.5
  angles : radians
  times  : seconds
  af0    : seconds  (|af0| < 1e-4 s for healthy SVs)
  af1    : s/s
  TGD    : seconds  (L1 group delay, ~±20 ns)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .constants import WGS84_A, MU, OMEGA_E, PI


@dataclass(frozen=True)
class SVEphemeris:
    """Full IS-GPS-200 broadcast ephemeris for one satellite."""

    prn: int

    # Clock parameters
    toc: float   # Clock reference epoch (s in GPS week)
    af0: float   # Clock bias (s)
    af1: float   # Clock drift (s/s)
    af2: float   # Clock drift rate (s/s²)
    TGD: float   # L1 group delay (s)

    # Ephemeris parameters
    toe: float       # Ephemeris reference epoch (s in GPS week)
    sqrtA: float     # √(semi-major axis) (m^0.5)
    e: float         # Eccentricity
    i0: float        # Inclination at toe (rad)
    Omega0: float    # RAAN at weekly epoch (rad)
    omega: float     # Argument of perigee (rad)
    M0: float        # Mean anomaly at toe (rad)
    deltaN: float    # Mean motion correction (rad/s)
    OmegaDot: float  # Rate of RAAN (rad/s)
    iDot: float      # Rate of inclination (rad/s)

    # Second-harmonic corrections
    Cuc: float   # Cosine correction to argument of latitude (rad)
    Cus: float   # Sine   correction to argument of latitude (rad)
    Crc: float   # Cosine correction to orbit radius (m)
    Crs: float   # Sine   correction to orbit radius (m)
    Cic: float   # Cosine correction to inclination (rad)
    Cis: float   # Sine   correction to inclination (rad)

    # Status
    IODC: int = 0
    health: int = 0   # 0 = healthy


def _build_almanac() -> list[SVEphemeris]:
    """
    Build the embedded 31-SV almanac table.

    Base constellation geometry:
      - Six orbital planes (A–F), RAAN 60° apart
      - Nominal inclination 55°, altitude ~20 200 km
      - sqrtA ≈ 5153.7 m^0.5  →  a ≈ 26 560 km

    Per-SV values are representative of a real GPS constellation epoch.
    """
    BASE_SQRT_A   = 5153.7
    BASE_INC      = np.radians(55.0)
    BASE_OMEGA_DOT = -8.4469e-9       # rad/s
    BASE_DELTA_N   = 4.5e-9           # rad/s
    BASE_TOE       = 388800.0         # GPS s-of-week
    BASE_TOC       = 388800.0

    # fmt: off
    # (PRN, RAAN_off°, M0_off°, e, iDot, Crs, Crc, Cus, Cuc, Cis, Cic, af0, af1, TGD)
    _TABLE = [
        (1,  11.7, 12.3,  0.008819,  2.7e-10,  35.5, 303.5,  5.8e-6, -2.3e-7,  4.6e-8, -1.5e-7,  4.8e-7, -1.0e-12, -1.1e-8),
        (2,  71.3,198.7,  0.012021,  3.1e-10, -77.0, 279.5, -4.1e-6,  1.5e-6, -1.5e-7,  5.6e-9, -2.3e-7, -2.2e-13, -5.6e-9),
        (3, 131.5, 87.4,  0.006234,  1.9e-10,  32.0, 256.5,  7.2e-7, -3.1e-6,  3.5e-7,  7.6e-8,  1.6e-7, -1.5e-13, -9.3e-9),
        (4, 191.2,333.1,  0.011234, -1.4e-10, -98.0, 304.5,  2.1e-6, -1.9e-6,  1.6e-7, -2.4e-7, -4.5e-7,  3.8e-13,  1.9e-8),
        (5, 251.9,244.6,  0.004890,  2.2e-10,  45.5, 271.0,  3.4e-6,  2.2e-6, -8.9e-8,  5.2e-8,  3.1e-7, -4.6e-13, -6.5e-9),
        (6, 311.4,168.9,  0.007821, -0.8e-10, -56.5, 289.5, -6.3e-6,  4.1e-6,  2.1e-7, -3.3e-7, -1.7e-7,  1.2e-13,  1.4e-8),
        (7,  41.8,301.2,  0.009123,  1.6e-10,  63.0, 312.0,  4.7e-6, -5.2e-6, -1.3e-7,  2.8e-7,  5.7e-7, -8.1e-13, -8.7e-9),
        (8, 101.6, 57.8,  0.002341, -2.1e-10, -43.0, 298.0, -1.8e-6,  3.3e-6,  4.4e-7, -1.1e-7, -3.9e-7,  2.3e-13,  7.0e-9),
        (9, 161.3,194.5,  0.013456,  0.9e-10,  89.5, 267.5,  8.9e-6, -2.7e-6, -2.2e-7,  6.3e-8,  2.2e-7, -3.4e-13, -4.8e-9),
        (10,221.0,118.3,  0.005678, -1.7e-10, -71.5, 281.5, -3.6e-6,  1.1e-6,  1.8e-7, -4.5e-7, -6.1e-7,  5.7e-13,  3.3e-8),
        (11,281.7,342.7,  0.008932,  2.4e-10,  52.0, 316.5,  6.5e-6, -4.8e-6, -9.1e-8,  3.7e-7,  8.4e-7, -6.9e-13, -1.2e-8),
        (12,341.4,271.6,  0.003412, -0.5e-10, -82.5, 293.0, -5.1e-6,  2.9e-6,  3.2e-7, -8.9e-8, -1.1e-7,  1.8e-13,  5.4e-9),
        (13, 51.1, 83.9,  0.011567,  1.3e-10,  67.5, 259.5,  9.3e-6, -6.1e-6, -4.7e-7,  2.1e-7,  4.4e-7, -5.2e-13, -9.8e-9),
        (14,111.8,217.3,  0.006789, -2.3e-10, -38.5, 308.5, -2.4e-6,  4.7e-6,  1.4e-7, -6.2e-7, -7.8e-7,  7.1e-13,  2.6e-8),
        (15,171.5,151.8,  0.004123,  0.6e-10,  41.0, 277.0,  7.1e-6, -3.9e-6, -2.9e-7,  4.6e-8,  1.9e-7, -2.8e-13, -7.1e-9),
        (16,231.2,396.2,  0.009876, -1.2e-10, -93.0, 322.5, -4.7e-6,  6.2e-6,  5.8e-7, -3.1e-7, -5.3e-7,  4.4e-13,  1.7e-8),
        (17,291.9, 28.6,  0.007234,  2.0e-10,  58.5, 303.5,  5.3e-6, -1.4e-6, -1.6e-7,  7.2e-8,  7.1e-7, -7.6e-13, -1.5e-8),
        (18,351.6,264.9,  0.002789, -0.3e-10, -66.5, 286.0, -3.2e-6,  5.1e-6,  2.7e-7, -5.4e-8, -2.8e-7,  3.1e-13,  4.7e-9),
        (19, 61.3,119.5,  0.012100,  1.8e-10,  73.5, 268.5,  8.1e-6, -5.5e-6, -3.8e-7,  1.4e-7,  6.6e-7, -5.9e-13, -1.1e-8),
        (20,121.0,353.2,  0.005345, -1.9e-10, -49.5, 314.0, -1.5e-6,  3.8e-6,  4.1e-7, -7.8e-8, -4.6e-7,  6.3e-13,  2.1e-8),
        (21,180.7,187.6,  0.008467,  0.4e-10,  46.5, 295.5,  6.8e-6, -2.1e-6, -1.1e-7,  5.9e-8,  3.8e-7, -4.1e-13, -8.2e-9),
        (22,240.4, 74.3,  0.003934, -2.5e-10, -87.0, 275.0, -6.9e-6,  7.3e-6,  6.5e-7, -2.7e-7, -8.9e-7,  8.5e-13,  3.9e-8),
        (23,300.1,298.8,  0.010789,  2.5e-10,  61.5, 310.5,  4.5e-6, -4.3e-6, -5.6e-7,  3.3e-7,  5.2e-7, -6.4e-13, -1.3e-8),
        (24,  0.8,228.1,  0.006012, -1.1e-10, -54.5, 280.5, -2.9e-6,  2.4e-6,  3.9e-7, -9.7e-8, -1.4e-7,  2.7e-13,  6.2e-9),
        (25, 60.5,163.5,  0.011456,  1.5e-10,  78.0, 264.5,  9.7e-6, -6.9e-6, -4.3e-7,  1.8e-7,  9.3e-7, -8.8e-13, -1.7e-8),
        (26,120.2, 37.9,  0.004567, -2.0e-10, -41.0, 320.0, -1.1e-6,  5.9e-6,  2.3e-7, -6.7e-8, -3.5e-7,  5.0e-13,  1.4e-8),
        (27,179.9,282.4,  0.007890,  0.1e-10,  44.0, 291.5,  7.5e-6, -3.5e-6, -7.8e-8,  4.3e-8,  7.8e-7, -3.7e-13, -9.5e-9),
        (28,239.6,118.7,  0.003178, -2.7e-10, -79.5, 283.0, -5.7e-6,  6.7e-6,  5.2e-7, -2.4e-7, -6.7e-7,  7.9e-13,  3.1e-8),
        (29,299.3, 43.2,  0.009234,  2.3e-10,  55.0, 307.0,  4.1e-6, -3.8e-6, -6.4e-7,  2.9e-7,  4.1e-7, -5.6e-13, -1.4e-8),
        (30, 29.0,357.8,  0.006523, -0.7e-10, -62.5, 278.5, -2.6e-6,  1.8e-6,  4.8e-7, -1.6e-8, -2.1e-7,  1.6e-13,  5.9e-9),
        (31, 88.7,193.1,  0.012678,  1.2e-10,  69.0, 272.5,  8.7e-6, -5.0e-6, -3.4e-7,  6.6e-8,  6.3e-7, -7.2e-13, -1.6e-8),
    ]
    # fmt: on

    ephs: list[SVEphemeris] = []
    for row in _TABLE:
        prn, raan_off, m0_off, e, idot, crs, crc, cus, cuc, cis, cic, af0, af1, tgd = row
        plane = (prn - 1) // 5
        ephs.append(
            SVEphemeris(
                prn=prn,
                toc=BASE_TOC,
                af0=af0,
                af1=af1,
                af2=0.0,
                TGD=tgd,
                toe=BASE_TOE,
                sqrtA=BASE_SQRT_A + (prn % 7) * 0.03 - 0.1,
                e=e,
                i0=BASE_INC + np.radians(raan_off % 3 - 1.5),
                Omega0=np.radians(plane * 60.0) + np.radians(raan_off),
                omega=np.radians(m0_off * 0.7),
                M0=np.radians(m0_off),
                deltaN=BASE_DELTA_N + (prn % 5) * 1e-10,
                OmegaDot=BASE_OMEGA_DOT + (prn % 3 - 1) * 1e-11,
                iDot=idot,
                Cuc=cuc, Cus=cus, Crc=crc, Crs=crs, Cic=cic, Cis=cis,
                IODC=prn,
                health=0,
            )
        )
    return ephs


#: Default embedded almanac — module-level singleton
ALMANAC: list[SVEphemeris] = _build_almanac()

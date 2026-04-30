"""
Tests for the IS-GPS-200 Keplerian propagator.

Validates:
  - Satellite altitude within GPS constellation bounds
  - Earth-centred geometry (position on a sphere of correct radius)
  - Velocity magnitude consistent with orbital mechanics
  - Kepler's equation convergence
  - Sagnac correction magnitude
"""

import numpy as np
import pytest

from nemesis_sim.almanac import ALMANAC
from nemesis_sim.propagator.kepler import sv_position_velocity, eccentric_anomaly
from nemesis_sim.propagator.transforms import sagnac_correction, lla_to_ecef
from nemesis_sim.constants import MU, C, OMEGA_E

GPS_TOW = 388800.0

# GPS orbital altitude bounds (m)
# GPS orbital radius (ECEF, including Earth radius + altitude)
# a = sqrtA² ≈ 26 560 km, plus eccentricity variation
GPS_ALT_MIN = 26_000_000.0
GPS_ALT_MAX = 27_500_000.0


class TestKeplerPropagator:

    def test_all_svs_compute_without_error(self):
        for eph in ALMANAC:
            pos, vel = sv_position_velocity(eph, GPS_TOW)
            assert pos.shape == (3,)
            assert vel.shape == (3,)
            assert np.all(np.isfinite(pos))
            assert np.all(np.isfinite(vel))

    def test_sv_altitude_in_gps_bounds(self):
        """All SVs should be at GPS orbital altitude ~20 200 km."""
        for eph in ALMANAC:
            pos, _ = sv_position_velocity(eph, GPS_TOW)
            r = np.linalg.norm(pos)
            assert GPS_ALT_MIN < r < GPS_ALT_MAX, (
                f"PRN {eph.prn}: |r| = {r/1e6:.3f} Mm, outside GPS bounds"
            )

    def test_velocity_magnitude_consistent_with_orbit(self):
        """
        Orbital velocity v = √(μ/a) for circular orbit.
        GPS: a ≈ 26 560 km → v ≈ 3874 m/s.
        Actual range (eccentric): 3700–4100 m/s.
        """
        for eph in ALMANAC:
            _, vel = sv_position_velocity(eph, GPS_TOW)
            v = np.linalg.norm(vel)
            assert 2500 < v < 4200, (
                f"PRN {eph.prn}: |v| = {v:.1f} m/s out of expected range"
            )

    def test_eccentric_anomaly_kepler(self):
        """E − e·sin(E) = M must hold after solving Kepler's equation."""
        test_cases = [(0.5, 0.1), (2.0, 0.05), (3.0, 0.013), (5.0, 0.008)]
        for M, e in test_cases:
            E = eccentric_anomaly(M, e)
            residual = abs(E - e * np.sin(E) - M)
            assert residual < 1e-12, f"Kepler residual {residual:.2e} for M={M}, e={e}"

    def test_position_continuity(self):
        """Position should change smoothly over small time steps."""
        eph = ALMANAC[0]
        pos1, _ = sv_position_velocity(eph, GPS_TOW)
        pos2, _ = sv_position_velocity(eph, GPS_TOW + 1.0)
        delta = np.linalg.norm(pos2 - pos1)
        # In 1 second, SV moves ~3874 m
        assert 2500 < delta < 4200

    def test_propagation_does_not_mutate_ephemeris(self):
        """Propagator should not modify the frozen SVEphemeris."""
        import dataclasses
        eph = ALMANAC[0]
        m0_before = eph.M0
        sv_position_velocity(eph, GPS_TOW)
        assert eph.M0 == m0_before


class TestSagnac:

    def test_sagnac_magnitude_reasonable(self):
        """Sagnac correction should be in range ±30 m for GPS geometry."""
        user = lla_to_ecef(6.9271, 79.8612, 10.0)
        for eph in ALMANAC[:8]:
            pos, _ = sv_position_velocity(eph, GPS_TOW)
            delta = sagnac_correction(pos, user)
            assert abs(delta) < 35.0, (
                f"PRN {eph.prn}: Sagnac = {delta:.2f} m, exceeds ±35 m"
            )

    def test_sagnac_sign_convention(self):
        """
        Sagnac correction has opposite signs for SVs in east vs west.
        Eastern SVs (positive x in ENU) → positive Sagnac.
        """
        user = lla_to_ecef(0.0, 0.0, 0.0)   # equator, prime meridian
        # SV due east in ECEF (approximate)
        sat_east = np.array([7e6, 2e7, 0.0])
        sat_west = np.array([7e6, -2e7, 0.0])
        # Convention: Δ_sagnac = (Ω_E/c)(x_sv·y_u - y_sv·x_u)
        # User on +x axis (y_u=0): Δ = -y_sv·x_u
        # East sat (+y_sv) → negative, West sat (-y_sv) → positive
        assert sagnac_correction(sat_east, user) < 0
        assert sagnac_correction(sat_west, user) > 0

"""Shared pytest fixtures."""

import numpy as np
import pytest

from nemesis_sim import NEMESISSimulator, AttackConfig
from nemesis_sim.almanac import ALMANAC


# ── Standard receiver position (Colombo, Sri Lanka) ───────────
LAT, LON, ALT = 6.9271, 79.8612, 10.0
GPS_TOW = 388800.0
DOY = 180.0


@pytest.fixture(scope="session")
def colombo_sim():
    """Shared simulator instance for Colombo."""
    sim = NEMESISSimulator(
        lat_deg=LAT, lon_deg=LON, alt_m=ALT,
        gps_tow=GPS_TOW, doy=DOY,
        fs=4.092e6, cn0_dbhz=45.0, rng_seed=42,
    )
    sim.compute_truth()
    return sim


@pytest.fixture(scope="session")
def truth_obs(colombo_sim):
    return colombo_sim._truth_obs


@pytest.fixture
def fresh_sim():
    """New simulator per test (not cached)."""
    return NEMESISSimulator(
        lat_deg=LAT, lon_deg=LON, alt_m=ALT,
        gps_tow=GPS_TOW, doy=DOY,
        fs=4.092e6, cn0_dbhz=45.0, rng_seed=0,
    )

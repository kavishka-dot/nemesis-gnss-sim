"""
Tests for atmospheric delay models.

Klobuchar: validates against IS-GPS-200 reference bounds.
Troposphere: validates NMF mapping function limits and ZHD/ZWD magnitudes.
"""

import numpy as np
import pytest

from nemesis_sim.atmosphere.klobuchar import ionospheric_delay_s, ionospheric_delay_m
from nemesis_sim.atmosphere.troposphere import (
    tropospheric_delay_s,
    tropospheric_delay_m,
    gpt_met,
)
from nemesis_sim.constants import C


class TestKlobuchar:

    def test_minimum_delay_is_5ns(self):
        """IS-GPS-200: minimum ionospheric delay is 5 ns (nighttime)."""
        delay = ionospheric_delay_s(
            lat_deg=0.0, lon_deg=0.0,
            el_deg=90.0, az_deg=0.0,
            gps_tow=0.0,   # local midnight
        )
        assert delay >= 5e-9

    def test_delay_positive(self):
        """Ionospheric delay is always positive (group delay)."""
        test_cases = [
            (6.9, 79.8, 45.0, 180.0, 388800.0),
            (51.5, 0.1, 30.0, 90.0, 43200.0),
            (-33.9, 151.2, 60.0, 270.0, 21600.0),
            (0.0, 0.0, 5.0, 0.0, 50400.0),
        ]
        for lat, lon, el, az, tow in test_cases:
            d = ionospheric_delay_s(lat, lon, el, az, tow)
            assert d > 0, f"Negative iono delay at ({lat},{lon}) el={el}"

    def test_delay_increases_at_lower_elevation(self):
        """Lower elevation → longer path through ionosphere → more delay."""
        d_high = ionospheric_delay_s(6.9, 79.8, 70.0, 0.0, 388800.0)
        d_low  = ionospheric_delay_s(6.9, 79.8, 15.0, 0.0, 388800.0)
        assert d_low > d_high

    def test_delay_in_metres_consistency(self):
        s_delay = ionospheric_delay_s(6.9, 79.8, 45.0, 90.0, 388800.0)
        m_delay = ionospheric_delay_m(6.9, 79.8, 45.0, 90.0, 388800.0)
        assert m_delay == pytest.approx(C * s_delay, rel=1e-9)

    def test_typical_l1_range(self):
        """Typical L1 iono delay: 1–30 m (daytime, mid-latitude)."""
        d = ionospheric_delay_m(40.0, 0.0, 45.0, 180.0, 50400.0)   # local noon
        assert 1.0 < d < 30.0, f"Iono delay {d:.2f} m outside typical range"


class TestTroposphere:

    def test_delay_positive(self):
        for el in [10, 20, 45, 90]:
            d = tropospheric_delay_m(6.9, 10.0, el, doy=180)
            assert d > 0

    def test_delay_increases_at_lower_elevation(self):
        d_high = tropospheric_delay_m(6.9, 0.0, 80.0)
        d_low  = tropospheric_delay_m(6.9, 0.0, 10.0)
        assert d_low > d_high * 3   # should be much larger at low elevation

    def test_zenith_delay_magnitude(self):
        """Zenith delay should be roughly 2.3–2.6 m for sea-level tropics."""
        d = tropospheric_delay_m(6.9, 0.0, 90.0, doy=180)
        assert 2.0 < d < 3.5, f"Zenith tropo delay {d:.3f} m outside expected range"

    def test_altitude_reduces_delay(self):
        """Higher altitude → less atmosphere above → smaller delay."""
        d_sea   = tropospheric_delay_m(45.0, 0.0, 30.0)
        d_high  = tropospheric_delay_m(45.0, 2000.0, 30.0)
        assert d_high < d_sea

    def test_gpt_pressure_reasonable(self):
        """GPT pressure should be close to ISA at sea level (~1013 hPa)."""
        P, T, e = gpt_met(lat_deg=45.0, alt_m=0.0, doy=180.0)
        assert 950 < P < 1050, f"GPT P = {P:.1f} hPa at sea level"
        assert 250 < T < 320,  f"GPT T = {T:.1f} K"
        assert e > 0

    def test_seconds_to_metres_consistency(self):
        s = tropospheric_delay_s(6.9, 10.0, 45.0)
        m = tropospheric_delay_m(6.9, 10.0, 45.0)
        assert m == pytest.approx(C * s, rel=1e-9)

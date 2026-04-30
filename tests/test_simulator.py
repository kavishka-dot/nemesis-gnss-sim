"""
End-to-end tests for NEMESISSimulator.
"""

import numpy as np
import pytest

from nemesis_sim import NEMESISSimulator, AttackConfig


LAT, LON, ALT = 6.9271, 79.8612, 10.0
GPS_TOW = 388800.0


class TestSimulator:

    def test_compute_truth_returns_nonempty_list(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        obs = sim.compute_truth()
        assert len(obs) > 0

    def test_visible_svs_sorted_by_elevation(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        obs = sim.compute_truth()
        elevations = [sv.el_deg for sv in obs]
        assert elevations == sorted(elevations, reverse=True)

    def test_all_visible_svs_above_mask(self):
        el_mask = 10.0
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW,
                               el_mask_deg=el_mask)
        obs = sim.compute_truth()
        for sv in obs:
            assert sv.el_deg >= el_mask

    def test_pseudoranges_in_gps_range(self):
        """GPS pseudoranges: 20 000–26 000 km."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        obs = sim.compute_truth()
        for sv in obs:
            assert 19_000_000 < sv.pseudorange_m < 27_000_000, (
                f"PRN {sv.prn}: ρ = {sv.pseudorange_m/1e6:.3f} Mm out of range"
            )

    def test_doppler_in_expected_range(self):
        """GPS Doppler at L1: typically ±5000 Hz for stationary receiver."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        obs = sim.compute_truth()
        for sv in obs:
            assert abs(sv.doppler_hz) < 5500, (
                f"PRN {sv.prn}: Doppler = {sv.doppler_hz:.1f} Hz out of range"
            )

    def test_iq_output_shape(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW,
                               fs=4.092e6)
        sim.compute_truth()
        iq = sim.generate_iq(duration_ms=10.0)
        expected_n = int(4.092e6 * 10.0 / 1e3)
        assert iq.shape == (expected_n,)
        assert iq.dtype == np.complex128

    def test_iq_power_reasonable(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW,
                               cn0_dbhz=45.0)
        sim.compute_truth()
        iq = sim.generate_iq(duration_ms=50.0, add_noise=True)
        pwr_db = 10 * np.log10(np.mean(np.abs(iq) ** 2))
        assert -5 < pwr_db < 20, f"IQ power {pwr_db:.2f} dBFS outside expected range"

    def test_no_noise_iq_lower_power_variance(self):
        """Noise-free IQ should have lower amplitude variance than noisy."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW,
                               rng_seed=1)
        sim.compute_truth()
        iq_clean = sim.generate_iq(duration_ms=10.0, add_noise=False)
        iq_noisy = sim.generate_iq(duration_ms=10.0, add_noise=True)
        assert np.std(np.abs(iq_noisy)) > np.std(np.abs(iq_clean))

    def test_summary_json_serialisable(self):
        import json
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        sim.compute_truth()
        s = sim.summary()
        assert json.dumps(s)   # should not raise

    def test_n_visible_property(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        sim.compute_truth()
        assert sim.n_visible == len(sim._truth_obs)

    def test_rng_seed_reproducibility(self):
        """Same seed must produce identical IQ."""
        def make_iq(seed):
            sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT,
                                   gps_tow=GPS_TOW, rng_seed=seed)
            sim.compute_truth()
            return sim.generate_iq(duration_ms=5.0, add_noise=True)

        iq1 = make_iq(42)
        iq2 = make_iq(42)
        np.testing.assert_array_equal(iq1, iq2)

    def test_attack_none_returns_truth(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        truth = sim.compute_truth()
        attacked = sim.apply_attack(AttackConfig(attack_type="none"))
        for tv, av in zip(truth, attacked):
            assert tv.pseudorange_m == av.pseudorange_m

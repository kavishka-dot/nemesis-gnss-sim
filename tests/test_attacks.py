"""
Tests for NEMESIS attack models.

Validates the physical correctness of each attack:
  - Meaconing: Δρ = c·Δτ, Doppler unchanged
  - Slow Drift: bias = rate·Δt, Doppler shift = -rate·f_L1/c
  - Adversarial: per-SV pseudoranges geometrically consistent with false position
"""

import numpy as np
import pytest

from nemesis_sim import NEMESISSimulator, AttackConfig
from nemesis_sim.attacks import MeaconingAttack, SlowDriftAttack, AdversarialAttack
from nemesis_sim.constants import C, L1_FREQ

LAT, LON, ALT = 6.9271, 79.8612, 10.0
GPS_TOW = 388800.0
DOY = 180.0


@pytest.fixture(scope="module")
def sim_with_truth():
    sim = NEMESISSimulator(
        lat_deg=LAT, lon_deg=LON, alt_m=ALT,
        gps_tow=GPS_TOW, doy=DOY,
        fs=4.092e6, cn0_dbhz=45.0, rng_seed=0,
    )
    sim.compute_truth()
    return sim


class TestMeaconing:

    def test_pseudorange_offset_equals_c_times_delay(self, sim_with_truth):
        delay_s = 1e-4
        cfg = AttackConfig(attack_type="meaconing", meaconing_delay_s=delay_s)
        truth   = sim_with_truth._truth_obs
        attacked = sim_with_truth.apply_attack(cfg)

        expected_delta = C * delay_s
        for tv, av in zip(truth, attacked):
            delta = av.pseudorange_m - tv.pseudorange_m
            assert delta == pytest.approx(expected_delta, rel=1e-9), (
                f"PRN {tv.prn}: Δρ = {delta:.4f} m, expected {expected_delta:.4f} m"
            )

    def test_doppler_unchanged(self, sim_with_truth):
        cfg = AttackConfig(attack_type="meaconing", meaconing_delay_s=1e-4)
        truth   = sim_with_truth._truth_obs
        attacked = sim_with_truth.apply_attack(cfg)

        for tv, av in zip(truth, attacked):
            assert av.doppler_hz == pytest.approx(tv.doppler_hz, abs=1e-9), (
                f"PRN {tv.prn}: Doppler changed during meaconing"
            )

    def test_zero_delay_is_identity(self, sim_with_truth):
        cfg = AttackConfig(attack_type="meaconing", meaconing_delay_s=0.0)
        truth   = sim_with_truth._truth_obs
        attacked = sim_with_truth.apply_attack(cfg)
        for tv, av in zip(truth, attacked):
            assert av.pseudorange_m == pytest.approx(tv.pseudorange_m, abs=1e-6)


class TestSlowDrift:

    def test_bias_equals_rate_times_dt(self):
        """b_i(t) = rate · (t − t_start)"""
        rate = 3.0   # m/s
        t_start = GPS_TOW
        dt = 100.0   # seconds

        sim = NEMESISSimulator(
            lat_deg=LAT, lon_deg=LON, alt_m=ALT,
            gps_tow=GPS_TOW + dt, doy=DOY,
        )
        truth = sim.compute_truth()
        cfg = AttackConfig(
            attack_type="slow_drift",
            drift_rate_m_s=rate,
            drift_start_s=t_start,
        )
        attacked = sim.apply_attack(cfg)

        expected_bias = rate * dt
        for tv, av in zip(truth, attacked):
            delta = av.pseudorange_m - tv.pseudorange_m
            assert delta == pytest.approx(expected_bias, rel=1e-9)

    def test_doppler_shift_consistent_with_drift(self):
        """Doppler shift must equal -rate * f_L1 / c."""
        rate = 2.0
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW + 50.0)
        truth = sim.compute_truth()
        cfg = AttackConfig(attack_type="slow_drift", drift_rate_m_s=rate, drift_start_s=GPS_TOW)
        attacked = sim.apply_attack(cfg)

        expected_dop_shift = -rate * L1_FREQ / C
        for tv, av in zip(truth, attacked):
            assert av.doppler_hz - tv.doppler_hz == pytest.approx(expected_dop_shift, rel=1e-6)

    def test_no_drift_before_start(self):
        """Before t_start, drift bias must be zero."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW - 10.0)
        truth = sim.compute_truth()
        cfg = AttackConfig(attack_type="slow_drift", drift_rate_m_s=5.0, drift_start_s=GPS_TOW)
        attacked = sim.apply_attack(cfg)
        for tv, av in zip(truth, attacked):
            assert av.pseudorange_m == pytest.approx(tv.pseudorange_m, abs=1e-6)

    def test_per_sv_override(self):
        """Per-SV drift rates should override the global rate."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW + 10.0)
        truth = sim.compute_truth()
        prn0 = truth[0].prn
        cfg = AttackConfig(
            attack_type="slow_drift",
            drift_rate_m_s=1.0,
            drift_per_sv={prn0: 10.0},
            drift_start_s=GPS_TOW,
        )
        attacked = sim.apply_attack(cfg)
        delta_sv0 = attacked[0].pseudorange_m - truth[0].pseudorange_m
        delta_sv1 = attacked[1].pseudorange_m - truth[1].pseudorange_m
        assert delta_sv0 == pytest.approx(10.0 * 10.0, rel=1e-6)
        assert delta_sv1 == pytest.approx(1.0 * 10.0, rel=1e-6)


class TestAdversarial:

    def test_pseudoranges_differ_from_truth(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        truth = sim.compute_truth()
        cfg = AttackConfig(attack_type="adversarial", false_lat=7.2, false_lon=80.1, false_alt=10.0)
        attacked = sim.apply_attack(cfg)
        deltas = [abs(av.pseudorange_m - tv.pseudorange_m)
                  for tv, av in zip(truth, attacked)]
        assert all(d > 100 for d in deltas), "Adversarial pseudoranges too close to truth"

    def test_per_sv_deltas_are_not_uniform(self):
        """Unlike meaconing, adversarial pseudorange offsets must vary per SV."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        truth = sim.compute_truth()
        cfg = AttackConfig(attack_type="adversarial", false_lat=7.2, false_lon=80.1)
        attacked = sim.apply_attack(cfg)
        deltas = [av.pseudorange_m - tv.pseudorange_m for tv, av in zip(truth, attacked)]
        assert max(deltas) - min(deltas) > 1000, (
            "Adversarial per-SV deltas are suspiciously uniform (should vary by geometry)"
        )

    def test_output_list_same_length_as_input(self):
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        truth = sim.compute_truth()
        cfg = AttackConfig(attack_type="adversarial", false_lat=7.2, false_lon=80.1)
        attacked = sim.apply_attack(cfg)
        assert len(attacked) == len(truth)

    def test_truth_not_mutated(self):
        """apply_attack must never modify the truth observation list."""
        sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW)
        truth = sim.compute_truth()
        pseudo_before = [sv.pseudorange_m for sv in truth]
        cfg = AttackConfig(attack_type="adversarial", false_lat=7.2, false_lon=80.1)
        sim.apply_attack(cfg)
        pseudo_after = [sv.pseudorange_m for sv in truth]
        assert pseudo_before == pseudo_after, "Truth observations were mutated by attack"

"""
Physics validation script.

Prints a complete error budget for one representative epoch and checks
all terms against expected physical bounds. Run before any paper submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemesis_sim import NEMESISSimulator
from nemesis_sim.constants import C, L1_FREQ

LAT, LON, ALT = 6.9271, 79.8612, 10.0
GPS_TOW = 388800.0
DOY = 180.0

CHECKS = {
    "iono_m":      (1.0,  30.0,  "Klobuchar L1 delay (m)"),
    "tropo_m":     (2.0,  30.0,  "NMF troposphere slant delay (m)"),
    "sagnac_m":    (-35.0, 35.0, "Sagnac correction (m)"),
    "clock_err_m": (-500.0, 500.0, "SV clock correction (m)"),
}


def main() -> None:
    sim = NEMESISSimulator(
        lat_deg=LAT, lon_deg=LON, alt_m=ALT,
        gps_tow=GPS_TOW, doy=DOY, fs=4.092e6,
    )
    truth = sim.compute_truth()

    print(f"\nNEMESIS Physics Validation")
    print(f"Location : {LAT}°N  {LON}°E  {ALT} m")
    print(f"GPS TOW  : {GPS_TOW} s   DOY: {DOY}\n")

    print(f"{'PRN':>3}  {'El':>6}  {'Az':>6}  {'ρ (km)':>12}  "
          f"{'Dopp (Hz)':>10}  {'Clk (m)':>9}  "
          f"{'Iono (m)':>8}  {'Tropo (m)':>9}  {'Sagnac (m)':>10}")
    print("─" * 90)
    for sv in truth:
        print(f"{sv.prn:>3}  {sv.el_deg:>6.2f}  {sv.az_deg:>6.2f}  "
              f"{sv.pseudorange_m/1e3:>12.3f}  {sv.doppler_hz:>10.2f}  "
              f"{sv.clock_err_m:>9.4f}  {sv.iono_m:>8.3f}  "
              f"{sv.tropo_m:>9.3f}  {sv.sagnac_m:>10.4f}")

    print(f"\n{'─'*60}")
    print("Validation checks:")
    all_pass = True
    for sv in truth:
        for field, (lo, hi, label) in CHECKS.items():
            val = getattr(sv, field)
            ok = lo <= val <= hi
            if not ok:
                print(f"  FAIL  PRN{sv.prn:2d}  {label}: {val:.4f} outside [{lo}, {hi}]")
                all_pass = False

    if all_pass:
        print("  All checks PASSED")

    # Attack delta validation
    print(f"\nAttack delta checks:")
    from nemesis_sim import AttackConfig
    from nemesis_sim.constants import C

    # Meaconing: Δρ must equal c·Δτ
    delay = 1e-4
    cfg = AttackConfig(attack_type="meaconing", meaconing_delay_s=delay)
    atk = sim.apply_attack(cfg)
    expected = C * delay
    for tv, av in zip(truth, atk):
        delta = av.pseudorange_m - tv.pseudorange_m
        err = abs(delta - expected)
        if err > 1e-3:
            print(f"  FAIL  Meaconing PRN{tv.prn}: Δρ error = {err:.6f} m")
            all_pass = False
    print(f"  Meaconing Δρ = {expected:.2f} m  ✓")

    # Slow drift: 100s at 2 m/s = 200 m
    sim2 = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT, gps_tow=GPS_TOW+100)
    truth2 = sim2.compute_truth()
    cfg2 = AttackConfig(attack_type="slow_drift", drift_rate_m_s=2.0, drift_start_s=GPS_TOW)
    atk2 = sim2.apply_attack(cfg2)
    for tv, av in zip(truth2, atk2):
        delta = av.pseudorange_m - tv.pseudorange_m
        if abs(delta - 200.0) > 1e-6:
            print(f"  FAIL  SlowDrift PRN{tv.prn}: Δρ = {delta:.6f} m (expected 200.0)")
            all_pass = False
    print(f"  SlowDrift 100s@2m/s Δρ = 200.00 m  ✓")

    print(f"\n{'PASS' if all_pass else 'FAIL'}  ({len(truth)} SVs checked)\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Command-line interface for the NEMESIS GNSS simulator.

Entry point: nemesis-sim  (defined in pyproject.toml)

Colab/Jupyter safe: kernel .json arguments are stripped before parsing.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .simulator import NEMESISSimulator
from .attacks import AttackConfig
from .io import save_int16, save_cf32


_BANNER = r"""
  _   _ _____ __  __ _____ ____ ___ ____
 | \ | | ____|  \/  | ____/ ___|_ _/ ___|
 |  \| |  _| | |\/| |  _| \___ \| |\___ \
 | |\  | |___| |  | | |___ ___) | | ___) |
 |_| \_|_____|_|  |_|_____|____/___|____/
"""

def _print_banner() -> None:
    from . import __version__
    print(_BANNER)
    print(f"  version {__version__}    GPS L1 C/A Signal Simulator")
    print(f"  IS-GPS-200  |  Klobuchar Iono  |  Neill MF Tropo  |  WGS-84")
    print(f"  Attacks: Meaconing  |  Slow Drift  |  Adversarial")
    print(f"  https://github.com/kavishka-dot/nemesis-gnss-sim")
    print()



def _clean_argv(argv: list[str]) -> list[str]:
    """Strip Jupyter kernel connection args (-f /path/kernel-xxx.json)."""
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "-f":
            skip_next = True
            continue
        if arg.endswith(".json") and "kernel" in arg:
            continue
        cleaned.append(arg)
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nemesis-sim",
        description="NEMESIS GPS L1 C/A signal simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Receiver position
    pos = p.add_argument_group("Receiver position")
    pos.add_argument("--lat",     type=float, default=6.9271,   metavar="DEG")
    pos.add_argument("--lon",     type=float, default=79.8612,  metavar="DEG")
    pos.add_argument("--alt",     type=float, default=10.0,     metavar="M")

    # Time
    tm = p.add_argument_group("Time")
    tm.add_argument("--tow",  type=float, default=388800.0, metavar="S",
                    help="GPS time of week (seconds)")
    tm.add_argument("--doy",  type=float, default=180.0,    metavar="DAY",
                    help="Day of year (1–365)")

    # Signal
    sig = p.add_argument_group("Signal")
    sig.add_argument("--fs",       type=float, default=4.092e6, metavar="HZ",
                     help="Sample rate (Hz)")
    sig.add_argument("--cn0",      type=float, default=45.0,    metavar="dBHz")
    sig.add_argument("--dur",      type=float, default=1000.0,  metavar="MS",
                     help="IQ duration (milliseconds)")
    sig.add_argument("--el-mask",  type=float, default=5.0,     metavar="DEG")

    # Attack
    atk = p.add_argument_group("Attack")
    atk.add_argument("--attack", default="none",
                     choices=["none", "meaconing", "slow_drift", "adversarial"])
    atk.add_argument("--meacon-delay",  type=float, default=1e-4, metavar="S")
    atk.add_argument("--drift-rate",    type=float, default=2.0,  metavar="M/S")
    atk.add_argument("--false-lat",     type=float, default=7.0,  metavar="DEG")
    atk.add_argument("--false-lon",     type=float, default=80.0, metavar="DEG")
    atk.add_argument("--false-alt",     type=float, default=10.0, metavar="M")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--out",     type=str, default=None, metavar="FILE")
    out.add_argument("--fmt",     type=str, default="int16",
                     choices=["int16", "cf32"])
    out.add_argument("--summary", action="store_true",
                     help="Print JSON summary to stdout")
    out.add_argument("--no-noise", action="store_true",
                     help="Disable AWGN")

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = _clean_argv(argv)
    _print_banner()

    parser = build_parser()
    a = parser.parse_args(argv)

    sim = NEMESISSimulator(
        lat_deg     = a.lat,
        lon_deg     = a.lon,
        alt_m       = a.alt,
        gps_tow     = a.tow,
        doy         = a.doy,
        el_mask_deg = a.el_mask,
        fs          = a.fs,
        cn0_dbhz    = a.cn0,
    )

    truth = sim.compute_truth()

    # ── Print observation table ──────────────────────────────
    print(f"\n NEMESIS Simulator — {a.lat:.4f}°N  {a.lon:.4f}°E")
    print(f" GPS TOW: {a.tow:.1f}s   DOY: {a.doy:.0f}   fs: {a.fs/1e6:.3f} MHz\n")
    print(f" {'PRN':>3}  {'El':>6}  {'Az':>6}  {'ρ (km)':>12}  "
          f"{'Dopp (Hz)':>10}  {'Clk (m)':>9}  "
          f"{'Iono (m)':>8}  {'Tropo (m)':>9}  {'Sagnac (m)':>10}")
    print(" " + "─" * 90)
    for sv in truth:
        print(f" {sv.prn:>3}  {sv.el_deg:>6.2f}  {sv.az_deg:>6.2f}  "
              f"{sv.pseudorange_m/1e3:>12.3f}  {sv.doppler_hz:>10.2f}  "
              f"{sv.clock_err_m:>9.4f}  {sv.iono_m:>8.3f}  "
              f"{sv.tropo_m:>9.3f}  {sv.sagnac_m:>10.4f}")

    # ── Apply attack ─────────────────────────────────────────
    if a.attack != "none":
        cfg = AttackConfig(attack_type=a.attack)
        if a.attack == "meaconing":
            cfg.meaconing_delay_s = a.meacon_delay
        elif a.attack == "slow_drift":
            cfg.drift_rate_m_s = a.drift_rate
            cfg.drift_start_s  = a.tow
        elif a.attack == "adversarial":
            cfg.false_lat = a.false_lat
            cfg.false_lon = a.false_lon
            cfg.false_alt = a.false_alt

        attacked = sim.apply_attack(cfg)
        print(f"\n ATTACK: {a.attack.upper()}")
        print(f" {'PRN':>3}  {'ΔPSR (m)':>12}  {'ΔDopp (Hz)':>12}")
        for tv, av in zip(truth, attacked):
            print(f" {tv.prn:>3}  "
                  f"{av.pseudorange_m - tv.pseudorange_m:>12.4f}  "
                  f"{av.doppler_hz - tv.doppler_hz:>12.4f}")

    # ── JSON summary ─────────────────────────────────────────
    if a.summary:
        print("\n" + json.dumps(sim.summary(use_attacked=(a.attack != "none")), indent=2))

    # ── IQ generation ────────────────────────────────────────
    if a.out:
        print(f"\n Generating {a.dur} ms IQ ({int(a.fs * a.dur / 1e3):,} samples)...")
        iq = sim.generate_iq(
            duration_ms  = a.dur,
            use_attacked = (a.attack != "none"),
            add_noise    = not a.no_noise,
        )
        pwr = 10 * np.log10(np.mean(np.abs(iq) ** 2))
        print(f" IQ power: {pwr:.2f} dBFS")
        if a.fmt == "int16":
            save_int16(iq, a.out)
        else:
            save_cf32(iq, a.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


_BANNER = r"""
  _   _ _____ __  __ _____ ____ ___ ____
 | \ | | ____|  \/  | ____/ ___|_ _/ ___|
 |  \| |  _| | |\/| |  _| \___ \| |\___ \
 | |\  | |___| |  | | |___ ___) | | ___) |
 |_| \_|_____|_|  |_|_____|____/___|____/
"""

def _print_banner() -> None:
    from . import __version__
    print(_BANNER)
    print(f"  version {__version__}    GPS L1 C/A Signal Simulator")
    print(f"  IS-GPS-200  |  Klobuchar Iono  |  Neill MF Tropo  |  WGS-84")
    print(f"  Attacks: Meaconing  |  Slow Drift  |  Adversarial")
    print(f"  https://github.com/kavishka-dot/nemesis-gnss-sim")
    print()

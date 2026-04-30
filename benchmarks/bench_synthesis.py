"""
IQ synthesis throughput benchmark.

Measures samples/second across sample rates and durations.
Run with: python benchmarks/bench_synthesis.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemesis_sim import NEMESISSimulator

LAT, LON, ALT = 6.9271, 79.8612, 10.0
GPS_TOW = 388800.0


def bench(fs: float, duration_ms: float, n_repeat: int = 3) -> dict:
    sim = NEMESISSimulator(lat_deg=LAT, lon_deg=LON, alt_m=ALT,
                           gps_tow=GPS_TOW, fs=fs, cn0_dbhz=45.0)
    sim.compute_truth()
    n_svs = sim.n_visible

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        iq = sim.generate_iq(duration_ms=duration_ms, add_noise=True)
        times.append(time.perf_counter() - t0)

    n_samples = len(iq)
    best_t = min(times)
    return {
        "fs_mhz":      fs / 1e6,
        "duration_ms": duration_ms,
        "n_samples":   n_samples,
        "n_svs":       n_svs,
        "wall_s":      best_t,
        "msps":        n_samples / best_t / 1e6,
        "realtime_x":  (duration_ms / 1e3) / best_t,
    }


def main() -> None:
    print("NEMESIS IQ Synthesis Benchmark")
    print(f"{'FS (MHz)':>10}  {'Dur (ms)':>10}  {'Samples':>10}  "
          f"{'SVs':>5}  {'Wall (s)':>10}  {'MSps':>8}  {'RT×':>8}")
    print("─" * 75)

    for fs in [2.046e6, 4.092e6, 8.184e6, 16.368e6]:
        for dur in [100.0, 1000.0]:
            r = bench(fs, dur)
            print(f"{r['fs_mhz']:>10.3f}  {r['duration_ms']:>10.1f}  "
                  f"{r['n_samples']:>10,}  {r['n_svs']:>5}  "
                  f"{r['wall_s']:>10.3f}  {r['msps']:>8.2f}  {r['realtime_x']:>8.2f}×")


if __name__ == "__main__":
    main()

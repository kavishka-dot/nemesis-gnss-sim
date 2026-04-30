"""
Batch IQ dataset generator for NEMESIS training pipeline.

Produces labeled IQ chunks with a manifest.json for all four attack classes.

Usage
-----
python scripts/generate_dataset.py \\
    --out-dir /data/nemesis_iq \\
    --duration-ms 1000 \\
    --tow-start 388800 --tow-end 389800 --tow-step 100 \\
    --attacks all

Output structure
----------------
out_dir/
  clearsky/
    colombo_388800.bin
    colombo_388900.bin
    ...
  meaconing/
    ...
  slow_drift/
    ...
  adversarial/
    ...
  manifest.json      ← label, path, n_samples, tow, location, attack_params
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Colab-safe argv
_argv = [a for a in sys.argv[1:] if a != "-f" and ".json" not in a]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nemesis_sim import NEMESISSimulator, AttackConfig, save_int16

# ── Default locations ──────────────────────────────────────────
DEFAULT_LOCATIONS = [
    {"name": "colombo",     "lat":  6.9271, "lon":  79.8612, "alt": 10.0},
    {"name": "london",      "lat": 51.5074, "lon":  -0.1278, "alt": 10.0},
    {"name": "singapore",   "lat":  1.3521, "lon": 103.8198, "alt": 10.0},
    {"name": "new_york",    "lat": 40.7128, "lon": -74.0060, "alt": 10.0},
    {"name": "sydney",      "lat":-33.8688, "lon": 151.2093, "alt": 10.0},
]

ATTACK_CONFIGS = {
    "clearsky": AttackConfig(attack_type="none"),
    "meaconing": AttackConfig(attack_type="meaconing", meaconing_delay_s=1e-4),
    "slow_drift": AttackConfig(attack_type="slow_drift", drift_rate_m_s=2.0),
    "adversarial": AttackConfig(attack_type="adversarial",
                                false_lat=7.2, false_lon=80.1, false_alt=10.0),
}

ATTACK_LABELS = {"clearsky": 0, "meaconing": 1, "slow_drift": 2, "adversarial": 3}


def generate_dataset(
    out_dir: Path,
    tow_values: list[float],
    attacks: list[str],
    duration_ms: float,
    fs: float,
    cn0_dbhz: float,
    locations: list[dict],
    doy: float,
    fmt: str,
    verbose: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = len(locations) * len(tow_values) * len(attacks)
    done = 0

    for loc in locations:
        for tow in tow_values:
            # One simulator per (location, tow) — reuse across attacks
            sim = NEMESISSimulator(
                lat_deg=loc["lat"], lon_deg=loc["lon"], alt_m=loc["alt"],
                gps_tow=tow, doy=doy, fs=fs, cn0_dbhz=cn0_dbhz, rng_seed=int(tow),
            )
            sim.compute_truth()

            for atk_name in attacks:
                cfg = ATTACK_CONFIGS[atk_name]
                if atk_name == "slow_drift":
                    cfg = AttackConfig(attack_type="slow_drift",
                                       drift_rate_m_s=2.0, drift_start_s=tow)

                sim.apply_attack(cfg)
                iq = sim.generate_iq(duration_ms=duration_ms, use_attacked=True)

                atk_dir = out_dir / atk_name
                atk_dir.mkdir(exist_ok=True)
                fname = f"{loc['name']}_{int(tow)}.bin"
                fpath = atk_dir / fname

                save_int16(iq, fpath, verbose=False)

                manifest.append({
                    "file":     str(fpath.relative_to(out_dir)),
                    "label":    ATTACK_LABELS[atk_name],
                    "attack":   atk_name,
                    "location": loc["name"],
                    "lat":      loc["lat"],
                    "lon":      loc["lon"],
                    "alt_m":    loc["alt"],
                    "tow":      tow,
                    "doy":      doy,
                    "n_samples":len(iq),
                    "fs_hz":    fs,
                    "cn0_dbhz": cn0_dbhz,
                    "n_visible":sim.n_visible,
                })

                done += 1
                if verbose:
                    print(f"  [{done:4d}/{total}]  {atk_name:12s}  "
                          f"{loc['name']:12s}  TOW={tow:.0f}  "
                          f"SVs={sim.n_visible}  {len(iq):,} samples")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDataset complete: {done} files, manifest → {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="NEMESIS IQ dataset generator")
    p.add_argument("--out-dir",      default="nemesis_dataset")
    p.add_argument("--tow-start",    type=float, default=388800.0)
    p.add_argument("--tow-end",      type=float, default=389800.0)
    p.add_argument("--tow-step",     type=float, default=100.0)
    p.add_argument("--attacks",      default="all",
                   help="Comma-separated list or 'all'")
    p.add_argument("--duration-ms",  type=float, default=1000.0)
    p.add_argument("--fs",           type=float, default=4.092e6)
    p.add_argument("--cn0",          type=float, default=45.0)
    p.add_argument("--doy",          type=float, default=180.0)
    p.add_argument("--fmt",          default="int16", choices=["int16", "cf32"])
    p.add_argument("--quiet",        action="store_true")
    a = p.parse_args(_argv)

    attacks = list(ATTACK_CONFIGS.keys()) if a.attacks == "all" \
              else a.attacks.split(",")
    tow_values = list(np.arange(a.tow_start, a.tow_end + 1, a.tow_step))

    print(f"Generating NEMESIS IQ dataset")
    print(f"  Output  : {a.out_dir}")
    print(f"  TOW     : {a.tow_start:.0f} – {a.tow_end:.0f} s (step {a.tow_step:.0f})")
    print(f"  Attacks : {attacks}")
    print(f"  Duration: {a.duration_ms} ms  |  fs: {a.fs/1e6:.3f} MHz")
    print(f"  Locations: {len(DEFAULT_LOCATIONS)}\n")

    generate_dataset(
        out_dir     = Path(a.out_dir),
        tow_values  = tow_values,
        attacks     = attacks,
        duration_ms = a.duration_ms,
        fs          = a.fs,
        cn0_dbhz    = a.cn0,
        locations   = DEFAULT_LOCATIONS,
        doy         = a.doy,
        fmt         = a.fmt,
        verbose     = not a.quiet,
    )


if __name__ == "__main__":
    main()

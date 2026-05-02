# NEMESIS GNSS Simulator

Geodetic-accuracy GPS L1 C/A signal simulator with physically rigorous spoofing attack models,
built for the [NEMESIS](https://github.com/kavishka-dot) research pipeline.

<img width="2469" height="969" alt="image" src="https://github.com/user-attachments/assets/fe480603-5cc7-473c-8912-a1e8df9c8372" />


```python
from nemesis_sim import NEMESISSimulator, AttackConfig

sim = NEMESISSimulator(lat_deg=6.9271, lon_deg=79.8612, alt_m=10.0, gps_tow=388800.0)
sim.compute_truth()
sim.apply_attack(AttackConfig(attack_type="slow_drift", drift_rate_m_s=2.0))
iq = sim.generate_iq(duration_ms=1000.0)   # complex128 at 4.092 MHz
```

---

## Features

**Signal model (IS-GPS-200 compliant)**
- Full 16-parameter Keplerian propagator with 6 harmonic corrections (Cuc, Cus, Crc, Crs, Cic, Cis)
- Satellite clock: `af0 + af1·tk + af2·tk² + F·e·√A·sin(Ek)` relativistic term
- Sagnac / Earth-rotation correction during signal transit
- Klobuchar ionospheric delay (GPS week 2238 broadcast α/β coefficients)
- Neill Mapping Function troposphere + GPT standard atmosphere (seasonal, latitude-dependent)
- Correct C/A Gold code generation for all 32 PRNs (IS-GPS-200 Table 3-Ia)
- BPSK(1) baseband IQ synthesis at arbitrary sample rate
- Transparent RINEX navigation file support (natively handles `.rnx`, `.gz`, and `.7z`)

**Interactive Web GUI**
- Launch a local lightweight map-based UI with `nemesis-sim --gui`
- Click anywhere on the globe to generate RF signals instantly
- Live visualization of visible satellites and synthesized baseband IQ
- Direct browser download of generated signals as interleaved `.bin` files

**Ephemeris sources - your choice**

| Mode | How | When to use |
|---|---|---|
| `embedded` | Built-in 31-SV almanac, zero setup | Training data, Colab, offline use |
| `rinex` | Real broadcast ephemeris from RINEX 2/3 file | Paper validation, geodetic accuracy |

**Attack classes (NEMESIS taxonomy)**

| Class | Model |
|---|---|
| `meaconing` | Uniform capture-and-rebroadcast delay `Δτ` across all SVs |
| `slow_drift` | Per-SV linear pseudorange ramp `b_i(t) = ṙ_i·(t − t₀)` with consistent Doppler |
| `adversarial` | Full synthetic constellation synthesised at a false geodetic position |

**Output formats**
- `int16` interleaved (Pluto / bladeRF / gps-sdr-sim compatible)
- `complex64` (GNU Radio / SigMF compatible)

---

## Installation

```bash
# Minimal (NumPy only)
pip install nemesis-gnss-sim

# With visualisation tools
pip install "nemesis-gnss-sim[viz]"

# Development
git clone https://github.com/kavishka-dot/nemesis-gnss-sim
cd nemesis-gnss-sim
pip install -e ".[dev]"
```

**Colab (no install needed):**
```python
!git clone https://github.com/kavishka-dot/nemesis-gnss-sim
import sys; sys.path.insert(0, "nemesis-gnss-sim")
from nemesis_sim import NEMESISSimulator
```

---

## Quick Start

### Embedded mode (default — works anywhere)

```python
from nemesis_sim import NEMESISSimulator, AttackConfig
from nemesis_sim.io import save_int16

sim = NEMESISSimulator(
    lat_deg=6.9271, lon_deg=79.8612, alt_m=10.0,
    gps_tow=388800.0, doy=180.0, fs=4.092e6, cn0_dbhz=45.0,
)
sim.compute_truth()

for label, cfg in [
    ("clearsky",    AttackConfig("none")),
    ("meaconing",   AttackConfig("meaconing",   meaconing_delay_s=1e-4)),
    ("slow_drift",  AttackConfig("slow_drift",  drift_rate_m_s=2.0, drift_start_s=388800.0)),
    ("adversarial", AttackConfig("adversarial", false_lat=7.2, false_lon=80.1)),
]:
    sim.apply_attack(cfg)
    iq = sim.generate_iq(duration_ms=1000.0)
    save_int16(iq, f"{label}.bin")
```

### RINEX mode (real broadcast ephemeris)

Download a daily RINEX navigation file from [NASA CDDIS](https://cddis.nasa.gov/archive/gnss/data/daily/)
or [IGS](https://igs.bkg.bund.de/):

```python
from nemesis_sim import NEMESISSimulator

sim = NEMESISSimulator(
    lat_deg=6.9271, lon_deg=79.8612, alt_m=10.0,
    gps_tow=388800.0,
    rinex_path="BRDC00IGS_R_20251200000_01D_MN.rnx",   # RINEX 2 or 3
)
sim.compute_truth()
iq = sim.generate_iq(duration_ms=1000.0)
```

---

## CLI

```bash
<<<<<<< Updated upstream
# Embedded mode (default)
=======
# Launch Interactive Web GUI
nemesis-sim --gui

# Clearsky
>>>>>>> Stashed changes
nemesis-sim --lat 6.9271 --lon 79.8612 --out clearsky.bin

# RINEX mode
nemesis-sim --lat 6.9271 --lon 79.8612 \
            --ephemeris rinex --rinex-file BRDC00IGS_R_20251200000_01D_MN.rnx \
            --out clearsky_rinex.bin

# Meaconing attack (100 µs delay)
nemesis-sim --lat 6.9271 --lon 79.8612 \
            --attack meaconing --meacon-delay 1e-4 \
            --out meaconing.bin

# Slow drift (2 m/s)
nemesis-sim --lat 6.9271 --lon 79.8612 \
            --attack slow_drift --drift-rate 2.0 \
            --out drift.bin

# Adversarial (false position 27 km away)
nemesis-sim --lat 6.9271 --lon 79.8612 \
            --attack adversarial --false-lat 7.2 --false-lon 80.1 \
            --out adversarial.bin
```

**CLI output:**
```
  _   _ _____ __  __ _____ ____ ___ ____
 | \ | | ____|  \/  | ____/ ___|_ _/ ___|
 |  \| |  _| | |\/| |  _| \___ \| |\___ \
 | |\  | |___| |  | | |___ ___) | | ___) |
 |_| \_|_____|_|  |_|_____|____/___|____/

  version 0.1.0    GPS L1 C/A Signal Simulator
  IS-GPS-200  |  Klobuchar Iono  |  Neill MF Tropo  |  WGS-84
  Attacks: Meaconing  |  Slow Drift  |  Adversarial
  https://github.com/kavishka-dot/nemesis-gnss-sim

 [ephemeris]  embedded  —  31 SVs loaded
 [visible]    16 satellites above 5.0° mask

 PRN      El      Az        ρ (km)   Dopp (Hz)    Clk (m)  Iono (m)  Tropo (m)  Sagnac (m)
 ──────────────────────────────────────────────────────────────────────────────────────────
  24   68.74  337.06     20624.917    -1037.51    40.7704     3.105      2.486      4.4894
   7   68.06    1.59     20413.493     1619.75  -178.8717     3.107      2.497     -0.3257
  ...
```

---

## Ephemeris Modes

### Embedded (default)

A built-in 31-SV almanac derived from IS-GPS-200 reference constellation parameters.

- Zero setup, no files needed
- Works fully offline and in Colab
- Reproducible: same output for same parameters every time
- Best for: training dataset generation, attack model development, quick experiments

### RINEX

Real GPS broadcast ephemeris loaded from a RINEX 2.11 or RINEX 3.x navigation file.
When multiple records exist per PRN (files typically contain entries every 2 hours),
the record closest to the requested GPS TOW is selected automatically.

- Satellite positions accurate to ~1 m
- Tied to a real calendar date
- Best for: paper validation, hardware-in-the-loop testing, peer-review credibility

**Getting a RINEX file:**
```bash
# IGS open mirror (no registration required)
# https://igs.bkg.bund.de/root_ftp/IGS/BRDC/YYYY/DDD/
# Example: day 120 of 2025
# Download BRDC00IGS_R_20251200000_01D_MN.rnx.gz → extract → .rnx
```

```python
# Inspect a RINEX file before using it
from nemesis_sim import rinex_summary
print(rinex_summary("BRDC00IGS_R_20251200000_01D_MN.rnx"))
# {'path': '...', 'n_records': 312, 'prns': [1,2,...,32], 'healthy': 296, ...}
```

---

## Signal Model

Full equation reference: [`docs/signal_model.md`](docs/signal_model.md)

The pseudorange model follows IS-GPS-200 §20.3.3:

```
ρ = r_geo + Δ_sagnac − c·dT_sv + Δ_iono + Δ_tropo + ε
```

| Term | Implementation |
|---|---|
| `r_geo` | Iterative signal-transit-time correction (3 iterations) |
| `Δ_sagnac` | `(Ω_E/c)(x_sv·y_u − y_sv·x_u)` |
| `c·dT_sv` | IS-GPS-200 §20.3.3.3.3 + relativistic `F·e·√A·sin(Ek)` |
| `Δ_iono` | Klobuchar (GPS week 2238 α/β) |
| `Δ_tropo` | Neill MF + GPT (hydrostatic + wet, seasonal) |

---

## Dataset Generation

```bash
python scripts/generate_dataset.py \
    --tow-start 388800 --tow-end 389800 --tow-step 100 \
    --attacks all \
    --duration-ms 1000 \
    --out-dir /data/nemesis_iq
```

Produces a labeled dataset with a `manifest.json` for direct use with the NEMESIS training pipeline.
Labels: `0=clearsky`, `1=meaconing`, `2=slow_drift`, `3=adversarial`.

---

## Testing

```bash
pytest tests/ -v --cov=nemesis_sim
```

**78 tests** covering propagator physics, C/A code properties, atmospheric models,
all three attack classes, RINEX 2/3 parsing, and end-to-end IQ generation.

---

## Comparison with gps-sdr-sim

| Feature | [gps-sdr-sim](https://github.com/osqzss/gps-sdr-sim) | nemesis-gnss-sim |
|---|---|---|
| Language | C (compile required) | Pure Python, pip installable |
| Ephemeris | RINEX only (required) | Embedded or RINEX (your choice) |
| Spoofing attacks | None | Meaconing, Slow Drift, Adversarial |
| Troposphere | Saastamoinen (fixed) | Neill MF + GPT (seasonal) |
| Test suite | None | 78 tests |
| Python API | None | Full `NEMESISSimulator` class |
| Colab ready | No | Yes |
| Labeled dataset generation | No | `scripts/generate_dataset.py` |
| Maintenance | Archived | Active |

---

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{nemesis_gnss_sim,
  author  = {Kavishka Gihan},
  title   = {{NEMESIS GNSS Simulator}},
  year    = {2025},
  url     = {https://github.com/kavishka-dot/nemesis-gnss-sim},
}
```

---

## License

MIT: see [`LICENSE`](LICENSE).

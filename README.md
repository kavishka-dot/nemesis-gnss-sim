# NEMESIS GNSS Simulator

<img width="2474" height="951" alt="image" src="https://github.com/user-attachments/assets/97a6d05a-fa5c-459b-8efe-21fc370f5eb0" />


Geodetic-accuracy GPS L1 C/A signal simulator with physically rigorous spoofing attack models,
built for the [NEMESIS](https://github.com/kavishka-dot) research pipeline.

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

### Generate all four IQ files

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

### CLI

```bash
# Clearsky
nemesis-sim --lat 6.9271 --lon 79.8612 --out clearsky.bin

# Meaconing (100 µs delay)
nemesis-sim --lat 6.9271 --lon 79.8612 --attack meaconing --meacon-delay 1e-4 --out meaconing.bin

# Slow drift (2 m/s)
nemesis-sim --lat 6.9271 --lon 79.8612 --attack slow_drift --drift-rate 2.0 --out drift.bin

# Adversarial (false position 27 km away)
nemesis-sim --lat 6.9271 --lon 79.8612 --attack adversarial \
            --false-lat 7.2 --false-lon 80.1 --out adversarial.bin
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
    --locations scripts/locations.csv \
    --tow-start 388800 --tow-end 389800 --tow-step 100 \
    --attacks all \
    --duration-ms 1000 \
    --out-dir /data/nemesis_iq
```

Produces a labeled dataset with a `manifest.json` for direct use with the NEMESIS training pipeline.

---

## Testing

```bash
pytest tests/ -v --cov=nemesis_sim
```

Key test: `tests/test_propagator.py` validates SV positions against IS-GPS-200 Appendix II
reference to < 1 mm.

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

MIT — see [`LICENSE`](LICENSE).

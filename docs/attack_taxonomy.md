# NEMESIS Attack Taxonomy

Three spoofing attack classes, matching the NEMESIS paper (IEEE ACES 2025).

## Class 1: Meaconing

**Definition:** A meaconer captures authentic GPS signals and rebroadcasts
them with a uniform time delay Δτ.

**Pseudorange model:**
$$\rho_i^\text{atk} = \rho_i^\text{true} + c \cdot \Delta\tau \quad \forall i$$

**Doppler:** Unchanged (rebroadcast at original carrier rate).

**Key NEMESIS detection signature:**
The Doppler-to-pseudorange-rate ratio is inconsistent. Authentic signals
satisfy $\dot{\rho}_i \approx -\lambda f_{d,i}$; meaconed signals violate this.

**Parameters:** `meaconing_delay_s` (typical: 1e-5 to 1e-3 s)

---

## Class 2: Slow Drift

**Definition:** Each satellite's pseudorange is corrupted with a time-linear bias:

$$b_i(t) = \dot{r}_i \cdot (t - t_0) \quad t \geq t_0$$

The corresponding Doppler is shifted consistently:
$$f_{d,i}^\text{atk} = f_{d,i}^\text{true} - \frac{\dot{r}_i \cdot f_{L1}}{c}$$

**Key NEMESIS detection signature:**
The per-SV drift rates are uniform (or follow a pattern), unlike the
independently varying natural Doppler rates of authentic signals.

**Parameters:** `drift_rate_m_s`, `drift_start_s`, optional `drift_per_sv`

---

## Class 3: Adversarial

**Definition:** The spoofer synthesises a complete GPS constellation consistent
with the victim being at a false geodetic position $(φ_f, λ_f, h_f)$.

For each visible SV, the adversarial pseudorange is:
$$\rho_i^\text{atk} = \rho_i^\text{false} = r_i(φ_f,λ_f,h_f) + \Delta_\text{iono} + \Delta_\text{tropo} + \ldots$$

**Key NEMESIS detection signature:**
Per-SV pseudorange offsets are not uniform (unlike meaconing) and not
time-linear (unlike slow drift). The wavelet-domain JEPA encoder learns
the statistical signature of this synthetic constellation geometry.

**Parameters:** `false_lat`, `false_lon`, `false_alt`

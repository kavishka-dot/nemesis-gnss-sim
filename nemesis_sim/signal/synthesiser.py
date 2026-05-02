"""
GPS L1 C/A baseband IQ synthesiser.

Generates composite complex baseband signal from a list of SVObs.
Each satellite contributes:
  - C/A code spread at 1.023 MHz chip rate
  - Navigation message at 50 bps (simplified deterministic pattern)
  - BPSK carrier at Doppler-shifted L1 frequency
  - Amplitude set by C/N₀

The composite is AWGN-corrupted at a configurable noise figure.
"""

from __future__ import annotations

import numpy as np

from ..constants import CA_CHIP_RATE, CA_CODE_LEN, L1_FREQ, NAV_BIT_RATE, PI, C
from ..observations import SVObs
from .ca_code import get_ca_code


def synthesise_iq(
    obs: list[SVObs],
    fs: float,
    duration_ms: float,
    cn0_dbhz: float = 45.0,
    add_noise: bool = True,
    noise_figure_db: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthesise composite GPS L1 C/A baseband IQ signal.

    Parameters
    ----------
    obs             : list of SVObs (visible satellites)
    fs              : sample rate (Hz)
    duration_ms     : signal duration (milliseconds)
    cn0_dbhz        : carrier-to-noise density (dB-Hz), applied per SV
    add_noise       : add AWGN if True
    noise_figure_db : receiver noise figure (dB), default 3 dB
    rng             : NumPy Generator for reproducibility; None → default_rng()

    Returns
    -------
    iq : complex128 array of shape (N,)
         Normalised so noise power ≈ noise_figure_linear.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = int(fs * duration_ms / 1e3)
    t = np.arange(N, dtype=np.float64) / fs
    iq = np.zeros(N, dtype=np.complex128)

    chips_per_sample = CA_CHIP_RATE / fs

    for sv in obs:
        ca: np.ndarray = get_ca_code(sv.prn).astype(np.float64)

        # Code phase at start of epoch (chips, mod 1023)
        code_phase = (sv.pseudorange_m / C) * CA_CHIP_RATE % CA_CODE_LEN

        # Per-sample chip index
        chip_idx = (
            np.floor(code_phase + chips_per_sample * np.arange(N))
            .astype(np.int32) % CA_CODE_LEN
        )
        ca_samples = ca[chip_idx]

        # Navigation message: 50 bps, PRN-seeded deterministic pattern
        nav = _nav_bits(sv.prn, chip_idx, t)

        # Signal amplitude from C/N₀ = A² / (2·N₀),  N₀ = 1/fs (normalised)
        A = np.sqrt(2.0 * 10.0 ** (cn0_dbhz / 10.0) / fs)

        # Carrier: Doppler + initial phase from pseudorange
        phi0 = (sv.pseudorange_m / C) * L1_FREQ * 2.0 * PI
        carrier = np.exp(1j * (2.0 * PI * sv.doppler_hz * t + phi0))

        iq += A * ca_samples * nav * carrier

    # AWGN
    if add_noise:
        nf_linear = 10.0 ** (noise_figure_db / 10.0)
        sigma = np.sqrt(nf_linear / 2.0)
        iq += sigma * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    return iq


def _nav_bits(prn: int, chip_idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Generate navigation message bit sequence (±1) for N samples.

    Uses a deterministic PRN-seeded pattern (alternating sign per bit epoch)
    to avoid the need for a full 1500-bit frame generator while preserving
    the 20 ms bit transition structure visible in the correlator output.
    """
    chips_per_bit = int(CA_CHIP_RATE / NAV_BIT_RATE)   # 20 460 chips / bit
    N = len(t)
    nav = np.empty(N, dtype=np.float64)
    for k in range(N):
        bit_idx = int(chip_idx[k] / chips_per_bit + t[k] * NAV_BIT_RATE) % 20
        nav[k] = 1.0 - 2.0 * ((prn * 7 + bit_idx) % 2)
    return nav

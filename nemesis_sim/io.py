"""
IQ file I/O helpers.

Supported formats:
  - int16 interleaved : I₀ Q₀ I₁ Q₁ … (Pluto, bladeRF, gps-sdr-sim)
  - complex64         : float32 I/Q pairs (GNU Radio, SigMF)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_int16(
    iq: np.ndarray,
    path: str | Path,
    scale: float = 2047.0,
    verbose: bool = True,
) -> None:
    """
    Save IQ as interleaved signed 16-bit integers.

    Format: I₀ Q₀ I₁ Q₁ … (little-endian)
    Compatible with ADALM-Pluto, bladeRF, HackRF, and gps-sdr-sim.

    Parameters
    ----------
    iq      : complex128 or complex64 array
    path    : output file path (.bin)
    scale   : amplitude scale factor (default 2047 → full 12-bit range)
    verbose : print save confirmation
    """
    path = Path(path)
    buf = np.empty(2 * len(iq), dtype=np.int16)
    buf[0::2] = np.clip(np.real(iq) * scale, -32768, 32767).astype(np.int16)
    buf[1::2] = np.clip(np.imag(iq) * scale, -32768, 32767).astype(np.int16)
    buf.tofile(path)
    if verbose:
        print(f"  [int16]  {path}  —  {len(iq):,} samples  ({buf.nbytes / 1e6:.2f} MB)")


def save_cf32(
    iq: np.ndarray,
    path: str | Path,
    verbose: bool = True,
) -> None:
    """
    Save IQ as interleaved float32 complex (GNU Radio / SigMF format).

    Parameters
    ----------
    iq   : complex array
    path : output file path (.cf32 or .iq)
    """
    path = Path(path)
    iq.astype(np.complex64).tofile(path)
    if verbose:
        sz = len(iq) * 8
        print(f"  [cf32]   {path}  —  {len(iq):,} samples  ({sz / 1e6:.2f} MB)")


def load_int16(
    path: str | Path,
    scale: float = 2047.0,
) -> np.ndarray:
    """
    Load interleaved int16 IQ file → complex64 array.

    Parameters
    ----------
    path  : input file path
    scale : scale factor used when saving (default 2047.0)
    """
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2 != 0:
        raise ValueError(f"File '{path}' has odd number of int16 samples.")
    iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / scale
    return iq.astype(np.complex64)


def load_cf32(path: str | Path) -> np.ndarray:
    """Load float32 complex IQ file → complex64 array."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError(f"File '{path}' has odd number of float32 values.")
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

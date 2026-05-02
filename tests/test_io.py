import numpy as np
import tempfile
from pathlib import Path
from nemesis_sim.io import save_int16, load_int16, save_cf32, load_cf32

def test_int16_io():
    iq = np.array([1.0+2.0j, -0.5-0.1j], dtype=np.complex128)
    
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = Path(f.name)
    
    try:
        save_int16(iq, path, scale=1000.0, verbose=False)
        iq_loaded = load_int16(path, scale=1000.0)
        
        np.testing.assert_allclose(np.real(iq), np.real(iq_loaded), atol=1e-3)
        np.testing.assert_allclose(np.imag(iq), np.imag(iq_loaded), atol=1e-3)
    finally:
        path.unlink(missing_ok=True)

def test_cf32_io():
    iq = np.array([1.0+2.0j, -0.5-0.1j], dtype=np.complex64)
    with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
        path = Path(f.name)
    try:
        save_cf32(iq, path, verbose=False)
        iq_loaded = load_cf32(path)
        np.testing.assert_allclose(iq, iq_loaded)
    finally:
        path.unlink(missing_ok=True)

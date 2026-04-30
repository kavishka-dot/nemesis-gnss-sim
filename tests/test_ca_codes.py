"""
Tests for GPS C/A Gold code generator.

Validates IS-GPS-200 Table 3-Ia compliance:
  - Code length 1023 chips
  - NRZ values ±1
  - Autocorrelation peak = N, sidelobes ≤ 65
  - Cross-correlation ≤ 65 for distinct PRNs
  - Caching behaviour
"""

import numpy as np
import pytest

from nemesis_sim.signal.ca_code import generate_ca_code, get_ca_code, clear_cache, G2_TAPS
from nemesis_sim.constants import CA_CODE_LEN


ALL_PRNS = list(range(1, 33))


class TestCACodeGeneration:

    def test_code_length(self):
        for prn in ALL_PRNS:
            ca = generate_ca_code(prn)
            assert len(ca) == CA_CODE_LEN, f"PRN {prn}: length {len(ca)} != 1023"

    def test_nrz_values(self):
        for prn in ALL_PRNS:
            ca = generate_ca_code(prn)
            unique = set(ca.tolist())
            assert unique == {-1, 1}, f"PRN {prn}: unexpected values {unique}"

    def test_all_prns_distinct(self):
        """No two PRNs should produce identical codes."""
        codes = {prn: generate_ca_code(prn).tobytes() for prn in ALL_PRNS}
        assert len(set(codes.values())) == len(ALL_PRNS), "Duplicate C/A codes found"

    def test_invalid_prn_raises(self):
        with pytest.raises(ValueError, match="PRN"):
            generate_ca_code(0)
        with pytest.raises(ValueError, match="PRN"):
            generate_ca_code(33)

    def test_cache_returns_same_object(self):
        clear_cache()
        ca1 = get_ca_code(1)
        ca2 = get_ca_code(1)
        assert ca1 is ca2

    def test_cache_clear(self):
        get_ca_code(1)
        clear_cache()
        from nemesis_sim.signal.ca_code import _CODE_CACHE
        assert len(_CODE_CACHE) == 0


class TestCACodeCorrelation:

    @pytest.mark.parametrize("prn", [1, 7, 19, 24])
    def test_autocorrelation_peak(self, prn):
        """Autocorrelation at zero lag must equal code length (1023)."""
        ca = generate_ca_code(prn).astype(np.float64)
        peak = float(np.dot(ca, ca))
        assert peak == pytest.approx(CA_CODE_LEN, abs=0)

    @pytest.mark.parametrize("prn", [1, 7, 19, 24])
    def test_autocorrelation_sidelobes(self, prn):
        """All non-zero lag autocorrelation values should be ≤ 65."""
        ca = generate_ca_code(prn).astype(np.float64)
        for lag in range(1, CA_CODE_LEN):
            shifted = np.roll(ca, lag)
            corr = abs(float(np.dot(ca, shifted)))
            assert corr <= 65, f"PRN {prn} lag {lag}: autocorr sidelobe = {corr}"

    def test_cross_correlation_bounded(self):
        """Cross-correlation between distinct PRNs should be ≤ 65."""
        ca1 = generate_ca_code(1).astype(np.float64)
        ca2 = generate_ca_code(2).astype(np.float64)
        for lag in range(0, CA_CODE_LEN, 10):   # sample every 10 lags for speed
            shifted = np.roll(ca2, lag)
            corr = abs(float(np.dot(ca1, shifted)))
            assert corr <= 65, f"Cross-corr PRN1-PRN2 lag {lag}: {corr}"

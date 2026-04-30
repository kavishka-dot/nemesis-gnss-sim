"""
Tests for the RINEX 2/3 navigation file parser.

Uses embedded minimal RINEX strings — no external files needed.
Validates:
  - RINEX 2 and RINEX 3 parsing produce valid SVEphemeris objects
  - Field values match known reference values
  - select_closest picks the right epoch
  - NEMESISSimulator accepts rinex_path without breaking
  - Error cases (missing file, no GPS records)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nemesis_sim.rinex import load_rinex, select_closest, rinex_summary
from nemesis_sim.almanac import SVEphemeris


# ── Minimal valid RINEX 2.11 navigation file ──────────────────────────────────
_RINEX2_CONTENT = """\
     2.11           N: GPS NAV DATA                         RINEX VERSION / TYPE
                                                            END OF HEADER
 1 25  1  4  0  0  0.0 4.824583977460D-04 8.526512829121D-12 0.000000000000D+00
    1.300000000000D+01-9.406250000000D+01 4.398997369764D-09-2.832492814207D+00
   -4.994869232178D-06 1.381169795990D-02 9.564310312271D-06 5.153630000000D+03
    3.888000000000D+05 1.117587089539D-07-1.648921278001D+00-3.166496753693D-08
    9.581489412318D-01 2.988125000000D+02-2.129848857588D+00-8.020674426610D-09
   -3.660317799066D-10 1.000000000000D+00 2.346000000000D+03 0.000000000000D+00
    2.000000000000D+00 0.000000000000D+00-1.862645149231D-09 1.300000000000D+01
    3.820200000000D+05 4.000000000000D+00
24 25  1  4  0  0  0.0-1.375228911638D-04 3.183231456205D-12 0.000000000000D+00
    8.000000000000D+00-1.409375000000D+02 4.840390418619D-09 2.047850421524D+00
   -7.264316082001D-06 6.011649919860D-03 9.246915578842D-06 5.153625000000D+03
    3.888000000000D+05-3.166496753693D-08 5.782648285341D-01 3.725290298462D-09
    9.619710346613D-01 2.805625000000D+02-7.417052745592D-01-8.024803978890D-09
   -3.077363992620D-10 1.000000000000D+00 2.346000000000D+03 0.000000000000D+00
    2.000000000000D+00 0.000000000000D+00 5.587935447693D-09 8.000000000000D+00
    3.820200000000D+05 4.000000000000D+00
"""

# ── Minimal valid RINEX 3.04 navigation file ──────────────────────────────────
_RINEX3_CONTENT = """\
     3.04           N: GNSS NAV DATA    M: Mixed            RINEX VERSION / TYPE
                                                            END OF HEADER
G01 2025 01 04 00 00 00 4.824583977460E-04 8.526512829121E-12 0.000000000000E+00
     1.300000000000E+01-9.406250000000E+01 4.398997369764E-09-2.832492814207E+00
    -4.994869232178E-06 1.381169795990E-02 9.564310312271E-06 5.153630000000E+03
     3.888000000000E+05 1.117587089539E-07-1.648921278001E+00-3.166496753693E-08
     9.581489412318E-01 2.988125000000E+02-2.129848857588E+00-8.020674426610E-09
    -3.660317799066E-10 1.000000000000E+00 2.346000000000E+03 0.000000000000E+00
     2.000000000000E+00 0.000000000000E+00-1.862645149231E-09 1.300000000000E+01
     3.820200000000E+05 4.000000000000E+00 0.000000000000E+00 0.000000000000E+00
G24 2025 01 04 00 00 00-1.375228911638E-04 3.183231456205E-12 0.000000000000E+00
     8.000000000000E+00-1.409375000000E+02 4.840390418619E-09 2.047850421524E+00
    -7.264316082001E-06 6.011649919860E-03 9.246915578842E-06 5.153625000000E+03
     3.888000000000E+05-3.166496753693E-08 5.782648285341E-01 3.729290298462E-09
     9.619710346613E-01 2.805625000000E+02-7.417052745592E-01-8.024803978890E-09
    -3.077363992620E-10 1.000000000000E+00 2.346000000000E+03 0.000000000000E+00
     2.000000000000E+00 0.000000000000E+00 5.587935447693E-09 8.000000000000E+00
     3.820200000000E+05 4.000000000000E+00 0.000000000000E+00 0.000000000000E+00
"""


def _write_temp(content: str, suffix: str = ".nav") -> Path:
    """Write content to a temporary file and return path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix,
                                      delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ── RINEX 2 tests ─────────────────────────────────────────────────────────────

class TestRINEX2:

    def setup_method(self):
        self.path = _write_temp(_RINEX2_CONTENT, ".nav")
        self.ephs = load_rinex(self.path)

    def teardown_method(self):
        self.path.unlink(missing_ok=True)

    def test_correct_number_of_records(self):
        assert len(self.ephs) == 2

    def test_prns_parsed_correctly(self):
        prns = {e.prn for e in self.ephs}
        assert prns == {1, 24}

    def test_returns_sv_ephemeris_instances(self):
        for e in self.ephs:
            assert isinstance(e, SVEphemeris)

    def test_sqrta_in_gps_range(self):
        for e in self.ephs:
            assert 5100 < e.sqrtA < 5200

    def test_eccentricity_positive_and_small(self):
        for e in self.ephs:
            assert 0 < e.e < 0.05

    def test_toe_positive(self):
        for e in self.ephs:
            assert e.toe > 0

    def test_prn1_af0_correct(self):
        prn1 = next(e for e in self.ephs if e.prn == 1)
        assert prn1.af0 == pytest.approx(4.824583977460e-4, rel=1e-6)

    def test_prn1_sqrta_correct(self):
        prn1 = next(e for e in self.ephs if e.prn == 1)
        assert prn1.sqrtA == pytest.approx(5153.63, rel=1e-4)


# ── RINEX 3 tests ─────────────────────────────────────────────────────────────

class TestRINEX3:

    def setup_method(self):
        self.path = _write_temp(_RINEX3_CONTENT, ".rnx")
        self.ephs = load_rinex(self.path)

    def teardown_method(self):
        self.path.unlink(missing_ok=True)

    def test_correct_number_of_records(self):
        assert len(self.ephs) == 2

    def test_prns_parsed_correctly(self):
        prns = {e.prn for e in self.ephs}
        assert prns == {1, 24}

    def test_sqrta_in_gps_range(self):
        for e in self.ephs:
            assert 5100 < e.sqrtA < 5200

    def test_prn1_af0_matches_rinex2(self):
        """RINEX 3 and RINEX 2 files have same PRN1 af0 — values should match."""
        r2 = load_rinex(_write_temp(_RINEX2_CONTENT))
        r3 = self.ephs
        af0_r2 = next(e.af0 for e in r2 if e.prn == 1)
        af0_r3 = next(e.af0 for e in r3 if e.prn == 1)
        assert af0_r2 == pytest.approx(af0_r3, rel=1e-6)


# ── select_closest tests ──────────────────────────────────────────────────────

class TestSelectClosest:

    def test_returns_one_per_prn(self):
        path = _write_temp(_RINEX2_CONTENT)
        ephs = load_rinex(path)
        # Add a duplicate PRN 1 with different TOE
        from dataclasses import replace
        eph_dup = SVEphemeris(**{**ephs[0].__dict__, 'toe': ephs[0].toe + 7200})
        ephs_with_dup = ephs + [eph_dup]
        selected = select_closest(ephs_with_dup, ephs[0].toe + 3600)
        prns = [e.prn for e in selected]
        assert len(prns) == len(set(prns)), "Duplicate PRNs in select_closest output"

    def test_selects_nearest_toe(self):
        path = _write_temp(_RINEX2_CONTENT)
        ephs = load_rinex(path)
        prn1 = next(e for e in ephs if e.prn == 1)
        from dataclasses import replace
        eph_far  = SVEphemeris(**{**prn1.__dict__, 'toe': prn1.toe + 7200})
        eph_near = SVEphemeris(**{**prn1.__dict__, 'toe': prn1.toe + 1000})
        target_tow = prn1.toe + 1100
        selected = select_closest([eph_far, eph_near], target_tow)
        prn1_sel = next(e for e in selected if e.prn == 1)
        assert prn1_sel.toe == eph_near.toe


# ── Error handling tests ──────────────────────────────────────────────────────

class TestErrorHandling:

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_rinex("/nonexistent/path/file.nav")

    def test_no_end_of_header_raises(self):
        path = _write_temp("     2.11           N: GPS NAV DATA\nsome data\n")
        with pytest.raises(ValueError, match="END OF HEADER"):
            load_rinex(path)
        path.unlink(missing_ok=True)

    def test_empty_file_raises(self):
        path = _write_temp("")
        with pytest.raises(ValueError):
            load_rinex(path)
        path.unlink(missing_ok=True)


# ── Integration test: simulator accepts RINEX ─────────────────────────────────

class TestSimulatorWithRINEX:

    def test_simulator_accepts_rinex_path(self):
        from nemesis_sim import NEMESISSimulator
        path = _write_temp(_RINEX2_CONTENT)
        sim = NEMESISSimulator(
            lat_deg=6.9271, lon_deg=79.8612, alt_m=10.0,
            gps_tow=388800.0, rinex_path=str(path),
        )
        obs = sim.compute_truth()
        assert isinstance(obs, list)
        path.unlink(missing_ok=True)

    def test_ephem_source_set_correctly(self):
        from nemesis_sim import NEMESISSimulator
        path = _write_temp(_RINEX2_CONTENT)
        sim = NEMESISSimulator(
            lat_deg=6.9271, lon_deg=79.8612,
            gps_tow=388800.0, rinex_path=str(path),
        )
        assert "rinex" in sim._ephem_source
        path.unlink(missing_ok=True)

    def test_embedded_source_unchanged(self):
        from nemesis_sim import NEMESISSimulator
        sim = NEMESISSimulator(lat_deg=6.9271, lon_deg=79.8612)
        assert sim._ephem_source == "embedded"

    def test_rinex_sim_generates_iq(self):
        from nemesis_sim import NEMESISSimulator
        path = _write_temp(_RINEX2_CONTENT)
        sim = NEMESISSimulator(
            lat_deg=6.9271, lon_deg=79.8612,
            gps_tow=388800.0, rinex_path=str(path),
        )
        sim.compute_truth()
        iq = sim.generate_iq(duration_ms=10.0)
        assert len(iq) > 0
        path.unlink(missing_ok=True)

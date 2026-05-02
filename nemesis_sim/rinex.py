"""
RINEX 2/3 Navigation File Parser.

Parses GPS broadcast ephemeris from RINEX 2.11 and RINEX 3.x navigation
files into SVEphemeris objects, which are drop-in replacements for the
embedded almanac.

RINEX files are available free from:
  - NASA CDDIS : https://cddis.nasa.gov/archive/gnss/data/daily/
  - IGS         : https://igs.bkg.bund.de/
  - UNAVCO      : https://data.unavco.org/

Typical filename: brdc1200.25n  (day 120 of 2025, GPS only)

Usage
-----
>>> from nemesis_sim.rinex import load_rinex
>>> ephs = load_rinex("brdc1200.25n")
>>> sim = NEMESISSimulator(..., almanac=ephs)
"""

from __future__ import annotations

from pathlib import Path

from .almanac import SVEphemeris

# ── RINEX field parser ────────────────────────────────────────────────────────

def _parse_d(s: str) -> float:
    """Parse RINEX Fortran double: '1.234D+05' → 123400.0"""
    return float(s.strip().replace("D", "e").replace("d", "e"))


def _safe_d(s: str, default: float = 0.0) -> float:
    """Parse RINEX double, return default on failure."""
    try:
        return _parse_d(s)
    except (ValueError, AttributeError):
        return default


# ── RINEX 2 parser ────────────────────────────────────────────────────────────

def _parse_rinex2_block(lines: list[str]) -> SVEphemeris | None:
    """
    Parse one RINEX 2.11 navigation record (8 lines).

    RINEX 2 record format:
      Line 0:  PRN / epoch / clock
      Lines 1-7: broadcast orbit parameters (4 fields × 19 chars each)
    """
    if len(lines) < 8:
        return None

    try:
        prn    = int(lines[0][0:2])
        # Clock parameters (line 0, cols 22-59)
        af0    = _parse_d(lines[0][22:41])
        af1    = _parse_d(lines[0][41:60])
        af2    = _parse_d(lines[0][60:79])
    except (ValueError, IndexError):
        return None

    def field(line_idx: int, field_idx: int) -> float:
        """Extract one 19-char field from broadcast orbit lines (1-indexed line)."""
        line = lines[line_idx]
        start = 3 + field_idx * 19
        end   = start + 19
        return _safe_d(line[start:end]) if end <= len(line) else 0.0

    try:
        # Line 1: IODE, Crs, deltaN, M0
        # (IODE ignored)
        Crs    = field(1, 1)
        deltaN = field(1, 2)
        M0     = field(1, 3)

        # Line 2: Cuc, e, Cus, sqrtA
        Cuc    = field(2, 0)
        e      = field(2, 1)
        Cus    = field(2, 2)
        sqrtA  = field(2, 3)

        # Line 3: toe, Cic, Omega0, Cis
        toe    = field(3, 0)
        Cic    = field(3, 1)
        Omega0 = field(3, 2)
        Cis    = field(3, 3)

        # Line 4: i0, Crc, omega, OmegaDot
        i0      = field(4, 0)
        Crc     = field(4, 1)
        omega   = field(4, 2)
        OmegaDot= field(4, 3)

        # Line 5: iDot, L2 codes (ignored), GPS week (ignored), L2P flag (ignored)
        iDot   = field(5, 0)

        # Line 6: SV accuracy (ignored), SV health, TGD, IODC
        health = int(field(6, 1))
        TGD    = field(6, 2)
        IODC   = int(field(6, 3))

        # Line 7: transmission time (ignored), fit interval (ignored)
        toc = toe   # approximate: use toe as toc

    except (ValueError, IndexError):
        return None

    return SVEphemeris(
        prn=prn, toc=toc, af0=af0, af1=af1, af2=af2, TGD=TGD,
        toe=toe, sqrtA=sqrtA, e=e, i0=i0, Omega0=Omega0, omega=omega,
        M0=M0, deltaN=deltaN, OmegaDot=OmegaDot, iDot=iDot,
        Cuc=Cuc, Cus=Cus, Crc=Crc, Crs=Crs, Cic=Cic, Cis=Cis,
        IODC=IODC, health=health,
    )


# ── RINEX 3 parser ────────────────────────────────────────────────────────────

def _parse_rinex3_block(lines: list[str]) -> SVEphemeris | None:
    """
    Parse one RINEX 3.x GPS navigation record.

    RINEX 3 format:
      Line 0:  'G{PRN} YYYY MM DD HH MM SS.S  af0  af1  af2'
      Lines 1+: '     field1  field2  field3  field4'
    """
    if len(lines) < 8:
        return None

    try:
        header = lines[0]
        prn    = int(header[1:3])
        af0    = _parse_d(header[23:42])
        af1    = _parse_d(header[42:61])
        af2    = _parse_d(header[61:80])
    except (ValueError, IndexError):
        return None

    def field3(line_idx: int, field_idx: int) -> float:
        line  = lines[line_idx]
        start = 4 + field_idx * 19
        end   = start + 19
        return _safe_d(line[start:end]) if end <= len(line) else 0.0

    try:
        Crs    = field3(1, 1);  deltaN = field3(1, 2);  M0     = field3(1, 3)
        Cuc    = field3(2, 0);  e      = field3(2, 1);  Cus    = field3(2, 2);  sqrtA  = field3(2, 3)
        toe    = field3(3, 0);  Cic    = field3(3, 1);  Omega0 = field3(3, 2);  Cis    = field3(3, 3)
        i0     = field3(4, 0);  Crc    = field3(4, 1);  omega  = field3(4, 2);  OmegaDot = field3(4, 3)
        iDot   = field3(5, 0)
        health = int(field3(6, 1))
        TGD    = field3(6, 2)
        IODC   = int(field3(6, 3))
        toc    = toe
    except (ValueError, IndexError):
        return None

    return SVEphemeris(
        prn=prn, toc=toc, af0=af0, af1=af1, af2=af2, TGD=TGD,
        toe=toe, sqrtA=sqrtA, e=e, i0=i0, Omega0=Omega0, omega=omega,
        M0=M0, deltaN=deltaN, OmegaDot=OmegaDot, iDot=iDot,
        Cuc=Cuc, Cus=Cus, Crc=Crc, Crs=Crs, Cic=Cic, Cis=Cis,
        IODC=IODC, health=health,
    )


# ── Version detector ──────────────────────────────────────────────────────────

def _detect_version(first_line: str) -> int:
    """Return RINEX major version (2 or 3) from header line."""
    try:
        ver = float(first_line[:9].strip())
        return int(ver)
    except ValueError:
        return 2   # default to 2 if unreadable


# ── Public API ────────────────────────────────────────────────────────────────

def load_rinex(path: str | Path) -> list[SVEphemeris]:
    """
    Load GPS broadcast ephemeris from a RINEX 2 or RINEX 3 navigation file.

    Parameters
    ----------
    path : path to RINEX navigation file (.nav, .rnx, .25n, .24n, etc.)

    Returns
    -------
    List of SVEphemeris objects, one per valid SV record found.
    Only GPS satellites (PRN 1–32, system identifier 'G') are returned.
    If multiple records exist for the same PRN, all are kept — the caller
    (or NEMESISSimulator) selects the most appropriate one by TOE proximity.

    Raises
    ------
    FileNotFoundError : if path does not exist
    ValueError        : if file does not appear to be a RINEX navigation file
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RINEX file not found: {path}")

    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"Empty file: {path}")

    # ── Detect version ──────────────────────────────────────────────────
    version = _detect_version(lines[0])

    # ── Find END OF HEADER ──────────────────────────────────────────────
    header_end = 0
    for i, line in enumerate(lines):
        if "END OF HEADER" in line:
            header_end = i + 1
            break

    if header_end == 0:
        raise ValueError(f"No 'END OF HEADER' found in {path}. Is this a RINEX nav file?")

    data_lines = lines[header_end:]

    # ── Parse records ───────────────────────────────────────────────────
    ephs: list[SVEphemeris] = []

    if version >= 3:
        ephs = _parse_rinex3_records(data_lines)
    else:
        ephs = _parse_rinex2_records(data_lines)

    if not ephs:
        raise ValueError(
            f"No valid GPS ephemeris records found in {path}. "
            f"Ensure this is a GPS-only or mixed RINEX navigation file."
        )

    return ephs


def _parse_rinex2_records(lines: list[str]) -> list[SVEphemeris]:
    """Split RINEX 2 data section into 8-line blocks and parse each."""
    ephs: list[SVEphemeris] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Record header: starts with 2-digit PRN in cols 0-1
        if len(line) >= 2 and line[0:2].strip().isdigit():
            block = lines[i:i+8]
            eph = _parse_rinex2_block(block)
            if eph is not None and 1 <= eph.prn <= 32:
                ephs.append(eph)
            i += 8
        else:
            i += 1
    return ephs


def _parse_rinex3_records(lines: list[str]) -> list[SVEphemeris]:
    """Split RINEX 3 data section into blocks starting with 'G' and parse each."""
    ephs: list[SVEphemeris] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # GPS record header starts with 'G' followed by 2-digit PRN
        if line.startswith("G") and len(line) > 2 and line[1:3].strip().isdigit():
            block = lines[i:i+8]
            eph = _parse_rinex3_block(block)
            if eph is not None and 1 <= eph.prn <= 32:
                ephs.append(eph)
            i += 8
        else:
            i += 1
    return ephs


# ── Ephemeris selection ───────────────────────────────────────────────────────

def select_closest(ephs: list[SVEphemeris], gps_tow: float) -> list[SVEphemeris]:
    """
    When a RINEX file contains multiple records per PRN (common — files
    contain entries every 2 hours), select the one whose TOE is closest
    to the requested GPS time of week.

    Parameters
    ----------
    ephs    : all parsed SVEphemeris records
    gps_tow : GPS time of week at simulation epoch (seconds)

    Returns
    -------
    One SVEphemeris per PRN, the one with smallest |TOE - gps_tow|.
    """
    from .constants import HALF_WEEK
    best: dict[int, SVEphemeris] = {}
    for eph in ephs:
        dt = abs(eph.toe - gps_tow)
        if dt > HALF_WEEK:
            dt = abs(dt - 2 * HALF_WEEK)
        if eph.prn not in best or dt < abs(best[eph.prn].toe - gps_tow):
            best[eph.prn] = eph
    return list(best.values())


# ── Summary ───────────────────────────────────────────────────────────────────

def rinex_summary(path: str | Path) -> dict:
    """
    Quick summary of a RINEX file without full parse.
    Returns dict with version, n_records, prns, tow_range.
    """
    ephs = load_rinex(path)
    toes = [e.toe for e in ephs]
    return {
        "path":      str(path),
        "n_records": len(ephs),
        "prns":      sorted({e.prn for e in ephs}),
        "toe_min":   min(toes),
        "toe_max":   max(toes),
        "healthy":   sum(1 for e in ephs if e.health == 0),
    }

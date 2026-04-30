"""
Physical constants and system parameters.

All values per IS-GPS-200 and WGS-84 unless otherwise noted.
"""

# ── Fundamental constants ──────────────────────────────────────────────────
C: float = 2.99792458e8          # Speed of light (m/s)
PI: float = 3.141592653589793

# ── Earth model (WGS-84) ──────────────────────────────────────────────────
MU: float         = 3.986004418e14   # Earth gravitational constant (m³/s²)
OMEGA_E: float    = 7.2921151467e-5  # Earth rotation rate (rad/s)
WGS84_A: float    = 6378137.0        # Semi-major axis (m)
WGS84_F: float    = 1 / 298.257223563
WGS84_B: float    = WGS84_A * (1 - WGS84_F)
WGS84_E2: float   = 1 - (WGS84_B / WGS84_A) ** 2   # First eccentricity squared
WGS84_EP2: float  = (WGS84_A / WGS84_B) ** 2 - 1   # Second eccentricity squared

# ── GPS signal ─────────────────────────────────────────────────────────────
L1_FREQ: float    = 1.57542e9        # L1 carrier frequency (Hz)
L2_FREQ: float    = 1.22760e9        # L2 carrier frequency (Hz)
CA_CHIP_RATE: float = 1.023e6        # C/A code chip rate (chips/s)
CA_CODE_LEN: int  = 1023             # C/A code length (chips)
NAV_BIT_RATE: float = 50.0           # Navigation message bit rate (bit/s)

# ── Relativistic correction constant (IS-GPS-200 §20.3.3.3.3) ─────────────
F_REL: float = -4.442807633e-10      # s / √m

# ── GPS time ───────────────────────────────────────────────────────────────
GPS_WEEK_SECONDS: float = 604800.0   # Seconds per GPS week
HALF_WEEK: float        = 302400.0   # Half-week rollover threshold (s)

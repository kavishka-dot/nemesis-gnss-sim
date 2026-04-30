"""
NEMESISSimulator — top-level API.

Orchestrates the full pipeline:
  1. compute_truth()   : geodetically-accurate pseudoranges for all visible SVs
  2. apply_attack(cfg) : inject spoofing attack (meaconing / slow_drift / adversarial)
  3. generate_iq()     : synthesise composite L1 C/A baseband IQ

Example
-------
>>> from nemesis_sim import NEMESISSimulator, AttackConfig
>>> sim = NEMESISSimulator(lat_deg=6.9271, lon_deg=79.8612, alt_m=10.0, gps_tow=388800.0)
>>> sim.compute_truth()
>>> sim.apply_attack(AttackConfig("slow_drift", drift_rate_m_s=2.0, drift_start_s=388800.0))
>>> iq = sim.generate_iq(duration_ms=1000.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import C, L1_FREQ
from .almanac import SVEphemeris, ALMANAC
from .observations import SVObs, compute_observations
from .attacks import AttackConfig, apply_attack
from .signal.synthesiser import synthesise_iq
from .propagator.transforms import lla_to_ecef


class NEMESISSimulator:
    """
    End-to-end GPS L1 C/A signal simulator.

    Parameters
    ----------
    lat_deg   : receiver geodetic latitude (degrees)
    lon_deg   : receiver longitude (degrees)
    alt_m     : receiver height above ellipsoid (metres)
    gps_tow   : GPS time of week at simulation epoch (seconds)
    doy       : day of year for troposphere seasonality (1–365)
    el_mask_deg : elevation mask angle (degrees)
    fs        : baseband sample rate (Hz), default 4 × chip rate
    cn0_dbhz  : carrier-to-noise density (dB-Hz) applied to all SVs
    user_vel  : receiver ECEF velocity (3,) m/s, default stationary
    rng_seed  : NumPy random seed for reproducible IQ noise
    """

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float = 0.0,
        gps_tow: float = 388800.0,
        doy: float = 180.0,
        el_mask_deg: float = 5.0,
        fs: float = 4.092e6,
        cn0_dbhz: float = 45.0,
        user_vel: Optional[np.ndarray] = None,
        rng_seed: int = 42,
        rinex_path: Optional[str] = None,
        almanac: Optional[list] = None,
    ) -> None:
        self.lat         = lat_deg
        self.lon         = lon_deg
        self.alt         = alt_m
        self.tow         = gps_tow
        self.doy         = doy
        self.el_mask     = el_mask_deg
        self.fs          = fs
        self.cn0         = cn0_dbhz
        self.user_vel    = user_vel if user_vel is not None else np.zeros(3)
        self.user_ecef   = lla_to_ecef(lat_deg, lon_deg, alt_m)
        self._rng        = np.random.default_rng(rng_seed)

        # ── Ephemeris source ──────────────────────────────────────────
        if rinex_path is not None:
            from .rinex import load_rinex, select_closest
            all_ephs = load_rinex(rinex_path)
            self._almanac      = select_closest(all_ephs, gps_tow)
            self._ephem_source = f"rinex:{rinex_path}"
            self._ephem_n      = len(all_ephs)
        elif almanac is not None:
            self._almanac      = almanac
            self._ephem_source = "custom"
            self._ephem_n      = len(almanac)
        else:
            from .almanac import ALMANAC
            self._almanac      = ALMANAC
            self._ephem_source = "embedded"
            self._ephem_n      = len(ALMANAC)

        self._truth_obs:  list[SVObs] | None = None
        self._attack_obs: list[SVObs] | None = None
        self._last_cfg:   AttackConfig | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def compute_truth(self) -> list[SVObs]:
        """
        Compute clearsky observations for all visible SVs.

        Caches the result internally; call again to force recomputation
        (e.g. after updating self.tow).
        """
        self._truth_obs = compute_observations(
            user_ecef   = self.user_ecef,
            user_vel    = self.user_vel,
            lat_deg     = self.lat,
            lon_deg     = self.lon,
            alt_m       = self.alt,
            gps_tow     = self.tow,
            doy         = self.doy,
            el_mask_deg = self.el_mask,
            almanac     = self._almanac,
        )
        self._attack_obs = None
        return self._truth_obs

    def apply_attack(self, cfg: AttackConfig) -> list[SVObs]:
        """
        Apply a spoofing attack to the cached truth observations.

        Parameters
        ----------
        cfg : AttackConfig describing the attack type and parameters

        Returns
        -------
        Attacked SVObs list (also cached internally for generate_iq).
        """
        if self._truth_obs is None:
            self.compute_truth()

        self._attack_obs = apply_attack(
            obs       = self._truth_obs,     # type: ignore[arg-type]
            cfg       = cfg,
            gps_tow   = self.tow,
            user_ecef = self.user_ecef,
            user_vel  = self.user_vel,
            lat_deg   = self.lat,
            lon_deg   = self.lon,
            alt_m     = self.alt,
            doy       = self.doy,
        )
        self._last_cfg = cfg
        return self._attack_obs

    def generate_iq(
        self,
        duration_ms: float = 1.0,
        use_attacked: bool = True,
        add_noise: bool = True,
        noise_figure_db: float = 3.0,
    ) -> np.ndarray:
        """
        Synthesise composite GPS L1 C/A baseband IQ signal.

        Parameters
        ----------
        duration_ms     : signal duration (milliseconds)
        use_attacked    : use attack observations if available, else truth
        add_noise       : add AWGN
        noise_figure_db : receiver noise figure (dB)

        Returns
        -------
        complex128 ndarray of shape (N,) at self.fs sample rate
        """
        if use_attacked and self._attack_obs is not None:
            obs = self._attack_obs
        elif self._truth_obs is not None:
            obs = self._truth_obs
        else:
            obs = self.compute_truth()

        return synthesise_iq(
            obs             = obs,
            fs              = self.fs,
            duration_ms     = duration_ms,
            cn0_dbhz        = self.cn0,
            add_noise       = add_noise,
            noise_figure_db = noise_figure_db,
            rng             = self._rng,
        )

    def summary(self, use_attacked: bool = False) -> dict:
        """
        Return a JSON-serialisable summary of the current simulation state.
        """
        obs = (self._attack_obs if use_attacked else self._truth_obs) or []
        return {
            "location":  {"lat": self.lat, "lon": self.lon, "alt_m": self.alt},
            "ephemeris": {"source": self._ephem_source, "n_svs": self._ephem_n},
            "gps_tow":   self.tow,
            "doy":       self.doy,
            "fs_mhz":    self.fs / 1e6,
            "cn0_dbhz":  self.cn0,
            "n_visible": len(obs),
            "attack":    self._last_cfg.attack_type if self._last_cfg else "none",
            "satellites": [
                {
                    "prn":           sv.prn,
                    "elevation_deg": round(sv.el_deg, 4),
                    "azimuth_deg":   round(sv.az_deg, 4),
                    "pseudorange_m": round(sv.pseudorange_m, 4),
                    "range_true_m":  round(sv.range_true_m, 4),
                    "doppler_hz":    round(sv.doppler_hz, 4),
                    "clock_err_m":   round(sv.clock_err_m, 6),
                    "iono_m":        round(sv.iono_m, 4),
                    "tropo_m":       round(sv.tropo_m, 4),
                    "sagnac_m":      round(sv.sagnac_m, 6),
                }
                for sv in obs
            ],
        }

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def n_visible(self) -> int:
        """Number of visible satellites (requires compute_truth() first)."""
        return len(self._truth_obs) if self._truth_obs else 0

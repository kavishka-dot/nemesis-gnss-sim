# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2025-01-01

### Added
- Full IS-GPS-200 Keplerian propagator with 16 orbital elements and 6 harmonic corrections
- Satellite clock model: af0 + af1·tk + af2·tk² + relativistic periodic term
- Sagnac / Earth-rotation correction during signal transit
- Klobuchar ionospheric delay with GPS week 2238 broadcast α/β coefficients
- Neill Mapping Function troposphere with GPT standard atmosphere
- Correct C/A Gold code generator for all 32 PRNs (IS-GPS-200 Table 3-Ia)
- BPSK(1) baseband IQ synthesis at arbitrary sample rate
- Three NEMESIS attack classes: Meaconing, Slow Drift, Adversarial
- int16 and complex64 output formats
- CLI entry point `nemesis-sim`
- Batch dataset generation script

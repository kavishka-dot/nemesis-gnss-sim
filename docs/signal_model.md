# GPS L1 C/A Signal Model

Full IS-GPS-200 compliant pseudorange measurement model implemented in `nemesis_sim`.

## Pseudorange Equation

$$\rho = r_\text{geo} + \Delta_\text{Sagnac} - c \cdot dT_{sv} + \Delta_\text{iono} + \Delta_\text{tropo}$$

| Term | Module | Reference |
|---|---|---|
| $r_\text{geo}$ | `propagator/kepler.py` | IS-GPS-200 §20.3.3.4.3 |
| $\Delta_\text{Sagnac}$ | `propagator/transforms.py` | IS-GPS-200 §20.3.3.4.3 (iv) |
| $c \cdot dT_{sv}$ | `propagator/clock.py` | IS-GPS-200 §20.3.3.3.3 |
| $\Delta_\text{iono}$ | `atmosphere/klobuchar.py` | IS-GPS-200 §20.3.3.5.2.5 |
| $\Delta_\text{tropo}$ | `atmosphere/troposphere.py` | Neill (1996) |

## Satellite Clock

$$dT_{sv} = a_{f0} + a_{f1}(t - t_{oc}) + a_{f2}(t - t_{oc})^2 + \Delta T_{rel} - T_{GD}$$

$$\Delta T_{rel} = F \cdot e \cdot \sqrt{A} \cdot \sin E_k$$

where $F = -4.44280763 \times 10^{-10}$ s/√m.

## Sagnac Correction

$$\Delta_\text{Sagnac} = \frac{\Omega_E}{c}(x_{sv} y_u - y_{sv} x_u)$$

Accounts for Earth's rotation during the ~67 ms signal transit.
Magnitude: ±30 m depending on SV-receiver geometry.

## Klobuchar Ionosphere

Standard GPS broadcast model with 8 coefficients (α₀–α₃, β₀–β₃).
Accuracy: ~50% of actual ionospheric delay under nominal conditions.
Embedded coefficients from GPS week 2238.

## Neill Mapping Function

$$\tau_\text{tropo} = ZHD \cdot m_h(\epsilon) + ZWD \cdot m_w(\epsilon)$$

where ZHD is the Saastamoinen zenith hydrostatic delay and ZWD uses
the Askne-Nordius approximation. Meteorological inputs from GPT (latitude + seasonal).

## Attack Models

See [`attack_taxonomy.md`](attack_taxonomy.md) for the NEMESIS attack class definitions.

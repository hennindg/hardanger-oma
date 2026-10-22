# Hardanger Bridge – reference data (checked against the source papers)

Literature values for the Hardanger Bridge used in this project. Every number was checked on
2026-10-14 against the source papers listed in Section 8 (the papers themselves are not included;
see the references). An earlier, unchecked compilation of these values contained errors; where a
value was corrected, it is marked **FIXED** and the wrong value is given, so others using the same
numbers can spot them.

Legend:
- ✔ checked, matches the paper
- **FIXED** was wrong in the earlier compilation, corrected here (wrong value given)
- ⚠ NOT in any of the papers (estimate or assumption) – do not use as data

---

## 1. Geometry

| Quantity | Value | Source | Check |
|---|---|---|---|
| Main span L | 1310 m | Øiseth et al. 2010; Petersen & Øiseth 2017 | ✔ (Fenerci & Øiseth 2018 say 1308 m) |
| Total length | 1380 m | Øiseth et al. 2010 | ✔ |
| Deck width B | 18.3 m | all papers | ✔ |
| Deck depth D | 3.33 m (2010 design paper) / **3.25 m (as built)** | Øiseth 2010 p. 735; Siedziako 2017 Fig. 1d; Banday 2024 Fig. 1. Data paper rounds to 3.2 m | ✔ both genuine; use 3.25 m for the as-built bridge, 3.33 m when using Øiseth 2010 coefficients |
| Distance between cable planes | 14.5 m | Øiseth 2010; Banday 2024 | ✔ |
| Tower height | 186 m | Øiseth 2010 | ✔ |
| Clearance under girder | 55 m | Øiseth 2010 ("headroom under the girder") | ✔ |
| Girder | steel closed box, partially streamlined, guide vanes | Øiseth 2010; Siedziako 2017 | ✔ |
| Cross-section dimensions (mm) | 1120, 4500, 4500, 430, 3250, 700 along the top; 3250 depth; 18300 width | Banday 2024 Fig. 1 | **FIXED**: these are horizontal layout dimensions of the drawing, not "plate widths". Plate thicknesses are not given |
| Hanger spacing | – | – | ⚠ 20 m in the earlier compilation is NOT in the papers |

## 2. Mass and material properties

### 2.1 Modal equivalent mass (Øiseth et al. 2010, Table 2, design-stage ALVSAT model) ✔

Øiseth et al. (2010, p. 734) explain that the main cables and the girder edges move together vertically
and the towers' mass is negligible, so the modal equivalent mass (and mass moment of inertia) is the
same for all vertical (torsional) modes.

| Quantity | Value | Check |
|---|---|---|
| Vertical modal equivalent mass m̃_z | **12,820 kg/m** (all vertical modes, includes cables) | ✔ |
| Torsional modal equivalent mass moment of inertia m̃_θ | **426,000 kg·m²/m** (all torsional modes) | ✔ |
| Radius of gyration √(m̃_θ / m̃_z) | ≈ 5.8 m (arithmetic) | ✔ |
| Horizontal modal equivalent mass | 9,400–12,480 kg/m (modes 1, 2, 5, 11, 15, 16); modes 8 and 9: 34,550 and 34,190 kg/m | ✔ |

### 2.2 FE model updating (Petersen & Øiseth 2017, Table 3) ✔

| Parameter | Unit | Initial | Updated | Change |
|---|---|---|---|---|
| E-modulus girder | MPa | 2.100e5 | 2.205e5 | +5.0 % |
| Density girder | kg/m³ | 7850 | 7585 | −3.4 % |
| Poisson ratio girder | – | 0.300 | 0.251 | −16.2 % |
| E-modulus main cable | MPa | 2.000e5 | 1.890e5 | −5.5 % |
| Density main cable | kg/m³ | 7850 | 7535 | −4.0 % |
| E-modulus pylon | MPa | 4.000e4 | 3.876e4 | −3.1 % |
| Density pylon | kg/m³ | 2500 | 2269 | −9.2 % |
| Non-structural mass girder | kg/m² | 220 | 188 | −14.5 % |
| Spring side, vertical | N/m | 1.000e8 | 7.787e9 | +7687 % |
| Spring centre, longitudinal | N/m | 1.000e9 | 2.319e9 | +132 % |
| Spring centre, vertical | N/m | 1.000e8 | 5.576e9 | +5476 % |
| Spring centre, rotation | Nm/rad | 1.000e9 | 1.080e7 | −98.9 % |
| Hanger tension | kN | 903.9 | 931.4 | +3.0 % |

⚠ The earlier compilation's estimate "as-built equivalent mass ~12.0–12.5 t/m" is the author's own
reasoning (updated densities are lower), not a value from a paper.

### 2.3 Rough cross-checks – ⚠ none of these are from the papers

| Estimate | Value | Problem |
|---|---|---|
| Non-structural mass per length | ≈ 3,440 kg/m | arithmetic 188 kg/m² × 18.3 m; assumes it acts over the full width |
| Deck mass from hanger tension | ≈ 9,500 kg/m | ⚠ needs the hanger spacing, which is not in the papers |
| Section-model scaled mass | ≈ 11,350 kg/m | ⚠ assumes the 7.72 kg / 1.7 m model (Siedziako 2017) is mass-scaled, which is not stated. Øiseth 2010 gives a different section-model mass (8793 g, Hansen 2006 tests) without its length. Do not use |
| Cable mass | ≈ 3,300 kg/m | ⚠ derived from the uncertain values above |

## 3. Identified modes from measured data (Petersen & Øiseth 2017) ✔

### 3.1 System identification settings ✔
- 20 triaxial accelerometers (60 channels) + 9 anemometers.
- 120 min recording, 13 January 2016, mean wind 3–6 m/s.
- Low-pass Chebyshev type I, cut-off 2 Hz; resampled 200 Hz → 5 Hz.
- Cov-SSI, max time lag 20 s (100 time steps), stabilization diagram up to order 200.
- 30 modes, 0.05–1.31 Hz.

### 3.2 Identified modes and FE comparison (Tables 1–2) ✔

Stored as data in `literature/petersen_oiseth_2017_modes.csv` (identified frequencies and types).

| Mode | Type | f identified [Hz] | f initial FE [Hz] (err) | f updated FE [Hz] (err) | MAC initial | MAC updated |
|---|---|---|---|---|---|---|
| 1 | Horizontal | 0.052 | 0.051 (−1.67 %) | 0.051 (−1.48 %) | 0.999 | 0.999 |
| 2 | Horizontal | 0.105 | 0.101 (−3.82 %) | 0.105 (−0.86 %) | 0.993 | 0.992 |
| 3 | Vertical | 0.119 | 0.110 (−7.44 %) | 0.111 (−6.57 %) | 0.988 | 0.988 |
| 4 | Vertical | 0.142 | 0.141 (−0.91 %) | 0.141 (−0.58 %) | 0.994 | 0.993 |
| 5 | Horizontal | 0.183 | 0.177 (−3.26 %) | 0.186 (1.28 %) | 0.993 | 0.994 |
| 6 | Vertical | 0.206 | 0.200 (−2.97 %) | 0.201 (−2.55 %) | 0.990 | 0.990 |
| 7 | Vertical | 0.212 | 0.210 (−0.93 %) | 0.212 (−0.35 %) | 0.996 | 0.996 |
| 8 | Cable/horizontal | 0.230 | 0.234 (1.70 %) | 0.236 (2.84 %) | 0.962 | 0.975 |
| 9 | Vertical | 0.276 | 0.274 (−0.61 %) | 0.277 (0.29 %) | 0.998 | 0.998 |
| 10 | Horizontal | 0.318 | 0.306 (−3.71 %) | 0.320 (0.64 %) | 0.991 | 0.990 |
| 11 | Vertical | 0.333 | 0.328 (−1.49 %) | 0.332 (−0.44 %) | 0.993 | 0.993 |
| 12 | Torsion | 0.374 | 0.358 (−4.36 %) | 0.374 (−0.05 %) | 0.964 | 0.963 |
| 13 | Vertical | 0.401 | 0.396 (−1.30 %) | 0.401 (0.10 %) | 0.998 | 0.998 |
| 14 | Cable/horizontal | 0.418 | 0.387 (−7.33 %) | 0.395 (−5.42 %) | 0.773 | 0.903 |
| 15 | Horizontal | 0.464 | 0.449 (−3.24 %) | 0.464 (0.04 %) | 0.691 | 0.700 |
| 16 | Vertical | 0.471 | 0.459 (−2.68 %) | 0.466 (−1.09 %) | 0.983 | 0.985 |
| 17 | Pylon | 0.516 | 0.501 (−2.91 %) | 0.512 (−0.74 %) | 0.606 | 0.866 |
| 18 | Pylon | 0.529 | 0.508 (−4.10 %) | 0.519 (−2.01 %) | 0.740 | 0.804 |
| 19 | Vertical | 0.547 | 0.535 (−2.26 %) | 0.546 (−0.35 %) | 0.998 | 0.998 |
| 20 | Torsion | 0.560 | 0.512 (−8.57 %) | 0.558 (−0.35 %) | 0.974 | 0.975 |
| 21 | Vertical | 0.628 | 0.610 (−2.75 %) | 0.627 (−0.17 %) | 0.959 | 0.851 |
| 22 | Vertical | 0.715 | 0.695 (−2.88 %) | 0.712 (−0.37 %) | 0.997 | 0.995 |
| 23 | Vertical | 0.808 | 0.768 (−4.86 %) | 0.790 (−2.17 %) | 0.944 | 0.963 |
| 24 | Torsion | 0.828 | 0.769 (−7.07 %) | 0.837 (1.10 %) | 0.986 | 0.978 |
| 25 | Vertical | 0.905 | 0.870 (−3.81 %) | 0.896 (−0.97 %) | 0.995 | 0.995 |
| 26 | Vertical | 1.007 | 1.015 (0.83 %) | 0.967 (−3.91 %) | 0.436 | 0.678 |
| 27 | Torsion | 1.069 | 0.991 (−7.29 %) | 1.068 (−0.12 %) | 0.992 | 0.993 |
| 28 | Vertical | 1.112 | 1.066 (−4.15 %) | 1.101 (−0.96 %) | 0.990 | 0.988 |
| 29 | Vertical | 1.227 | 1.184 (−3.53 %) | 1.235 (0.60 %) | 0.912 | 0.764 |
| 30 | Torsion | 1.314 | 1.221 (−7.09 %) | 1.311 (−0.18 %) | 0.979 | 0.989 |

Lowest horizontal, vertical and torsional modes: 0.052, 0.119 and 0.374 Hz (Fig. 4). ✔

## 4. Design-stage modal model (Øiseth et al. 2010, Table 2, ALVSAT) ✔

Mode numbering differs from Section 3. f = ω / 2π (conversions checked). Types from Fig. 4 and
the text ("the first 8 horizontal, 8 vertical and 8 torsional").
The damping ratios are **design assumptions** (the bridge was under construction), not measurements.

| Mode | Type | ω [rad/s] | f [Hz] | Damping ξ | Modal equivalent mass / inertia |
|---|---|---|---|---|---|
| 1 | Horizontal | 0.32 | 0.051 | 0.005 | 10,470 kg/m |
| 2 | Horizontal | 0.62 | 0.099 | 0.006 | 9,400 kg/m |
| 3 | Vertical | 0.69 | 0.110 | 0.005 | 12,820 kg/m |
| 4 | Vertical | 0.89 | 0.142 | 0.006 | 12,820 kg/m |
| 5 | Horizontal | 1.06 | 0.169 | 0.007 | 9,510 kg/m |
| 6 | Vertical | 1.27 | 0.202 | 0.007 | 12,820 kg/m |
| 7 | Vertical | 1.34 | 0.213 | 0.007 | 12,820 kg/m |
| 8 | Horizontal | 1.49 | 0.237 | 0.009 | 34,550 kg/m (⚠ "cable-dominated" is an interpretation, not stated) |
| 9 | Horizontal | 1.57 | 0.250 | 0.010 | 34,190 kg/m (⚠ same) |
| 10 | Vertical | 1.74 | 0.277 | 0.010 | 12,820 kg/m |
| 11 | Horizontal | 1.86 | 0.296 | 0.015 | 9,710 kg/m |
| 12 | Vertical | 2.10 | 0.334 | 0.015 | 12,820 kg/m |
| 13 | Torsional | 2.23 | 0.355 | 0.005 | 426,000 kg·m²/m |
| 14 | Vertical | 2.53 | 0.403 | 0.020 | 12,820 kg/m |
| 15 | Horizontal | 2.49 | 0.396 | 0.020 | 12,480 kg/m |
| 16 | Horizontal | 2.63 | 0.419 | 0.025 | 11,020 kg/m |
| 17 | Vertical | 2.97 | 0.473 | 0.025 | 12,820 kg/m |
| 18 | Torsional | 3.37 | 0.536 | 0.006 | 426,000 kg·m²/m |
| 19 | Torsional | 5.10 | 0.812 | 0.009 | 426,000 kg·m²/m |
| 20 | Torsional | 6.76 | 1.076 | 0.012 | 426,000 kg·m²/m |
| 21 | Torsional | 8.45 | 1.345 | 0.015 | 426,000 kg·m²/m |
| 22 | Torsional | 10.14 | 1.614 | 0.020 | 426,000 kg·m²/m |
| 23 | Torsional | 11.83 | 1.883 | 0.025 | 426,000 kg·m²/m |
| 24 | Torsional | 13.515 | 2.151 | 0.030 | 426,000 kg·m²/m |

Siedziako & Øiseth 2017 assumed lognormal modal damping, mean 0.5 %, CoV 0.4 (citing
Davenport & Larose 1989), for the flutter analysis. ✔

## 5. Aerodynamic properties

### 5.1 Static force coefficients (Hansen et al. 2006, given in Øiseth 2010 below Eq. 16)

| C̄_D | C′_D | C̄_L | C′_L | C̄_M | C′_M |
|---|---|---|---|---|---|
| 0.70 | 0 | **−0.25** | 2.4 | 0.01 | 0.74 |

- **FIXED**: C̄_L is negative (earlier compilation: 0.25 ⚠). C̄_M = +0.01 is correct.
- **ADDED**: reference lengths. C_D is referenced to the depth **D**, C_L to the width B, C_M to
  B². Every equation contains (D/B)·C̄_D (Øiseth 2010, Eqs. 7, 11, 16).
- Quasi-steady heave (vertical) damping force: q_z = ½ρV²B · (−(C′_L + (D/B)C̄_D)) · ż/V (Eq. 11),
  so the effective coefficient is C′_L + (D/B)C̄_D = 2.4 + (3.33/18.3)·0.70 = **2.53**.

### 5.2 Modified quasi-steady coefficients (Øiseth 2010, Eqs. 13–14, 18)

Aerodynamic derivatives approximated as X*ᵢ(K) = xᵢ·(1/K) for i = **1, 2, 5** and
xᵢ·(1/K)² for i = **3, 4, 6** (**FIXED**: the earlier compilation listed only i = 1, 2 and 3, 4).

| h1 | h2 | h3 | h4 | a1 | a2 | a3 | a4 |
|---|---|---|---|---|---|---|---|
| **−2.734** | 0.206 | 2.271 | **−0.208** | **−0.823** | **−0.258** | 0.726 | **−0.037** |

**FIXED**: five minus signs were lost in the earlier compilation (h1, h4, a1, a2, a4).
The heave damping term in Eq. 14 is h1·ż/V, so −h1 = **2.73** is the modified quasi-steady
counterpart of 2.53 in Section 5.1.

Most important for the stability limit: **A1*, A2*, A3* and, to some extent, H1*** (Øiseth 2010,
Section 3 and Table 5). **FIXED**: the earlier compilation also listed H3*, which the paper does not name
(Table 5 shows a moderate ±9 % effect).

### 5.3 Stability limit ✔

| Source | Critical flutter speed |
|---|---|
| Øiseth 2010, aerodynamic derivatives, modes 4, 6, 13 | 78 m/s (ω_cr = 1.63 rad/s) |
| Øiseth 2010, aerodynamic derivatives, all 24 modes | 79 m/s (ω_cr = 1.63 rad/s) |
| Øiseth 2010, classical quasi-steady, modes 4, 6, 13 | 44 m/s (underestimates by > 40 %) |
| Øiseth 2010, modified quasi-steady, modes 4, 6, 13 | 74 m/s |
| Siedziako 2017, free-vibration ADs (with details) | 80.93 ± 5.48 m/s |
| Siedziako 2017, forced-vibration ADs, no details | 61.36 ± 2.00 m/s |
| Siedziako 2017, forced-vibration ADs, with details | 70.05 ± 2.79 m/s |
| Siedziako 2017, **measured** in free-vibration test, with / without details | 79.5 / 71.6 m/s (**ADDED**) |
| Design 10-min mean wind, 50-year return period | 38 m/s |

## 6. Wind field

### 6.1 Design spectra (Øiseth 2010, Eq. 15) ✔

ω circular frequency, V mean wind, z height, κ = 0.0031 (roughness), Δx separation:

- S_uu(ω) = 40.58 · V z κ / (1 + 9.74 ω z / V)^(5/3) · exp(−1.4 Δx ω / V)
- S_ww(ω) = 0.82 · V z κ / (1 + 0.79 ω z / V)^(5/3) · exp(−Δx ω / V)
- S_uw(ω) = **−**2.23 · V z κ / (1 + 1.67 ω z / V)^(7/3) · exp(−Δx ω / V)  (negative sign confirmed)

Air density ρ = 1.25 kg/m³. ✔

### 6.2 Long-term site wind model (Fenerci & Øiseth 2018; used by Banday et al. 2024) ✔

Checked against Fenerci & Øiseth (2018) on 2026-10-14.

- Data: anemometers on the hangers, 8 m above the girder, monitored since December 2013;
  **10-min averaging**, 15,386 10-min recordings, recordings below 3 m/s discarded. One-point
  statistics from the midspan sensor A6; two-point statistics from the sensor pairs A3–A4, A4–A5.
- Turbulence spectra (Eq. 5): Kaimal type, S_n f / σ_n² = A_n f_z / (1 + 1.5 A_n f_z)^(5/3),
  f_z = f z / U, with z = 68 m; normalized co-spectra of Davenport type C_n = exp(−K_n f Δx / U), n = u, w.
- The six parameters (σ_u, σ_w, A_u, A_w, K_u, K_w) are **lognormal**. **μ̃ and σ̃ are the mean and
  standard deviation of ln(x)** (Eq. 6 and the text below it). This resolves the ⚠ of the earlier compilation
  file: the median σ_u is exp(0.122 + 0.039 U), e.g. 1.67 m/s at 10 m/s (I_u ≈ 17 %) and 2.46 m/s at
  20 m/s (I_u ≈ 12 %).
- The parameters were fitted to data above 10 m/s (the statistics stabilise there). The paper
  does not cover lower wind speeds well.

Final lognormal parameters (Table 5) and correlation coefficients (Table 6):

| | σ_u | σ_w | A_u | A_w | K_u | K_w |
|---|---|---|---|---|---|---|
| East μ̃ | 0.122 + 0.039U | −0.657 + 0.032U | 2.67 + 0.0248U | 0.7076 | 1.9385 | 1.7932 |
| East σ̃ | 0.2566 | 0.2632 | 0.4538 | 0.4466 | 0.2652 | 0.3423 |
| West μ̃ | 0.122 + 0.039U | −0.657 + 0.032U | 2.407 + 0.048U | 1.2075 | 2.1093 | 2.1633 |
| West σ̃ | 0.3159 | 0.3021 | 0.5282 | 0.4943 | 0.268 | 0.3322 |

Non-zero correlations (Table 6), East / West: σ_u–σ_w 0.7608 / 0.8148; σ_u–A_u 0.2641 / 0.4087;
σ_w–A_w 0.2571 / 0.2851; A_u–A_w 0.1633 / 0.3065; K_u–K_w 0.3261 / 0.4725. All other pairs set to 0.

Notes:
- Banday et al. (2024) use σ̃ = 0.28 for σ_u. That is the value in Fenerci & Øiseth's **Table 4**
  (conditional statistics, same for east and west); the **final** Table 5 gives 0.2566 (east) and
  0.3159 (west). Both values are in the paper.
- ⚠ The Weibull distribution of the mean wind speed (λ = 5.1941, k = 1.7946) is **not** in
  Fenerci & Øiseth (2018); this paper only gives distributions conditional on U. The Weibull
  values are from Banday et al. (2024), Table 1, who cite the 2018 paper for the model in general.
- Design 10-min mean wind speed (50-year return period): Fenerci & Øiseth (2018) give **39 m/s**,
  Øiseth et al. (2010) give **38 m/s**. Both cite the Statens vegvesen (2006) design basis.

Natural frequencies from an FE model (Fenerci & Øiseth 2018, Table 7, citing Fenerci & Øiseth 2017):
0.05 Hz symmetric lateral, 0.098 Hz asymmetric lateral, 0.11 Hz **asymmetric vertical**, 0.14 Hz
symmetric vertical, 0.36 Hz symmetric torsional. ✔ (confirms that the ~0.12 Hz mode is the first
antisymmetric vertical mode, as found in our data)

Damage-relevant wind speeds for buffeting fatigue: 5–15 m/s (Banday 2024, Fig. 2). ✔

## 7. Buffeting response (for sanity checks of measured RMS)

Values read from figures, so approximate:
- Siedziako 2017, Fig. 5, midspan std at 50 m/s (first 20 modes): horizontal ≈ 0.9–1.1 m
  (forced-vibration data; the free-vibration curve is extrapolated), vertical ≈ 0.55–0.62 m,
  torsional ≈ 0.013–0.016 rad. ✔
- Øiseth 2010, Fig. 14, std at 40 m/s: vertical ≈ **0.36 m at midspan** (maximum ≈ 0.45 m at
  x/L ≈ 0.2 and 0.8), torsional ≈ **0.007–0.009 rad** at midspan. **FIXED** (earlier compilation: 0.4 m and 0.01 rad).
- Banday 2024, Table 2: 100-year fatigue damage from buffeting 0.0066 / 0.0052 / 0.003 at points
  A / B / C. ✔

## 8. Sources (all checked)

1. Øiseth, O., Rönnquist, A. & Sigbjörnsson, R. (2010). Simplified prediction of wind-induced
   response and stability limit of slender long-span suspension bridges, based on modified
   quasi-steady theory: A case study. J. Wind Eng. Ind. Aerodyn. 98, 730–741.
   Erratum: JWEIA 99 (2011) 894 – only adds a missing reference, no numbers change.
   File: `1-s2.0-S0167610510000711-mainext.pdf`.
2. Petersen, Ø.W. & Øiseth, O. (2017). Finite element model updating of a long span suspension
   bridge. ICESD 2017, Reykjavík (also Springer 2019). File: `AID_24.pdf`.
3. Siedziako, B. & Øiseth, O. (2017). On the importance of cross-sectional details in the wind
   tunnel testing of bridge deck section models. Procedia Eng. 199, 3145–3151 (EURODYN 2017).
   File: `1-s2.0-S1877705817340924-main.pdf`.
4. Banday, Z.Z., Fenerci, A., Lystad, T.M. & Øiseth, O. (2024). Fatigue life estimation for long
   span suspension bridges due to buffeting. IABMAS 2024, pp. 2747–2754. File: `Buffeting.pdf`.
5. Fenerci, A. & Øiseth, O. (2018). Site-specific data-driven probabilistic wind field modeling
   for the wind-induced response prediction of cable-supported bridges. J. Wind Eng. Ind.
   Aerodyn. 181, 161–179. File: `1-s2.0-S0167610518304252-main.pdf`. Note: this paper gives the
   main span as **1308 m**; Øiseth 2010 and Petersen & Øiseth 2017 give 1310 m.
6. Fenerci, A. et al. (2021). Data set from long-term wind and acceleration monitoring of the
   Hardanger Bridge. J. Struct. Eng. 147(5), 04721003 (cited by Banday 2024).

Not used: Fenerci, Lystad & Øiseth (2023), JWEIA 242, 105583 – Gjemnessund Bridge, a different
bridge. File: `1-s2.0-S0167610523002866-main.pdf`.

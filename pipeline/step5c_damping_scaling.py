"""
Hardanger Bridge project - Step 5c
Check if the wind-induced increase of damping follows the quasi-steady U/f scaling.

Quasi-steady theory for vertical (heave) motion: a vertical deck velocity z' changes the
angle of attack by z'/U, which gives a lift + drag force proportional to -U z'. That is an
aerodynamic damping force. For one mode, with uniform wind and mass along the span:

    zeta_aero = rho * B * C_heave * U / (8 * pi * m * f),    C_heave = CL' + (D/B) * CD

  rho = air density, B = deck width, D = deck depth, CL' = dCL/dalpha (lift slope, referred
  to B), CD = drag coefficient (referred to D), m = vertical modal equivalent mass per unit
  length, f = natural frequency. The form of the force is Eq. 11 in Oiseth et al. (2010).

So the slope d(zeta)/dU should be K / f, with the SAME K = rho B C_heave / (8 pi m) for all
vertical modes. This script checks that in two ways:
  1. slope * f should be constant (fit K through the origin of slope vs. 1/f),
  2. a power law  slope = C * f^p  fitted in log-log should give p close to -1.
Then K is turned into a measured C_heave and compared with the wind tunnel values.

Needs results/step5/regression_wind_temperature.csv from pipeline/step5b_wind_temperature.py.

How to use:
  python -m pipeline.step5c_damping_scaling
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

# results\step5_all = all data sets (1,485 recordings); results\step5 = Sept 2014 + storm Urd only
INPUT_FILE = RESULTS_FOLDER / "step5_all" / "regression_wind_temperature.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

# Damping slope to use: "damp_b_pt_per_10ms" (wind + temperature model, step 5b)
# or "damp_b_wind_only_pt_per_10ms" (wind only). Step 5b showed they are almost equal.
SLOPE_COLUMN = "damp_b_pt_per_10ms"

# Modes left out of the fit (median frequency in Hz, matched within 0.003 Hz).
# 0.117 Hz: the doubtful vertical mode (very high damping, frequency drops with wind,
# maybe two close modes mixed together). It is still drawn, as a hollow marker.
EXCLUDE_FREQUENCIES = [0.117]

# Physical constants for converting K into C_heave.
# All values below are checked against the papers, see literature/reference_values.md
AIR_DENSITY = 1.25          # kg/m^3, Oiseth et al. (2010) p. 735
DECK_WIDTH = 18.3           # m, all papers

# Vertical modal equivalent evenly distributed mass, in kg/m. Oiseth et al. (2010), Table 2:
# 12,820 kg/m for ALL vertical modes, because the main cables and the girder edges move together
# vertically (so it includes the cables). Design-stage model (ALVSAT); the updated FE model of
# Petersen & Oiseth (2017) has 3-15 % lower densities, so the as-built value is probably a little
# lower. None = only report K and C_heave per unit mass.
MASS_PER_LENGTH = 12820

# Wind tunnel values to compare with (Hansen et al. 2006 tests, given in Oiseth et al. 2010):
LIFT_SLOPE = 2.4            # CL' (referred to B), below Eq. 16
DRAG_COEFFICIENT = 0.70     # mean CD (referred to D), below Eq. 16
DECK_DEPTH_2010 = 3.33      # m, the depth used with these coefficients in Oiseth et al. (2010)
MQS_H1 = -2.734             # modified quasi-steady h1 (fitted to the wind tunnel H1*), Eq. 18;
                            # the heave damping term is h1 * z'/V, so -h1 is comparable to C_heave

# ==================================================================


def read_slopes(path):
    """Read the damping slopes from step 5b. Returns a list of dictionaries, one per mode."""
    modes = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index(SLOPE_COLUMN)
        for row in reader:
            modes.append({
                "track": int(row[0]),
                "f": float(row[1]),
                "type": row[2],
                "slope": float(row[col]),         # %-points per 10 m/s
                "se": float(row[col + 1]),        # its standard error (next column)
            })
    return modes


def is_excluded(f):
    """True if this mode frequency is in EXCLUDE_FREQUENCIES."""
    return any(abs(f - fx) < 0.003 for fx in EXCLUDE_FREQUENCIES)


def fit_k_over_f(f, slope, se):
    """
    Weighted least squares fit of slope = K / f (a straight line through the origin in
    slope vs. 1/f). Weights 1 / se^2, so precise slopes count more.
    Returns K, its standard error, and the reduced chi-square
    (about 1 = the scatter around K/f is as large as the error bars predict).
    """
    x = 1 / f
    w = 1 / se ** 2
    K = np.sum(w * x * slope) / np.sum(w * x ** 2)
    se_K = 1 / np.sqrt(np.sum(w * x ** 2))
    chi2_reduced = np.sum(w * (slope - K * x) ** 2) / (len(f) - 1)
    return K, se_K, chi2_reduced


def fit_power_law(f, slope, se):
    """
    Weighted least squares fit of log(slope) = log(C) + p log(f).
    The error of log(slope) is about se / slope. Only positive slopes can be used.
    Returns C, p and the standard error of p.
    """
    ok = slope > 0
    x = np.log(f[ok])
    y = np.log(slope[ok])
    w = (slope[ok] / se[ok]) ** 2
    X = np.column_stack([np.ones(len(x)), x])
    W = np.diag(w)
    covariance = np.linalg.inv(X.T @ W @ X)
    beta = covariance @ X.T @ W @ y
    # Scale the covariance with the actual scatter (reduced chi-square), so the error of p
    # is not too optimistic if the points scatter more than their error bars
    chi2_reduced = np.sum(w * (y - X @ beta) ** 2) / (len(x) - 2)
    se_p = np.sqrt(covariance[1, 1] * max(chi2_reduced, 1.0))
    return np.exp(beta[0]), beta[1], se_p


# --------------------------- Main program ---------------------------

modes = read_slopes(INPUT_FILE)
vertical = [m for m in modes if m["type"] == "vertical"]
used = [m for m in vertical if not is_excluded(m["f"])]

f = np.array([m["f"] for m in used])
slope = np.array([m["slope"] for m in used])
se = np.array([m["se"] for m in used])

K, se_K, chi2 = fit_k_over_f(f, slope, se)
C, p, se_p = fit_power_law(f, slope, se)

print(f"Vertical modes used in the fit: {len(used)} "
      f"(left out: {', '.join(f'{fx} Hz' for fx in EXCLUDE_FREQUENCIES)})\n")
print(" f [Hz]   slope [%-pt/10 m/s]    slope * f [%-pt Hz/10 m/s]")
for m in vertical:
    note = "   (left out)" if is_excluded(m["f"]) else ""
    print(f"  {m['f']:.3f}     {m['slope']:+5.2f} +- {m['se']:.2f}          {m['slope'] * m['f']:.3f}{note}")

print(f"\n1) slope = K / f :   K = {K:.3f} +- {se_K:.3f} %-pt Hz per 10 m/s,   "
      f"reduced chi-square = {chi2:.2f}")
print(f"2) slope = C f^p :   p = {p:.2f} +- {se_p:.2f}   (quasi-steady theory: p = -1)")

# K in SI units: zeta (as a fraction) per (U / f), i.e. per metre
K_si = K / 100 / 10
print(f"\nK = {K_si:.2e} 1/m   (zeta_aero = K * U / f with U in m/s, f in Hz)")
heave_per_mass = 8 * np.pi * K_si / (AIR_DENSITY * DECK_WIDTH)
print(f"C_heave / m = 8 pi K / (rho B) = {heave_per_mass:.2e} m/kg")
if MASS_PER_LENGTH is not None:
    # Measured C_heave, with an uncertainty from the standard error of K only
    # (the uncertainty of the mass is not included)
    C_measured = heave_per_mass * MASS_PER_LENGTH
    C_measured_se = C_measured * se_K / K
    C_quasi_steady = LIFT_SLOPE + (DECK_DEPTH_2010 / DECK_WIDTH) * DRAG_COEFFICIENT
    C_mqs = -MQS_H1
    print(f"\nWith m = {MASS_PER_LENGTH:.0f} kg/m (Oiseth et al. 2010, Table 2):")
    print(f"  measured from damping slopes:   C_heave = {C_measured:.2f} +- {C_measured_se:.2f}")
    print(f"  quasi-steady, CL' + (D/B) CD:   C_heave = {LIFT_SLOPE} + ({DECK_DEPTH_2010}/{DECK_WIDTH}) * "
          f"{DRAG_COEFFICIENT} = {C_quasi_steady:.2f}   (ratio measured/this = {C_measured / C_quasi_steady:.2f})")
    print(f"  modified quasi-steady, -h1:     C_heave = {C_mqs:.2f}   "
          f"(ratio measured/this = {C_measured / C_mqs:.2f})")
    print("  Note: a lower as-built mass would lower the measured value in proportion.")
else:
    print("Set MASS_PER_LENGTH to convert this into C_heave.")

# Save the fit, so later scripts can use K without copying the number by hand
with open(OUTPUT_FOLDER / "damping_scaling_fit.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K_per_m", "se_K_per_m", "power_law_exponent_p", "se_p", "n_modes", "slope_column"])
    writer.writerow([f"{K_si:.4e}", f"{se_K / 1000:.4e}", f"{p:.3f}", f"{se_p:.3f}", len(used), SLOPE_COLUMN])

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
markers = {"vertical": "o", "lateral": "s", "torsion": "^"}
colors = {"vertical": "k", "lateral": "tab:blue", "torsion": "tab:green"}

for m in modes:
    face = "none" if (m["type"] != "vertical" or is_excluded(m["f"])) else colors[m["type"]]
    style = dict(fmt=markers[m["type"]], color=colors[m["type"]], markerfacecolor=face,
                 markersize=6, capsize=2)
    ax1.errorbar(m["f"], m["slope"], yerr=2 * m["se"], **style)
    ax2.errorbar(m["f"], m["slope"] * m["f"], yerr=2 * m["se"] * m["f"], **style)

f_line = np.linspace(0.09, 0.95, 200)
ax1.plot(f_line, K / f_line, "r-", label=f"K / f,  K = {K:.3f}")
ax1.plot(f_line, C * f_line ** p, "r--", label=f"C f^p,  p = {p:.2f} +- {se_p:.2f}")
ax2.axhline(K, color="r", label=f"K = {K:.3f} +- {se_K:.3f}")
ax2.axhspan(K - 2 * se_K, K + 2 * se_K, color="r", alpha=0.1)

for name in markers:
    ax1.plot([], [], markers[name], color=colors[name], label=name)
ax1.plot([], [], "o", color="k", markerfacecolor="none", label="left out of fit")

ax1.set_xlabel("Mode frequency f [Hz]")
ax1.set_ylabel("Damping slope [%-points per 10 m/s]")
ax1.set_title("Damping slope vs. frequency")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax2.set_xlabel("Mode frequency f [Hz]")
ax2.set_ylabel("Slope x f  [%-points Hz per 10 m/s]")
ax2.set_title("Slope x f (constant if damping scales with U / f)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
fig.suptitle("Quasi-steady U/f scaling of aerodynamic damping   (error bars = 2 standard errors; "
             "filled = vertical modes used in fit)")
fig.savefig(OUTPUT_FOLDER / "damping_scaling.png", dpi=100)

print(f"\nPlot saved in {OUTPUT_FOLDER}")
plt.show()

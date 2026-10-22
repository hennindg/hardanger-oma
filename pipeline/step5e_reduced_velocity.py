"""
Hardanger Bridge project - Step 5e
Test whether the aerodynamic damping of the vertical modes is governed by the reduced velocity.

Theory (Scanlan's aerodynamic derivatives, as in Oiseth et al. 2010, Eq. 12): the motion-induced
vertical force is q_z = 1/2 rho U^2 B K H1*(K) z'/U with K = B omega / U. For one mode with
modal mass m per unit length this adds a damping ratio

    zeta_aero = - rho B^2 H1*(V_r) / (4 m),     V_r = U / (B omega)  (reduced velocity, = 1/K)

So the aerodynamic damping of every vertical mode depends ONLY on V_r. Quasi-steady theory gives
H1* = -C_heave V_r, i.e. one straight line for all modes:  zeta_aero = rho B^2 C_heave V_r / (4 m).

Test:
  - If step 5c's deviation from U/f scaling is a reduced-velocity effect, all modes follow ONE
    common curve: where two modes cover the same V_r they give the same local coefficient
        C_local = (4 m / (rho B^2)) * d(zeta)/d(V_r)
  - If it is mode-specific (e.g. different modal mass or cable participation), the modes give
    different C_local at the same V_r.

Method: for each vertical mode, the recordings are split into windows of V_r (equal numbers of
recordings); in each window a straight line zeta = a + s V_r is fitted, and C_local = (4m/(rho B^2)) s.
Using SLOPES (not damping values) means the unknown structural damping drops out.

Needs results/step5_all/modes_vs_wind.csv.

How to use:
  python -m pipeline.step5e_reduced_velocity
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

AIR_DENSITY = 1.25          # kg/m^3, Oiseth et al. (2010)
B = 18.3                    # deck width (m)
MASS = 12820.0              # vertical modal equivalent mass (kg/m), Oiseth et al. (2010), Table 2

# Wind tunnel values to compare with (Oiseth et al. 2010)
C_QUASI_STEADY = 2.4 + (3.33 / 18.3) * 0.70     # CL' + (D/B) CD, quasi-steady (Eq. 11, 16)
C_MODIFIED_QS = 2.734                            # -h1, modified quasi-steady (Eq. 18)
WIND_TUNNEL_MIN_VR = 0.7    # lowest V/(B omega) with wind tunnel H1* data (Siedziako & Oiseth 2017,
                            # Table 1: forced vibration 0.69-10.43; Oiseth 2010 Fig. 5 starts near 1)

# Number of V_r windows per mode (each with the same number of recordings)
N_WINDOWS = 5

# Only recordings with at least this wind speed (below it the wind speed is uncertain and
# the aerodynamic damping is tiny)
MIN_WIND_SPEED = 2.0

# The 0.119 Hz mode has extra damping not explained by quasi-steady theory (see
# studies/mode_0119/inspect_0117.py); it is drawn but not used in the summary numbers.
SPECIAL_MODE = 0.119

# ==================================================================


def line_fit(x, y):
    """Straight line y = a + s x. Returns a, s and the standard error of s."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    s2 = np.sum(residuals ** 2) / (len(x) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta[0], beta[1], se[1]


def read_vertical_modes():
    """Reduced velocity and damping (fraction) of every vertical mode in every recording."""
    modes = {}
    with open(MODES_VS_WIND, newline="") as f:
        for row in csv.DictReader(f):
            if row["type"] != "vertical":
                continue
            U = float(row["mean_wind_speed_ms"])
            if np.isnan(U) or U < MIN_WIND_SPEED:
                continue
            f_mode = float(row["freq_Hz"])
            V_r = U / (B * 2 * np.pi * f_mode)        # reduced velocity U / (B omega)
            modes.setdefault(float(row["track_median_freq_Hz"]), []).append(
                (V_r, float(row["damping_percent"]) / 100, U))
    return {f0: np.array(v) for f0, v in sorted(modes.items())}


# --------------------------- Main program ---------------------------

coefficient = 4 * MASS / (AIR_DENSITY * B ** 2)     # converts d(zeta)/d(V_r) into C
modes = read_vertical_modes()

print(f"C_local = (4 m / (rho B^2)) * d(zeta)/d(V_r), with m = {MASS:.0f} kg/m.  "
      f"Wind tunnel: quasi-steady {C_QUASI_STEADY:.2f}, modified quasi-steady {C_MODIFIED_QS:.2f}\n")
print(" mode f [Hz]   V_r range      whole-mode C     C_local per V_r window (V_r: C +- se)")

windows = []          # (f0, V_r median, C_local, se) for all windows
whole = {}            # f0 -> (intercept zeta_s, C, se)
for f0, data in modes.items():
    V_r, zeta = data[:, 0], data[:, 1]
    a, s, se = line_fit(V_r, zeta)
    whole[f0] = (a, coefficient * s, coefficient * se)
    order = np.argsort(V_r)
    text = []
    for part in np.array_split(order, N_WINDOWS):
        a_w, s_w, se_w = line_fit(V_r[part], zeta[part])
        windows.append((f0, np.median(V_r[part]), coefficient * s_w, coefficient * se_w,
                        np.median(data[part, 2])))
        text.append(f"{np.median(V_r[part]):.2f}: {coefficient * s_w:5.2f}+-{coefficient * se_w:.2f}")
    print(f"   {f0:.3f}   {np.percentile(V_r, 5):.2f}-{np.percentile(V_r, 95):.2f}     "
          f"{coefficient * s:5.2f} +- {coefficient * se:.2f}     " + "   ".join(text))

windows = np.array(windows)
usable = np.abs(windows[:, 0] - SPECIAL_MODE) > 0.003

# Test 1: does C_local depend on V_r? (one straight line through all windows of all modes)
w = 1 / windows[usable, 3] ** 2
X = np.column_stack([np.ones(np.sum(usable)), windows[usable, 1]])
W = np.diag(w)
cov = np.linalg.inv(X.T @ W @ X)
beta = cov @ X.T @ W @ windows[usable, 2]
residuals = windows[usable, 2] - X @ beta
chi2 = np.sum(w * residuals ** 2) / (np.sum(usable) - 2)
print(f"\nWeighted line through all windows (without the {SPECIAL_MODE} Hz mode):")
print(f"   C_local = {beta[0]:.2f} + {beta[1]:.2f} * V_r   (slope +- {np.sqrt(cov[1, 1] * max(chi2, 1)):.2f}),"
      f" reduced chi-square = {chi2:.1f}")

# Test 1b: is it V_r or simply the wind speed U? Within a mode, low V_r also means low U.
# Across modes the same V_r occurs at very different U, so both can be tested together:
#   C_local = a + b V_r + c U   (weighted). The variable that matters keeps a clear coefficient.
U_window = windows[usable, 4]
X2 = np.column_stack([np.ones(np.sum(usable)), windows[usable, 1], U_window])
cov2 = np.linalg.inv(X2.T @ W @ X2)
beta2 = cov2 @ X2.T @ W @ windows[usable, 2]
chi2_2 = np.sum(w * (windows[usable, 2] - X2 @ beta2) ** 2) / (np.sum(usable) - 3)
se2 = np.sqrt(np.diag(cov2) * max(chi2_2, 1))
print(f"   C_local = {beta2[0]:.2f} + {beta2[1]:.2f} V_r + {beta2[2]:.3f} U:  "
      f"t(V_r) = {beta2[1] / se2[1]:+.1f},  t(U) = {beta2[2] / se2[2]:+.1f}   (|t| > 2: the variable matters)")
X3 = np.column_stack([np.ones(np.sum(usable)), U_window])
cov3 = np.linalg.inv(X3.T @ W @ X3)
beta3 = cov3 @ X3.T @ W @ windows[usable, 2]
chi2_3 = np.sum(w * (windows[usable, 2] - X3 @ beta3) ** 2) / (np.sum(usable) - 2)
print(f"   reduced chi-square:  V_r only {chi2:.2f},  U only {chi2_3:.2f},  V_r + U {chi2_2:.2f}  (lower = better)")

# Test 2: do different modes agree at the same V_r? Compare windows that overlap in V_r.
print("\nWindows of different modes at similar V_r (within 15 %):")
pairs, differences = 0, []
rows = windows[usable]
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if rows[i, 0] == rows[j, 0]:
            continue
        if abs(rows[i, 1] - rows[j, 1]) / rows[i, 1] < 0.15:
            pairs += 1
            differences.append((rows[i, 2] - rows[j, 2]) / np.hypot(rows[i, 3], rows[j, 3]))
differences = np.array(differences)
print(f"   {pairs} pairs; difference in C_local / combined standard error: median |z| = "
      f"{np.median(np.abs(differences)):.1f}, {100 * np.mean(np.abs(differences) < 2):.0f} % within 2 standard errors "
      f"(about 95 % if the modes agree)")

# ---- Plots ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
vr_line = np.linspace(0, 1.9, 50)
for k, (f0, data) in enumerate(modes.items()):
    V_r, zeta = data[:, 0], data[:, 1]
    zeta_s = whole[f0][0]
    # Binned medians of the aerodynamic part (damping minus the fitted structural damping)
    order = np.argsort(V_r)
    bins = np.array_split(order, 12)
    vr_bin = [np.median(V_r[b]) for b in bins]
    z_bin = [100 * np.median(zeta[b] - zeta_s) for b in bins]
    style = "s--" if abs(f0 - SPECIAL_MODE) < 0.003 else "o-"
    ax1.plot(vr_bin, z_bin, style, color=colors[k], markersize=4, linewidth=1, label=f"{f0:.3f} Hz")
    sel = windows[:, 0] == f0
    ax2.errorbar(windows[sel, 1], windows[sel, 2], yerr=2 * windows[sel, 3], fmt=style[0] + "-",
                 color=colors[k], markersize=4, capsize=2, linewidth=0.8, label=f"{f0:.3f} Hz")

for ax, scale in [(ax1, 100 / coefficient), (ax2, None)]:
    if scale is not None:
        ax.plot(vr_line, scale * C_QUASI_STEADY * vr_line, "k-", linewidth=2, label="quasi-steady (2.53)")
        ax.plot(vr_line, scale * C_MODIFIED_QS * vr_line, "k--", linewidth=1.5, label="modified quasi-steady (2.73)")
    else:
        ax.axhline(C_QUASI_STEADY, color="k", linewidth=2, label="quasi-steady (2.53)")
        ax.axhline(C_MODIFIED_QS, color="k", linestyle="--", linewidth=1.5, label="modified quasi-steady (2.73)")
    ax.axvspan(0, WIND_TUNNEL_MIN_VR, color="tab:orange", alpha=0.08, label="no wind tunnel data")
    ax.set_xlabel("Reduced velocity $V_r = U / (B \\omega)$")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.9)
ax1.set_ylabel("Aerodynamic damping $\\zeta - \\zeta_s$ [%]   (binned medians)")
ax1.set_title("All vertical modes vs. reduced velocity (one curve if $V_r$ governs)")
ax2.set_ylabel("$C_{local} = (4m/\\rho B^2)\\, d\\zeta/dV_r$")
ax2.set_title(f"Local damping coefficient in {N_WINDOWS} $V_r$ windows per mode (error bars 2 se)")
ax2.set_ylim(-1, 5)
ax1.legend(fontsize=7, ncol=2)
fig.suptitle("Is the aerodynamic damping of the vertical modes a function of the reduced velocity only?  "
             f"(m = {MASS:.0f} kg/m; dashed squares = {SPECIAL_MODE} Hz mode)")
fig.savefig(OUTPUT_FOLDER / "reduced_velocity_test.png", dpi=95)

print(f"\nPlot saved in {OUTPUT_FOLDER}")
plt.show()

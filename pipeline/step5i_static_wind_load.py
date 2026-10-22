"""
Hardanger Bridge project - Step 5i
Is the rise of the vertical frequencies with wind caused by the STATIC wind load?

Hypothesis: the mean wind gives a static lift force on the deck,
    L = 1/2 rho U^2 B C_L(alpha),   C_L(alpha) = CL_mean + CL_slope * alpha
(alpha = mean inclination of the incoming wind, positive upwards). A downward lift (L < 0) adds to
the dead load, which raises the cable tension and with it the vertical frequencies. For a mode
whose stiffness comes from the cable tension, frequency^2 is proportional to the tension, so
    delta_f / f  ~  -1/2 * L / q_dead          (q_dead = dead load per unit length)
This predicts:
  1. the frequency change grows with U^2, not U,
  2. its size follows from the wind tunnel coefficients (CL_mean = -0.25, CL_slope = 2.4 per rad),
  3. it can differ between east and west winds, because the terrain tilts the wind differently
     (alpha), and C_L is sensitive to alpha.

What the script does:
  1. Mean flow inclination alpha of every recording from the anemometers: atan(mean w / U),
     averaged over the working span anemometers (cached in recording_inclination.csv).
  2. Predicted static-load frequency change s = -1/2 L / q_dead [%] per recording.
  3. Per mode: compares the models  f = a + b U + c T,  f = a + b U^2 + c T,  f = a + k s + c T
     (R^2, and k with a block bootstrap 95 % interval; k = 1 means the hypothesis explains the
     whole effect, for a purely cable-dominated mode).
  4. East/west: mean alpha per direction and the predicted vs. measured frequency change.

How to use:
  python -m pipeline.step5i_static_wind_load
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording
from hardanger.regression import load_table, least_squares
from oma.block_bootstrap import block_bootstrap, summarize
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

SPAN_ANEMOMETERS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
MIN_WIND_SPEED = 3.0      # inclination is not meaningful in very weak wind

AIR_DENSITY = 1.25        # kg/m^3 (Oiseth et al. 2010)
B = 18.3                  # deck width (m)
CL_MEAN = -0.25           # mean lift coefficient at zero inclination (Oiseth et al. 2010, Eq. 16)
CL_SLOPE = 2.4            # dCL/dalpha per radian (Oiseth et al. 2010)
# Dead load per unit length. ASSUMPTION: approximated by the vertical modal equivalent mass
# (12,820 kg/m, Oiseth et al. 2010, includes the cables) times g. Not given in the papers directly.
DEAD_LOAD = 12820.0 * 9.81    # N/m

EAST_SECTOR = (0.0, 180.0)    # deg, "East" winds (assumed: direction the wind comes FROM)
N_BOOT = 500
RANDOM_SEED = 5

# ==================================================================


def recording_inclination(path):
    """
    Mean flow inclination (deg, positive = wind blowing upwards) of one recording:
    atan(mean w / mean U) per working span anemometer, averaged over the anemometers.
    """
    rec = load_recording(path)
    angles = []
    for name in SPAN_ANEMOMETERS:
        data = rec["sensors"].get(name)
        if data is None or np.isnan(data).any():
            continue
        U, w = np.mean(data[:, 1]), np.mean(data[:, 2])
        if U >= MIN_WIND_SPEED:
            angles.append(np.degrees(np.arctan2(w, U)))
    return float(np.mean(angles)) if angles else np.nan


def inclinations(file_names):
    """Inclination of every recording, from the cache file if possible."""
    cache_file = OUTPUT_FOLDER / "recording_inclination.csv"
    cache = {}
    if cache_file.exists():
        with open(cache_file, newline="") as f:
            cache = {r["file_name"]: float(r["inclination_deg"]) for r in csv.DictReader(f)}
    missing = [n for n in file_names if n not in cache]
    if missing:
        print(f"Reading the flow inclination from {len(missing)} recordings ...")
        mat_files = {p.stem: p for p in DATA_FOLDER.rglob("*.mat") if "results" not in p.parts}
        for n in missing:
            cache[n] = recording_inclination(mat_files[n])
        with open(cache_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "inclination_deg"])
            for n, a in sorted(cache.items()):
                writer.writerow([n, f"{a:.3f}"])
    return np.array([cache[n] for n in file_names])


def static_load_prediction(U, alpha_deg):
    """Predicted frequency change [%] from the static lift: -1/2 L / q_dead."""
    lift = 0.5 * AIR_DENSITY * U ** 2 * B * (CL_MEAN + CL_SLOPE * np.radians(alpha_deg))
    return 100 * (-0.5) * lift / DEAD_LOAD


def model_fits(table, rows, mode, s):
    """
    R^2 of the three models and the coefficient k of the static-load model, for one mode.
    Frequency in % of its mean. Uses only rows with valid U, T, s and frequency.
    """
    U, T, f, ss = table["U"][rows], table["T"][rows], mode["freq"][rows], s[rows]
    ok = ~np.isnan(U) & ~np.isnan(T) & ~np.isnan(f) & ~np.isnan(ss)
    U, T, ss = U[ok], T[ok], ss[ok]
    y = 100 * f[ok] / np.mean(f[ok])
    r2, k = [], np.nan
    for x in (U, U ** 2, ss):
        X = np.column_stack([np.ones(len(y)), x, T])
        beta, _ = least_squares(X, y)
        residuals = y - X @ beta
        r2.append(1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))
        k = beta[1]                     # last loop: coefficient of s
    return np.array(r2 + [k])


# --------------------------- Main program ---------------------------

table = load_table(MODES_VS_WIND)
alpha = inclinations(table["file_name"])
s = static_load_prediction(table["U"], alpha)
is_east = (table["direction"] >= EAST_SECTOR[0]) & (table["direction"] < EAST_SECTOR[1])
valid = ~np.isnan(alpha)

print("\nMean flow inclination (deg, + = upwards), recordings with U >= 6 m/s:")
for label, sel in [("East", is_east), ("West", ~is_east)]:
    use = sel & valid & (table["U"] >= 6)
    print(f"  {label}: median {np.median(alpha[use]):+.2f} deg (IQR {np.percentile(alpha[use], 25):+.2f} to "
          f"{np.percentile(alpha[use], 75):+.2f}), n = {np.sum(use)};  C_L = {CL_MEAN:+.2f} + {CL_SLOPE} * alpha = "
          f"{CL_MEAN + CL_SLOPE * np.radians(np.median(alpha[use])):+.3f}")
for U_example in [10, 15, 20]:
    e = static_load_prediction(U_example, np.median(alpha[is_east & valid & (table["U"] >= 6)]))
    w = static_load_prediction(U_example, np.median(alpha[~is_east & valid & (table["U"] >= 6)]))
    print(f"  predicted static-load frequency change at {U_example} m/s: East {e:+.2f} %, West {w:+.2f} %")

# ---- Model comparison per mode, with a block bootstrap of k ----
rows_all = np.arange(len(table["file_name"]))
estimates = np.array([model_fits(table, rows_all, m, s) for m in table["modes"]])
samples = block_bootstrap(table["day"], lambda r: np.array([model_fits(table, r, m, s)[3] for m in table["modes"]]),
                          N_BOOT, np.random.default_rng(RANDOM_SEED))
_, k_low, k_high = summarize(samples)

print("\nPer mode (all recordings, temperature in every model):  R^2 of  f ~ U,  f ~ U^2,  f ~ static load s;"
      "  k = coefficient of s (1 = fully explained), 95 % bootstrap interval")
print(" track  f [Hz]  type        R2(U)   R2(U^2)  R2(s)      k       95 % interval")
rows_out = []
for i, mode in enumerate(table["modes"]):
    r2_u, r2_u2, r2_s, k = estimates[i]
    print(f"  {mode['track']:3d}   {mode['f0']:.3f}  {mode['type']:9s}  {r2_u:.3f}   {r2_u2:.3f}   {r2_s:.3f}    "
          f"{k:+6.2f}   [{k_low[i]:+.2f}, {k_high[i]:+.2f}]")
    rows_out.append([mode["track"], mode["f0"], mode["type"], r2_u, r2_u2, r2_s, k, k_low[i], k_high[i]])

with open(OUTPUT_FOLDER / "static_wind_load.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "R2_U", "R2_U2", "R2_static_load", "k", "k_ci95_low",
                     "k_ci95_high"])
    for r in rows_out:
        writer.writerow([v if isinstance(v, str) else f"{v:.4g}" for v in r])

# ---- Plots ----
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
for label, sel, color in [("East", is_east, "tab:red"), ("West", ~is_east, "tab:blue")]:
    use = sel & valid
    ax1.plot(table["U"][use], alpha[use], ".", color=color, markersize=3, alpha=0.5, label=label)
    ax2.plot(table["U"][use], s[use], ".", color=color, markersize=3, alpha=0.5, label=label)
ax1.set_xlabel("Mean wind speed U [m/s]")
ax1.set_ylabel("Mean flow inclination [deg]  (+ = upwards)")
ax1.set_title("Mean flow inclination at the anemometers")
ax2.set_xlabel("Mean wind speed U [m/s]")
ax2.set_ylabel("Predicted static-load frequency change s [%]")
ax2.set_title("s = -1/2 L / q_dead,  L = 1/2 rho U^2 B (CL + CL' alpha)")
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=4)
fig1.savefig(OUTPUT_FOLDER / "static_wind_load_inclination.png", dpi=100)

vertical = [m for m in table["modes"] if m["type"] == "vertical"]
n_cols = 4
n_rows = int(np.ceil(len(vertical) / n_cols))
fig2, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows), squeeze=False, layout="constrained")
for k_ax in range(n_rows * n_cols):
    ax = axes[k_ax // n_cols][k_ax % n_cols]
    if k_ax >= len(vertical):
        ax.axis("off")
        continue
    mode = vertical[k_ax]
    ok = valid & ~np.isnan(table["T"]) & ~np.isnan(mode["freq"])
    y = 100 * mode["freq"][ok] / np.mean(mode["freq"][ok])
    X = np.column_stack([np.ones(np.sum(ok)), s[ok], table["T"][ok]])
    beta, _ = least_squares(X, y)
    y_corr = y - beta[2] * (table["T"][ok] - np.mean(table["T"][ok])) - np.mean(y)   # temperature removed
    for label, sel, color in [("East", is_east[ok], "tab:red"), ("West", ~is_east[ok], "tab:blue")]:
        order = np.argsort(s[ok][sel])
        for part in np.array_split(order, 10):
            ax.plot(np.median(s[ok][sel][part]), np.median(y_corr[sel][part]), "o", color=color, markersize=4)
    line = np.linspace(np.nanmin(s), np.nanmax(s), 10)
    ax.plot(line, line - np.median(line) + 0 * line, "k--", linewidth=0.8)
    ax.plot(line, beta[1] * line - beta[1] * np.mean(s[ok]), "k-", linewidth=1)
    i = table["modes"].index(mode)
    ax.set_title(f"{mode['f0']:.3f} Hz: k = {estimates[i][3]:+.2f} [{k_low[i]:+.2f}, {k_high[i]:+.2f}]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("predicted s [%]", fontsize=8)
    ax.set_ylabel("measured f change [%] (T removed)", fontsize=8)
axes[0][0].plot([], [], "o", color="tab:red", label="East (binned)")
axes[0][0].plot([], [], "o", color="tab:blue", label="West (binned)")
axes[0][0].plot([], [], "k--", label="slope 1 (fully explained)")
axes[0][0].plot([], [], "k-", label="fitted slope k")
axes[0][0].legend(fontsize=7)
fig2.suptitle("Vertical modes: measured frequency change vs. static-wind-load prediction (temperature removed)")
fig2.savefig(OUTPUT_FOLDER / "static_wind_load_modes.png", dpi=90)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

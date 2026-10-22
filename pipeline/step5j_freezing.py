"""
Hardanger Bridge project - Step 5j
Freezing test: do the frequencies stiffen extra below about 0 degC?

Background: on many bridges the natural frequencies rise much faster with falling temperature
once it freezes (e.g. frozen asphalt, bearings, expansion joints). A linear temperature term
cannot describe that, and the leftover could look like other effects - step 5i found that
west-wind recordings at low wind (mostly winter) have higher frequencies than east-wind ones.

Test per mode (frequency in % of its mean):
  linear model:  f = a + b U^2 + c T
  hinge model:   f = a + b U^2 + c T + d * max(0, T0 - T)
d is the EXTRA slope below the kink temperature T0. Freezing predicts d < 0 (frequency rises
faster as T falls below T0). U^2 removes the static wind-load effect (step 5i).
  1. T0 is scanned; the residual sum of squares over all vertical modes shows where a kink fits best.
  2. d at the chosen T0 with a block bootstrap 95 % interval (days resampled).
  3. Does the hinge model remove the west - east offset at low wind?

The temperature is the mean sonic (air) temperature of the recording - a proxy; the structure
lags behind the air temperature.

How to use:
  python -m pipeline.step5j_freezing
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.regression import load_table, least_squares
from oma.block_bootstrap import block_bootstrap, summarize
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

# Kink temperatures to scan (degC), and the one used for the bootstrap (None = best from the scan)
T0_SCAN = np.arange(-3.0, 6.01, 0.5)
T0_FIXED = None

LOW_WIND = 6.0                  # m/s, "low wind" for the west - east offset check
EAST_SECTOR = (0.0, 180.0)      # deg (assumed: direction the wind comes FROM)
N_BOOT = 500
RANDOM_SEED = 6

# ==================================================================


def valid_rows(table, rows, mode):
    """Rows (of 'rows') with a valid wind speed, temperature and frequency of this mode."""
    ok = ~np.isnan(table["U"][rows]) & ~np.isnan(table["T"][rows]) & ~np.isnan(mode["freq"][rows])
    return rows[ok]


def design(table, rows, T0):
    """Model matrix: [1, U^2, T] (linear) or [1, U^2, T, max(0, T0 - T)] (hinge, if T0 is given)."""
    U, T = table["U"][rows], table["T"][rows]
    columns = [np.ones(len(rows)), U ** 2, T]
    if T0 is not None:
        columns.append(np.maximum(0.0, T0 - T))
    return np.column_stack(columns)


def fit_mode(table, rows, mode, T0):
    """Least-squares fit of one mode. Returns beta, the residuals and the rows used."""
    rows = valid_rows(table, rows, mode)
    f = mode["freq"][rows]
    y = 100 * f / np.mean(f)
    # With (almost) no recordings below T0 the hinge slope cannot be estimated (a bootstrap
    # sample may contain none): then fit the linear model and report d as NaN
    if T0 is not None and np.sum(table["T"][rows] < T0) < 5:
        X = design(table, rows, None)
        beta, _ = least_squares(X, y)
        return np.append(beta, np.nan), y - X @ beta, rows
    X = design(table, rows, T0)
    beta, _ = least_squares(X, y)
    return beta, y - X @ beta, rows


def low_wind_offset(table, rows, residuals):
    """Mean residual of west minus east recordings at U < LOW_WIND (in % of the frequency)."""
    U, direction = table["U"][rows], table["direction"][rows]
    east = (direction >= EAST_SECTOR[0]) & (direction < EAST_SECTOR[1])
    low = U < LOW_WIND
    if np.sum(low & east) < 5 or np.sum(low & ~east) < 5:
        return np.nan
    return np.mean(residuals[low & ~east]) - np.mean(residuals[low & east])


def statistics(table, rows, modes, T0):
    """For each mode: d (hinge slope), and the low-wind west - east offset in the linear and hinge models."""
    out = []
    for mode in modes:
        beta_h, res_h, r_h = fit_mode(table, rows, mode, T0)
        _, res_l, r_l = fit_mode(table, rows, mode, None)
        out.extend([beta_h[3], low_wind_offset(table, r_l, res_l), low_wind_offset(table, r_h, res_h)])
    return np.array(out)


# --------------------------- Main program ---------------------------

table = load_table(MODES_VS_WIND)
T = table["T"]
print(f"Temperatures: {np.sum(T < 0)} recordings below 0 degC, {np.sum(T < 2)} below 2 degC, "
      f"{np.sum(~np.isnan(T))} in total (min {np.nanmin(T):.1f} degC)")
east = (table["direction"] >= EAST_SECTOR[0]) & (table["direction"] < EAST_SECTOR[1])
low = table["U"] < LOW_WIND
print(f"Low wind (U < {LOW_WIND} m/s): East median T {np.nanmedian(T[low & east]):.1f} degC (n = {np.sum(low & east)}), "
      f"West median T {np.nanmedian(T[low & ~east]):.1f} degC (n = {np.sum(low & ~east)})")

modes = table["modes"]
vertical = [m for m in modes if m["type"] == "vertical"]
all_rows = np.arange(len(table["file_name"]))

# ---- 1. Scan the kink temperature ----
print("\nScan of the kink temperature T0 (sum over vertical modes of the residual variance, relative to linear):")
ss_linear = sum(np.sum(fit_mode(table, all_rows, m, None)[1] ** 2) for m in vertical)
scan = []
for T0 in T0_SCAN:
    ss = sum(np.sum(fit_mode(table, all_rows, m, T0)[1] ** 2) for m in vertical)
    scan.append(ss / ss_linear)
scan = np.array(scan)
best_T0 = T0_SCAN[np.argmin(scan)]
print("   T0:    " + " ".join(f"{t:5.1f}" for t in T0_SCAN))
print("   ratio: " + " ".join(f"{r:5.3f}" for r in scan))
T0 = best_T0 if T0_FIXED is None else T0_FIXED
print(f"   best T0 = {best_T0:.1f} degC (residual variance {100 * (1 - scan.min()):.1f} % lower than linear); using T0 = {T0:.1f}")

# ---- 2. Hinge slope d and 3. low-wind offset, with block bootstrap ----
estimate = statistics(table, all_rows, modes, T0)
samples = block_bootstrap(table["day"], lambda r: statistics(table, r, modes, T0), N_BOOT,
                          np.random.default_rng(RANDOM_SEED))
_, low_ci, high_ci = summarize(samples)

print(f"\nPer mode: d = extra frequency slope below {T0:.1f} degC [% per degC] (freezing: d < 0), and the")
print(f"west - east frequency offset at U < {LOW_WIND} m/s [%] in the linear and hinge models (95 % bootstrap intervals)")
print(" track  f [Hz]  type          d [%/degC]   95 % interval        offset linear      offset hinge")
rows_out = []
for k, mode in enumerate(modes):
    d, off_l, off_h = estimate[3 * k:3 * k + 3]
    lo, hi = low_ci[3 * k:3 * k + 3], high_ci[3 * k:3 * k + 3]
    flag = "*" if hi[0] < 0 or lo[0] > 0 else " "
    print(f"  {mode['track']:3d}   {mode['f0']:.3f}  {mode['type']:9s}  {d:+7.3f}    [{lo[0]:+.3f}, {hi[0]:+.3f}] {flag}"
          f"   {off_l:+.3f} [{lo[1]:+.2f}, {hi[1]:+.2f}]   {off_h:+.3f} [{lo[2]:+.2f}, {hi[2]:+.2f}]")
    rows_out.append([mode["track"], mode["f0"], mode["type"], d, lo[0], hi[0], off_l, lo[1], hi[1], off_h, lo[2], hi[2]])

with open(OUTPUT_FOLDER / "freezing_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", f"d_below_{T0:.1f}C_percent_per_degC", "ci_low", "ci_high",
                     "lowwind_offset_W_minus_E_linear_percent", "ci_low", "ci_high",
                     "lowwind_offset_W_minus_E_hinge_percent", "ci_low", "ci_high"])
    for r in rows_out:
        writer.writerow([v if isinstance(v, str) else f"{v:.4g}" for v in r])

# ---- Plots: residuals of the model WITHOUT temperature vs. temperature (shows the shape) ----
n_cols = 5
n_rows = int(np.ceil(len(modes) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 2.9 * n_rows), squeeze=False, layout="constrained")
for k in range(n_rows * n_cols):
    ax = axes[k // n_cols][k % n_cols]
    if k >= len(modes):
        ax.axis("off")
        continue
    mode = modes[k]
    rows = valid_rows(table, all_rows, mode)
    f = mode["freq"][rows]
    y = 100 * f / np.mean(f)
    X = np.column_stack([np.ones(len(rows)), table["U"][rows] ** 2])
    beta, _ = least_squares(X, y)
    y_wind_removed = y - X @ beta                        # wind (U^2) effect removed, temperature left in
    T_rows = table["T"][rows]
    is_east = east[rows]
    for sel, color in [(is_east, "tab:red"), (~is_east, "tab:blue")]:
        ax.plot(T_rows[sel], y_wind_removed[sel], ".", color=color, markersize=2, alpha=0.25)
    edges = np.arange(-8, 29, 2)
    centres, medians = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        in_bin = (T_rows >= a) & (T_rows < b)
        if np.sum(in_bin) >= 8:
            centres.append((a + b) / 2)
            medians.append(np.median(y_wind_removed[in_bin]))
    ax.plot(centres, medians, "k-o", markersize=3, linewidth=1.2)
    ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
    d, lo, hi = estimate[3 * k], low_ci[3 * k], high_ci[3 * k]
    ax.set_title(f"{mode['f0']:.3f} Hz {mode['type']}\nd = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] %/degC", fontsize=8)
    ax.set_ylim(np.percentile(y_wind_removed, 1), np.percentile(y_wind_removed, 99))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if k % n_cols == 0:
        ax.set_ylabel("f change [%] (wind removed)", fontsize=8)
    if k >= len(modes) - n_cols:
        ax.set_xlabel("Mean air temperature [degC]", fontsize=8)
axes[0][0].plot([], [], ".", color="tab:red", label="East wind")
axes[0][0].plot([], [], ".", color="tab:blue", label="West wind")
axes[0][0].plot([], [], "k-o", label="2-degC bin median")
axes[0][0].legend(fontsize=7, markerscale=3)
fig.suptitle(f"Frequency vs. temperature after removing the wind effect (U^2). A kink near 0 degC would mean "
             f"freezing; d = extra slope below {T0:.1f} degC (95 % bootstrap)")
fig.savefig(OUTPUT_FOLDER / "freezing_test.png", dpi=90)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

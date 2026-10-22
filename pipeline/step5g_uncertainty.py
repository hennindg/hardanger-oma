"""
Hardanger Bridge project - Step 5g
Realistic uncertainties for the main results of steps 5b and 5c (block bootstrap by day).

The ordinary standard errors of steps 5b-5d assume that all ~1,450 recordings are independent.
They are not: recordings of the same day share the same storm, temperature and traffic. Here
whole DAYS are resampled (block bootstrap, see oma/block_bootstrap.py) and the analysis of steps 5b
and 5c is repeated N_BOOT times:
  - per mode: frequency trend with wind b_f [% per 10 m/s] and temperature c_f [% per degC]
    (at constant temperature / wind), and damping trend with wind b_z [%-points per 10 m/s]
  - over the vertical modes: K, the power law exponent p, and C_heave (step 5c)
The ratio bootstrap / ordinary standard error shows how much the ordinary errors underestimate.

How to use:
  python -m pipeline.step5g_uncertainty
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.regression import load_table, mode_coefficients, scaling_fit
from oma.block_bootstrap import block_bootstrap, summarize
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

N_BOOT = 1000
RANDOM_SEED = 2

# The 0.119 Hz mode is left out of the K / p / C_heave fit, as in step 5c
EXCLUDE_FROM_SCALING = [0.119]

# ==================================================================


def all_results(table, rows):
    """
    Everything we want an uncertainty for, from the recordings in 'rows':
    [b_f, c_f, b_z for every mode] + [K, p, C_heave]. Also returns the ordinary standard errors
    of the per-mode coefficients.
    """
    values, errors = [], []
    scaling = ([], [], [])
    for mode in table["modes"]:
        v, e = mode_coefficients(table, rows, mode)
        values.extend(v)
        errors.extend(e)
        if mode["type"] == "vertical" and all(abs(mode["f0"] - x) > 0.003 for x in EXCLUDE_FROM_SCALING):
            scaling[0].append(mode["f0"])
            scaling[1].append(v[2])
            scaling[2].append(e[2])
    K, p, C = scaling_fit(*scaling)
    return np.array(values + [K, p, C]), np.array(errors)


# --------------------------- Main program ---------------------------

table = load_table(MODES_VS_WIND)
n_modes = len(table["modes"])
n_days = len(np.unique(table["day"]))
print(f"{len(table['file_name'])} recordings on {n_days} different days; {N_BOOT} bootstrap repetitions\n")

all_rows = np.arange(len(table["file_name"]))
estimate, ordinary_se = all_results(table, all_rows)

rng = np.random.default_rng(RANDOM_SEED)
samples = block_bootstrap(table["day"], lambda rows: all_results(table, rows)[0], N_BOOT, rng)
boot_se, low, high = summarize(samples)

names = ["b_f", "c_f", "b_z"]
units = {"b_f": "%/10 m/s", "c_f": "%/degC", "b_z": "%-pt/10 m/s"}
print(" track  f [Hz]  type       quantity       estimate   ordinary se   bootstrap se   ratio   95 % interval")
rows_out = []
for k, mode in enumerate(table["modes"]):
    for j, name in enumerate(names):
        i = 3 * k + j
        ratio = boot_se[i] / ordinary_se[i]
        print(f"  {mode['track']:3d}   {mode['f0']:.3f}  {mode['type']:9s}  {name} [{units[name]:11s}] "
              f"{estimate[i]:+8.3f}    {ordinary_se[i]:8.3f}     {boot_se[i]:8.3f}     {ratio:5.1f}   "
              f"[{low[i]:+.3f}, {high[i]:+.3f}]")
        rows_out.append([mode["track"], mode["f0"], mode["type"], name, units[name], estimate[i],
                         ordinary_se[i], boot_se[i], low[i], high[i]])

i0 = 3 * n_modes
print(f"\nDamping scaling over the vertical modes (step 5c), estimate and 95 % bootstrap interval:")
print(f"  K       = {estimate[i0]:.2e} 1/m   [{low[i0]:.2e}, {high[i0]:.2e}]")
print(f"  p       = {estimate[i0 + 1]:+.2f}       [{low[i0 + 1]:+.2f}, {high[i0 + 1]:+.2f}]   (quasi-steady: -1)")
print(f"  C_heave = {estimate[i0 + 2]:.2f}        [{low[i0 + 2]:.2f}, {high[i0 + 2]:.2f}]   "
      f"(wind tunnel: 2.53 quasi-steady, 2.73 modified quasi-steady)")
for label, i in [("K", i0), ("p", i0 + 1), ("C_heave", i0 + 2)]:
    rows_out.append(["all vertical", "", "", label, "", estimate[i], "", boot_se[i], low[i], high[i]])

with open(OUTPUT_FOLDER / "bootstrap_uncertainty.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "quantity", "unit", "estimate", "ordinary_se",
                     "bootstrap_se", "ci95_low", "ci95_high"])
    for r in rows_out:
        writer.writerow([v if isinstance(v, str) else f"{v:.5g}" for v in r])

ratios = boot_se[:i0] / ordinary_se
print(f"\nBootstrap / ordinary standard error over all per-mode coefficients: median {np.median(ratios):.1f}, "
      f"range {ratios.min():.1f} - {ratios.max():.1f}")

# ---- Plot: estimates with ordinary 95 % (2 se) and bootstrap 95 % intervals ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
f0 = np.array([m["f0"] for m in table["modes"]])
markers = {"vertical": "o", "lateral": "s", "torsion": "^"}
for j, (ax, name, title) in enumerate(zip(axes, names, ["Frequency vs. wind (at constant T)",
                                                         "Frequency vs. temperature (at constant U)",
                                                         "Damping vs. wind (at constant T)"])):
    for k, mode in enumerate(table["modes"]):
        i = 3 * k + j
        m = markers.get(mode["type"], "o")
        ax.errorbar(f0[k] - 0.005, estimate[i], yerr=2 * ordinary_se[k * 3 + j], fmt=m, color="0.6",
                    markersize=4, capsize=2)
        ax.errorbar(f0[k] + 0.005, estimate[i], yerr=[[estimate[i] - low[i]], [high[i] - estimate[i]]],
                    fmt=m, color="k", markersize=4, capsize=2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Mode frequency [Hz]")
    ax.set_ylabel(f"{name} [{units[name]}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
axes[0].plot([], [], "o", color="0.6", label="ordinary 95 % (2 se)")
axes[0].plot([], [], "o", color="k", label="block bootstrap 95 % (by day)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Uncertainty of the wind and temperature coefficients: ordinary vs. block bootstrap "
             f"({n_days} days, {N_BOOT} repetitions)")
fig.savefig(OUTPUT_FOLDER / "bootstrap_uncertainty.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

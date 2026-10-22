"""
Hardanger Bridge project - Step 5h
Does the wind direction matter? East winds vs. west winds.

The two main wind directions at the bridge come from opposite sides of the fjord, over different
terrain (Fenerci & Oiseth 2018 found different turbulence for east and west winds). The deck cross
section is not exactly symmetric (3 % crossfall, railings), so the aerodynamics could also differ.

For each direction separately, the step 5b / 5c analysis is repeated:
  - per mode: frequency vs. wind b_f and temperature c_f, damping vs. wind b_z
  - over the vertical modes: K, power law exponent p, C_heave
and the difference West - East is given with a block bootstrap 95 % interval (days resampled,
see oma/block_bootstrap.py). A difference whose interval does not contain 0 is significant.

ASSUMPTION (as in step 5d): the anemometer angle is the direction the wind comes FROM,
measured from the global x-axis; 0-180 deg = "East". The labels may be swapped if the angle
means the direction the wind blows TO - the comparison itself is not affected.

How to use:
  python -m pipeline.step5h_direction
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

EAST_SECTOR = (0.0, 180.0)          # deg, "East" winds (see the assumption above)

# Only recordings in this wind speed range (m/s) are compared. West winds are mostly weaker than
# east winds, and the wind trends are not linear over the whole range (e.g. the damping is below
# quasi-steady theory at low wind, step 5f), so both directions must cover the SAME range.
# None = all recordings.
COMMON_WIND_RANGE = (6.0, 15.0)
N_BOOT = 1000
RANDOM_SEED = 4
EXCLUDE_FROM_SCALING = [0.119]      # as in step 5c

# ==================================================================


def side_results(table, rows):
    """
    Per-mode coefficients and the scaling fit, for East and West separately, from 'rows'.
    Returns one vector: [East results..., West results..., West - East...].
    """
    if COMMON_WIND_RANGE is not None:
        U = table["U"][rows]
        rows = rows[(U >= COMMON_WIND_RANGE[0]) & (U <= COMMON_WIND_RANGE[1])]
    direction = table["direction"][rows]
    east = rows[(direction >= EAST_SECTOR[0]) & (direction < EAST_SECTOR[1])]
    west = rows[~((direction >= EAST_SECTOR[0]) & (direction < EAST_SECTOR[1])) & ~np.isnan(direction)]
    out = []
    for side_rows in (east, west):
        values = []
        scaling = ([], [], [])
        for mode in table["modes"]:
            v, e = mode_coefficients(table, side_rows, mode)
            values.extend(v)
            if mode["type"] == "vertical" and all(abs(mode["f0"] - x) > 0.003 for x in EXCLUDE_FROM_SCALING):
                scaling[0].append(mode["f0"])
                scaling[1].append(v[2])
                scaling[2].append(e[2])
        values.extend(scaling_fit(*scaling))
        out.append(np.array(values))
    return np.concatenate([out[0], out[1], out[1] - out[0]])


# --------------------------- Main program ---------------------------

table = load_table(MODES_VS_WIND)
is_east = (table["direction"] >= EAST_SECTOR[0]) & (table["direction"] < EAST_SECTOR[1])
in_range = np.ones(len(table["U"]), dtype=bool)
if COMMON_WIND_RANGE is not None:
    in_range = (table["U"] >= COMMON_WIND_RANGE[0]) & (table["U"] <= COMMON_WIND_RANGE[1])
    print(f"Only recordings with {COMMON_WIND_RANGE[0]} <= U <= {COMMON_WIND_RANGE[1]} m/s")
for label, sel in [("East", is_east & in_range), ("West", ~is_east & ~np.isnan(table["direction"]) & in_range)]:
    U, T = table["U"][sel], table["T"][sel]
    print(f"{label}: {np.sum(sel)} recordings on {len(np.unique(table['day'][sel]))} days, "
          f"U median {np.nanmedian(U):.1f} m/s (95 %: {np.nanpercentile(U, 95):.1f}), "
          f"T median {np.nanmedian(T):.1f} degC")

all_rows = np.arange(len(table["file_name"]))
estimate = side_results(table, all_rows)
samples = block_bootstrap(table["day"], lambda rows: side_results(table, rows), N_BOOT,
                          np.random.default_rng(RANDOM_SEED))
_, low, high = summarize(samples)

n_modes = len(table["modes"])
n = 3 * n_modes + 3               # results per side
names = ["b_f [%/10 m/s]", "c_f [%/degC]", "b_z [%-pt/10 m/s]"]
print(f"\nEast, West and West - East (95 % block bootstrap interval of the difference; * = significant)")
rows_out = []
for j, name in enumerate(names):
    print(f"\n{name}")
    print(" track  f [Hz]  type          East      West      West - East   95 % interval")
    for k, mode in enumerate(table["modes"]):
        i = 3 * k + j
        e, w, d = estimate[i], estimate[n + i], estimate[2 * n + i]
        lo, hi = low[2 * n + i], high[2 * n + i]
        flag = "*" if lo > 0 or hi < 0 else " "
        print(f"  {mode['track']:3d}   {mode['f0']:.3f}  {mode['type']:9s} {e:+8.3f}  {w:+8.3f}   {d:+8.3f}     "
              f"[{lo:+.3f}, {hi:+.3f}] {flag}")
        rows_out.append([mode["track"], mode["f0"], mode["type"], name, e, w, d, lo, hi])

print("\nDamping scaling over the vertical modes:")
for j, label in enumerate(["K [1/m]", "p", "C_heave"]):
    i = 3 * n_modes + j
    e, w, d = estimate[i], estimate[n + i], estimate[2 * n + i]
    print(f"  {label:8s} East {e:.3g} [{low[i]:.3g}, {high[i]:.3g}]   West {w:.3g} [{low[n + i]:.3g}, {high[n + i]:.3g}]"
          f"   West - East {d:+.3g} [{low[2 * n + i]:+.3g}, {high[2 * n + i]:+.3g}]")
    rows_out.append(["all vertical", "", "", label, e, w, d, low[2 * n + i], high[2 * n + i]])

with open(OUTPUT_FOLDER / "direction_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "quantity", "east", "west", "west_minus_east",
                     "diff_ci95_low", "diff_ci95_high"])
    for r in rows_out:
        writer.writerow([v if isinstance(v, str) else f"{v:.5g}" for v in r])

# ---- Plot: East and West coefficients with bootstrap 95 % intervals ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
f0 = np.array([m["f0"] for m in table["modes"]])
markers = {"vertical": "o", "lateral": "s", "torsion": "^"}
for j, ax in enumerate(axes):
    for k, mode in enumerate(table["modes"]):
        m = markers.get(mode["type"], "o")
        for offset, base, color in [(-0.005, 0, "tab:red"), (0.005, n, "tab:blue")]:
            i = base + 3 * k + j
            ax.errorbar(f0[k] + offset, estimate[i], yerr=[[estimate[i] - low[i]], [high[i] - estimate[i]]],
                        fmt=m, color=color, markersize=4, capsize=2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Mode frequency [Hz]")
    ax.set_ylabel(names[j])
    ax.grid(True, alpha=0.3)
axes[0].set_title("Frequency vs. wind (at constant T)")
axes[1].set_title("Frequency vs. temperature (at constant U)")
axes[2].set_title("Damping vs. wind (at constant T)")
axes[0].plot([], [], "o", color="tab:red", label="East winds")
axes[0].plot([], [], "o", color="tab:blue", label="West winds")
axes[0].legend(fontsize=8)
fig.suptitle("East vs. west winds (error bars: block bootstrap 95 % by day)")
fig.savefig(OUTPUT_FOLDER / "direction_comparison.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

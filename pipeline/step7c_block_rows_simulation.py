"""
Hardanger Bridge project - Step 7c
Which number of block rows gives UNBIASED damping? A simulation test with known damping.

Step 7b showed that the damping slopes (damping vs. wind) depend on the number of block rows:
80 block rows gave steeper slopes (C_heave 2.51), 160 block rows flatter ones (2.06), with the
baseline 120 in between (2.27). On real data we cannot tell which one is right. Here we simulate
records of a bridge-like system whose damping we KNOW, and identify them exactly as in step 7a
(SSI-cov + stability check + clustering with the step 3 settings), with 80, 120 and 160 block rows.

Simulated system (ASSUMPTIONS, a simplified Hardanger girder, not an FE model):
  - the real girder sensor layout: 16 accelerometers (H1-H9 east/west, H2 west only, H8 east only),
    y and z components = 32 channels, sampled at 2 Hz, 1860 s per record (like the LowFreq data)
  - sinusoidal mode shapes over the main span (L = 1310 m): lateral modes in y, vertical modes in z
    (east = west), torsion modes in z (east up, west down). Frequencies like the tracked modes.
  - each mode is a single-degree-of-freedom oscillator driven by its own white-noise force
    (the modes are therefore uncorrelated, as SSI assumes); random response level per mode
  - true damping = structural part + quasi-steady aerodynamic part:
        vertical:  zeta = ZETA_STRUCTURAL + rho B C_heave U / (8 pi m f)     (same formula as step 5c)
        lateral:   zeta = ZETA_STRUCTURAL + rho D C_D U / (4 pi m_y f)       (drag damping)
        torsion:   zeta = ZETA_STRUCTURAL                                     (no wind effect assumed)
    with C_heave = C_HEAVE_TRUE. So high wind = high damping of the low vertical modes, as on the bridge.
  - 5 % white measurement noise on every channel

Then, per number of block rows: damping error of each identified mode, damping slope per mode
(least squares damping vs U) compared with the true slope, and C_heave from the slopes with the
same weighted K/f fit as step 5c (scaling_fit in hardanger/regression.py).

The simulation is slow (~1-2 min per record, three SVDs of a large Toeplitz matrix), so the
identified modes are saved per record in results/step7c/records/ and reused when the script runs again.

How to use:
  python -m pipeline.step7c_block_rows_simulation
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from oma.ssi_cov import ssi_cov, check_stability, mac
from oma.mode_clustering import cluster_poles
from hardanger.simulated_girder import (MODES, FS, C_HEAVE_TRUE, true_damping, mode_shape,
                                        simulate_record, match_modes)
from hardanger.regression import least_squares, scaling_fit
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

OUTPUT_FOLDER = RESULTS_FOLDER / "step7c"

# Records: this many simulated records at each mean wind speed (m/s)
WIND_SPEEDS = [2, 6, 10, 14, 18, 22]
RECORDS_PER_SPEED = 8
RANDOM_SEED = 7
# The simulated system (modes, sensors, true damping with C_heave = 2.5, noise) is defined in
# hardanger/simulated_girder.py, so step 7d can use exactly the same system.

# Identification: the same settings as step 3 / 4a / 7a baseline, only the block rows change
BLOCK_ROWS_RUNS = [80, 120, 160]
ORDERS = list(range(2, 161, 2))
MAX_DAMPING = 0.10
TOL_FREQ, TOL_DAMPING, TOL_MAC = 0.01, 0.05, 0.95
MAX_DISTANCE = 0.05
MIN_ORDERS = 20

# Matching an identified mode to a true mode: |df|/f + (1 - MAC) below this
MATCH_DISTANCE = 0.20

# ==================================================================


def identify(y, block_rows):
    """SSI-cov + stability check + clustering with the step 3 settings. Returns freq, damping, shapes."""
    poles, _ = ssi_cov(y, FS, block_rows, ORDERS, MAX_DAMPING)
    poles = check_stability(poles, TOL_FREQ, TOL_DAMPING, TOL_MAC)
    modes, _ = cluster_poles(poles, MAX_DISTANCE, MIN_ORDERS)
    freq = np.array([m["freq"] for m in modes])
    damping = np.array([m["damping"] for m in modes])
    shapes = np.array([m["shape"] for m in modes])
    return freq, damping, shapes


def run_or_load_records(shapes):
    """
    Simulate and identify all records (or load the ones already done).
    Returns U [n_records] and identified damping / frequency [n_block_rows, n_records, n_modes] (NaN = missed).
    """
    folder = OUTPUT_FOLDER / "records"
    folder.mkdir(parents=True, exist_ok=True)
    true_freqs = np.array([m[1] for m in MODES])
    rng = np.random.default_rng(RANDOM_SEED)
    U_list, damping_list, freq_list = [], [], []
    count = 0
    for U in WIND_SPEEDS:
        for r in range(RECORDS_PER_SPEED):
            count += 1
            # One random generator per record, so a rerun gives the same record even if earlier ones are loaded
            record_rng = np.random.default_rng(rng.integers(1e9))
            path = folder / f"U{U:02d}_record{r}.npz"
            if path.exists():
                data = np.load(path)
            else:
                y = simulate_record(record_rng, U, shapes)
                damping_found = np.full((len(BLOCK_ROWS_RUNS), len(MODES)), np.nan)
                freq_found = np.full((len(BLOCK_ROWS_RUNS), len(MODES)), np.nan)
                for b, block_rows in enumerate(BLOCK_ROWS_RUNS):
                    freq, damping, id_shapes = identify(y, block_rows)
                    found = match_modes(freq, id_shapes, true_freqs, shapes, MATCH_DISTANCE)
                    for k in range(len(MODES)):
                        if found[k] >= 0:
                            damping_found[b, k] = damping[found[k]]
                            freq_found[b, k] = freq[found[k]]
                np.savez(path, damping=damping_found, freq=freq_found)
                data = np.load(path)
                print(f"[{count}/{len(WIND_SPEEDS) * RECORDS_PER_SPEED}] U = {U} m/s, record {r}: "
                      f"found {np.sum(~np.isnan(damping_found), axis=1)} of {len(MODES)} modes "
                      f"(block rows {BLOCK_ROWS_RUNS})", flush=True)
            U_list.append(U)
            damping_list.append(data["damping"])
            freq_list.append(data["freq"])
    return np.array(U_list, dtype=float), np.stack(damping_list, axis=1), np.stack(freq_list, axis=1)


def damping_slope(U, damping):
    """Least-squares slope of damping vs U, in %-points per 10 m/s, with its standard error."""
    ok = ~np.isnan(damping)
    X = np.column_stack([np.ones(np.sum(ok)), U[ok]])
    beta, se = least_squares(X, 100 * damping[ok])
    return 10 * beta[1], 10 * se[1]


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
shapes = np.array([mode_shape(t, n) for t, f, n in MODES])
U, damping_id, freq_id = run_or_load_records(shapes)
true_freqs = np.array([m[1] for m in MODES])
true_zeta = np.array([[true_damping(t, f, u) for t, f, n in MODES] for u in U])   # [n_records, n_modes]

print(f"\n{len(U)} simulated records, true C_heave = {C_HEAVE_TRUE}")
print("Damping error = identified - true (mean over records, %-points); slope = damping vs U (%-pt per 10 m/s)")
header = f" {'mode':16s}" + "".join(f"  rows{b:<3d} found / error / slope" for b in BLOCK_ROWS_RUNS) + "   true slope"
print(header)

rows = []
slopes = np.zeros((len(BLOCK_ROWS_RUNS), len(MODES)))
slope_errors = np.zeros((len(BLOCK_ROWS_RUNS), len(MODES)))
for k, (mode_type, f, n) in enumerate(MODES):
    true_slope = 100 * 10 * (true_damping(mode_type, f, 10) - true_damping(mode_type, f, 0)) / 10
    line = f" {mode_type:8s} {f:.3f}"
    for b in range(len(BLOCK_ROWS_RUNS)):
        error = 100 * (damping_id[b, :, k] - true_zeta[:, k])
        found = np.mean(~np.isnan(error))
        slopes[b, k], slope_errors[b, k] = damping_slope(U, damping_id[b, :, k])
        line += f"   {100 * found:5.0f} % {np.nanmean(error):+6.3f} {slopes[b, k]:+6.3f}      "
        rows.append([mode_type, f, BLOCK_ROWS_RUNS[b], found, np.nanmean(error), np.nanstd(error),
                     slopes[b, k], slope_errors[b, k], true_slope])
    print(line + f"   {true_slope:+6.3f}")

# C_heave from the vertical slopes, as in step 5c (0.119 Hz left out there too)
vertical = np.array([t == "vertical" and f != 0.119 for t, f, n in MODES])
C_estimates = []
for b, block_rows in enumerate(BLOCK_ROWS_RUNS):
    _, _, C = scaling_fit(true_freqs[vertical], slopes[b, vertical], slope_errors[b, vertical])
    C_estimates.append(C)
    print(f"block rows {block_rows}: C_heave = {C:.2f}  (true {C_HEAVE_TRUE})")

with open(OUTPUT_FOLDER / "block_rows_simulation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mode_type", "freq_Hz", "block_rows", "share_found", "damping_error_mean_pt",
                     "damping_error_std_pt", "slope_pt_per_10ms", "slope_se", "true_slope_pt_per_10ms"])
    for r in rows:
        writer.writerow([r[0], r[1], r[2], f"{r[3]:.3f}"] + [f"{v:.4f}" for v in r[4:]])
    for b, block_rows in enumerate(BLOCK_ROWS_RUNS):
        writer.writerow(["C_heave", "", block_rows, "", f"{C_estimates[b]:.3f}", "", "", "", C_HEAVE_TRUE])

# ---- Plots ----
colors = ["tab:blue", "tab:green", "tab:orange"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")

# 1: damping error vs true damping, all modes and records (binned means)
ax = axes[0]
bins = np.array([0.0, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.045])
for b, block_rows in enumerate(BLOCK_ROWS_RUNS):
    z_true = 100 * true_zeta.ravel()
    error = 100 * (damping_id[b].ravel() - true_zeta.ravel())
    ok = ~np.isnan(error)
    centres, means, ses = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = ok & (z_true >= 100 * low) & (z_true < 100 * high)
        if np.sum(inside) >= 5:
            centres.append(np.mean(z_true[inside]))
            means.append(np.mean(error[inside]))
            ses.append(np.std(error[inside]) / np.sqrt(np.sum(inside)))
    ax.errorbar(centres, means, yerr=ses, fmt="o-", color=colors[b], capsize=3, label=f"{block_rows} block rows")
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlabel("True damping ratio [%]")
ax.set_ylabel("Identified - true damping [%-points] (mean +- se)")
ax.set_title("Damping error vs. true damping (all modes)")
ax.legend()
ax.grid(True, alpha=0.3)

# 2: identified vs true slope for each vertical mode
ax = axes[1]
for b, block_rows in enumerate(BLOCK_ROWS_RUNS):
    ax.errorbar(true_freqs[vertical] * (1 + 0.01 * (b - 1)), slopes[b, vertical], yerr=slope_errors[b, vertical],
                fmt="o", color=colors[b], capsize=2, markersize=4, label=f"{block_rows} block rows")
f_line = np.linspace(0.13, 0.92, 200)
true_line = [100 * 10 * (true_damping("vertical", f, 10) - true_damping("vertical", f, 0)) / 10 for f in f_line]
ax.plot(f_line, true_line, "k-", label=f"true (C_heave = {C_HEAVE_TRUE})")
ax.set_xlabel("Mode frequency [Hz]")
ax.set_ylabel("Damping slope [%-points per 10 m/s]")
ax.set_title("Vertical modes: identified vs. true damping slope")
ax.legend()
ax.grid(True, alpha=0.3)

# 3: C_heave per number of block rows, with the step 7b values on real data for comparison
ax = axes[2]
ax.plot(BLOCK_ROWS_RUNS, C_estimates, "o-", color="tab:blue", label="simulation (identified)")
ax.axhline(C_HEAVE_TRUE, color="k", linestyle="--", label="simulation (true)")
ax.plot(BLOCK_ROWS_RUNS, [2.51, 2.27, 2.06], "s-", color="tab:red", label="real data, step 7b (150 recordings)")
ax.set_xlabel("Number of block rows (max. time lag in s at 2 Hz)")
ax.set_ylabel("C_heave")
ax.set_xticks(BLOCK_ROWS_RUNS)
ax.set_title("Heave coefficient from the damping slopes")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(f"Step 7c: SSI-cov damping with 80 / 120 / 160 block rows on simulated records "
             f"({len(U)} records, {len(MODES)} modes, known quasi-steady damping)")
fig.savefig(OUTPUT_FOLDER / "block_rows_simulation.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

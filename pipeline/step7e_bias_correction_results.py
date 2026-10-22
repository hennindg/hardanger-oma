"""
Hardanger Bridge project - Step 7e
Results of the damping bias correction (step 7d).

  1. Validation: the correction applied to simulated records with known damping (step 7c system,
     true C_heave = 2.5). Damping error before / after correction, and C_heave before / after.
  2. Calibration check: does each simulated twin look like its real recording?
     (ratio of channel standard deviations, first singular value spectra for three examples)
  3. Real recordings: the estimated bias as a function of the identified damping.
  4. Real recordings: damping slopes and C_heave before / after correction for 80, 120 and 160 block
     rows, with a block bootstrap (by day) of the corrected 120-row result. If the correction works,
     the three corrected values should agree.

Needs the results of pipeline/step7a_robustness_run.py and pipeline/step7d_bias_correction.py.

How to use:
  python -m pipeline.step7e_bias_correction_results
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from oma.block_bootstrap import block_bootstrap, summarize
from hardanger.regression import least_squares, scaling_fit
from hardanger.robustness import step5_tracks, match_variants, build_table, conclusions
from hardanger import simulated_girder as girder
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

STEP7D_FOLDER = RESULTS_FOLDER / "step7d"
ROBUSTNESS_FOLDER = RESULTS_FOLDER / "robustness"
MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step7d"

# Variants before / after correction, with their number of block rows
PAIRS = [("rows80", "rows80_corrected", 80), ("baseline", "baseline_corrected", 120),
         ("rows160", "rows160_corrected", 160)]
MATCH_DISTANCE = 0.20
EXCLUDE_FROM_SCALING = [0.119]       # as in step 5c
N_BOOT = 500
RANDOM_SEED = 12
WIND_TUNNEL_C_HEAVE = 2.53           # quasi-steady, Oiseth et al. (2010)

# Bins of the identified damping (%) for the bias plot
DAMPING_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# ==================================================================


def damping_slope(U, damping):
    """Least-squares slope of damping (fraction) vs U, in %-points per 10 m/s, with its standard error."""
    ok = ~np.isnan(damping)
    X = np.column_stack([np.ones(np.sum(ok)), U[ok]])
    beta, se = least_squares(X, 100 * damping[ok])
    return 10 * beta[1], 10 * se[1]


def validation_results():
    """Identified and corrected damping of the simulated records, with the true damping. Arrays [n_records, n_modes]."""
    U, identified, corrected = [], [], []
    for path in sorted((STEP7D_FOLDER / "validation").glob("*.npz")):
        data = np.load(path)
        U.append(float(data["U"]))
        identified.append(data["identified"])
        corrected.append(data["corrected"])
    U = np.array(U)
    true = np.array([[girder.true_damping(t, f, u) for t, f, n in girder.MODES] for u in U])
    return U, np.array(identified), np.array(corrected), true


def simulated_C_heave(U, damping):
    """C_heave from the vertical-mode damping slopes of the simulated records (0.119 Hz left out, as step 5c)."""
    freqs, slopes, errors = [], [], []
    for k, (mode_type, f, n) in enumerate(girder.MODES):
        if mode_type == "vertical" and f != 0.119:
            slope, se = damping_slope(U, damping[:, k])
            freqs.append(f)
            slopes.append(slope)
            errors.append(se)
    return scaling_fit(freqs, slopes, errors)[2]


def binned_mean(x, y, bins):
    """Mean of y and its standard error in bins of x (bins with at least 5 values)."""
    centres, means, ses = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (x >= low) & (x < high) & ~np.isnan(y)
        if np.sum(inside) >= 5:
            centres.append(np.mean(x[inside]))
            means.append(np.mean(y[inside]))
            ses.append(np.std(y[inside]) / np.sqrt(np.sum(inside)))
    return np.array(centres), np.array(means), np.array(ses)


# --------------------------- Main program ---------------------------

# ---- 1. Validation on simulated records ----
U_sim, z_identified, z_corrected, z_true = validation_results()
C_sim_before = simulated_C_heave(U_sim, z_identified)
C_sim_after = simulated_C_heave(U_sim, z_corrected)
error_before = 100 * (z_identified - z_true)
error_after = 100 * (z_corrected - z_true)
print(f"1. Validation on {len(U_sim)} simulated records (true C_heave = {girder.C_HEAVE_TRUE}):")
print(f"   damping error (mean over all modes and records): before {np.nanmean(error_before):+.3f} %-pt, "
      f"after {np.nanmean(error_after):+.3f} %-pt")
for low, high in [(0, 1), (1, 2), (2, 5)]:
    inside = (100 * z_true >= low) & (100 * z_true < high)
    print(f"   true damping {low}-{high} %: before {np.nanmean(error_before[inside]):+.3f}, "
          f"after {np.nanmean(error_after[inside]):+.3f} %-pt  (random error per record after: "
          f"{np.nanstd(error_after[inside]):.2f} %-pt)")
print(f"   C_heave: before {C_sim_before:.2f}, after {C_sim_after:.2f}")

# ---- 2. Calibration check ----
calibration = {}
for path in sorted((STEP7D_FOLDER / "calibration").glob("*.npz")):
    calibration[path.stem] = np.load(path)
std_ratio = np.array([np.median(c["std_twin"] / c["std_real"]) for c in calibration.values()])
noise_share = np.array([np.median(c["noise"] / c["std_real"]) for c in calibration.values()])
print(f"\n2. Calibration check on {len(calibration)} recordings: twin / real channel std = "
      f"median {np.median(std_ratio):.2f} (5-95 %: {np.percentile(std_ratio, 5):.2f}-{np.percentile(std_ratio, 95):.2f}); "
      f"estimated noise = {100 * np.median(noise_share):.1f} % of the channel std (median)")

# ---- 3-4. Real recordings ----
with open(ROBUSTNESS_FOLDER / "recording_conditions.csv", newline="") as f:
    conditions = {r["file_name"]: r for r in csv.DictReader(f)}
variants = [v for pair in PAIRS for v in pair[:2]]
files = sorted(f for f in conditions if all((ROBUSTNESS_FOLDER / v / f"{f}_modes.npz").exists() for v in variants))
track_freqs, tracks_info = step5_tracks(MODES_VS_WIND)
track_ids = sorted(tracks_info)
values, _, _ = match_variants(ROBUSTNESS_FOLDER, files, variants, track_freqs, track_ids, MATCH_DISTANCE)
tables = {v: build_table(files, conditions, tracks_info, values[v]) for v in variants}
rows_all = np.arange(len(files))
results = {v: conclusions(tables[v], rows_all, EXCLUDE_FROM_SCALING) for v in variants}
modes = tables["baseline"]["modes"]
n_m = len(modes)

# Bias against the identified damping (120 block rows), all modes
identified_real = np.concatenate([values["baseline"][t][1] for t in track_ids])
corrected_real = np.concatenate([values["baseline_corrected"][t][1] for t in track_ids])
bias_real = identified_real - corrected_real
centres, mean_bias, se_bias = binned_mean(identified_real, bias_real, DAMPING_BINS)
print(f"\n3. Real recordings ({len(files)}), 120 block rows: mean bias (identified - corrected) per identified damping")
for c, m, s in zip(centres, mean_bias, se_bias):
    print(f"   identified {c:.2f} %: bias {m:+.3f} +- {s:.3f} %-pt")

samples = block_bootstrap(tables["baseline_corrected"]["day"],
                          lambda rows: conclusions(tables["baseline_corrected"], rows, EXCLUDE_FROM_SCALING),
                          N_BOOT, np.random.default_rng(RANDOM_SEED))
boot_se, low, high = summarize(samples)
print(f"\n4. C_heave on the real recordings (wind tunnel {WIND_TUNNEL_C_HEAVE}):")
for before, after, rows in PAIRS:
    print(f"   {rows} block rows: before {results[before][2 * n_m]:.2f}, after correction {results[after][2 * n_m]:.2f}")
print(f"   corrected, 120 block rows: 95 % bootstrap interval [{low[2 * n_m]:.2f}, {high[2 * n_m]:.2f}]")

with open(OUTPUT_FOLDER / "bias_correction_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["quantity", "block_rows", "before", "after"])
    writer.writerow(["simulation C_heave (true 2.5)", 120, f"{C_sim_before:.3f}", f"{C_sim_after:.3f}"])
    writer.writerow(["simulation mean damping error [%-pt]", 120, f"{np.nanmean(error_before):.4f}",
                     f"{np.nanmean(error_after):.4f}"])
    for before, after, rows in PAIRS:
        writer.writerow(["real C_heave", rows, f"{results[before][2 * n_m]:.3f}", f"{results[after][2 * n_m]:.3f}"])
    writer.writerow(["real C_heave corrected 95 % low / high", 120, f"{low[2 * n_m]:.3f}", f"{high[2 * n_m]:.3f}"])
    for j, mode in enumerate(modes):
        writer.writerow([f"real b_z {mode['f0']:.3f} Hz [%-pt/10 m/s]", 120, f"{results['baseline'][j]:.4f}",
                         f"{results['baseline_corrected'][j]:.4f}"])

# ---- Plots ----
fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

ax = axes[0][0]
for errors, label, color in [(error_before, "identified", "tab:orange"), (error_after, "after correction", "tab:blue")]:
    c, m, s = binned_mean(100 * z_true.ravel(), errors.ravel(), [0.0, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5])
    ax.errorbar(c, m, yerr=s, fmt="o-", color=color, capsize=3, label=label)
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlabel("True damping ratio [%]")
ax.set_ylabel("Error vs. true damping [%-points] (mean +- se)")
ax.set_title(f"1. Validation on {len(U_sim)} simulated records (120 block rows)\n"
             f"C_heave: identified {C_sim_before:.2f}, corrected {C_sim_after:.2f}, true {girder.C_HEAVE_TRUE}")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0][1]
U_of = {name: float(conditions[name]["mean_wind_speed_ms"]) for name in calibration if name in conditions}
examples = sorted(U_of, key=U_of.get)
examples = [examples[0], examples[len(examples) // 2], examples[-1]]
for name, color in zip(examples, ["tab:green", "tab:blue", "tab:red"]):
    c = calibration[name]
    ax.semilogy(c["sv_freqs"], c["sv_real"], color=color, label=f"real, U = {U_of[name]:.1f} m/s")
    ax.semilogy(c["sv_freqs"], c["sv_twin"], color=color, linestyle="--", alpha=0.8, label="twin")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("First singular value of the spectral matrix")
ax.set_title(f"2. Calibration: real recording vs. simulated twin\n"
             f"channel std twin / real: median {np.median(std_ratio):.2f} over {len(calibration)} recordings")
ax.set_xlim(0, 1)
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

ax = axes[1][0]
ax.plot(identified_real, bias_real, ".", color="0.7", markersize=2, label="one mode in one recording")
ax.errorbar(centres, mean_bias, yerr=se_bias, fmt="ko-", capsize=3, label="mean per damping bin")
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlim(0, 6)
ax.set_ylim(-3, 3)
ax.set_xlabel("Identified damping ratio [%] (120 block rows)")
ax.set_ylabel("Estimated bias = identified - corrected [%-points]")
ax.set_title(f"3. Estimated SSI damping bias of the real recordings ({len(files)} recordings)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1][1]
rows_list = [p[2] for p in PAIRS]
ax.plot(rows_list, [results[p[0]][2 * n_m] for p in PAIRS], "s-", color="tab:orange", label="identified (step 7b)")
ax.plot(rows_list, [results[p[1]][2 * n_m] for p in PAIRS], "o-", color="tab:blue", label="after bias correction")
ax.errorbar([120], [results["baseline_corrected"][2 * n_m]],
            yerr=[[results["baseline_corrected"][2 * n_m] - low[2 * n_m]], [high[2 * n_m] - results["baseline_corrected"][2 * n_m]]],
            fmt="none", color="tab:blue", capsize=5, label="95 % bootstrap (corrected, 120 rows)")
ax.axhline(WIND_TUNNEL_C_HEAVE, color="tab:green", linestyle="--", label=f"wind tunnel, quasi-steady ({WIND_TUNNEL_C_HEAVE})")
ax.set_xticks(rows_list)
ax.set_xlabel("Number of block rows")
ax.set_ylabel("$C_{heave}$")
ax.set_title("4. Heave coefficient from the real damping slopes")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Step 7d/7e: correcting the SSI damping bias with a simulated twin of every recording")
fig.savefig(OUTPUT_FOLDER / "bias_correction.png", dpi=100)
print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

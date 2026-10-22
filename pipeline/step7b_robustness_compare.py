"""
Hardanger Bridge project - Step 7b
Robustness check, part 2: how much do the identified modes and the main conclusions change with
the SSI / mode-picking settings? (Needs the results of pipeline/step7a_robustness_run.py.)

  1. The baseline variant must reproduce step 4 (same settings). Its modes are linked to the step 5
     tracks by their (identical) frequency, so we know which physical mode each one is.
  2. The modes of every variant are matched to the baseline modes of the same recording
     (distance |df|/f + (1 - MAC) <= 0.2, as in the tracking).
  3. Per variant and mode: how often the mode is still found, change in frequency and damping.
  4. The main conclusions recomputed for every variant on the same recordings:
     damping slope with wind b_z, temperature coefficient c_f, and C_heave (step 5c fit).
     The change caused by the settings is compared with the statistical uncertainty of the baseline
     (block bootstrap by day, 95 % interval).

How to use:
  python -m pipeline.step7b_robustness_compare
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from oma.block_bootstrap import block_bootstrap, summarize
from hardanger.robustness import load_variant, step5_tracks, match_variants, build_table, conclusions
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

ROBUSTNESS_FOLDER = RESULTS_FOLDER / "robustness"
MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"

VARIANTS = ["baseline", "rows80", "rows160", "order120", "order200", "tol_relaxed", "tol_strict",
            "dist0.03", "dist0.08", "minord15", "minord30"]
MATCH_DISTANCE = 0.20
EXCLUDE_FROM_SCALING = [0.119]       # as in step 5c
N_BOOT = 500
RANDOM_SEED = 10

# ==================================================================


# --------------------------- Main program ---------------------------

with open(ROBUSTNESS_FOLDER / "recording_conditions.csv", newline="") as f:
    conditions = {r["file_name"]: r for r in csv.DictReader(f)}
files = sorted(f for f in conditions if all((ROBUSTNESS_FOLDER / v / f"{f}_modes.npz").exists() for v in VARIANTS))
print(f"{len(files)} recordings with results for all {len(VARIANTS)} variants")

track_freqs, tracks_info = step5_tracks(MODES_VS_WIND)
track_ids = sorted(tracks_info)

# ---- 1-3. Link the baseline to the step 5 tracks, match every variant to the baseline ----
values, n_matched, n_expected = match_variants(ROBUSTNESS_FOLDER, files, VARIANTS, track_freqs, track_ids, MATCH_DISTANCE)
print(f"Baseline reproduces step 4: {n_matched} of {n_expected} tracked modes have identical frequencies")

print("\nPer variant (median over the tracked modes):  found = share of baseline modes still found;"
      "  |df| = median |frequency change| [%],  |dzeta| = median |damping change| [%-points]")
print(" variant        found     |df| [%]    |dzeta| [%-pt]   modes per recording")
summary = {}
for v in VARIANTS:
    found, df, dz = [], [], []
    for t in track_ids:
        bf, bz = values["baseline"][t]
        vf, vz = values[v][t]
        has_base = ~np.isnan(bf)
        if np.sum(has_base) < 10:
            continue
        found.append(np.mean(~np.isnan(vf[has_base])))
        both = has_base & ~np.isnan(vf)
        df.append(np.median(np.abs(100 * (vf[both] - bf[both]) / bf[both])))
        dz.append(np.median(np.abs(vz[both] - bz[both])))
    n_modes = np.mean([len(load_variant(ROBUSTNESS_FOLDER, v, f)["freq"]) for f in files])
    summary[v] = (np.median(found), np.median(df), np.median(dz))
    print(f"  {v:12s}  {100 * np.median(found):5.1f} %    {np.median(df):7.3f}       {np.median(dz):6.3f}          {n_modes:5.1f}")

# ---- 4. Main conclusions per variant, compared with the bootstrap uncertainty of the baseline ----
tables = {v: build_table(files, conditions, tracks_info, values[v]) for v in VARIANTS}
rows_all = np.arange(len(files))
results = {v: conclusions(tables[v], rows_all, EXCLUDE_FROM_SCALING) for v in VARIANTS}
samples = block_bootstrap(tables["baseline"]["day"], lambda rows: conclusions(tables["baseline"], rows, EXCLUDE_FROM_SCALING), N_BOOT,
                          np.random.default_rng(RANDOM_SEED))
boot_se, low, high = summarize(samples)

modes = tables["baseline"]["modes"]
n_m = len(modes)
print(f"\nMain conclusions on these {len(files)} recordings: baseline value, 95 % bootstrap interval, and the")
print("largest change over all variants (in units of the bootstrap standard error)")
print(" quantity                         baseline      95 % interval        largest change   (variant)")
rows_out = []
for j, mode in enumerate(modes):
    for label, i in [(f"b_z {mode['f0']:.3f} Hz [%-pt/10 m/s]", j), (f"c_f {mode['f0']:.3f} Hz [%/degC]", n_m + j)]:
        changes = {v: results[v][i] - results["baseline"][i] for v in VARIANTS[1:]}
        worst = max(changes, key=lambda v: abs(changes[v]) if not np.isnan(changes[v]) else -1)
        ratio = abs(changes[worst]) / boot_se[i] if boot_se[i] > 0 else np.nan
        rows_out.append([label, results["baseline"][i], low[i], high[i], changes[worst], ratio, worst])
i = 2 * n_m
changes = {v: results[v][i] - results["baseline"][i] for v in VARIANTS[1:]}
worst = max(changes, key=lambda v: abs(changes[v]))
rows_out.append(["C_heave", results["baseline"][i], low[i], high[i], changes[worst], abs(changes[worst]) / boot_se[i], worst])
for r in rows_out:
    if r[0].startswith("b_z") or r[0] == "C_heave":
        print(f"  {r[0]:30s}  {r[1]:+8.3f}   [{r[2]:+.3f}, {r[3]:+.3f}]    {r[4]:+8.3f} ({r[5]:.1f} se)   {r[6]}")
ratios = np.array([r[5] for r in rows_out if not np.isnan(r[5])])
print(f"\nOver all {len(ratios)} coefficients: largest setting-induced change = median {np.median(ratios):.2f}, "
      f"max {ratios.max():.2f} bootstrap standard errors")
print("C_heave per variant: " + ", ".join(f"{v} {results[v][2 * n_m]:.2f}" for v in VARIANTS))

with open(ROBUSTNESS_FOLDER / "robustness_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["variant", "share_of_modes_found", "median_abs_freq_change_percent", "median_abs_damping_change_pt",
                     "C_heave"])
    for v in VARIANTS:
        writer.writerow([v] + [f"{x:.4f}" for x in summary[v]] + [f"{results[v][2 * n_m]:.3f}"])
    writer.writerow([])
    writer.writerow(["quantity", "baseline", "ci95_low", "ci95_high", "largest_change", "largest_change_in_se", "variant"])
    for r in rows_out:
        writer.writerow([r[0]] + [f"{x:.4f}" for x in r[1:6]] + [r[6]])

# ---- Plots ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
ax = axes[0]
x = np.arange(len(VARIANTS))
ax.bar(x - 0.2, [100 * summary[v][0] for v in VARIANTS], 0.4, label="baseline modes found [%]")
ax2 = ax.twinx()
ax2.bar(x + 0.2, [summary[v][2] for v in VARIANTS], 0.4, color="tab:orange", label="median |damping change| [%-pt]")
ax.set_xticks(x)
ax.set_xticklabels(VARIANTS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Baseline modes found [%]")
ax2.set_ylabel("Median |damping change| [%-points]", color="tab:orange")
ax.set_ylim(0, 105)
ax.set_title("Mode detection and damping change per variant")

ax = axes[1]
ax.errorbar([0], [results["baseline"][2 * n_m]], yerr=[[results["baseline"][2 * n_m] - low[2 * n_m]],
            [high[2 * n_m] - results["baseline"][2 * n_m]]], fmt="ks", capsize=4, label="baseline (95 % bootstrap)")
ax.plot(x[1:], [results[v][2 * n_m] for v in VARIANTS[1:]], "o", color="tab:blue", label="variants")
ax.axhline(2.53, color="tab:green", linestyle="--", label="wind tunnel, quasi-steady (2.53)")
ax.set_xticks(x)
ax.set_xticklabels(VARIANTS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("$C_{heave}$")
ax.set_title("Aerodynamic damping coefficient per variant")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[2]
f0 = np.array([m["f0"] for m in modes])
ax.errorbar(f0, results["baseline"][:n_m], yerr=[results["baseline"][:n_m] - low[:n_m], high[:n_m] - results["baseline"][:n_m]],
            fmt="ks", capsize=2, markersize=4, label="baseline (95 % bootstrap)")
for v in VARIANTS[1:]:
    ax.plot(f0, results[v][:n_m], ".", markersize=5, alpha=0.7, label=v)
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlabel("Mode frequency [Hz]")
ax.set_ylabel("Damping slope b_z [%-points per 10 m/s]")
ax.set_title("Damping vs. wind per mode: variants vs. baseline")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
fig.suptitle(f"Robustness of the results to the SSI / mode-picking settings ({len(files)} recordings)")
fig.savefig(ROBUSTNESS_FOLDER / "robustness.png", dpi=100)

print(f"\nResults saved in {ROBUSTNESS_FOLDER}")
plt.show()

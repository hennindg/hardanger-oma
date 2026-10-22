"""
Hardanger Bridge project - Step 6
Environmental normalization and novelty detection (towards damage detection).

Natural frequencies change with wind and temperature (steps 5-5j). A damage would also change
them, typically by lowering them. To see a damage, the environmental effects must be removed first
("normalization"); what is left (the residual) should be noise only, as long as the structure is
unchanged.

What it does:
  1. Per mode, a normalization model is fitted on a REFERENCE period (training):
         f [% of mean] = a + b1 U + b2 U^2 + c T
     U^2 = static wind load (step 5i), T = temperature (step 5b).
  2. The model is applied to the later TEST period (as in real monitoring). Residual scatter is
     compared for: no normalization, temperature only, and the full model.
  3. Control charts:
     - per mode: residual with +-3 sigma limits from the reference period,
     - all selected modes together: Hotelling T^2 = r^T S^-1 r (r = residual vector, S its
       covariance in the reference period). Alarm limit = 99 % level of T^2 in the reference period.
     The bridge is assumed undamaged, so every alarm in the test period is a FALSE alarm.
  4. Detection limit: an artificial frequency drop of x % is added to all selected modes in the
     test period; the detection rate is computed as a function of x, for single recordings and for
     daily averages.

How to use:
  python -m pipeline.step6_normalization
"""

from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from hardanger.regression import load_table
from hardanger.novelty_detection import residuals_for_mode, hotelling, daily_means
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step6"

# Reference (training) period: recordings before this date; test period: from this date on
TRAIN_END = "2016-06-01"

# Modes used for novelty detection (track median frequency, matched within 0.003 Hz):
# vertical and torsion modes found in >= 95 % of the recordings, without the 0.119 Hz mode
# (unexplained behaviour, see studies/mode_0119/inspect_0117.py)
SELECTED_MODES = [0.142, 0.206, 0.212, 0.276, 0.333, 0.371, 0.401, 0.417, 0.470, 0.547, 0.626, 0.713, 0.901]

# Alarm limits
SIGMA_LIMIT = 3.0              # per-mode control chart: +- 3 standard deviations
T2_QUANTILE = 0.99             # multivariate chart: 99 % level of the reference period

# Artificial frequency drops (%) for the detection limit
DAMAGE_LEVELS = np.arange(0.0, 2.01, 0.1)

# ==================================================================


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
table = load_table(MODES_VS_WIND)
times = np.array([datetime.strptime(d, "%Y-%m-%d") for d in table["day"]])
train = table["day"] < TRAIN_END
test = ~train
print(f"Reference period (before {TRAIN_END}): {np.sum(train)} recordings on {len(np.unique(table['day'][train]))} days")
print(f"Test period (from {TRAIN_END}):        {np.sum(test)} recordings on {len(np.unique(table['day'][test]))} days")
print(f"Wind: reference median {np.nanmedian(table['U'][train]):.1f} m/s, test {np.nanmedian(table['U'][test]):.1f} m/s;"
      f"  temperature: reference {np.nanmin(table['T'][train]):.0f}...{np.nanmax(table['T'][train]):.0f} degC, "
      f"test {np.nanmin(table['T'][test]):.0f}...{np.nanmax(table['T'][test]):.0f} degC")

modes = [m for m in table["modes"] if any(abs(m["f0"] - f) < 0.003 for f in SELECTED_MODES)]
names = [f"{m['f0']:.3f}" for m in modes]

# ---- 1-2. Normalization models and residual scatter ----
models = {"none": [], "T only": ["T"], "U + U^2 + T": ["U", "U2", "T"]}
residuals = {}
print("\nResidual standard deviation [%] per mode (reference / test period):")
print(" mode [Hz]   " + "   ".join(f"{name:>17s}" for name in models))
std_table = []
for mode in modes:
    mean_freq = np.nanmean(mode["freq"][train])
    row = []
    for name, variables in models.items():
        r = residuals_for_mode(table, mode, train, variables, mean_freq)
        residuals[(mode["f0"], name)] = r
        row.append((np.nanstd(r[train]), np.nanstd(r[test])))
    std_table.append(row)
    print(f"   {mode['f0']:.3f}    " + "   ".join(f"{a:7.3f} / {b:7.3f}" for a, b in row))

# Full-model residual matrix [n_recordings, n_modes]
R = np.column_stack([residuals[(m["f0"], "U + U^2 + T")] for m in modes])
complete = ~np.isnan(R).any(axis=1)
print(f"\n{np.sum(complete & train)} reference and {np.sum(complete & test)} test recordings have all "
      f"{len(modes)} selected modes")

# ---- 3a. Per-mode control charts: false alarm rate in the test period ----
sigma = np.nanstd(R[train], axis=0)
mu = np.nanmean(R[train], axis=0)
print(f"\nPer-mode control chart (+-{SIGMA_LIMIT} sigma from the reference period):")
print(" mode [Hz]   sigma [%]   test mean shift [%]   false alarms in test")
for k, mode in enumerate(modes):
    r = R[test, k]
    r = r[~np.isnan(r)]
    alarms = np.mean(np.abs(r - mu[k]) > SIGMA_LIMIT * sigma[k])
    print(f"   {mode['f0']:.3f}      {sigma[k]:.3f}         {np.mean(r) - mu[k]:+.3f}              {100 * alarms:.1f} %")

# ---- 3b. Hotelling T^2 ----
R_train = R[train & complete]
R_test = R[test & complete]
days_test = table["day"][test & complete]
days_train = table["day"][train & complete]
mean_vec = np.mean(R_train, axis=0)
S_inv = np.linalg.inv(np.cov(R_train, rowvar=False))
T2_train = hotelling(R_train, mean_vec, S_inv)
T2_test = hotelling(R_test, mean_vec, S_inv)
limit = np.quantile(T2_train, T2_QUANTILE)

# Daily averages: average residual vectors per day, own covariance and limit from the reference days
_, D_train = daily_means(R_train, days_train)
labels_test, D_test = daily_means(R_test, days_test)
D_mean = np.mean(D_train, axis=0)
D_S_inv = np.linalg.inv(np.cov(D_train, rowvar=False))
T2_day_train = hotelling(D_train, D_mean, D_S_inv)
T2_day_test = hotelling(D_test, D_mean, D_S_inv)
limit_day = np.quantile(T2_day_train, T2_QUANTILE)

print(f"\nHotelling T^2 over {len(modes)} modes, limit = {100 * T2_QUANTILE:.0f} % level of the reference period:")
print(f"   single recordings: limit {limit:.1f}; false alarms in test {100 * np.mean(T2_test > limit):.1f} % "
      f"(expected {100 * (1 - T2_QUANTILE):.0f} % if the test period behaves like the reference)")
print(f"   daily averages:    limit {limit_day:.1f}; false alarms in test {100 * np.mean(T2_day_test > limit_day):.1f} % "
      f"of {len(D_test)} days")

# ---- 4. Detection limit: artificial uniform frequency drop of x % in all selected modes ----
rate_single, rate_day = [], []
for x in DAMAGE_LEVELS:
    rate_single.append(np.mean(hotelling(R_test - x, mean_vec, S_inv) > limit))
    rate_day.append(np.mean(hotelling(D_test - x, D_mean, D_S_inv) > limit_day))
rate_single, rate_day = np.array(rate_single), np.array(rate_day)


def level_for(rate, target):
    """Smallest damage level with a detection rate of at least 'target'."""
    hits = np.where(rate >= target)[0]
    return DAMAGE_LEVELS[hits[0]] if len(hits) else np.nan


print("\nDetection of a uniform frequency drop in all selected modes (test period):")
for target in (0.5, 0.9):
    print(f"   detected in {100 * target:.0f} % of cases: single recordings at {level_for(rate_single, target):.1f} %, "
          f"daily averages at {level_for(rate_day, target):.1f} % frequency drop")

with open(OUTPUT_FOLDER / "normalization_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mode_Hz"] + [f"std_{n}_{p}" for n in models for p in ("ref", "test")])
    for mode, row in zip(modes, std_table):
        writer.writerow([f"{mode['f0']:.3f}"] + [f"{v:.4f}" for pair in row for v in pair])
with open(OUTPUT_FOLDER / "detection_curve.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frequency_drop_percent", "detection_rate_single", "detection_rate_daily"])
    for x, a, b in zip(DAMAGE_LEVELS, rate_single, rate_day):
        writer.writerow([f"{x:.2f}", f"{a:.3f}", f"{b:.3f}"])

# ---- Plots ----
# Figure 1: residual scatter per mode for the three models
fig1, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
x = np.arange(len(modes))
for j, (name, color) in enumerate(zip(models, ["0.75", "tab:orange", "tab:blue"])):
    ax.bar(x + (j - 1) * 0.27, [row[j][1] for row in std_table], width=0.27, color=color, label=name)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_xlabel("Mode [Hz]")
ax.set_ylabel("Residual standard deviation, test period [%]")
ax.set_title("How much of the frequency scatter the normalization removes (model fitted on the reference period)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig1.savefig(OUTPUT_FOLDER / "normalization_scatter.png", dpi=100)

# Figure 2: residual control charts of a few modes over time
show = [0, 3, 6, 9, 12] if len(modes) > 12 else list(range(min(5, len(modes))))
fig2, axes = plt.subplots(len(show), 1, figsize=(14, 2.3 * len(show)), sharex=True, layout="constrained")
for ax, k in zip(axes, show):
    for sel, color, label in [(train, "0.5", "reference"), (test, "tab:blue", "test")]:
        ax.plot(times[sel], R[sel, k], ".", color=color, markersize=2, label=label)
    for sign in (-1, 1):
        ax.axhline(mu[k] + sign * SIGMA_LIMIT * sigma[k], color="tab:red", linewidth=0.8)
    ax.set_ylabel(f"{names[k]} Hz\nresidual [%]", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=7, markerscale=4, loc="upper left")
axes[0].set_title(f"Normalized frequency residuals with +-{SIGMA_LIMIT:.0f} sigma limits (reference period before {TRAIN_END})")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig2.savefig(OUTPUT_FOLDER / "control_charts.png", dpi=90)

# Figure 3: Hotelling T^2 chart and detection curve
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
ax1.semilogy(times[train & complete], T2_train, ".", color="0.5", markersize=2, label="reference")
ax1.semilogy(times[test & complete], T2_test, ".", color="tab:blue", markersize=2, label="test")
ax1.axhline(limit, color="tab:red", label=f"{100 * T2_QUANTILE:.0f} % limit")
ax1.set_ylabel(f"Hotelling $T^2$ ({len(modes)} modes)")
ax1.set_title("Multivariate novelty index of the normalized frequencies (single recordings)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.legend(markerscale=4)
ax1.grid(True, alpha=0.3)
ax2.plot(DAMAGE_LEVELS, 100 * rate_single, "o-", label="single recordings")
ax2.plot(DAMAGE_LEVELS, 100 * rate_day, "s-", label="daily averages")
ax2.axhline(100 * (1 - T2_QUANTILE), color="k", linewidth=0.6, linestyle=":")
ax2.set_xlabel("Artificial frequency drop in all selected modes [%]")
ax2.set_ylabel("Detection rate in the test period [%]")
ax2.set_title("Detection limit")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig3.savefig(OUTPUT_FOLDER / "novelty_detection.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

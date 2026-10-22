"""
Hardanger Bridge project - Step 6b
Detection limits for more realistic damage patterns.

Step 6 tested a UNIFORM frequency drop in all modes. A real damage changes each mode by a
different amount. By Rayleigh's quotient, a stiffness loss lowers omega^2 of a mode in proportion
to the share of the mode's strain energy that is stored in the damaged part:
    delta(omega^2) / omega^2  =  - (damage) * (strain energy in damaged part) / (total strain energy)
so delta f / f is about half of that.

The vertical frequencies are almost evenly spaced (~0.07 Hz apart), i.e. the bridge behaves like a
TENSIONED BEAM (cables + girder bending) with nearly sinusoidal shapes, k = n pi / L and f ~ k.
For such a beam the bending curvature is ~ k^2 phi, so the bending strain energy density is
~ k^4 phi^2 ~ f^4 phi^2, while the bending share of the total stiffness grows ~ k^2 ~ f^2.
This gives the damage patterns (relative frequency drop per mode):

  uniform            equal in all modes                      (loss of cable stiffness / tension)
  single mode        one mode only                           (bound: what one mode can detect)
  girder bending     ~ f^2                                   (bending stiffness loss over the span)
  local bending      ~ phi(x_d)^2 f^2                        (bending damage at section x_d)

phi(x_d) is taken from our MEASURED vertical mode shapes at the sections with an east/west sensor
pair (median over reference recordings). Torsion modes get zero weight in the bending patterns
(conservative). ASSUMPTIONS: tensioned-beam scaling, sinusoidal curvature; no FE model is used.

Each pattern is scaled so that the MOST AFFECTED mode drops by x %. The detection rate (Hotelling
T^2 of step 6, single recordings and daily averages) is computed for increasing x; reported is the
x needed for 90 % detection.

How to use:
  python -m pipeline.step6b_damage_patterns
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.regression import load_table
from hardanger.data import section_mode_shape
from hardanger.novelty_detection import residuals_for_mode, hotelling, daily_means
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
STEP4_FOLDER = RESULTS_FOLDER / "step4"
OUTPUT_FOLDER = RESULTS_FOLDER / "step6"

# Same as step 6
TRAIN_END = "2016-06-01"
SELECTED_MODES = [0.142, 0.206, 0.212, 0.276, 0.333, 0.371, 0.401, 0.417, 0.470, 0.547, 0.626, 0.713, 0.901]
MODEL_VARIABLES = ["U", "U2", "T"]
T2_QUANTILE = 0.99
SIGMA_LIMIT = 3.0

# Mode shapes: median over this many reference recordings with moderate wind
N_SHAPE_RECORDINGS = 60
SHAPE_WIND_RANGE = (5.0, 15.0)
RANDOM_SEED = 7

# Frequency drop of the most affected mode (%), and the required detection rate
DROP_LEVELS = np.arange(0.0, 10.01, 0.05)
TARGET_RATE = 0.9

# ==================================================================


def channels_from_names(names, xs):
    """Rebuild the channel descriptions (as in girder_channels) from the saved names 'H1 East y'."""
    channels = []
    for name, x in zip(names, xs):
        section, side, component = str(name).split()
        channels.append({"sensor": f"{section} {side}", "section": section, "side": side,
                         "component": component, "x": float(x)})
    return channels


def measured_phi_squared(table, modes, data_set_of, rows):
    """
    Squared vertical mode shape at the paired sections, per mode: median over the given recordings.
    Each shape is normalized to max |vertical| = 1 over the paired sections.
    Returns the section x positions and phi^2 [n_modes, n_sections].
    """
    values = {}          # (mode index, x) -> list of phi^2
    for r in rows:
        name = table["file_name"][r]
        npz = np.load(STEP4_FOLDER / data_set_of[name] / "modes" / f"{name}_modes.npz")
        channels = channels_from_names(npz["channel_names"], npz["channel_x"])
        for k, mode in enumerate(modes):
            f = mode["freq"][r]
            if np.isnan(f):
                continue
            index = int(np.argmin(np.abs(npz["freq"] - f)))
            x, lateral, vertical, torsion = section_mode_shape(npz["shapes"][index], channels)
            paired = ~np.isnan(torsion)                 # sections with both an east and a west sensor
            v = vertical[paired] / np.max(np.abs(vertical[paired]))
            for xi, vi in zip(np.round(x[paired]), v):
                values.setdefault((k, xi), []).append(vi ** 2)
    sections = sorted({xi for (_, xi) in values})
    phi2 = np.array([[np.median(values.get((k, xi), [np.nan])) for xi in sections] for k in range(len(modes))])
    return np.array(sections), phi2


def detection_level(R_test, D_test, weights, mean_vec, S_inv, limit, D_mean, D_S_inv, limit_day):
    """
    Smallest drop x (of the most affected mode) detected in TARGET_RATE of the test recordings / days.
    weights: relative frequency drop per mode (max = 1). A drop of x % lowers the residual by x * w.
    """
    found = [np.nan, np.nan]
    for x in DROP_LEVELS:
        rates = (np.mean(hotelling(R_test - x * weights, mean_vec, S_inv) > limit),
                 np.mean(hotelling(D_test - x * weights, D_mean, D_S_inv) > limit_day))
        for i in range(2):
            if np.isnan(found[i]) and rates[i] >= TARGET_RATE:
                found[i] = x
        if not np.isnan(found).any():
            break
    return found


# --------------------------- Main program ---------------------------

table = load_table(MODES_VS_WIND)
data_set_of = {}
with open(MODES_VS_WIND, newline="") as f:
    for row in csv.DictReader(f):
        data_set_of[row["file_name"]] = row["data_set"]

train = table["day"] < TRAIN_END
test = ~train
modes = [m for m in table["modes"] if any(abs(m["f0"] - f) < 0.003 for f in SELECTED_MODES)]
f0 = np.array([m["f0"] for m in modes])
is_torsion = np.array([m["type"] == "torsion" for m in modes])

# ---- Residuals and T^2 set-up, exactly as in step 6 ----
R = np.column_stack([residuals_for_mode(table, m, train, MODEL_VARIABLES, np.nanmean(m["freq"][train]))
                     for m in modes])
complete = ~np.isnan(R).any(axis=1)
R_train, R_test = R[train & complete], R[test & complete]
mean_vec = np.mean(R_train, axis=0)
S_inv = np.linalg.inv(np.cov(R_train, rowvar=False))
limit = np.quantile(hotelling(R_train, mean_vec, S_inv), T2_QUANTILE)
_, D_train = daily_means(R_train, table["day"][train & complete])
_, D_test = daily_means(R_test, table["day"][test & complete])
D_mean = np.mean(D_train, axis=0)
D_S_inv = np.linalg.inv(np.cov(D_train, rowvar=False))
limit_day = np.quantile(hotelling(D_train, D_mean, D_S_inv), T2_QUANTILE)
sigma = np.std(R_train, axis=0)

# ---- Measured mode shapes at the paired sections ----
rng = np.random.default_rng(RANDOM_SEED)
candidates = np.where(train & complete & (table["U"] >= SHAPE_WIND_RANGE[0]) & (table["U"] <= SHAPE_WIND_RANGE[1]))[0]
shape_rows = rng.choice(candidates, size=min(N_SHAPE_RECORDINGS, len(candidates)), replace=False)
sections, phi2 = measured_phi_squared(table, modes, data_set_of, shape_rows)
print(f"Mode shapes from {len(shape_rows)} reference recordings; paired sections at x = "
      + ", ".join(f"{x:.0f}" for x in sections) + " m")
print("phi^2 (vertical, normalized) per mode and section:")
for k, m in enumerate(modes):
    print(f"   {m['f0']:.3f} Hz {m['type']:9s} " + " ".join(f"{v:5.2f}" for v in phi2[k]))

# ---- Damage patterns ----
scenarios = {}
scenarios["uniform (cable stiffness)"] = np.ones(len(modes))
bending = np.where(is_torsion, 0.0, f0 ** 2)
scenarios["girder bending, whole span (~f^2)"] = bending / bending.max()
for j, x in enumerate(sections):
    w = np.where(is_torsion, 0.0, np.nan_to_num(phi2[:, j]) * f0 ** 2)
    scenarios[f"local bending at x = {x:+.0f} m"] = w / w.max()

print(f"\nFrequency drop of the MOST AFFECTED mode needed for {100 * TARGET_RATE:.0f} % detection (Hotelling T^2):")
print(" scenario                                     most affected mode    single recordings   daily averages")
rows_out = []
results = {}
for name, w in scenarios.items():
    single, daily = detection_level(R_test, D_test, w, mean_vec, S_inv, limit, D_mean, D_S_inv, limit_day)
    results[name] = (single, daily)
    most = f0[np.argmax(w)]
    print(f"  {name:42s}   {most:.3f} Hz             {single:5.2f} %             {daily:5.2f} %")
    rows_out.append([name, f"{most:.3f}", single, daily] + list(w))

# Single-mode damage: T^2 and the mode's own control chart (one-sided: frequency drop)
print(f"\nSingle-mode drop: needed drop for {100 * TARGET_RATE:.0f} % detection")
print(" mode [Hz]   T^2 single   T^2 daily   own 3-sigma chart (single recordings)")
single_mode = []
for k, m in enumerate(modes):
    w = np.zeros(len(modes))
    w[k] = 1.0
    t2_single, t2_daily = detection_level(R_test, D_test, w, mean_vec, S_inv, limit, D_mean, D_S_inv, limit_day)
    own = np.nan
    for x in DROP_LEVELS:
        if np.mean(R_test[:, k] - x < mean_vec[k] - SIGMA_LIMIT * sigma[k]) >= TARGET_RATE:
            own = x
            break
    single_mode.append((t2_single, t2_daily, own))
    print(f"   {m['f0']:.3f}      {t2_single:5.2f} %     {t2_daily:5.2f} %       {own:5.2f} %")
    rows_out.append([f"single mode {m['f0']:.3f}", f"{m['f0']:.3f}", t2_single, t2_daily] + list(w))

with open(OUTPUT_FOLDER / "damage_patterns.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scenario", "most_affected_mode_Hz", f"drop_for_{int(100 * TARGET_RATE)}pct_single",
                     f"drop_for_{int(100 * TARGET_RATE)}pct_daily"] + [f"weight_{m['f0']:.3f}" for m in modes])
    for r in rows_out:
        writer.writerow([v if isinstance(v, str) else f"{v:.3f}" for v in r])

# ---- Plots ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
ax = axes[0]
for name in ["uniform (cable stiffness)", "girder bending, whole span (~f^2)",
             f"local bending at x = {sections[0]:+.0f} m", f"local bending at x = {sections[len(sections) // 2]:+.0f} m"]:
    ax.plot(f0, scenarios[name], "o-", markersize=4, label=name)
ax.set_xlabel("Mode frequency [Hz]")
ax.set_ylabel("Relative frequency drop (most affected mode = 1)")
ax.set_title("Damage patterns")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[1]
local = [results[f"local bending at x = {x:+.0f} m"] for x in sections]
ax.plot(sections, [r[0] for r in local], "o-", label="single recordings")
ax.plot(sections, [r[1] for r in local], "s-", label="daily averages")
for name, style in [("uniform (cable stiffness)", ":"), ("girder bending, whole span (~f^2)", "--")]:
    ax.axhline(results[name][0], color="k", linestyle=style, linewidth=0.8, label=f"{name.split(' (')[0]}, single")
ax.set_xlabel("Damage location x along the bridge [m]")
ax.set_ylabel(f"Drop of most affected mode for {100 * TARGET_RATE:.0f} % detection [%]")
ax.set_title("Local bending damage: detection limit vs. location")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[2]
x = np.arange(len(modes))
ax.bar(x - 0.27, [s[0] for s in single_mode], 0.27, label="$T^2$, single recordings")
ax.bar(x, [s[1] for s in single_mode], 0.27, label="$T^2$, daily averages")
ax.bar(x + 0.27, [s[2] for s in single_mode], 0.27, label="own 3$\\sigma$ chart, single")
ax.set_xticks(x)
ax.set_xticklabels([f"{f:.3f}" for f in f0], rotation=45, fontsize=8)
ax.set_xlabel("Mode [Hz]")
ax.set_ylabel(f"Drop for {100 * TARGET_RATE:.0f} % detection [%]")
ax.set_title("Drop in ONE mode only")
ax.legend(fontsize=7)
ax.grid(True, axis="y", alpha=0.3)
fig.suptitle("Detection limits for different damage patterns (normalized frequencies, test period; "
             "tensioned-beam assumption, measured mode shapes)")
fig.savefig(OUTPUT_FOLDER / "damage_patterns.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

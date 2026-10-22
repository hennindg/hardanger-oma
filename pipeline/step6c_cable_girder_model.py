"""
Hardanger Bridge project - Step 6c
A simple cable-girder model of the main span, calibrated to the measured frequencies, to turn
the detection limits of step 6 (frequency drops) into STIFFNESS losses.

Theory: linearised deflection theory of suspension bridges. The main span (length L) is a beam
with bending stiffness EI hanging from cables with horizontal force H, mass m per unit length.
For the ANTISYMMETRIC vertical modes the cable is not stretched, and with sinusoidal shapes
sin(n pi x'/L), k = n pi / L (n = 2, 4, 6, ...):
        m omega^2 = H k^2 + EI k^4      ->      omega^2 / k^2 = H/m + (EI/m) k^2
a straight line in k^2: intercept H/m, slope EI/m. SYMMETRIC modes (n = 1, 3, 5, ...) also stretch
the cable, which raises their frequency (most for low n); they are used as a check only.

Share of the bending stiffness in mode n:   beta_n = EI k^4 / (m omega^2),   cable share 1 - beta_n.
Rayleigh's quotient then gives the frequency drop for a stiffness loss delta (fraction):
  global EI loss              delta f / f = -1/2 delta beta_n
  global cable stiffness loss delta f / f = -1/2 delta (1 - beta_n)
  local EI loss over length l at position x' (from the tower end):
                              delta f / f = -1/2 delta beta_n (2 l / L) sin^2(n pi x' / L)
(strain energy density of a sine mode ~ sin^2; averaging sin^2 over the span gives 1/2).

ASSUMPTIONS: main span only (side spans and towers ignored), sinusoidal shapes, uniform m, EI, H.
Torsion modes are left out of the damage patterns (weight 0, conservative).

How to use:
  python -m pipeline.step6c_cable_girder_model
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.regression import load_table
from hardanger.data import section_mode_shape
from hardanger.novelty_detection import residuals_for_mode, t2_setup, detection_level, daily_means
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
STEP4_FOLDER = RESULTS_FOLDER / "step4"
OUTPUT_FOLDER = RESULTS_FOLDER / "step6"

MAIN_SPAN = 1310.0          # m (Oiseth et al. 2010; Petersen & Oiseth 2017)
MASS = 12820.0              # kg/m, vertical modal equivalent mass (Oiseth et al. 2010, Table 2)

# "Still-air" frequencies: median over recordings with lower wind speed than this (m/s)
LOW_WIND = 5.0

# Mode shapes for the symmetry check: median over this many low-wind recordings
N_SHAPE_RECORDINGS = 40
RANDOM_SEED = 8

# Same as step 6 (normalization and T^2 detection)
TRAIN_END = "2016-06-01"
SELECTED_MODES = [0.142, 0.206, 0.212, 0.276, 0.333, 0.371, 0.401, 0.417, 0.470, 0.547, 0.626, 0.713, 0.901]
MODEL_VARIABLES = ["U", "U2", "T"]
T2_QUANTILE = 0.99
TARGET_RATE = 0.9

# Modes left out of the fit (frequency, matched within 0.003 Hz). The 0.119 Hz antisymmetric mode
# (n = 2) is also badly predicted by both FE models in the literature (-6.6 % Petersen & Oiseth 2017,
# -7.6 % Oiseth 2010); it probably involves physics outside this model (towers, side spans, bearings).
EXCLUDE_FROM_FIT = [0.119]

# Damage scenarios
DAMAGE_LOCATIONS = np.arange(-635.0, 636.0, 10.0)       # m, centre of a local damage (x = 0 at midspan)
STIFFNESS_LEVELS = np.arange(0.0, 100.01, 0.5)          # % stiffness loss scanned (global damage)
# For a SHORT local damage the frequency drop is proportional to (EI loss in %) x (damaged length in m),
# so the detection limit is given as this product, in %*m: e.g. 3000 %*m = 30 m fully lost = 60 m at 50 %
LOCAL_LEVELS = np.arange(0.0, 50001.0, 100.0)           # %*m scanned

# ==================================================================


def channels_from_names(names, xs):
    """Rebuild the channel descriptions (as in girder_channels) from the saved names 'H1 East y'."""
    channels = []
    for name, x in zip(names, xs):
        section, side, component = str(name).split()
        channels.append({"sensor": f"{section} {side}", "section": section, "side": side,
                         "component": component, "x": float(x)})
    return channels


def symmetry_index(table, mode, data_set_of, rows):
    """
    How symmetric the vertical mode shape is about midspan: sum phi(x) phi(-x) / sum phi(x)^2 over the
    mirrored sections (+-120, +-240, +-480 m). +1 = symmetric, -1 = antisymmetric. Median over 'rows'.
    """
    values = []
    for r in rows:
        f = mode["freq"][r]
        if np.isnan(f):
            continue
        name = table["file_name"][r]
        npz = np.load(STEP4_FOLDER / data_set_of[name] / "modes" / f"{name}_modes.npz")
        index = int(np.argmin(np.abs(npz["freq"] - f)))
        x, lateral, vertical, torsion = section_mode_shape(npz["shapes"][index],
                                                           channels_from_names(npz["channel_names"], npz["channel_x"]))
        shape = {round(xi): vi for xi, vi, ti in zip(x, vertical, torsion) if not np.isnan(ti)}
        pairs = [(shape[d], shape[-d]) for d in (120, 240, 480) if d in shape and -d in shape]
        if len(pairs) >= 2:
            a = np.array(pairs)
            values.append(np.sum(a[:, 0] * a[:, 1]) / np.sum(a ** 2) * 2)
    return np.median(values)


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
table = load_table(MODES_VS_WIND)
data_set_of = {}
with open(MODES_VS_WIND, newline="") as f:
    for row in csv.DictReader(f):
        data_set_of[row["file_name"]] = row["data_set"]

vertical = [m for m in table["modes"] if m["type"] == "vertical"]
low = table["U"] < LOW_WIND
rng = np.random.default_rng(RANDOM_SEED)
shape_rows = rng.choice(np.where(low)[0], size=min(N_SHAPE_RECORDINGS, np.sum(low)), replace=False)

# ---- 1-2. Still-air frequency, symmetry and half-wave number of each vertical mode ----
f_low = np.array([np.nanmedian(m["freq"][low]) for m in vertical])
sym = np.array([symmetry_index(table, m, data_set_of, shape_rows) for m in vertical])
is_antisym = sym < 0
n = np.zeros(len(vertical), dtype=int)
n[np.where(is_antisym)[0][np.argsort(f_low[is_antisym])]] = 2 * np.arange(1, np.sum(is_antisym) + 1)
n[np.where(~is_antisym)[0][np.argsort(f_low[~is_antisym])]] = 2 * np.arange(np.sum(~is_antisym)) + 1

print(f"Vertical modes (frequency = median at U < {LOW_WIND} m/s):")
print(" track median f [Hz]  f low wind   symmetry index   type            n")
for m, f, s, ni in zip(vertical, f_low, sym, n):
    print(f"   {m['f0']:.3f}           {f:.4f}       {s:+.2f}          {'antisymmetric' if s < 0 else 'symmetric    '}  {ni:3d}")

# ---- 3. Fit H/m and EI/m to the antisymmetric modes: omega^2/k^2 = H/m + (EI/m) k^2 ----
k = n * np.pi / MAIN_SPAN
omega2 = (2 * np.pi * f_low) ** 2
in_fit = is_antisym & np.array([all(abs(m["f0"] - x) > 0.003 for x in EXCLUDE_FROM_FIT) for m in vertical])
A = np.column_stack([np.ones(np.sum(in_fit)), k[in_fit] ** 2])
(H_m, EI_m), *_ = np.linalg.lstsq(A, omega2[in_fit] / k[in_fit] ** 2, rcond=None)
predicted = np.sqrt(H_m * k ** 2 + EI_m * k ** 4) / (2 * np.pi)
print(f"\nFit to the {np.sum(in_fit)} antisymmetric modes (without {EXCLUDE_FROM_FIT} Hz):  H/m = {H_m:.1f} m^2/s^2,  EI/m = {EI_m:.3e} m^4/s^2")
print(f"   with m = {MASS:.0f} kg/m:  H = {H_m * MASS / 1e6:.0f} MN (both cables),  EI = {EI_m * MASS:.3e} N m^2")
print(" mode [Hz]   n   measured   model     error    bending share beta")
beta = EI_m * k ** 4 / omega2
for m, ni, f, fp, b, a in zip(vertical, n, f_low, predicted, beta, is_antisym):
    note = "" if a else "  (symmetric: cable stretching not in the model)"
    print(f"   {m['f0']:.3f}   {ni:3d}   {f:.4f}    {fp:.4f}   {100 * (fp - f) / f:+5.1f} %    {b:.3f}{note}")

# ---- 4. Detection set-up (as step 6) ----
train = table["day"] < TRAIN_END
test = ~train
modes = [m for m in table["modes"] if any(abs(m["f0"] - f) < 0.003 for f in SELECTED_MODES)]
R = np.column_stack([residuals_for_mode(table, m, train, MODEL_VARIABLES, np.nanmean(m["freq"][train])) for m in modes])
complete = ~np.isnan(R).any(axis=1)
setup = t2_setup(R[train & complete], table["day"][train & complete], T2_QUANTILE)
R_test = R[test & complete]
_, D_test = daily_means(R_test, table["day"][test & complete])

# beta and n for each selected mode (torsion modes: weight 0)
beta_sel = np.zeros(len(modes))
n_sel = np.zeros(len(modes), dtype=int)
for j, m in enumerate(modes):
    for i, v in enumerate(vertical):
        if v is m:
            beta_sel[j], n_sel[j] = beta[i], n[i]

# Weights = frequency drop [%] per 1 % stiffness loss
scenarios = {
    "girder bending EI, whole span": 0.5 * beta_sel,
    "cable stiffness, whole span": 0.5 * np.where(n_sel > 0, 1 - beta_sel, 0.0),
}
print(f"\nStiffness loss needed for {100 * TARGET_RATE:.0f} % detection (Hotelling T^2, test period):")
print(" scenario                              single recordings   daily averages   largest frequency drop at that loss")
results = {}
for name, w in scenarios.items():
    single, daily = detection_level(R_test, D_test, w, setup, STIFFNESS_LEVELS, TARGET_RATE)
    results[name] = (single, daily)
    print(f"  {name:36s}   {single:6.1f} %            {daily:6.1f} %          {single * w.max():.2f} % "
          f"(mode {modes[int(np.argmax(w))]['f0']:.3f} Hz)")

# Local EI damage: frequency drop [%] per (1 % EI loss x 1 m damaged length)
local_single, local_daily = [], []
for x in DAMAGE_LOCATIONS:
    x_from_tower = x + MAIN_SPAN / 2
    w = 0.5 * beta_sel * (2 / MAIN_SPAN) * np.sin(n_sel * np.pi * x_from_tower / MAIN_SPAN) ** 2
    single, daily = detection_level(R_test, D_test, w, setup, LOCAL_LEVELS, TARGET_RATE)
    local_single.append(single)
    local_daily.append(daily)
local_single, local_daily = np.array(local_single), np.array(local_daily)
best, worst = np.nanmin(local_single), np.nanmax(local_single)
print("\nLocal girder bending damage, detection limit as (EI loss) x (damaged length):")
print(f"   single recordings: {best:.0f} - {worst:.0f} %*m  (e.g. {best / 100:.0f}-{worst / 100:.0f} m fully lost, "
      f"or {best / 50:.0f}-{worst / 50:.0f} m with 50 % EI loss)")
print(f"   daily averages:    {np.nanmin(local_daily):.0f} - {np.nanmax(local_daily):.0f} %*m")
print(f"   easiest at x = {DAMAGE_LOCATIONS[np.nanargmin(local_single)]:+.0f} m, hardest at "
      f"x = {DAMAGE_LOCATIONS[np.nanargmax(local_single)]:+.0f} m; not reached within {LOCAL_LEVELS[-1]:.0f} %*m "
      f"at {np.sum(np.isnan(local_single))} locations")

with open(OUTPUT_FOLDER / "cable_girder_model.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mode_Hz", "n", "symmetric", "f_low_wind_Hz", "f_model_Hz", "bending_share_beta"])
    for m, ni, s, fl, fp, b in zip(vertical, n, ~is_antisym, f_low, predicted, beta):
        writer.writerow([f"{m['f0']:.3f}", ni, bool(s), f"{fl:.5f}", f"{fp:.5f}", f"{b:.4f}"])
    writer.writerow([])
    writer.writerow(["scenario", f"stiffness_loss_for_{int(100 * TARGET_RATE)}pct_single", "daily"])
    for name, (a, b) in results.items():
        writer.writerow([name, a, b])
    for x, a, b in zip(DAMAGE_LOCATIONS, local_single, local_daily):
        writer.writerow([f"local EI damage at x = {x:+.0f} m, limit in %*m (EI loss x length)", a, b])

# ---- Plots ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
ax = axes[0]
ax.plot(k[is_antisym] ** 2, omega2[is_antisym] / k[is_antisym] ** 2, "o", color="tab:blue", label="antisymmetric (fitted)")
ax.plot(k[~is_antisym] ** 2, omega2[~is_antisym] / k[~is_antisym] ** 2, "s", color="tab:orange",
        label="symmetric (cable stretching, not fitted)")
kk = np.linspace(0, (k ** 2).max() * 1.05, 50)
ax.plot(kk, H_m + EI_m * kk, "k-", label=f"H/m + (EI/m) k$^2$")
for m, ki, yi, ni in zip(vertical, k, omega2 / k ** 2, n):
    ax.annotate(f"{m['f0']:.3f}", (ki ** 2, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("$k^2 = (n\\pi/L)^2$  [1/m$^2$]")
ax.set_ylabel("$\\omega^2 / k^2$  [m$^2$/s$^2$]")
ax.set_title("Cable-girder model: straight line for the antisymmetric modes")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[1]
order = np.argsort(f_low)
ax.bar(range(len(vertical)), beta[order], color=["tab:blue" if a else "tab:orange" for a in is_antisym[order]])
ax.set_xticks(range(len(vertical)))
ax.set_xticklabels([f"{f:.3f}\nn={ni}" for f, ni in zip(f_low[order], n[order])], fontsize=7)
ax.set_ylabel("Bending share of the modal stiffness $\\beta_n$")
ax.set_title("How much of each mode's stiffness comes from girder bending")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[2]
ax.plot(DAMAGE_LOCATIONS, local_single, "o-", markersize=3, label="single recordings")
ax.plot(DAMAGE_LOCATIONS, local_daily, "s-", markersize=3, label="daily averages")
for x in (-480, -240, -120, -7, 120, 240, 480):
    ax.axvline(x, color="0.85", linewidth=0.6, zorder=0)
ax.set_xlabel("Centre of the damaged zone x [m] (grey: sensor sections)")
ax.set_ylabel(f"(EI loss %) x (length m) for {100 * TARGET_RATE:.0f} % detection [%*m]")
ax.set_title("Local girder bending damage (e.g. 3000 %*m = 30 m fully lost)")
secondary = ax.secondary_yaxis("right", functions=(lambda v: v / 50, lambda v: v * 50))
secondary.set_ylabel("damaged length at 50 % EI loss [m]")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
fig.suptitle("Cable-girder model calibrated to the measured frequencies, and detection limits as stiffness losses "
             "(ASSUMPTIONS: main span only, sinusoidal shapes, uniform properties)")
fig.savefig(OUTPUT_FOLDER / "cable_girder_model.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

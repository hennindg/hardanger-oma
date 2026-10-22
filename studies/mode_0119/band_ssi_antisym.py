"""
Hardanger Bridge project - 2-mode SSI for the antisymmetric modes near 0.1 Hz

The 0.105 Hz antisymmetric LATERAL mode and the ~0.12 Hz antisymmetric VERTICAL mode have
almost the same spanwise shape and lie close together. In wind the vertical mode gets extra
damping and its frequency drops (see studies/mode_0119/inspect_0117.py). Here both modes are
identified with a 2-mode model in every recording.

How the 2-mode problem is made:
  1. SPATIAL filter: the girder sections are combined into two antisymmetric "modal
     coordinates" with weights w = x / 480 m (odd in x):
         q_y = sum w_i * y_i   (lateral),    q_z = sum w_i * z_i   (vertical)
     y_i, z_i are east/west averages, so torsion cancels; symmetric modes cancel because
     w is odd. If a section is bad, its mirror section is left out too (keeps w odd).
  2. BAND filter 0.07 - 0.17 Hz: removes the other antisymmetric modes (0.05 Hz lateral,
     0.2+ Hz modes).
     What is left are the two modes we want (checked: the SVD shows 4 states above noise).
  3. SSI-cov with a LOW model order (4 - 8, i.e. 2 modes plus a little room for filter and
     noise) and several numbers of block rows. Each (order, block rows) combination is one
     estimate; the median is the result and the spread shows how robust it is.
     (The usual stability check is not used: with ~7 % damping the damping estimate of the
     vertical mode changes more than 5 % between orders, so it would never be "stable".)

Mode shape = 2 complex numbers [q_y, q_z]. Coupling:
  lateral mode:  c = q_z / q_y   (vertical motion per unit lateral motion)
  vertical mode: c = q_y / q_z   (lateral motion per unit vertical motion)
  |c| = amount of coupling, angle(c) = phase between the two components.

How to use:
  python -m studies.mode_0119.band_ssi_antisym
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording, mean_wind, GRAVITY
from oma.ssi_cov import ssi_cov, bandpass_filter
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

DATA_FOLDERS = {
    "2014_9": DATA_FOLDER / "LowFreq_2014_9" / "2014_9",
    "StormUrd_LowFreq": DATA_FOLDER / "Events_StormUrd_LowFreq" / "StormUrd_LowFreq",
}
LABELS = {"2014_9": "Sept. 2014", "StormUrd_LowFreq": "Storm Urd"}
COLORS = {"2014_9": "tab:blue", "StormUrd_LowFreq": "tab:red"}

# Girder sections used, as mirror pairs (+x, -x). H2 and H8 have only one sensor each,
# so lateral/vertical cannot be separated from torsion there: they are not used.
MIRROR_PAIRS = [("H1", "H9"), ("H3", "H7"), ("H4", "H6")]
CENTRE_SECTION = "H5"          # x = -7 m, weight almost 0, included for completeness
WEIGHT_LENGTH = 480.0          # w = x / WEIGHT_LENGTH (m)

# Band filter (Hz), and where to look for the two modes (Hz)
BAND = (0.07, 0.17)
SEARCH_RANGE = (0.095, 0.140)

# Low-order models: every combination is one estimate.
# Block rows: 30 - 90 samples = 15 - 45 s = 1.5 - 4.5 periods of the 0.1 Hz mode.
ORDERS = [4, 6, 8]
BLOCK_ROWS_LIST = [30, 45, 60, 75, 90]
MAX_DAMPING = 0.15             # higher than elsewhere: the vertical mode reaches ~7-8 %

# A pole counts as lateral if |q_z / q_y| < this, vertical if |q_z / q_y| > 1 / this.
# Poles in between (mixed) are not used.
MIXED_LIMIT = 0.6

# Full-band results (step 5) and quasi-steady fit (step 5c), for comparison
MODES_VS_WIND = RESULTS_FOLDER / "step5" / "modes_vs_wind.csv"
SCALING_FIT = RESULTS_FOLDER / "step5" / "damping_scaling_fit.csv"

OUTPUT_FOLDER = RESULTS_FOLDER / "band_ssi_antisym"

# ==================================================================


def section_average(sensors, section, column):
    """East/west average of one component (1 = y, 2 = z) of a section, in m/s^2. None if bad."""
    values = []
    for side in ["East", "Vest"]:
        name = f"{section} {side}"
        if name not in sensors or np.isnan(sensors[name]).any():
            return None            # need both sides, otherwise torsion does not cancel
        values.append(sensors[name][:, column])
    return np.mean(values, axis=0) * GRAVITY


def antisymmetric_coordinates(rec):
    """
    Spatial filter: q_y and q_z as weighted sums with odd weights w = x / WEIGHT_LENGTH.
    Returns an array [n_samples, 2] and the list of sections used.
    """
    S = rec["sensors"]
    used = []
    q_y = 0.0
    q_z = 0.0
    sections = [s for pair in MIRROR_PAIRS for s in pair] + [CENTRE_SECTION]
    good = {s: section_average(S, s, 1) is not None for s in sections}
    for plus, minus in MIRROR_PAIRS:
        if not (good[plus] and good[minus]):
            continue                # leave out both mirror sections, so w stays odd
        used.extend([plus, minus])
    if good[CENTRE_SECTION]:
        used.append(CENTRE_SECTION)
    for s in used:
        w = rec["positions"][s + " East"][0] / WEIGHT_LENGTH
        q_y = q_y + w * section_average(S, s, 1)
        q_z = q_z + w * section_average(S, s, 2)
    return np.column_stack([q_y, q_z]), used


def two_mode_estimates(q, fs):
    """
    Run all low-order models on the filtered coordinates.
    Returns two lists (lateral, vertical) of candidate poles (model number, freq, damping, coupling c).
    """
    candidates = {"lateral": [], "vertical": []}
    model = 0
    for s in BLOCK_ROWS_LIST:
        poles, _ = ssi_cov(q, fs, s, ORDERS, MAX_DAMPING)
        for P in poles:
            model += 1
            for f, z, shape in zip(P["freq"], P["damping"], P["shapes"].T):
                if not SEARCH_RANGE[0] < f < SEARCH_RANGE[1]:
                    continue
                ratio = abs(shape[1] / shape[0])          # |q_z / q_y|
                if ratio < MIXED_LIMIT:
                    candidates["lateral"].append((model, f, z, shape[1] / shape[0]))
                elif ratio > 1 / MIXED_LIMIT:
                    candidates["vertical"].append((model, f, z, shape[0] / shape[1]))
    return candidates


def one_per_model(candidates):
    """
    Some models give two poles of the same kind (one of them spurious). Keep, for every
    model, only the pole closest to the median frequency of all candidates.
    """
    median_freq = np.median([c[1] for c in candidates])
    best = {}
    for c in candidates:
        model = c[0]
        if model not in best or abs(c[1] - median_freq) < abs(best[model][1] - median_freq):
            best[model] = c
    return [best[m][1:] for m in sorted(best)]          # drop the model number


def summarize(candidates):
    """
    One estimate per mode from all candidate poles: median frequency, damping, |c| and phase,
    plus the interquartile range (IQR) of frequency and damping as a robustness measure.
    The phase is averaged as a vector (so +179 and -179 deg give 180, not 0).
    Needs poles from at least 3 models.
    """
    if len(candidates) == 0:
        return None
    candidates = one_per_model(candidates)
    if len(candidates) < 3:
        return None
    f = np.array([c[0] for c in candidates])
    z = np.array([c[1] for c in candidates])
    c = np.array([c[2] for c in candidates])
    return {
        "freq": np.median(f), "freq_iqr": np.subtract(*np.percentile(f, [75, 25])),
        "damping": np.median(z), "damping_iqr": np.subtract(*np.percentile(z, [75, 25])),
        "coupling": np.median(np.abs(c)),
        "phase": np.degrees(np.angle(np.mean(c / np.abs(c)))),
        "n": len(f),
    }


def full_band_track(freq_target, kind):
    """Wind speed, frequency and damping of one step 5 track (for comparison)."""
    U, f, z = [], [], []
    with open(MODES_VS_WIND, newline="") as fh:
        for row in csv.DictReader(fh):
            if abs(float(row["track_median_freq_Hz"]) - freq_target) < 0.003 and row["type"] == kind:
                U.append(float(row["mean_wind_speed_ms"]))
                f.append(float(row["freq_Hz"]))
                z.append(float(row["damping_percent"]))
    return np.array(U), np.array(f), np.array(z)


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
n_models = len(ORDERS) * len(BLOCK_ROWS_LIST)

results = []
for name, folder in DATA_FOLDERS.items():
    for path in sorted(Path(folder).glob("*.mat")):
        rec = load_recording(path)
        q, used = antisymmetric_coordinates(rec)
        q = bandpass_filter(q, rec["fs"], BAND[0], BAND[1])
        candidates = two_mode_estimates(q, rec["fs"])
        r = {"data_set": name, "file_name": path.stem, "wind": mean_wind(rec)[0], "sections": used,
             "lateral": summarize(candidates["lateral"]), "vertical": summarize(candidates["vertical"])}
        results.append(r)

        text = f"{path.stem[7:26]}  U = {r['wind']:4.1f} m/s  ({len(used)} sections)"
        for kind in ["lateral", "vertical"]:
            m = r[kind]
            if m is None:
                text += f"   {kind}: not found" + " " * 31
            else:
                text += (f"   {kind}: {m['freq']:.4f} Hz {100 * m['damping']:4.2f} %"
                         f"  |c| {m['coupling']:.2f} {m['phase']:+4.0f} deg ({m['n']:2d}/{n_models})")
        print(text)

# ---- Table ----
with open(OUTPUT_FOLDER / "two_modes.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["data_set", "file_name", "mean_wind_speed_ms", "mode", "freq_Hz", "freq_iqr_Hz",
                     "damping_percent", "damping_iqr_percent", "coupling_amplitude",
                     "coupling_phase_deg", "n_estimates"])
    for r in results:
        for kind in ["lateral", "vertical"]:
            m = r[kind]
            if m is not None:
                writer.writerow([r["data_set"], r["file_name"], f"{r['wind']:.2f}", kind,
                                 f"{m['freq']:.5f}", f"{m['freq_iqr']:.5f}",
                                 f"{100 * m['damping']:.3f}", f"{100 * m['damping_iqr']:.3f}",
                                 f"{m['coupling']:.4f}", f"{m['phase']:.1f}", m["n"]])

# ---- Trend of each quantity with wind (straight line, for the printout) ----
print("\nStraight-line trends vs. mean wind speed (per 10 m/s):")
for kind in ["lateral", "vertical"]:
    rows = [(r["wind"], r[kind]) for r in results if r[kind] is not None]
    U = np.array([u for u, _ in rows])
    for key, scale, unit in [("freq", 1000, "mHz"), ("damping", 100, "%-pt"), ("coupling", 1, "")]:
        v = np.array([m[key] for _, m in rows]) * scale
        b, a = np.polyfit(U, v, 1)
        r_corr = np.corrcoef(U, v)[0, 1]
        print(f"  {kind:8s} {key:8s}: {10 * b:+.3f} {unit} per 10 m/s  (r = {r_corr:+.2f}, n = {len(U)})")

# ---- Plot ----
with open(SCALING_FIT, newline="") as fh:
    K = float(next(csv.DictReader(fh))["K_per_m"])

fig, axes = plt.subplots(2, 4, figsize=(18, 8), layout="constrained")
for row, kind, full_freq in [(0, "lateral", 0.105), (1, "vertical", 0.117)]:
    ax_f, ax_z, ax_c, ax_p = axes[row]
    U_full, f_full, z_full = full_band_track(full_freq, kind)
    ax_f.plot(U_full, f_full, "x", color="0.6", label="full-band SSI (step 5)")
    ax_z.plot(U_full, z_full, "x", color="0.6", label="full-band SSI (step 5)")
    for r in results:
        m = r[kind]
        if m is None:
            continue
        c = COLORS[r["data_set"]]
        # error bars = interquartile range over the 15 low-order models
        ax_f.errorbar(r["wind"], m["freq"], yerr=m["freq_iqr"] / 2, fmt="o", color=c, markersize=4, capsize=2)
        ax_z.errorbar(r["wind"], 100 * m["damping"], yerr=100 * m["damping_iqr"] / 2, fmt="o", color=c,
                      markersize=4, capsize=2)
        ax_c.plot(r["wind"], m["coupling"], "o", color=c, markersize=4)
        ax_p.plot(r["wind"], m["phase"], "o", color=c, markersize=4)
    if kind == "vertical":
        u = np.linspace(4, 24, 50)
        ax_z.plot(u, 100 * (0.0006 + K * u / 0.121), "k--", label="0.06 % + K U/f (other vertical modes)")
    other = "vertical" if kind == "lateral" else "lateral"
    ax_f.set_ylabel(f"{kind.upper()} MODE\nFrequency [Hz]")
    ax_z.set_ylabel("Damping [%]")
    ax_c.set_ylabel(f"|{other} / {kind}| in mode")
    ax_p.set_ylabel(f"phase of {other} vs {kind} [deg]")
    ax_p.set_ylim(-180, 180)
    ax_p.set_yticks(range(-180, 181, 45))
    ax_z.legend(fontsize=7)
    for ax in axes[row]:
        ax.grid(True, alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("Mean wind speed [m/s]")
for name, c in COLORS.items():
    axes[0][0].plot([], [], "o", color=c, label=f"2-mode SSI, {LABELS[name]}")
axes[0][0].legend(fontsize=7)
fig.suptitle("2-mode SSI on antisymmetric coordinates (band "
             f"{BAND[0]}-{BAND[1]} Hz). Error bars: half the IQR over {n_models} low-order models")
fig.savefig(OUTPUT_FOLDER / "two_modes_vs_wind.png", dpi=90)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

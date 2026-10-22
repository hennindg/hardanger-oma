"""
Hardanger Bridge project - Step 5
Relate the identified natural frequencies and damping ratios to the mean wind, for all data sets.

What it does:
  1. Reads the per-recording modes from step 4a for all data sets in results/step4, and removes
     recordings that appear in two data sets (e.g. a storm event and its month).
  2. Builds reference modes by tracking the modes of the REFERENCE_DATA_SETS (pairwise clustering,
     as in step 4b), then assigns the modes of ALL recordings to these references
     (see oma/mode_tracking.py). This scales to thousands of recordings.
  3. Reads the mean temperature of each recording from the original files (stored in a CSV,
     so this is done only once).
  4. Plots frequency and damping of each tracked mode against mean wind speed (colour =
     temperature) with a straight-line fit, and the wind conditions that the data cover.
  5. Saves the tables used by steps 5b-5d.

How to use:
  python -m pipeline.step5_wind
"""

from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording, mean_temperature
from oma.mode_tracking import track_modes, build_reference_modes, assign_to_reference
from hardanger.literature import load_reference_modes, match_reference, reference_label
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

STEP4_FOLDER = RESULTS_FOLDER / "step4"

# Data sets to use (folder names in results/step4). None = all of them.
# If a recording is in two data sets, the one listed first in ORDER_FIRST keeps it.
DATA_SETS = None
ORDER_FIRST = ["2014_9", "StormUrd_LowFreq"]

# Reference modes are built from these data sets with the pairwise tracking of step 4b
REFERENCE_DATA_SETS = ["2014_9", "StormUrd_LowFreq"]
REFERENCE_MIN_FRACTION = 0.5     # a reference track must be found in half of these recordings

# Distance limit |df|/f + (1 - MAC), the same for building and assigning (step 4b: 0.20)
TRACK_MAX_DISTANCE = 0.20

# Where to save the results. results/step5 holds the earlier two-event analysis.
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

# Literature reference (Petersen & Oiseth 2017) was identified at a mean wind of 3-6 m/s.
# Our straight-line fit is evaluated at this wind speed to compare with it.
LITERATURE_WIND_SPEED = 4.5

# Recordings where fewer anemometers than this worked are drawn with open markers
MIN_ANEMOMETERS = 3

# Colour range for the temperature in the plots (degC)
TEMPERATURE_RANGE = (-5, 20)

# ==================================================================


def list_data_sets():
    """Data set names: ORDER_FIRST first, then all other folders in results/step4 with results."""
    available = sorted(p.parent.name for p in STEP4_FOLDER.glob("*/modes") if any(p.glob("*.npz")))
    wanted = available if DATA_SETS is None else [d for d in DATA_SETS if d in available]
    return [d for d in ORDER_FIRST if d in wanted] + [d for d in wanted if d not in ORDER_FIRST]


def load_all_recordings(data_sets):
    """
    Read the step 4a results of all data sets. A recording that is in two data sets is kept
    only once (in the first data set of the list). Returns a list of dictionaries, sorted by time.
    """
    recordings = {}
    for name in data_sets:
        n_new = 0
        for path in sorted((STEP4_FOLDER / name / "modes").glob("*_modes.npz")):
            data = np.load(path)
            file_name = str(data["file_name"])
            if file_name in recordings:
                continue                      # duplicate recording
            rec = {key: data[key] for key in data.files}
            rec["data_set"] = name
            rec["time"] = datetime.strptime(str(rec["start_time"]), "%d/%m/%Y %H:%M:%S.%f")
            recordings[file_name] = rec
            n_new += 1
        print(f"   {name}: {n_new} recordings")
    return sorted(recordings.values(), key=lambda r: r["time"])


def add_temperatures(recordings):
    """
    Mean temperature of each recording. Read from recording_conditions.csv if it was computed
    before; otherwise from the original .mat file (found by name anywhere in the project).
    """
    cache_file = OUTPUT_FOLDER / "recording_conditions.csv"
    cache = {}
    if cache_file.exists():
        with open(cache_file, newline="") as f:
            for row in csv.DictReader(f):
                cache[row["file_name"]] = float(row["mean_temperature_degC"])

    missing = [r for r in recordings if str(r["file_name"]) not in cache]
    if missing:
        print(f"Reading temperature from {len(missing)} original files ...")
        mat_files = {p.stem: p for p in DATA_FOLDER.rglob("*.mat") if "results" not in p.parts}
        for rec in missing:
            path = mat_files.get(str(rec["file_name"]))
            cache[str(rec["file_name"])] = mean_temperature(load_recording(path)) if path else np.nan

    for rec in recordings:
        rec["temperature"] = cache[str(rec["file_name"])]
    with open(cache_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "mean_temperature_degC"])
        for name, value in sorted(cache.items()):
            writer.writerow([name, f"{value:.2f}"])


def straight_line_fit(x, y):
    """
    Least-squares straight line y = a + b x through the points that are not NaN.
    Returns a, b and the correlation coefficient r (-1 ... 1, 0 = no linear trend).
    """
    ok = ~np.isnan(x) & ~np.isnan(y)
    b, a = np.polyfit(x[ok], y[ok], 1)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return a, b, r


def plot_against_wind(recordings, tracks, quantity, scale, ylabel, title):
    """
    One small plot per track: quantity ("freq" or "damping") vs. mean wind speed.
    Colour = mean temperature, open marker = few anemometers, black line = straight-line fit,
    green dashed line = literature frequency at low wind.
    """
    speeds = np.array([float(r["mean_wind_speed"]) for r in recordings])
    temperature = np.array([r["temperature"] for r in recordings])
    few_anemometers = np.array([int(r["n_anemometers"]) < MIN_ANEMOMETERS for r in recordings])

    n_cols = 5
    n_rows = int(np.ceil(len(tracks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 2.9 * n_rows), sharex=True,
                             squeeze=False, layout="constrained")

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols][k % n_cols]
        if k >= len(tracks):
            ax.axis("off")
            continue
        track = tracks[k]
        values = scale * track[quantity]
        filled = ~few_anemometers
        points = ax.scatter(speeds[filled], values[filled], c=temperature[filled], cmap="coolwarm",
                            vmin=TEMPERATURE_RANGE[0], vmax=TEMPERATURE_RANGE[1], s=6)
        ax.scatter(speeds[few_anemometers], values[few_anemometers], facecolors="none",
                   edgecolors="0.4", s=8, linewidths=0.5)

        a, b, r = straight_line_fit(speeds, values)
        u = np.array([np.nanmin(speeds), np.nanmax(speeds)])
        ax.plot(u, a + b * u, "k-", linewidth=1)

        ref = track["reference"]
        if quantity == "freq" and ref is not None:
            ax.axhline(ref["freq"], color="tab:green", linestyle="--", linewidth=1)
        if quantity == "freq":
            trend = f"{100 * 10 * b / a:+.1f} % per 10 m/s"
        else:
            trend = f"{10 * b:+.2f} %-pt per 10 m/s"
        ax.set_title(f"{k + 1}: {track['median_freq']:.3f} Hz {track['type']} ({track['n_found']})\n"
                     f"{trend}, r = {r:.2f}\n{reference_label(ref)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if k % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if k >= len(tracks) - n_cols:
            ax.set_xlabel("Mean wind speed [m/s]", fontsize=8)

    fig.colorbar(points, ax=axes, shrink=0.5, label="Mean temperature [degC]")
    fig.suptitle(f"{title}   ({len(recordings)} recordings; open grey = fewer than {MIN_ANEMOMETERS} "
                 "anemometers; green dashed = Petersen & Oiseth 2017, 3-6 m/s)")
    return fig


def plot_wind_conditions(recordings):
    """
    Which conditions do the data cover? Left: direction vs. speed. Middle: temperature vs. speed.
    Right: number of recordings per month.
    """
    speeds = np.array([float(r["mean_wind_speed"]) for r in recordings])
    direction = np.array([float(r["mean_wind_direction"]) for r in recordings])
    temperature = np.array([r["temperature"] for r in recordings])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")

    ax1.scatter(speeds, direction, c=temperature, cmap="coolwarm", vmin=TEMPERATURE_RANGE[0],
                vmax=TEMPERATURE_RANGE[1], s=6)
    ax1.axhline(90, color="k", linewidth=0.6, linestyle="--")
    ax1.axhline(270, color="k", linewidth=0.6, linestyle="--")
    ax1.set_ylim(0, 360)
    ax1.set_yticks(range(0, 361, 45))
    ax1.set_xlabel("Mean wind speed [m/s]")
    ax1.set_ylabel("Mean wind direction [deg from global x-axis]")
    ax1.set_title("Wind direction (dashed: perpendicular to the bridge)")

    ax2.plot(speeds, temperature, "o", color="tab:blue", markersize=2, alpha=0.5)
    ax2.set_xlabel("Mean wind speed [m/s]")
    ax2.set_ylabel("Mean sonic temperature [degC]")
    ax2.set_title("Temperature")

    months = [r["time"].strftime("%Y-%m") for r in recordings]
    labels = sorted(set(months))
    ax3.bar(range(len(labels)), [months.count(m) for m in labels], color="tab:blue")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=90, fontsize=7)
    ax3.set_ylabel("Number of recordings")
    ax3.set_title("Recordings per month")
    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
    return fig


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

data_sets = list_data_sets()
print(f"Reading {len(data_sets)} data sets (duplicate recordings removed):")
recordings = load_all_recordings(data_sets)
n_rec = len(recordings)
print(f"{n_rec} different recordings, {recordings[0]['time']:%Y-%m-%d} to {recordings[-1]['time']:%Y-%m-%d}")
add_temperatures(recordings)

# ---- Reference modes from the reference data sets (pairwise tracking, as in step 4b) ----
ref_recordings = [r for r in recordings if r["data_set"] in REFERENCE_DATA_SETS]
min_ref = int(np.ceil(REFERENCE_MIN_FRACTION * len(ref_recordings)))
ref_tracks, _, _, _ = track_modes(ref_recordings, TRACK_MAX_DISTANCE, min_ref)
references = build_reference_modes(ref_recordings, ref_tracks)
print(f"\n{len(references)} reference modes from {len(ref_recordings)} recordings of {REFERENCE_DATA_SETS}")

# ---- Assign the modes of all recordings to the reference modes ----
print("Assigning the modes of all recordings to the reference modes ...")
tracks = assign_to_reference(recordings, references, TRACK_MAX_DISTANCE)

literature = load_reference_modes()
for track in tracks:
    track["reference"] = match_reference(track["median_freq"], track["type"], literature)

speeds = np.array([float(r["mean_wind_speed"]) for r in recordings])

# ---- Trend table ----
rows = []
print(f"\n track  median f [Hz]  type       found          frequency trend         damping trend"
      "             ours at       literature")
print(f"                                  (of {n_rec})     [% per 10 m/s]  r       [%-pt per 10 m/s]  r"
      f"     {LITERATURE_WIND_SPEED} m/s    (difference)")
for t, track in enumerate(tracks):
    a_f, b_f, r_f = straight_line_fit(speeds, track["freq"])
    a_z, b_z, r_z = straight_line_fit(speeds, 100 * track["damping"])
    ref = track["reference"]
    f_low = a_f + b_f * LITERATURE_WIND_SPEED
    if ref is not None:
        difference = 100 * (f_low - ref["freq"]) / ref["freq"]
        ref_text = f"{reference_label(ref)} ({difference:+.1f} %)"
        ref_columns = [ref["mode"], ref["type"], ref["freq"], f"{difference:.2f}"]
    else:
        ref_text = reference_label(ref)
        ref_columns = ["", "", "", ""]
    print(f"  {t + 1:3d}   {track['median_freq']:8.4f}     {track['type']:9s}  {track['n_found']:5d}"
          f"          {100 * 10 * b_f / a_f:+6.2f}       {r_f:+.2f}      {10 * b_z:+6.2f}           {r_z:+.2f}"
          f"     {f_low:.4f}   {ref_text}")
    rows.append([t + 1, f"{track['median_freq']:.4f}", track["type"], track["n_found"],
                 f"{100 * 10 * b_f / a_f:.3f}", f"{r_f:.3f}", f"{10 * b_z:.3f}", f"{r_z:.3f}",
                 f"{f_low:.4f}"] + ref_columns)

with open(OUTPUT_FOLDER / "wind_trends.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "n_found",
                     "freq_change_percent_per_10ms", "r_freq",
                     "damping_change_percentpoints_per_10ms", "r_damping",
                     f"freq_fit_at_{LITERATURE_WIND_SPEED}ms_Hz",
                     "literature_mode", "literature_type", "literature_freq_Hz",
                     "difference_to_literature_percent"])
    writer.writerows(rows)

# Table of every tracked mode in every recording, with wind and temperature (input for steps 5b-5d)
with open(OUTPUT_FOLDER / "modes_vs_wind.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "track_median_freq_Hz", "type", "data_set", "file_name", "start_time_utc",
                     "mean_wind_speed_ms", "mean_wind_direction_deg", "n_anemometers",
                     "mean_temperature_degC", "freq_Hz", "damping_percent",
                     "literature_mode", "literature_type", "literature_freq_Hz"])
    for t, track in enumerate(tracks):
        ref = track["reference"]
        ref_columns = [ref["mode"], ref["type"], ref["freq"]] if ref else ["", "", ""]
        for r, rec in enumerate(recordings):
            if np.isnan(track["freq"][r]):
                continue
            writer.writerow([t + 1, f"{track['median_freq']:.4f}", track["type"], rec["data_set"],
                             str(rec["file_name"]), rec["time"].strftime("%Y-%m-%d %H:%M:%S"),
                             f"{float(rec['mean_wind_speed']):.2f}", f"{float(rec['mean_wind_direction']):.1f}",
                             int(rec["n_anemometers"]), f"{rec['temperature']:.1f}",
                             f"{track['freq'][r]:.5f}", f"{100 * track['damping'][r]:.3f}"]
                            + ref_columns)

# ---- Plots ----
fig1 = plot_wind_conditions(recordings)
fig1.savefig(OUTPUT_FOLDER / "wind_conditions.png", dpi=100)

fig2 = plot_against_wind(recordings, tracks, "freq", 1.0, "Frequency [Hz]",
                         "Natural frequency vs. mean wind speed")
fig2.savefig(OUTPUT_FOLDER / "frequency_vs_wind.png", dpi=90)

fig3 = plot_against_wind(recordings, tracks, "damping", 100.0, "Damping [%]",
                         "Damping ratio vs. mean wind speed")
fig3.savefig(OUTPUT_FOLDER / "damping_vs_wind.png", dpi=90)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

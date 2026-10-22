"""
Hardanger Bridge project - Step 4b
Mode tracking: link the modes identified in each recording (step 4a) into tracks,
and plot frequency and damping of each tracked mode over time.

Needs the result files from pipeline/step4a_run_all.py. Runs in a few seconds, so the
tracking settings can be tuned without rerunning SSI.

How to use:
  python -m pipeline.step4b_tracking
"""

from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from oma.mode_tracking import track_modes
from hardanger.literature import load_reference_modes, match_reference, reference_label
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

# Which data set to track: the name of the data folder used in step 4a,
# e.g. "2014_9" (September 2014) or "StormUrd_LowFreq" (storm Urd, December 2016)
DATA_SET = "StormUrd_LowFreq"

# Results of step 4a are read from results\step4\<DATA_SET>\modes,
# and the tracking plots and tables are saved in results\step4\<DATA_SET>
OUTPUT_FOLDER = RESULTS_FOLDER / "step4" / DATA_SET
MODES_FOLDER = OUTPUT_FOLDER / "modes"

# Modes from two recordings belong to the same track if their (average) distance
# |f_i - f_j| / f + (1 - MAC) is below this. 0.2 allows e.g. 2 % frequency change between
# recordings (wind affects the frequencies) together with MAC 0.82.
# Tried on September 2014: 0.1 split the 0.32 Hz lateral mode (whose shape estimate varies
# a lot between recordings) into small pieces. 0.2 and 0.3 still keep neighbouring modes
# such as 0.206 / 0.212 Hz and 0.802 / 0.816 Hz apart.
TRACK_MAX_DISTANCE = 0.20

# A track must be found in at least this fraction of the recordings
MIN_FRACTION_OF_RECORDINGS = 0.5

# ==================================================================


def load_all_results(folder):
    """
    Read all per-recording result files and sort them by start time.
    Returns a list of dictionaries (one per recording).
    """
    recordings = []
    for path in sorted(folder.glob("*_modes.npz")):
        data = np.load(path)
        rec = {key: data[key] for key in data.files}
        # Start time is text like "26/09/2014 18:20:01.440" (UTC)
        rec["time"] = datetime.strptime(str(rec["start_time"]), "%d/%m/%Y %H:%M:%S.%f")
        recordings.append(rec)
    recordings.sort(key=lambda r: r["time"])
    return recordings


def plot_overview(recordings, tracks, index, freq, in_track):
    """
    Top: every identified mode in every recording (frequency vs. time). Coloured = in a
    track, grey cross = not in any track. Bottom: mean wind speed of each recording.
    """
    times = [r["time"] for r in recordings]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    colors = plt.cm.tab10(np.arange(len(tracks)) % 10)

    untracked = np.where(in_track < 0)[0]
    ax1.plot([times[index[k][0]] for k in untracked], freq[untracked], "x", color="0.6",
             markersize=4, label="not in a track")
    for t, track in enumerate(tracks):
        ax1.plot(times, track["freq"], "o-", color=colors[t], markersize=4, linewidth=0.8)
        ax1.text(times[-1], track["median_freq"], f"  {t + 1}", fontsize=7, va="center",
                 color=colors[t])
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"All identified modes, {len(recordings)} recordings, {len(tracks)} tracks")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    speeds = [float(r["mean_wind_speed"]) for r in recordings]
    ax2.plot(times, speeds, "s-", color="k", markersize=4)
    ax2.set_ylabel("Mean wind speed [m/s]")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_track_grid(recordings, tracks, quantity, ylabel, scale):
    """
    One small plot per track: quantity ("freq" or "damping") vs. time, with the std from
    the pole clustering as error bars, and markers coloured by mean wind speed.
    """
    times = [r["time"] for r in recordings]
    speeds = np.array([float(r["mean_wind_speed"]) for r in recordings])
    n_cols = 5
    n_rows = int(np.ceil(len(tracks) / n_cols))
    # layout="constrained" makes room for the titles, labels and the colour bar automatically
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.6 * n_rows), sharex=True, squeeze=False,
                             layout="constrained")

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols][k % n_cols]
        if k >= len(tracks):
            ax.axis("off")
            continue
        track = tracks[k]
        values = scale * track[quantity]
        errors = scale * track[quantity + "_std"]
        ax.errorbar(times, values, yerr=errors, fmt="none", ecolor="0.6", capsize=2)
        points = ax.scatter(times, values, c=speeds, cmap="viridis", s=18, vmin=speeds.min(),
                            vmax=speeds.max(), zorder=3)
        ax.set_title(f"{k + 1}: {track['median_freq']:.3f} Hz, {track['type']} "
                     f"({track['n_found']}/{len(recordings)})\n{reference_label(track['reference'])}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        # At most 4 date labels per plot; matplotlib picks hours or days depending on the time span
        locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(True, alpha=0.3)
        if k % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)

    fig.colorbar(points, ax=axes, shrink=0.6, label="Mean wind speed [m/s]")
    return fig


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
recordings = load_all_results(MODES_FOLDER)
n_rec = len(recordings)
min_recordings = int(np.ceil(MIN_FRACTION_OF_RECORDINGS * n_rec))
print(f"{n_rec} recordings, {sum(len(r['freq']) for r in recordings)} identified modes in total")

tracks, index, freq, in_track = track_modes(recordings, TRACK_MAX_DISTANCE, min_recordings)
print(f"{len(tracks)} tracks found in at least {min_recordings} of {n_rec} recordings\n")

# Literature mode for each track (Petersen & Oiseth 2017, low wind), see hardanger/literature.py
reference = load_reference_modes()
for track in tracks:
    track["reference"] = match_reference(track["median_freq"], track["type"], reference)

print(" track   median f [Hz]   f range [Hz]       f spread   damping [%]: median (min - max)   found   type"
      "        literature (low wind)")
for t, track in enumerate(tracks):
    f = track["freq"]
    z = 100 * track["damping"]
    spread = 100 * (np.nanmax(f) - np.nanmin(f)) / np.nanmedian(f)
    print(f"  {t + 1:3d}     {track['median_freq']:8.4f}     {np.nanmin(f):.4f} - {np.nanmax(f):.4f}   "
          f"{spread:5.1f} %      {np.nanmedian(z):5.2f} ({np.nanmin(z):.2f} - {np.nanmax(z):.2f})"
          f"            {track['n_found']:2d}/{n_rec}   {track['type']:9s}   {reference_label(track['reference'])}")

# ---- Plots ----
fig1 = plot_overview(recordings, tracks, index, freq, in_track)
fig1.savefig(OUTPUT_FOLDER / "tracking_overview.png", dpi=110)

fig2 = plot_track_grid(recordings, tracks, "freq", "Frequency [Hz]", 1.0)
fig2.suptitle("Frequency of each tracked mode")
fig2.savefig(OUTPUT_FOLDER / "tracking_frequency.png", dpi=90)

fig3 = plot_track_grid(recordings, tracks, "damping", "Damping [%]", 100.0)
fig3.suptitle("Damping ratio of each tracked mode")
fig3.savefig(OUTPUT_FOLDER / "tracking_damping.png", dpi=90)

# ---- Save results ----
# .csv: one row per track and recording (only where the mode was found). Step 5 reads this.
with open(OUTPUT_FOLDER / "tracked_modes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "track_median_freq_Hz", "type", "file_name", "start_time_utc",
                     "mean_wind_speed_ms", "mean_wind_direction_deg",
                     "freq_Hz", "freq_std_Hz", "damping_percent", "damping_std_percent",
                     "literature_mode", "literature_type", "literature_freq_Hz"])
    for t, track in enumerate(tracks):
        ref = track["reference"]
        ref_columns = [ref["mode"], ref["type"], ref["freq"]] if ref else ["", "", ""]
        for r, rec in enumerate(recordings):
            if track["mode_number"][r] < 0:
                continue
            writer.writerow([t + 1, f"{track['median_freq']:.4f}", track["type"], str(rec["file_name"]),
                             rec["time"].strftime("%Y-%m-%d %H:%M:%S"),
                             f"{float(rec['mean_wind_speed']):.2f}", f"{float(rec['mean_wind_direction']):.1f}",
                             f"{track['freq'][r]:.5f}", f"{track['freq_std'][r]:.5f}",
                             f"{100 * track['damping'][r]:.3f}", f"{100 * track['damping_std'][r]:.3f}"]
                            + ref_columns)

# .npz: the same as matrices [n_tracks, n_recordings], NaN where a mode was not found
np.savez(
    OUTPUT_FOLDER / "tracked_modes.npz",
    file_names=np.array([str(r["file_name"]) for r in recordings]),
    start_times=np.array([r["time"].strftime("%Y-%m-%d %H:%M:%S") for r in recordings]),
    mean_wind_speed=np.array([float(r["mean_wind_speed"]) for r in recordings]),
    mean_wind_direction=np.array([float(r["mean_wind_direction"]) for r in recordings]),
    track_median_freq=np.array([t["median_freq"] for t in tracks]),
    track_type=np.array([t["type"] for t in tracks]),
    freq=np.array([t["freq"] for t in tracks]),
    freq_std=np.array([t["freq_std"] for t in tracks]),
    damping=np.array([t["damping"] for t in tracks]),
    damping_std=np.array([t["damping_std"] for t in tracks]),
)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

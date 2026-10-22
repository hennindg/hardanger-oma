"""
Hardanger Bridge project - is the 0.12 Hz mode amplitude dependent?

Step 5d showed that the frequency of the ~0.12 Hz antisymmetric vertical mode drops in very
turbulent (gusty) recordings. Turbulence makes the response larger, so the real cause could be
the RESPONSE AMPLITUDE (an amplitude-dependent, i.e. nonlinear, structural behaviour, for
example friction in the bearings or expansion joints) rather than the wind itself.

This script:
  1. Computes for every recording the displacement amplitude of the mode:
     - antisymmetric vertical: (z_H3 - z_H7) / 2, H3/H7 at x = +-240 m, near the antinodes,
     - control mode, symmetric vertical 0.143 Hz: z_H5 at midspan,
     band filtered around the mode; displacement std = acceleration std / omega^2
     (valid for a narrow-band response).
  2. Compares frequency and damping of the mode with mean wind speed U, turbulence intensity
     I_u (step 5d) and this amplitude, using several regression models.
  3. Does the same for the control mode. If the effect is a property of the 0.12 Hz mode,
     the control mode should not show it at similar amplitudes.

Needs: results/step5/modes_vs_wind.csv, results/step5d/turbulence_per_recording.csv,
       results/band_ssi_antisym/two_modes.csv

How to use:
  python -m studies.mode_0119.inspect_0117_amplitude
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording, GRAVITY
from oma.ssi_cov import bandpass_filter
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

DATA_FOLDERS = [
    DATA_FOLDER / "LowFreq_2014_9" / "2014_9",
    DATA_FOLDER / "Events_StormUrd_LowFreq" / "StormUrd_LowFreq",
]

# Modes to check. "plus"/"minus": sections whose vertical motion is combined as (plus - minus)/2
# (antisymmetric mode), or only "plus" for a symmetric mode (minus = None).
MODES = {
    "0.12 Hz antisym. vertical": {"track_freq": 0.1171, "plus": "H3", "minus": "H7",
                                  "band": (0.09, 0.15), "mode_freq": 0.119},
    "0.143 Hz sym. vertical (control)": {"track_freq": 0.1427, "plus": "H5", "minus": None,
                                         "band": (0.13, 0.16), "mode_freq": 0.142},
}

MODES_VS_WIND = RESULTS_FOLDER / "step5" / "modes_vs_wind.csv"
TURBULENCE = RESULTS_FOLDER / "step5d" / "turbulence_per_recording.csv"
TWO_MODE_SSI = RESULTS_FOLDER / "band_ssi_antisym" / "two_modes.csv"

OUTPUT_FOLDER = RESULTS_FOLDER / "inspect_0117"

# ==================================================================


def time_key_from_file_name(file_name):
    """'HB141M-2016-12-26_00-18-38_LowFreq' -> '2016-12-26 00:18:38' (same key as the step 5 tables)."""
    date, clock = file_name.split("_")[0][7:], file_name.split("_")[1]
    return f"{date} {clock.replace('-', ':')}"


def vertical_section(sensors, section):
    """East/west average of the vertical acceleration of a section (m/s^2)."""
    names = [section + " East", section + " Vest"]
    present = [sensors[n][:, 2] for n in names if n in sensors and not np.isnan(sensors[n]).any()]
    return np.mean(present, axis=0) * GRAVITY


def displacement_amplitude(sensors, fs, mode):
    """
    Displacement standard deviation (m) of one mode: band-filtered acceleration std / omega^2.
    """
    acc = vertical_section(sensors, mode["plus"])
    if mode["minus"] is not None:
        acc = (acc - vertical_section(sensors, mode["minus"])) / 2
    acc = bandpass_filter(acc[:, None], fs, mode["band"][0], mode["band"][1])[:, 0]
    omega = 2 * np.pi * mode["mode_freq"]
    return np.std(acc) / omega ** 2


def fit(columns, y):
    """
    Least squares y = a + sum(coef * column). Returns coefficients (without a), their
    standard errors, and R^2 (fraction of the variance of y explained by the model).
    """
    X = np.column_stack([np.ones(len(y))] + columns)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    s2 = np.sum(residuals ** 2) / (n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
    return beta[1:], se[1:], r2


def read_track(track_freq):
    """Frequency and damping of one step 5 track, keyed by recording time."""
    values = {}
    with open(MODES_VS_WIND, newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["track_median_freq_Hz"]) - track_freq) < 0.002:
                values[row["start_time_utc"]] = (float(row["mean_wind_speed_ms"]), float(row["freq_Hz"]),
                                                 float(row["damping_percent"]))
    return values


def print_models(label, U, I, amp, y, unit):
    """Print the regression models for one quantity (frequency or damping)."""
    print(f"  {label}:")
    for name, columns, names in [
        ("U only", [U], ["U"]),
        ("U + I_u", [U, I], ["U", "I_u"]),
        ("U + amplitude", [U, amp], ["U", "amp"]),
        ("U + I_u + amplitude", [U, I, amp], ["U", "I_u", "amp"]),
    ]:
        coef, se, r2 = fit(columns, y)
        # t = coefficient / standard error; |t| > 2 is roughly "significant"
        text = "   ".join(f"{n}: t = {c / s:+5.1f}" for n, c, s in zip(names, coef, se))
        print(f"    {name:22s} R2 = {r2:.2f}    {text}")
    coef, se, _ = fit([amp], y)
    # amp is in cm and y in % (or %-points), so coef[0] is already "per cm"
    print(f"    amplitude only: {coef[0]:+.3f} {unit} per cm of displacement std (t = {coef[0] / se[0]:+.1f})")


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ---- 1. Response amplitudes ----
amplitude = {name: {} for name in MODES}
for folder in DATA_FOLDERS:
    for path in sorted(Path(folder).glob("*.mat")):
        rec = load_recording(path)
        key = time_key_from_file_name(path.stem)
        for name, mode in MODES.items():
            amplitude[name][key] = displacement_amplitude(rec["sensors"], rec["fs"], mode)

turbulence = {}
with open(TURBULENCE, newline="") as f:
    for row in csv.DictReader(f):
        turbulence[row["start_time_utc"]] = float(row["mean_I_u"])

two_mode = {}
with open(TWO_MODE_SSI, newline="") as f:
    for row in csv.DictReader(f):
        if row["mode"] == "vertical":
            two_mode[time_key_from_file_name(row["file_name"])] = (
                float(row["mean_wind_speed_ms"]), float(row["freq_Hz"]), float(row["damping_percent"]))

# ---- 2. Regressions ----
fig, axes = plt.subplots(len(MODES), 3, figsize=(16, 4.6 * len(MODES)), layout="constrained", squeeze=False)
for row_number, (name, mode) in enumerate(MODES.items()):
    estimates = [("full-band SSI", read_track(mode["track_freq"]))]
    if mode["minus"] is not None:
        estimates.append(("2-mode SSI", two_mode))

    for est_name, values in estimates:
        keys = [k for k in sorted(values) if k in turbulence and k in amplitude[name]]
        U = np.array([values[k][0] for k in keys])
        freq = np.array([values[k][1] for k in keys])
        zeta = np.array([values[k][2] for k in keys])
        I = np.array([turbulence[k] for k in keys])
        amp = np.array([amplitude[name][k] for k in keys]) * 100      # cm

        print(f"\n=== {name}, {est_name} (n = {len(keys)}) ===")
        print(f"  displacement std {amp.min():.1f} - {amp.max():.1f} cm;  "
              f"corr(U, amp) = {np.corrcoef(U, amp)[0, 1]:+.2f}, corr(I_u, amp) = {np.corrcoef(I, amp)[0, 1]:+.2f}, "
              f"corr(U, I_u) = {np.corrcoef(U, I)[0, 1]:+.2f}")
        print_models("frequency", U, I, amp, 100 * freq / np.mean(freq), "%")
        print_models("damping", U, I, amp, zeta, "%-pt")

        if est_name != "full-band SSI":
            continue
        ax_f, ax_z, ax_a = axes[row_number]
        sc = ax_f.scatter(amp, freq, c=I, cmap="viridis", s=25, vmin=0.08, vmax=0.45)
        ax_z.scatter(amp, zeta, c=I, cmap="viridis", s=25, vmin=0.08, vmax=0.45)
        ax_a.scatter(U, amp, c=I, cmap="viridis", s=25, vmin=0.08, vmax=0.45)
        ax_f.set_ylabel(f"{name}\nFrequency [Hz]")
        ax_z.set_ylabel("Damping [%]")
        ax_a.set_ylabel("Displacement std [cm]")
        ax_f.set_xlabel("Displacement std of the mode [cm]")
        ax_z.set_xlabel("Displacement std of the mode [cm]")
        ax_a.set_xlabel("Mean wind speed U [m/s]")
        for ax in axes[row_number]:
            ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax_a, label="Turbulence intensity $I_u$")

fig.suptitle("Frequency and damping vs. response amplitude (full-band SSI, step 5). Colour = turbulence intensity")
fig.savefig(OUTPUT_FOLDER / "amplitude_dependence.png", dpi=95)

print(f"\nPlot saved in {OUTPUT_FOLDER}")
plt.show()

"""
Hardanger Bridge project - closer look at the 0.117 Hz vertical mode

In steps 3-5 this mode looked odd: very high damping (1-7 %), large scatter, and a frequency
that drops with wind while the other vertical modes go up. This script collects the checks:

  1. Spectra of antisymmetric vertical and lateral girder motion (H3 vs. H7, x = +-240 m)
     in low, medium and high wind: is there one mode or several close ones?
  2. Is there longitudinal girder motion in this mode? The deck x-accelerations are compared
     with what a TILTING accelerometer would measure (g * slope of the vertical mode shape).
  3. Damping of all vertical modes against U / f. Quasi-steady theory (step 5c) says
     zeta = zeta_structural + K * U / f. Does the 0.117 Hz mode follow the same line?

How to use:
  python -m studies.mode_0119.inspect_0117
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from hardanger.data import load_recording, GRAVITY
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

# Three recordings: low, medium and high mean wind speed
RECORDINGS = {
    "5.1 m/s (29.09.2014)": DATA_FOLDER / "LowFreq_2014_9" / "2014_9" / "HB141M-2014-09-29_16-32-55_LowFreq.mat",
    "12.7 m/s (26.09.2014)": DATA_FOLDER / "LowFreq_2014_9" / "2014_9" / "HB141M-2014-09-26_18-20-01_LowFreq.mat",
    "23.0 m/s (26.12.2016)": DATA_FOLDER / "Events_StormUrd_LowFreq" / "StormUrd_LowFreq" / "HB141M-2016-12-26_00-18-38_LowFreq.mat",
}

# Frequency band to look at (Hz) and the mode we are interested in
BAND = (0.08, 0.18)
TARGET_FREQ = 0.12

# Welch block length for the spectra (s). 400 s gives 0.0025 Hz resolution.
SEGMENT_SECONDS = 400

# Results from step 5 / 5c
MODES_VS_WIND = RESULTS_FOLDER / "step5" / "modes_vs_wind.csv"
SCALING_FIT = RESULTS_FOLDER / "step5" / "damping_scaling_fit.csv"

OUTPUT_FOLDER = RESULTS_FOLDER / "inspect_0117"

# ==================================================================


def section_average(sensors, section, column):
    """Average of the east and west sensor of a girder section (one of them if only one exists)."""
    names = [section + " East", section + " Vest"]
    present = [sensors[n][:, column] for n in names if n in sensors]
    return np.mean(present, axis=0)


def psd(x, fs):
    """Power spectral density with Welch's method."""
    x = x - np.mean(x)
    return signal.welch(x, fs=fs, nperseg=int(SEGMENT_SECONDS * fs))


def transfer_at(reference, response, fs, freq):
    """
    Frequency response from 'reference' to 'response' at one frequency: H = S_ref,resp / S_ref,ref.
    The real part tells how large the response is relative to the reference, with sign.
    """
    f, S_rr = signal.welch(reference, fs=fs, nperseg=int(SEGMENT_SECONDS * fs))
    _, S_rx = signal.csd(reference, response, fs=fs, nperseg=int(SEGMENT_SECONDS * fs))
    k = np.argmin(np.abs(f - freq))
    return np.real(S_rx[k] / S_rr[k])


def tracked_frequencies_in_band():
    """Median frequencies and types of the tracked modes (step 5) inside BAND."""
    found = {}
    with open(MODES_VS_WIND, newline="") as f:
        for row in csv.DictReader(f):
            fm = float(row["track_median_freq_Hz"])
            if BAND[0] < fm < BAND[1]:
                found[fm] = row["type"]
    return found


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
mode_lines = tracked_frequencies_in_band()

# ---- 1. Spectra of antisymmetric vertical and lateral motion ----
# H3 (x = +240 m) minus H7 (x = -240 m): symmetric modes cancel, antisymmetric modes add up.
fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, layout="constrained")
for ax, (label, path) in zip(axes, RECORDINGS.items()):
    rec = load_recording(path)
    S, fs = rec["sensors"], rec["fs"]
    anti_vertical = (section_average(S, "H3", 2) - section_average(S, "H7", 2)) / 2
    anti_lateral = (section_average(S, "H3", 1) - section_average(S, "H7", 1)) / 2
    sym_vertical = section_average(S, "H5", 2)

    for x, name, color in [(anti_vertical, "antisymmetric vertical (H3 - H7)", "tab:red"),
                           (anti_lateral, "antisymmetric lateral (H3 - H7)", "tab:blue"),
                           (sym_vertical, "vertical at midspan (H5)", "0.6")]:
        f, p = psd(x, fs)
        band = (f >= BAND[0]) & (f <= BAND[1])
        ax.semilogy(f[band], p[band], color=color, label=name)
    for fm, kind in mode_lines.items():
        ax.axvline(fm, color="k", linewidth=0.6, linestyle="--")
        ax.text(fm, ax.get_ylim()[1], f"{fm:.3f}\n{kind}", fontsize=6, ha="center", va="top")
    ax.set_title(f"Mean wind {label}")
    ax.set_xlabel("Frequency [Hz]")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("PSD [g$^2$/Hz]")      # load_recording gives accelerations in g
axes[0].legend(fontsize=7, loc="lower left")
fig1.suptitle("Spectra around the 0.117 Hz mode (dashed: tracked modes from step 5)")
fig1.savefig(OUTPUT_FOLDER / "spectra_low_mid_high_wind.png", dpi=100)

# ---- 2. Longitudinal motion or sensor tilt? ----
# For each girder section: x- and z-response relative to the antisymmetric vertical motion.
# A tilting accelerometer measures g * theta in x, with theta = d(z_displacement)/dx and
# z_displacement = z_acceleration / omega^2. So tilt predicts:  x = -(g / omega^2) * dz/dx.
# Rigid longitudinal motion of the girder would instead give the same x everywhere.
label, path = list(RECORDINGS.items())[-1]          # highest wind: strongest response
rec = load_recording(path)
S, fs = rec["sensors"], rec["fs"]
reference = (section_average(S, "H3", 2) - section_average(S, "H7", 2)) / 2
omega = 2 * np.pi * TARGET_FREQ

sections = ["H1", "H3", "H4", "H5", "H6", "H7", "H9"]       # sections with an east sensor and a pair
x_pos = np.array([rec["positions"][s + " East"][0] for s in sections])
x_measured = np.array([transfer_at(reference, S[s + " East"][:, 0], fs, TARGET_FREQ) for s in sections])
z_shape = np.array([transfer_at(reference, section_average(S, s, 2), fs, TARGET_FREQ) for s in sections])
x_tilt = -(GRAVITY / omega ** 2) * np.gradient(z_shape, x_pos)

print(f"Longitudinal (x) response at {TARGET_FREQ} Hz, {label}, relative to antisymmetric vertical:")
print("  section   x [m]   vertical shape   x measured   x predicted by tilt")
for s, xp, zs, xm, xt in zip(sections, x_pos, z_shape, x_measured, x_tilt):
    print(f"   {s:4s}   {xp:6.0f}      {zs:+.3f}          {xm:+.4f}        {xt:+.4f}")
tower_x = np.mean([transfer_at(reference, S[n][:, 0], fs, TARGET_FREQ) for n in ["H10 East", "H11 East"]])
print(f"  tower tops (H10, H11), x: {tower_x:+.4f}")

fig2, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
ax.plot(x_pos, x_measured, "ko-", label="measured x-acceleration")
ax.plot(x_pos, x_tilt, "r^--", label="predicted from tilt: -(g/$\\omega^2$) dz/dx")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("x along bridge [m]")
ax.set_ylabel("x-response / antisymmetric vertical response")
ax.set_title(f"Longitudinal motion or sensor tilt?  ({TARGET_FREQ} Hz, {label})")
ax.legend()
ax.grid(True, alpha=0.3)
fig2.savefig(OUTPUT_FOLDER / "longitudinal_vs_tilt.png", dpi=100)

# ---- 3. Damping of all vertical modes vs. U / f ----
with open(SCALING_FIT, newline="") as f:
    K = float(next(csv.DictReader(f))["K_per_m"])        # zeta per (U/f), from step 5c

U_over_f, zeta, freqs = [], [], []
with open(MODES_VS_WIND, newline="") as f:
    for row in csv.DictReader(f):
        if row["type"] != "vertical":
            continue
        fm = float(row["track_median_freq_Hz"])
        U_over_f.append(float(row["mean_wind_speed_ms"]) / float(row["freq_Hz"]))
        zeta.append(float(row["damping_percent"]) / 100)
        freqs.append(fm)
U_over_f, zeta, freqs = np.array(U_over_f), np.array(zeta), np.array(freqs)
target = np.abs(freqs - 0.117) < 0.003

# Structural damping = what is left after removing K U/f. Averaged over the other vertical modes.
zeta_structural_others = np.mean(zeta[~target] - K * U_over_f[~target])
zeta_structural_target = np.mean(zeta[target] - K * U_over_f[target])
print(f"\nDamping after removing the aerodynamic part K U/f (K = {K:.2e} 1/m from step 5c):")
print(f"  other vertical modes: {100 * zeta_structural_others:.2f} %")
print(f"  0.117 Hz mode:        {100 * zeta_structural_target:.2f} %")

fig3, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
scatter = ax.scatter(U_over_f[~target], 100 * zeta[~target], c=freqs[~target], cmap="viridis", s=15,
                     label="other vertical modes (colour = frequency)")
ax.plot(U_over_f[target], 100 * zeta[target], "rs", markersize=5, markerfacecolor="none",
        label="0.117 Hz mode")
line = np.linspace(0, U_over_f.max(), 50)
ax.plot(line, 100 * (zeta_structural_others + K * line), "k-",
        label=f"{100 * zeta_structural_others:.2f} % + K U/f  (K from step 5c, not fitted here)")
ax.set_xlabel("U / f  [m]")
ax.set_ylabel("Damping ratio [%]")
ax.set_title("Vertical modes: damping vs. U/f (quasi-steady aerodynamic damping)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig3.colorbar(scatter, label="Mode frequency [Hz]")
fig3.savefig(OUTPUT_FOLDER / "damping_vs_U_over_f.png", dpi=100)

print(f"\nPlots saved in {OUTPUT_FOLDER}")
plt.show()

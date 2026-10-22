"""
Hardanger Bridge project - Step 1
Plot the acceleration spectra from ONE 30-minute recording.

How to use:
  1. The folder LowFreq_2014_9\2014_9 has many .mat files, one per 30-minute recording.
  2. Put the path of one of them in FILE below.
  3. Run the script:  python -m pipeline.step1_spectra

The recordings are read with load_recording() in hardanger/data.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from hardanger.data import load_recording, PAIR_DISTANCE
from hardanger.paths import DATA_FOLDER


# ============================ SETTINGS ============================

# Path to one recording file (change this to one of your files)
FILE = DATA_FOLDER / "LowFreq_2014_9" / "2014_9" / "HB141M-2014-09-26_18-20-01_LowFreq.mat"

# Highest frequency on the plot (Hz). Can be at most half the sample rate (1 Hz for LowFreq).
FREQ_MAX = 1.0

# Length of each block used for averaging the spectrum (seconds).
# Frequency resolution is about 1 / SEGMENT_SECONDS, so 300 s gives about 0.003 Hz.
SEGMENT_SECONDS = 300

# ==================================================================

def spectrum(x, fs):
    """Power spectral density with Welch's method (split into blocks, average the spectra)."""
    x = x - np.mean(x)                              # remove the mean value
    block = min(len(x), int(SEGMENT_SECONDS * fs))  # number of samples per block
    freqs, psd = signal.welch(x, fs=fs, nperseg=block)
    return freqs, psd


# --------------------------- Main program ---------------------------

recording = load_recording(FILE)
fs = recording["fs"]
sensors = recording["sensors"]

print(f"Sample rate: {fs} Hz")
print("Sensors in this file:")
for name, data in sensors.items():
    if np.isnan(data).any():
        quality = "missing values (will be skipped)"
    else:
        quality = "ok"
    print(f"   {name:12s}  {data.shape[0]} samples  {quality}")

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

for name, data in sensors.items():
    parts = name.split()     # e.g. "H1 East" -> ["H1", "East"]
    label = parts[0]         # e.g. "H1"

    if not label.startswith("H"):
        continue             # skip the wind sensors (A1, A2, ...)
    if int(label[1:]) >= 10:
        continue             # skip the tower-top sensors (H10, H11)
    if np.isnan(data).any():
        continue             # skip sensors with bad data

    # Column 1 = y = lateral, column 2 = z = vertical
    freqs, psd = spectrum(data[:, 1], fs)
    axes[0].semilogy(freqs, psd, linewidth=0.8, label=name)

    freqs, psd = spectrum(data[:, 2], fs)
    axes[1].semilogy(freqs, psd, linewidth=0.8, label=name)

    # Torsion: difference in vertical acceleration between east and west side,
    # divided by the distance between them (gives rotational acceleration, rad/s^2)
    # Note: the west sensors are named in Norwegian, e.g. "H1 Vest"
    if parts[-1] == "East":
        west_name = label + " Vest"
        if west_name in sensors and not np.isnan(sensors[west_name]).any():
            rotation = (data[:, 2] - sensors[west_name][:, 2]) / PAIR_DISTANCE
            freqs, psd = spectrum(rotation, fs)
            axes[2].semilogy(freqs, psd, linewidth=0.8, label=label)

axes[0].set_title("Lateral acceleration (y)")
axes[1].set_title("Vertical acceleration (z)")
axes[2].set_title("Torsional acceleration (from east/west pairs)")
for ax in axes:
    ax.set_ylabel("PSD")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
axes[2].set_xlabel("Frequency [Hz]")
axes[2].set_xlim(0, FREQ_MAX)

fig.suptitle(FILE.name)
plt.tight_layout()
plt.show()

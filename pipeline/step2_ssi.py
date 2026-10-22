"""
Hardanger Bridge project - Step 2
Covariance-driven SSI (SSI-cov) on ONE 30-minute recording.

What it does:
  1. Reads the girder accelerations (lateral y and vertical z of H1 ... H9).
  2. Runs SSI-cov for a range of model orders.
  3. Draws the stabilization diagram (model order vs. frequency).
  4. Takes the stable poles at one chosen model order as the identified modes,
     and plots their mode shapes along the girder.
  5. Saves the results to the folder results/step2 (.npz for later steps, .csv to read yourself).

How to use:
  python -m pipeline.step2_ssi
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording, girder_channels, section_mode_shape, mean_wind, classify_mode
from hardanger.plots import plot_stabilization_diagram, plot_mode_shapes
from oma.ssi_cov import ssi_cov, check_stability, normalize_mode_shape, first_singular_value_spectrum
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

# Recording to analyse
FILE = DATA_FOLDER / "LowFreq_2014_9" / "2014_9" / "HB141M-2014-09-26_18-20-01_LowFreq.mat"

# Number of block rows s in the Toeplitz matrix.
# The covariances use time lags up to 2s-1 samples. The lowest peak in the step 1
# spectra is near 0.05 Hz, i.e. a period of about 20 s = 40 samples at 2 Hz.
# s = 120 samples = 60 s covers about 3 periods of that mode (and the Toeplitz
# matrix uses lags up to 120 s = 6 periods), which is enough to describe its decay.
# Larger s gives a bigger matrix (s * n_channels rows) and a slower SVD.
BLOCK_ROWS = 120

# Model orders to compute (must be even: each mode needs two states)
MIN_ORDER = 2
MAX_ORDER = 160
ORDER_STEP = 2

# Physical limits: poles with damping above this are thrown away
MAX_DAMPING = 0.10

# Stability criteria, compared with the previous model order
TOL_FREQ = 0.01      # relative frequency change < 1 %
TOL_DAMPING = 0.05   # relative damping change < 5 %
TOL_MAC = 0.95       # MAC with the previous pole > 0.95

# Model order where the stable poles are taken as the identified modes.
# Choose it in a region of the stabilization diagram where the columns of stable
# poles are clean. Step 3 will replace this with automatic clustering.
# For 2014-09-26 18:20, order 124 has the most stable poles between orders 100 and 140.
# Some modes drop out at single orders (e.g. 0.10 - 0.14 Hz at order 120), which is why
# picking at one order is fragile and step 3 is needed.
ORDER_FOR_MODES = 124

# Highest frequency on the plots (Hz). Nyquist = 1 Hz for LowFreq data.
FREQ_MAX = 1.0

# Block length for the background spectrum in the stabilization diagram (s)
SEGMENT_SECONDS = 300

# Where to save figures and results
OUTPUT_FOLDER = RESULTS_FOLDER / "step2"

# ==================================================================


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
file_name = Path(FILE).stem

print(f"Reading {file_name}")
recording = load_recording(FILE)
fs = recording["fs"]
y, channels = girder_channels(recording)
print(f"   {y.shape[0]} samples, {y.shape[1]} channels, sample rate {fs} Hz")

wind_speed, wind_direction, anemometers_used = mean_wind(recording)
print(f"   mean wind speed {wind_speed:.1f} m/s, direction {wind_direction:.0f} deg "
      f"(from {', '.join(anemometers_used)})")

print("Running SSI-cov ...")
orders = list(range(MIN_ORDER, MAX_ORDER + 1, ORDER_STEP))
poles, singular_values = ssi_cov(y, fs, BLOCK_ROWS, orders, MAX_DAMPING)
poles = check_stability(poles, TOL_FREQ, TOL_DAMPING, TOL_MAC)

# Identified modes = poles that are fully stable at the chosen model order
chosen = poles[orders.index(ORDER_FOR_MODES)]
modes = []
for p in np.where(chosen["stable_all"])[0]:
    shape = normalize_mode_shape(chosen["shapes"][:, p])
    x, lateral, vertical, torsion = section_mode_shape(shape, channels)
    modes.append({
        "freq": chosen["freq"][p],
        "damping": chosen["damping"][p],
        "shape": shape,
        "x": x, "lateral": lateral, "vertical": vertical, "torsion": torsion,
        "type": classify_mode(lateral, vertical, torsion),
    })

print(f"\nIdentified modes (stable poles at model order {ORDER_FOR_MODES}):")
print("   no.   freq [Hz]   damping [%]   type")
for k, m in enumerate(modes):
    print(f"   {k + 1:3d}   {m['freq']:9.4f}   {100 * m['damping']:11.2f}   {m['type']}")

# ---- Plots ----
freqs_sv, first_sv = first_singular_value_spectrum(y, fs, SEGMENT_SECONDS)
title = f"{file_name}   (s = {BLOCK_ROWS}, U = {wind_speed:.1f} m/s)"

fig1 = plot_stabilization_diagram(poles, freqs_sv, first_sv, [m["freq"] for m in modes], title,
                                  FREQ_MAX, order_line=ORDER_FOR_MODES)
fig1.savefig(OUTPUT_FOLDER / f"{file_name}_stabilization.png", dpi=110)

fig2 = plot_mode_shapes(modes, title)
fig2.savefig(OUTPUT_FOLDER / f"{file_name}_mode_shapes.png", dpi=90)

# ---- Save results ----
# .npz: everything later steps need, without rerunning SSI
np.savez(
    OUTPUT_FOLDER / f"{file_name}_ssi.npz",
    file_name=file_name,
    start_time=recording["start_time"],
    mean_wind_speed=wind_speed,
    mean_wind_direction=wind_direction,
    freq=np.array([m["freq"] for m in modes]),
    damping=np.array([m["damping"] for m in modes]),
    mode_type=np.array([m["type"] for m in modes]),
    shapes=np.array([m["shape"] for m in modes]),            # [n_modes, n_channels], complex
    channel_names=np.array([c["sensor"] + " " + c["component"] for c in channels]),
    channel_x=np.array([c["x"] for c in channels]),
    block_rows=BLOCK_ROWS,
    order_for_modes=ORDER_FOR_MODES,
)

# .csv: a table of the modes that opens in Excel
with open(OUTPUT_FOLDER / f"{file_name}_modes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start_time", "mean_wind_speed_ms", "mean_wind_direction_deg",
                     "mode", "freq_Hz", "damping_percent", "type"])
    for k, m in enumerate(modes):
        writer.writerow([recording["start_time"], f"{wind_speed:.2f}", f"{wind_direction:.1f}",
                         k + 1, f"{m['freq']:.5f}", f"{100 * m['damping']:.3f}", m["type"]])

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

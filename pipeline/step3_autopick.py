"""
Hardanger Bridge project - Step 3
Automatic mode picking: SSI-cov on ONE recording, then clustering of the stable poles
over all model orders (no manual choice of model order).

What it does:
  1. Reads the girder accelerations and runs SSI-cov (same as step 2).
  2. Clusters all stable poles (frequency + MAC distance), see oma/mode_clustering.py.
  3. Keeps clusters that appear at many model orders, removes damping outliers.
  4. Plots the clusters, the stabilization diagram and the mode shapes.
  5. Saves the results to the folder results/step3.

How to use:
  python -m pipeline.step3_autopick
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording, girder_channels, section_mode_shape, mean_wind, classify_mode
from hardanger.plots import plot_stabilization_diagram, plot_mode_shapes, plot_clusters
from oma.ssi_cov import ssi_cov, check_stability, normalize_mode_shape, first_singular_value_spectrum
from oma.mode_clustering import cluster_poles
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

# Recording to analyse
FILE = DATA_FOLDER / "LowFreq_2014_9" / "2014_9" / "HB141M-2014-09-26_18-20-01_LowFreq.mat"

# --- SSI settings (same as step 2, see the comments there) ---
BLOCK_ROWS = 120         # 60 s of time lags, about 3 periods of the 0.05 Hz mode
MIN_ORDER = 2
MAX_ORDER = 160
ORDER_STEP = 2
MAX_DAMPING = 0.10

# Stability criteria, compared with the previous model order
TOL_FREQ = 0.01          # relative frequency change < 1 %
TOL_DAMPING = 0.05       # relative damping change < 5 %
TOL_MAC = 0.95           # MAC with the previous pole > 0.95

# --- Clustering settings ---
# Distance between two poles = relative frequency difference + (1 - MAC).
# With single linkage this is the largest distance between NEIGHBOURING poles in a
# cluster (not between all poles). 0.05 means e.g. 2 % frequency difference with MAC 0.97.
# Tried on 2014-09-26 18:20: 0.02 splits the closely spaced modes near 0.53 Hz and
# 0.71 Hz into many small clusters that get rejected.
MAX_DISTANCE = 0.05

# A cluster is a mode only if it has poles from at least this many model orders.
# With orders 2, 4, ..., 160 there are 80 orders; 20 means the mode must be found
# at a quarter of them. Spurious poles usually show up at only a few orders.
MIN_ORDERS = 20

# --- Plots and output ---
FREQ_MAX = 1.0           # Nyquist = 1 Hz for LowFreq data
SEGMENT_SECONDS = 300    # block length for the background spectrum (s)
OUTPUT_FOLDER = RESULTS_FOLDER / "step3"

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

print("Clustering stable poles ...")
modes, all_stable = cluster_poles(poles, MAX_DISTANCE, MIN_ORDERS)
print(f"   {len(all_stable['freq'])} stable poles, "
      f"{len(np.unique(all_stable['cluster']))} clusters, {len(modes)} accepted as modes")

# Mode shapes per girder section, and lateral / vertical / torsion label
for m in modes:
    m["shape"] = normalize_mode_shape(m["shape"])
    m["x"], m["lateral"], m["vertical"], m["torsion"] = section_mode_shape(m["shape"], channels)
    m["type"] = classify_mode(m["lateral"], m["vertical"], m["torsion"])

print("\nIdentified modes:")
print("   no.   freq [Hz]  (std)       damping [%]  (std)   poles  orders   type")
for k, m in enumerate(modes):
    print(f"   {k + 1:3d}   {m['freq']:8.4f}  ({m['freq_std']:.4f})   "
          f"{100 * m['damping']:8.2f}  ({100 * m['damping_std']:.2f})   "
          f"{m['n_poles']:5d}  {m['n_orders']:6d}   {m['type']}")

# ---- Plots ----
freqs_sv, first_sv = first_singular_value_spectrum(y, fs, SEGMENT_SECONDS)
title = f"{file_name}   (U = {wind_speed:.1f} m/s, {len(modes)} modes)"

fig1 = plot_clusters(all_stable, modes, freqs_sv, first_sv, title, FREQ_MAX)
fig1.savefig(OUTPUT_FOLDER / f"{file_name}_clusters.png", dpi=110)

fig2 = plot_stabilization_diagram(poles, freqs_sv, first_sv, [m["freq"] for m in modes], title, FREQ_MAX)
fig2.savefig(OUTPUT_FOLDER / f"{file_name}_stabilization.png", dpi=110)

fig3 = plot_mode_shapes(modes, title)
fig3.savefig(OUTPUT_FOLDER / f"{file_name}_mode_shapes.png", dpi=90)

# ---- Save results ----
# .npz: everything later steps need, without rerunning SSI
np.savez(
    OUTPUT_FOLDER / f"{file_name}_modes.npz",
    file_name=file_name,
    start_time=recording["start_time"],
    mean_wind_speed=wind_speed,
    mean_wind_direction=wind_direction,
    freq=np.array([m["freq"] for m in modes]),
    freq_std=np.array([m["freq_std"] for m in modes]),
    damping=np.array([m["damping"] for m in modes]),
    damping_std=np.array([m["damping_std"] for m in modes]),
    n_orders=np.array([m["n_orders"] for m in modes]),
    mode_type=np.array([m["type"] for m in modes]),
    shapes=np.array([m["shape"] for m in modes]),            # [n_modes, n_channels], complex
    channel_names=np.array([c["sensor"] + " " + c["component"] for c in channels]),
    channel_x=np.array([c["x"] for c in channels]),
)

# .csv: a table of the modes that opens in Excel
with open(OUTPUT_FOLDER / f"{file_name}_modes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start_time", "mean_wind_speed_ms", "mean_wind_direction_deg", "mode",
                     "freq_Hz", "freq_std_Hz", "damping_percent", "damping_std_percent",
                     "n_orders", "type"])
    for k, m in enumerate(modes):
        writer.writerow([recording["start_time"], f"{wind_speed:.2f}", f"{wind_direction:.1f}", k + 1,
                         f"{m['freq']:.5f}", f"{m['freq_std']:.5f}",
                         f"{100 * m['damping']:.3f}", f"{100 * m['damping_std']:.3f}",
                         m["n_orders"], m["type"]])

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

"""
Hardanger Bridge project - Step 5f
Low-wind check: is the damping amplitude dependent when the wind is (almost) absent?

Background (step 5e): the local aerodynamic damping coefficient is lower at low wind. That could
be an aerodynamic (reduced velocity) effect, or a structural effect that shows up at low wind,
for example damping that depends on the vibration amplitude (friction in bearings, joints, cables).
In low wind the aerodynamic damping is small, so the structural damping can be looked at directly.

What it does:
  1. Takes all recordings with mean wind speed <= U_MAX (step 5, all data sets).
  2. For each tracked mode in those recordings, computes the vibration amplitude of the mode:
     - band-pass filter the girder accelerations around the mode (f +- 15 %),
     - modal filter: project them onto the identified shapes of all modes within f +- 30 %
       of this recording (separates close modes such as 0.206 / 0.212 Hz),
     - displacement std = acceleration std / omega^2 (narrow-band response). The mode shapes are
       scaled to 1 at their largest channel, so this is the displacement std at that point (m).
  3. Removes the small remaining aerodynamic damping with quasi-steady theory (confirmed in
     step 5e): zeta_aero = rho B^2 C V_r / (4 m), V_r = U / (B omega).
  4. Regression per mode: zeta_structural = a + c log10(amplitude)  (and with U as a check).
     c is the change in damping per tenfold increase in amplitude.

How to use:
  python -m pipeline.step5f_low_wind_amplitude
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording, girder_channels
from oma.ssi_cov import bandpass_filter
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
STEP4_FOLDER = RESULTS_FOLDER / "step4"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

# Recordings with at most this mean wind speed (m/s)
U_MAX = 5.0

# Band-pass around the mode, and the range of modes used in the modal filter (relative to f)
BAND_HALF_WIDTH = 0.15
FILTER_MODES_HALF_WIDTH = 0.30

# Quasi-steady aerodynamic damping removed from the damping (step 5c/5e, Oiseth et al. 2010)
AIR_DENSITY = 1.25
B = 18.3
MASS = 12820.0
C_HEAVE = 2.4 + (3.33 / 18.3) * 0.70      # 2.53, applied to vertical modes only

# ==================================================================


def find_mode(rec_npz, freq):
    """Index of the mode in a step 4a result file whose frequency equals freq (closest)."""
    return int(np.argmin(np.abs(rec_npz["freq"] - freq)))


def modal_amplitude(y, fs, rec_npz, index):
    """
    Displacement std (m) of mode 'index' of one recording, using a band-pass filter and a modal
    filter on the modes close in frequency. y: girder accelerations in the channel order of rec_npz.
    """
    f0 = rec_npz["freq"][index]
    # Upper band edge kept just below the Nyquist frequency (the 0.9 Hz mode + 15 % would exceed it)
    high = min(f0 * (1 + BAND_HALF_WIDTH), 0.98 * fs / 2)
    band = bandpass_filter(y, fs, f0 * (1 - BAND_HALF_WIDTH), high)
    nearby = np.where(np.abs(rec_npz["freq"] - f0) <= FILTER_MODES_HALF_WIDTH * f0)[0]
    shapes = np.real(rec_npz["shapes"][nearby]).T              # [n_channels, n_nearby]
    q = band @ np.linalg.pinv(shapes).T                        # modal accelerations [n_samples, n_nearby]
    q_mode = q[:, list(nearby).index(index)]
    return np.std(q_mode) / (2 * np.pi * f0) ** 2


def girder_matrix(mat_path, channel_names):
    """Girder accelerations (m/s^2) of a recording, in the channel order used in step 4a."""
    rec = load_recording(mat_path)
    y, channels = girder_channels(rec)
    names = [c["sensor"] + " " + c["component"] for c in channels]
    order = [names.index(str(n)) for n in channel_names]
    return y[:, order], rec["fs"]


def fit(columns, y):
    """Least squares y = a + sum(coef * column). Returns coefficients, standard errors and R^2."""
    X = np.column_stack([np.ones(len(y))] + columns)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    s2 = np.sum(residuals ** 2) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
    return beta, se, r2


# --------------------------- Main program ---------------------------

# Low-wind rows of the step 5 table, grouped by recording
rows_by_file = {}
with open(MODES_VS_WIND, newline="") as f:
    for row in csv.DictReader(f):
        U = float(row["mean_wind_speed_ms"])
        if not np.isnan(U) and U <= U_MAX:
            rows_by_file.setdefault((row["data_set"], row["file_name"]), []).append(row)
print(f"{len(rows_by_file)} recordings with U <= {U_MAX} m/s")

mat_files = {p.stem: p for p in DATA_FOLDER.rglob("*.mat") if "results" not in p.parts}
records = []          # one per (recording, mode)
for (data_set, file_name), rows in rows_by_file.items():
    rec_npz = np.load(STEP4_FOLDER / data_set / "modes" / f"{file_name}_modes.npz")
    y, fs = girder_matrix(mat_files[file_name], rec_npz["channel_names"])
    for row in rows:
        index = find_mode(rec_npz, float(row["freq_Hz"]))
        amp = modal_amplitude(y, fs, rec_npz, index)
        U = float(row["mean_wind_speed_ms"])
        freq = float(row["freq_Hz"])
        zeta = float(row["damping_percent"])
        if row["type"] == "vertical":
            V_r = U / (B * 2 * np.pi * freq)
            zeta = zeta - 100 * AIR_DENSITY * B ** 2 * C_HEAVE * V_r / (4 * MASS)
        records.append({"track": int(row["track"]), "f0": float(row["track_median_freq_Hz"]),
                        "type": row["type"], "U": U, "amp": amp, "zeta": zeta})

# ---- Regression per mode ----
tracks = sorted({(r["track"], r["f0"], r["type"]) for r in records})
print(f"\nStructural damping (quasi-steady aerodynamic part removed for vertical modes) vs. amplitude,"
      f" U <= {U_MAX} m/s")
print(" track  f [Hz]  type        n   amplitude [mm]        damping [%]        c [%-pt per decade]"
      "   with U: c          corr(U, log amp)")
summary = []
for t, f0, kind in tracks:
    sel = [r for r in records if r["track"] == t]
    amp = np.array([r["amp"] for r in sel])
    zeta = np.array([r["zeta"] for r in sel])
    U = np.array([r["U"] for r in sel])
    log_amp = np.log10(amp)
    beta, se, r2 = fit([log_amp], zeta)
    beta_u, se_u, _ = fit([log_amp, U], zeta)
    summary.append((t, f0, kind, len(sel), beta[1], se[1], beta_u[1], se_u[1]))
    print(f"  {t:3d}   {f0:.3f}  {kind:9s} {len(sel):4d}   {1000 * np.percentile(amp, 5):6.2f} - "
          f"{1000 * np.percentile(amp, 95):6.2f}      {np.median(zeta):5.2f} (IQR {np.subtract(*np.percentile(zeta, [75, 25])):.2f})"
          f"       {beta[1]:+6.3f} +- {se[1]:.3f}         {beta_u[1]:+6.3f} +- {se_u[1]:.3f}      "
          f"{np.corrcoef(U, log_amp)[0, 1]:+.2f}")

with open(OUTPUT_FOLDER / "low_wind_amplitude.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "n", "c_pt_per_decade", "se",
                     "c_with_U_pt_per_decade", "se"])
    for row in summary:
        writer.writerow(list(row[:4]) + [f"{v:.4f}" for v in row[4:]])

# ---- Plot: damping vs amplitude per mode, colour = wind speed ----
n_cols = 5
n_rows = int(np.ceil(len(tracks) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 2.9 * n_rows), squeeze=False, layout="constrained")
for k in range(n_rows * n_cols):
    ax = axes[k // n_cols][k % n_cols]
    if k >= len(tracks):
        ax.axis("off")
        continue
    t, f0, kind = tracks[k]
    sel = [r for r in records if r["track"] == t]
    amp = 1000 * np.array([r["amp"] for r in sel])
    zeta = np.array([r["zeta"] for r in sel])
    sc = ax.scatter(amp, zeta, c=[r["U"] for r in sel], cmap="viridis", vmin=0, vmax=U_MAX, s=7)
    row = [s for s in summary if s[0] == t][0]
    x = np.logspace(np.log10(amp.min()), np.log10(amp.max()), 20)
    beta, _, _ = fit([np.log10(amp / 1000)], zeta)
    ax.plot(x, beta[0] + beta[1] * np.log10(x / 1000), "k-")
    ax.set_xscale("log")
    ax.set_ylim(np.percentile(zeta, 1) - 0.2, np.percentile(zeta, 99) + 0.2)
    ax.set_title(f"{f0:.3f} Hz {kind} (n = {row[3]})\nc = {row[4]:+.2f} +- {row[5]:.2f} %-pt/decade", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.3)
    if k % n_cols == 0:
        ax.set_ylabel("Structural damping [%]", fontsize=8)
    if k >= len(tracks) - n_cols:
        ax.set_xlabel("Modal displacement std [mm]", fontsize=8)
fig.colorbar(sc, ax=axes, shrink=0.5, label="Mean wind speed [m/s]")
fig.suptitle(f"Low wind (U <= {U_MAX} m/s): damping (quasi-steady aerodynamic part removed for vertical modes)"
             " vs. vibration amplitude")
fig.savefig(OUTPUT_FOLDER / "low_wind_amplitude.png", dpi=90)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

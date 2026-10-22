"""
Hardanger Bridge project - Step 5d
Turbulence of the wind in each recording, compared with the site turbulence model of
Fenerci & Oiseth (2018), and used as an extra variable for the identified damping and frequency.

What it does:
  1. Splits every recording used in step 5 into 10-min segments (the averaging time used by
     Fenerci & Oiseth) and computes for each span anemometer: mean wind speed U, direction, and
     the standard deviations sigma_u (along-wind), sigma_v (cross-wind), sigma_w (vertical).
     Turbulence intensity I_u = sigma_u / U. Stored in a CSV, so this is done only once.
  2. Compares sigma_u, sigma_w and I_u at the midspan sensor A6 with the lognormal model of
     Fenerci & Oiseth (2018), Table 5, separately for east and west winds. The model is also
     corrected for our 2 Hz data, which cannot contain turbulence above 1 Hz.
  3. Regression per tracked mode: damping (and frequency) = a + b U + c I_u + d T, to see if
     turbulence explains part of the scatter that wind speed and temperature do not.

Needs results/step5_all/modes_vs_wind.csv from pipeline/step5_wind.py.

How to use:
  python -m pipeline.step5d_turbulence
"""

from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import load_recording
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================


# Results from step 5 (all data sets): which recordings to use, and their tracked modes
MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"

OUTPUT_FOLDER = RESULTS_FOLDER / "step5d_all"

# Averaging time for the turbulence statistics (s). 600 s = 10 min, as in Fenerci & Oiseth (2018).
SEGMENT_SECONDS = 600

# Span anemometers (A9 is on the tower top and is left out)
SPAN_ANEMOMETERS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

# Sensor used for the comparison with the model (Fenerci & Oiseth used the midspan sensor A6)
MODEL_SENSOR = "A6"

# Segments with a lower mean wind speed are not used (Fenerci & Oiseth discarded U < 3 m/s)
MIN_WIND_SPEED = 3.0

# Wind from directions in this range (deg) is called "East", the rest "West".
# ASSUMPTION: the anemometer angle is the direction the wind comes FROM, measured from the
# global x-axis (0 deg = along x, 90 deg = from +y = from the east side). Not stated in the papers.
EAST_SECTOR = (0.0, 180.0)

# Lognormal turbulence model, Fenerci & Oiseth (2018), Table 5 (final parameters).
# mu and sigma are the mean and standard deviation of ln(x); mu = mu_a + mu_b * U.
MODEL = {
    "East": {"sigma_u": (0.122, 0.039, 0.2566), "sigma_w": (-0.657, 0.032, 0.2632),
             "A_u": (2.67, 0.0248, 0.4538), "A_w": (0.7076, 0.0, 0.4466)},
    "West": {"sigma_u": (0.122, 0.039, 0.3159), "sigma_w": (-0.657, 0.032, 0.3021),
             "A_u": (2.407, 0.048, 0.5282), "A_w": (1.2075, 0.0, 0.4943)},
}
MODEL_HEIGHT = 68.0     # z in the Kaimal spectrum (m), Fenerci & Oiseth (2018), Eq. 5
MODEL_MIN_U = 10.0      # the model was fitted to data above 10 m/s

# Colours per group of data sets in the plots (data set name starts with the key)
GROUPS = {"2014_9": ("Sept. 2014", "tab:blue"), "StormUrd": ("Storm Urd", "tab:red"),
          "Events_Storm": ("other storms", "tab:orange"), "Events_Max_Wind": ("max wind", "tab:purple"),
          "Events_Low_Wind": ("low-wind events", "tab:green"), "LowFreq_2016": ("year 2016", "0.55")}

# ==================================================================


def group_of(data_set):
    """Label and colour of the group a data set belongs to."""
    for key, value in GROUPS.items():
        if data_set.startswith(key):
            return value
    return ("other", "k")


def segment_turbulence(data, fs):
    """
    Turbulence statistics for each 10-min segment of one anemometer.
    data: [n_samples, 4] with columns angle [deg], U [m/s], w [m/s], T [degC].
    Returns a list of dictionaries (one per full segment).

    The horizontal wind is split into a mean part along the mean wind direction and
    fluctuations: u along the mean direction, v across it. w is the vertical fluctuation.
    """
    n_per_segment = int(SEGMENT_SECONDS * fs)
    n_segments = data.shape[0] // n_per_segment       # only full segments
    results = []
    for k in range(n_segments):
        part = data[k * n_per_segment:(k + 1) * n_per_segment]
        angle = np.radians(part[:, 0])
        speed = part[:, 1]
        # Horizontal velocity components in the global x-y plane
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        # Mean wind direction and mean wind speed (of the mean vector)
        mean_direction = np.arctan2(np.mean(vy), np.mean(vx))
        U = np.hypot(np.mean(vx), np.mean(vy))
        # Rotate into along-wind (u) and cross-wind (v) components
        along = vx * np.cos(mean_direction) + vy * np.sin(mean_direction)
        across = -vx * np.sin(mean_direction) + vy * np.cos(mean_direction)
        results.append({
            "segment": k + 1,
            "U": U,
            "direction": np.degrees(mean_direction) % 360,
            "sigma_u": np.std(along),
            "sigma_v": np.std(across),
            "sigma_w": np.std(part[:, 2]),
        })
    return results


def wind_side(direction):
    """'East' or 'West', see EAST_SECTOR."""
    return "East" if EAST_SECTOR[0] <= direction < EAST_SECTOR[1] else "West"


def model_value(side, name, U, z=0.0):
    """
    Value of a lognormal model parameter at mean wind speed U.
    z = 0 gives the median; z = +-1.645 the 95 / 5 percentile; z = +-0.674 the 75 / 25 percentile.
    """
    mu_a, mu_b, sigma = MODEL[side][name]
    return np.exp(mu_a + mu_b * U + z * sigma)


def variance_fraction_above(f_cut, U, A):
    """
    Fraction of the turbulence variance above frequency f_cut for the Kaimal-type spectrum
    S f / sigma^2 = A f_z / (1 + 1.5 A f_z)^(5/3), f_z = f z / U  (Fenerci & Oiseth 2018, Eq. 5).
    Integrating the spectrum from f_cut to infinity gives (1 + 1.5 A f_cut z / U)^(-2/3).
    """
    return (1 + 1.5 * A * f_cut * MODEL_HEIGHT / U) ** (-2 / 3)


def least_squares(X, y):
    """Ordinary least squares y = X beta. Returns beta and its standard errors."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    s2 = np.sum(residuals ** 2) / (n - p)
    return beta, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


def compute_segments(recordings):
    """
    Turbulence statistics of all recordings (file name -> data set), read from the cache file
    turbulence_segments.csv if it exists, otherwise computed from the original .mat files.
    """
    cache_file = OUTPUT_FOLDER / "turbulence_segments.csv"
    keys = ["data_set", "file_name", "start_time_utc", "sensor", "segment", "side",
            "U", "direction", "sigma_u", "sigma_v", "sigma_w", "I_u", "I_w"]
    if cache_file.exists():
        with open(cache_file, newline="") as f:
            rows = list(csv.DictReader(f))
        done = {r["file_name"] for r in rows}
        if done >= set(recordings):
            print(f"Turbulence read from {cache_file.name}")
            return [{k: (r[k] if k in ("data_set", "file_name", "start_time_utc", "sensor", "side")
                         else float(r[k])) for k in keys} for r in rows if r["file_name"] in recordings]

    print(f"Computing turbulence for {len(recordings)} recordings ...")
    mat_files = {p.stem: p for p in DATA_FOLDER.rglob("*.mat") if "results" not in p.parts}
    segments = []
    for file_name, data_set in recordings.items():
        rec = load_recording(mat_files[file_name])
        start = datetime.strptime(rec["start_time"], "%d/%m/%Y %H:%M:%S.%f")
        for sensor in SPAN_ANEMOMETERS:
            data = rec["sensors"].get(sensor)
            if data is None or np.isnan(data).any():
                continue                      # sensor missing or poor quality
            for s in segment_turbulence(data, rec["fs"]):
                if s["U"] < MIN_WIND_SPEED:
                    continue
                s.update({"data_set": data_set, "file_name": file_name, "sensor": sensor,
                          "start_time_utc": start.strftime("%Y-%m-%d %H:%M:%S"),
                          "side": wind_side(s["direction"]), "I_u": s["sigma_u"] / s["U"],
                          "I_w": s["sigma_w"] / s["U"]})
                segments.append(s)

    with open(cache_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for s in segments:
            writer.writerow([s[k] if isinstance(s[k], str) else f"{s[k]:.4f}" for k in keys])
    return segments


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Recordings used in step 5, and the tracked modes
tracks = {}
recordings = {}
with open(MODES_VS_WIND, newline="") as f:
    for row in csv.DictReader(f):
        tracks.setdefault(int(row["track"]), []).append(row)
        recordings[row["file_name"]] = row["data_set"]

# ---- 1. Turbulence per 10-min segment and anemometer ----
segments = compute_segments(recordings)
nyquist = 1.0      # Hz, all LowFreq data are sampled at 2 Hz
print(f"{len(segments)} 10-min segments (all sensors), "
      f"{sum(s['sensor'] == MODEL_SENSOR for s in segments)} from {MODEL_SENSOR}; "
      f"{sum(s['side'] == 'East' for s in segments)} East, {sum(s['side'] == 'West' for s in segments)} West")

# ---- 2. Comparison with the Fenerci & Oiseth (2018) model at A6, east and west winds ----
print(f"\nExpected loss of turbulence variance above {nyquist:.0f} Hz (Kaimal spectrum, median A, East):")
for U in [10, 15, 20]:
    lost_u = variance_fraction_above(nyquist, U, model_value("East", "A_u", U))
    lost_w = variance_fraction_above(nyquist, U, model_value("East", "A_w", U))
    print(f"  U = {U} m/s: sigma_u {100 * (1 - np.sqrt(1 - lost_u)):.1f} % lower, "
          f"sigma_w {100 * (1 - np.sqrt(1 - lost_w)):.1f} % lower")

fig1, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
u_line = np.linspace(3, 28, 100)
for row, side in enumerate(["East", "West"]):
    a6 = [s for s in segments if s["sensor"] == MODEL_SENSOR and s["side"] == side]
    U_a6 = np.array([s["U"] for s in a6])
    in_range = U_a6 >= MODEL_MIN_U
    print(f"\nMeasured vs. model at {MODEL_SENSOR}, {side} winds ({len(a6)} segments, "
          f"{np.sum(in_range)} with U >= {MODEL_MIN_U} m/s):")
    for quantity in ["sigma_u", "sigma_w"]:
        if np.sum(in_range) < 5:
            break
        measured = np.array([s[quantity] for s in a6])[in_range]
        U = U_a6[in_range]
        A_name = "A_u" if quantity == "sigma_u" else "A_w"
        median = model_value(side, quantity, U)
        lost = variance_fraction_above(nyquist, U, model_value(side, A_name, U))
        low = model_value(side, quantity, U, -1.645) * np.sqrt(1 - lost)
        high = model_value(side, quantity, U, 1.645) * np.sqrt(1 - lost)
        inside = np.mean((measured >= low) & (measured <= high))
        print(f"  {quantity}: median ratio measured / model (corrected for 2 Hz) = "
              f"{np.median(measured / (median * np.sqrt(1 - lost))):.2f}; "
              f"{100 * inside:.0f} % inside the 5-95 % band (expected 90 %)")

    for col, (quantity, ylabel) in enumerate([("sigma_u", "$\\sigma_u$ [m/s]"), ("sigma_w", "$\\sigma_w$ [m/s]"),
                                              ("I_u", "$I_u = \\sigma_u / U$ [-]")]):
        ax = axes[row][col]
        base = "sigma_u" if quantity == "I_u" else quantity
        A_name = "A_w" if base == "sigma_w" else "A_u"
        divide = u_line if quantity == "I_u" else 1.0
        lost = variance_fraction_above(nyquist, u_line, model_value(side, A_name, u_line))
        ax.fill_between(u_line, model_value(side, base, u_line, -1.645) / divide,
                        model_value(side, base, u_line, 1.645) / divide, color="0.88", label="model 5-95 %")
        ax.fill_between(u_line, model_value(side, base, u_line, -0.674) / divide,
                        model_value(side, base, u_line, 0.674) / divide, color="0.75", label="model 25-75 %")
        ax.plot(u_line, model_value(side, base, u_line) * np.sqrt(1 - lost) / divide, "k-",
                label="model median (without > 1 Hz)")
        ax.axvspan(0, MODEL_MIN_U, color="tab:orange", alpha=0.07, label="below model fit range")
        drawn = set()
        for s in a6:
            label, color = group_of(s["data_set"])
            ax.plot(s["U"], s[quantity], ".", color=color, markersize=3, alpha=0.6,
                    label=None if label in drawn else label)
            drawn.add(label)
        ax.set_xlabel("10-min mean wind speed U [m/s]")
        ax.set_ylabel(f"{side} winds\n{ylabel}" if col == 0 else ylabel)
        ax.set_xlim(3, 28)
        if quantity == "I_u":
            ax.set_ylim(0, 0.7)
        ax.grid(True, alpha=0.3)
axes[0][0].legend(fontsize=7, loc="upper left", markerscale=3)
fig1.suptitle(f"Turbulence at {MODEL_SENSOR} (10-min segments) vs. Fenerci & Oiseth (2018) model, "
              "east (top) and west (bottom) winds")
fig1.savefig(OUTPUT_FOLDER / "turbulence_vs_model.png", dpi=95)

# ---- 3. Turbulence per recording, and regression of damping / frequency ----
per_recording = {}
for s in segments:
    per_recording.setdefault(s["start_time_utc"], []).append(s)
turbulence = {t: {"U": np.mean([s["U"] for s in v]), "I_u": np.mean([s["I_u"] for s in v]),
                  "I_w": np.mean([s["I_w"] for s in v]), "data_set": v[0]["data_set"]}
              for t, v in per_recording.items()}

with open(OUTPUT_FOLDER / "turbulence_per_recording.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start_time_utc", "mean_U_ms", "mean_I_u", "mean_I_w", "n_segments"])
    for t in sorted(turbulence):
        writer.writerow([t, f"{turbulence[t]['U']:.3f}", f"{turbulence[t]['I_u']:.4f}",
                         f"{turbulence[t]['I_w']:.4f}", len(per_recording[t])])

U_rec = np.array([turbulence[t]["U"] for t in sorted(turbulence)])
I_rec = np.array([turbulence[t]["I_u"] for t in sorted(turbulence)])
print(f"\n{len(turbulence)} recordings with turbulence statistics: I_u = {I_rec.min():.3f} - {I_rec.max():.3f}, "
      f"correlation(U, I_u) = {np.corrcoef(U_rec, I_rec)[0, 1]:+.2f}")

print("\nRegression per mode:  y = a + b U + c I_u + d T   (U, T from step 5; I_u = mean over sensors and segments)")
print("  c per 0.01 (= 1 %-point) of turbulence intensity, d per degC, +- 1 standard error")
print(" track  f [Hz]  type        n    damping: b [%-pt/10 m/s]  c [%-pt/0.01 I_u]"
      "      frequency: b [%/10 m/s]   c [%/0.01 I_u]    d [%/degC]")
results = []
for t in sorted(tracks):
    rows = [r for r in tracks[t] if r["start_time_utc"] in turbulence]
    U = np.array([float(r["mean_wind_speed_ms"]) for r in rows])
    T = np.array([float(r["mean_temperature_degC"]) for r in rows])
    I = np.array([turbulence[r["start_time_utc"]]["I_u"] for r in rows])
    zeta = np.array([float(r["damping_percent"]) for r in rows])
    freq = np.array([float(r["freq_Hz"]) for r in rows])
    ok = ~np.isnan(U) & ~np.isnan(T) & ~np.isnan(I)
    U, T, I, zeta, freq = U[ok], T[ok], I[ok], zeta[ok], freq[ok]
    X = np.column_stack([np.ones(len(U)), U, I, T])
    beta_z, se_z = least_squares(X, zeta)
    beta_f, se_f = least_squares(X, freq)
    f_mean = np.mean(freq)
    res = {"track": t, "f": float(rows[0]["track_median_freq_Hz"]), "type": rows[0]["type"], "n": len(U),
           "b_z": 10 * beta_z[1], "se_b_z": 10 * se_z[1],
           "c_z": 0.01 * beta_z[2], "se_c_z": 0.01 * se_z[2],
           "b_f": 100 * 10 * beta_f[1] / f_mean, "se_b_f": 100 * 10 * se_f[1] / f_mean,
           "c_f": 100 * 0.01 * beta_f[2] / f_mean, "se_c_f": 100 * 0.01 * se_f[2] / f_mean,
           "d_f": 100 * beta_f[3] / f_mean, "se_d_f": 100 * se_f[3] / f_mean}
    results.append(res)
    print(f"  {t:3d}   {res['f']:.3f}  {res['type']:9s} {res['n']:5d}     {res['b_z']:+5.2f} +- {res['se_b_z']:.2f}"
          f"     {res['c_z']:+6.3f} +- {res['se_c_z']:.3f}"
          f"        {res['b_f']:+5.2f} +- {res['se_b_f']:.2f}    {res['c_f']:+6.3f} +- {res['se_c_f']:.3f}"
          f"   {res['d_f']:+6.3f} +- {res['se_d_f']:.3f}")

with open(OUTPUT_FOLDER / "regression_turbulence.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "n", "damp_b_pt_per_10ms", "se",
                     "damp_c_pt_per_0.01Iu", "se", "freq_b_percent_per_10ms", "se",
                     "freq_c_percent_per_0.01Iu", "se", "freq_d_percent_per_degC", "se"])
    for r in results:
        writer.writerow([r["track"], r["f"], r["type"], r["n"]] +
                        [f"{r[k]:.4f}" for k in ["b_z", "se_b_z", "c_z", "se_c_z", "b_f", "se_b_f",
                                                 "c_f", "se_c_f", "d_f", "se_d_f"]])

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
drawn = set()
for t in sorted(turbulence):
    label, color = group_of(turbulence[t]["data_set"])
    ax1.plot(turbulence[t]["U"], turbulence[t]["I_u"], ".", color=color, markersize=4, alpha=0.7,
             label=None if label in drawn else label)
    drawn.add(label)
ax1.set_xlabel("Mean wind speed U [m/s]")
ax1.set_ylabel("Mean $I_u$ (all sensors) [-]")
ax1.set_title("Turbulence intensity per recording")
ax1.set_ylim(0, 0.8)
ax1.legend(fontsize=7, markerscale=3)
markers = {"vertical": "o", "lateral": "s", "torsion": "^"}
for r in results:
    m = markers.get(r["type"], "o")
    ax2.errorbar(r["f"], r["c_z"], yerr=2 * r["se_c_z"], fmt=m, color="k", capsize=2, markersize=5)
    ax3.errorbar(r["f"], r["c_f"], yerr=2 * r["se_c_f"], fmt=m, color="k", capsize=2, markersize=5)
for name, m in markers.items():
    ax2.plot([], [], m, color="k", label=name)
ax2.legend(fontsize=8)
ax2.set_title("Damping: effect of turbulence at constant U and T (c)")
ax2.set_ylabel("c [%-points damping per 0.01 $I_u$]")
ax3.set_title("Frequency: effect of turbulence at constant U and T (c)")
ax3.set_ylabel("c [% frequency per 0.01 $I_u$]")
for ax in (ax1, ax2, ax3):
    ax.grid(True, alpha=0.3)
for ax in (ax2, ax3):
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Mode frequency [Hz]")
fig2.suptitle("Regression y = a + b U + c $I_u$ + d T per tracked mode (error bars = 2 standard errors)")
fig2.savefig(OUTPUT_FOLDER / "regression_turbulence.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

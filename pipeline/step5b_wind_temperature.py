"""
Hardanger Bridge project - Step 5b
Separate the effect of wind speed and temperature on the natural frequencies (and damping).

Between the two events wind speed and temperature are mixed up (storm Urd is both windier
and colder than September 2014). A multiple linear regression

    f = a + b * U + c * T        (U = mean wind speed, T = mean temperature)

estimates the effect of wind at constant temperature (b) and of temperature at constant
wind (c) at the same time. The standard errors tell us if the data can really tell the
two apart: if U and T were perfectly correlated, the standard errors would be huge.

Needs results/step5/modes_vs_wind.csv from pipeline/step5_wind.py.

How to use:
  python -m pipeline.step5b_wind_temperature
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

# results\step5_all = all data sets (1,485 recordings); results\step5 = Sept 2014 + storm Urd only
INPUT_FILE = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "step5_all"

# Error bars in the plots: this many standard errors (2 = roughly a 95 % interval)
N_STANDARD_ERRORS = 2

# Marker per mode type in the plots
MARKERS = {"vertical": "o", "lateral": "s", "torsion": "^"}

# ==================================================================


def read_table(path):
    """Read modes_vs_wind.csv into a dictionary {track number: list of rows}."""
    tracks = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tracks.setdefault(int(row["track"]), []).append(row)
    return tracks


def least_squares(X, y):
    """
    Ordinary least squares y = X * beta.
    Returns the coefficients beta and their standard errors.
    Standard error: sqrt of the diagonal of  s^2 (X^T X)^-1, with s^2 = residual variance.
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    s2 = np.sum(residuals ** 2) / (n - p)
    covariance = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(covariance))


def fit_track(rows, column):
    """
    Fit 'column' (freq_Hz or damping_percent) of one track with
      model 1: a + b U
      model 2: a + b U + c T
    Returns a dictionary with the coefficients and standard errors of both models.
    """
    U = np.array([float(r["mean_wind_speed_ms"]) for r in rows])
    T = np.array([float(r["mean_temperature_degC"]) for r in rows])
    y = np.array([float(r[column]) for r in rows])
    # Leave out recordings without a valid wind speed or temperature (no working anemometer)
    ok = ~np.isnan(U) & ~np.isnan(T) & ~np.isnan(y)
    U, T, y = U[ok], T[ok], y[ok]
    ones = np.ones(len(y))

    beta1, se1 = least_squares(np.column_stack([ones, U]), y)
    beta2, se2 = least_squares(np.column_stack([ones, U, T]), y)
    return {
        "n": len(y), "mean": np.mean(y),
        "b_wind_only": beta1[1], "se_b_wind_only": se1[1],
        "b": beta2[1], "se_b": se2[1],
        "c": beta2[2], "se_c": se2[2],
        "corr_UT": np.corrcoef(U, T)[0, 1],
    }


def plot_coefficients(results, title, wind_label, temp_label, relative):
    """
    Left: wind coefficient vs. mode frequency, wind-only model and wind + temperature model.
    Right: temperature coefficient vs. mode frequency.
    relative = True: coefficients divided by the mean frequency of the mode (in %).
    Error bars = N_STANDARD_ERRORS standard errors.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for res in results:
        f0 = res["median_freq"]
        scale = 100 / res["mean"] if relative else 1.0
        marker = MARKERS.get(res["type"], "o")
        # wind-only model (grey, shifted slightly left so the error bars do not overlap)
        ax1.errorbar(f0 - 0.006, 10 * scale * res["b_wind_only"],
                     yerr=N_STANDARD_ERRORS * 10 * scale * res["se_b_wind_only"],
                     fmt=marker, color="0.6", markersize=5, capsize=2)
        # wind + temperature model (black)
        ax1.errorbar(f0 + 0.006, 10 * scale * res["b"], yerr=N_STANDARD_ERRORS * 10 * scale * res["se_b"],
                     fmt=marker, color="k", markersize=5, capsize=2)
        ax2.errorbar(f0, scale * res["c"], yerr=N_STANDARD_ERRORS * scale * res["se_c"],
                     fmt=marker, color="tab:red", markersize=5, capsize=2)

    # Legend entries
    ax1.plot([], [], "o", color="0.6", label="wind only:  a + b U")
    ax1.plot([], [], "o", color="k", label="wind + temperature:  a + b U + c T")
    for name, marker in MARKERS.items():
        ax2.plot([], [], marker, color="tab:red", label=name)

    for ax in (ax1, ax2):
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xlabel("Mode frequency [Hz]")
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel(wind_label)
    ax1.set_title("Effect of wind speed (b)")
    ax1.legend(fontsize=8)
    ax2.set_ylabel(temp_label)
    ax2.set_title("Effect of temperature at constant wind (c)")
    ax2.legend(fontsize=8)
    fig.suptitle(f"{title}   (error bars = {N_STANDARD_ERRORS} standard errors)")
    return fig


# --------------------------- Main program ---------------------------

tracks = read_table(INPUT_FILE)

# Correlation between wind speed and temperature over all recordings (each recording once)
recordings = {}
for rows in tracks.values():
    for r in rows:
        recordings[r["start_time_utc"]] = (float(r["mean_wind_speed_ms"]), float(r["mean_temperature_degC"]))
U_all = np.array([v[0] for v in recordings.values()])
T_all = np.array([v[1] for v in recordings.values()])
valid = ~np.isnan(U_all) & ~np.isnan(T_all)
print(f"{np.sum(~valid)} recordings without valid wind speed or temperature are left out")
U_all, T_all = U_all[valid], T_all[valid]
print(f"{len(U_all)} recordings: U = {U_all.min():.1f} - {U_all.max():.1f} m/s, "
      f"T = {T_all.min():.1f} - {T_all.max():.1f} degC, correlation(U, T) = {np.corrcoef(U_all, T_all)[0, 1]:.2f}\n")

freq_results = []
damp_results = []
for t in sorted(tracks):
    rows = tracks[t]
    info = {"track": t, "median_freq": float(rows[0]["track_median_freq_Hz"]), "type": rows[0]["type"]}
    freq_results.append({**info, **fit_track(rows, "freq_Hz")})
    damp_results.append({**info, **fit_track(rows, "damping_percent")})

print("FREQUENCY  (coefficients relative to the mean frequency of the mode, +- 1 standard error)")
print(" track  f [Hz]  type        n   wind only           wind + temperature")
print("                                [%/10 m/s]          b [%/10 m/s]       c [%/degC]")
for res in freq_results:
    s = 100 / res["mean"]
    print(f"  {res['track']:3d}   {res['median_freq']:.3f}  {res['type']:9s}  {res['n']:3d}"
          f"   {10 * s * res['b_wind_only']:+5.2f} +- {10 * s * res['se_b_wind_only']:.2f}"
          f"     {10 * s * res['b']:+5.2f} +- {10 * s * res['se_b']:.2f}"
          f"     {s * res['c']:+6.3f} +- {s * res['se_c']:.3f}")

print("\nDAMPING  (absolute, in percentage points, +- 1 standard error)")
print(" track  f [Hz]  type        n   wind only           wind + temperature")
print("                                [%-pt/10 m/s]       b [%-pt/10 m/s]    c [%-pt/degC]")
for res in damp_results:
    print(f"  {res['track']:3d}   {res['median_freq']:.3f}  {res['type']:9s}  {res['n']:3d}"
          f"   {10 * res['b_wind_only']:+5.2f} +- {10 * res['se_b_wind_only']:.2f}"
          f"     {10 * res['b']:+5.2f} +- {10 * res['se_b']:.2f}"
          f"     {res['c']:+6.3f} +- {res['se_c']:.3f}")

# ---- Plots ----
fig1 = plot_coefficients(freq_results, "Natural frequency", "b  [% per 10 m/s]", "c  [% per degC]", True)
fig1.savefig(OUTPUT_FOLDER / "regression_frequency.png", dpi=100)
fig2 = plot_coefficients(damp_results, "Damping ratio", "b  [%-points per 10 m/s]", "c  [%-points per degC]", False)
fig2.savefig(OUTPUT_FOLDER / "regression_damping.png", dpi=100)

# ---- Save table ----
with open(OUTPUT_FOLDER / "regression_wind_temperature.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["track", "median_freq_Hz", "type", "n_recordings",
                     "freq_b_wind_only_percent_per_10ms", "se",
                     "freq_b_percent_per_10ms", "se", "freq_c_percent_per_degC", "se",
                     "damp_b_wind_only_pt_per_10ms", "se",
                     "damp_b_pt_per_10ms", "se", "damp_c_pt_per_degC", "se"])
    for fr, dr in zip(freq_results, damp_results):
        s = 100 / fr["mean"]
        writer.writerow([fr["track"], fr["median_freq"], fr["type"], fr["n"],
                         f"{10 * s * fr['b_wind_only']:.3f}", f"{10 * s * fr['se_b_wind_only']:.3f}",
                         f"{10 * s * fr['b']:.3f}", f"{10 * s * fr['se_b']:.3f}",
                         f"{s * fr['c']:.4f}", f"{s * fr['se_c']:.4f}",
                         f"{10 * dr['b_wind_only']:.3f}", f"{10 * dr['se_b_wind_only']:.3f}",
                         f"{10 * dr['b']:.3f}", f"{10 * dr['se_b']:.3f}",
                         f"{dr['c']:.4f}", f"{dr['se_c']:.4f}"])

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

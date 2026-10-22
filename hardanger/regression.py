"""
Shared regression tools for the step 5 results (used by steps 5g and 5h).

Reads results/step5_all/modes_vs_wind.csv into one table (one row per recording, one column
per tracked mode) and computes the same quantities as steps 5b and 5c:
  - per mode: frequency = a + b U + c T  ->  b [% per 10 m/s], c [% per degC]
              damping   = a + b U + c T  ->  b [%-points per 10 m/s]
  - over the vertical modes: K (damping slope = K / f), power law exponent p, and
    C_heave = 8 pi K m / (rho B)  (quasi-steady heave damping coefficient, step 5c)

This file is imported by the step scripts; it does nothing when run on its own.
"""

import csv

import numpy as np


AIR_DENSITY = 1.25      # kg/m^3, Oiseth et al. (2010)
B = 18.3                # deck width (m)
MASS = 12820.0          # vertical modal equivalent mass (kg/m), Oiseth et al. (2010), Table 2


def load_table(path):
    """
    One row per recording: file name, day, U, T, direction; and for each tracked mode the
    frequency and damping in every recording (NaN where the mode was not found).
    """
    recordings = {}
    modes = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["file_name"]
            if name not in recordings:
                recordings[name] = (row["start_time_utc"][:10], float(row["mean_wind_speed_ms"]),
                                    float(row["mean_temperature_degC"]), float(row["mean_wind_direction_deg"]))
            t = int(row["track"])
            if t not in modes:
                modes[t] = {"track": t, "f0": float(row["track_median_freq_Hz"]), "type": row["type"], "values": {}}
            modes[t]["values"][name] = (float(row["freq_Hz"]), float(row["damping_percent"]))

    names = sorted(recordings)
    table = {
        "file_name": np.array(names),
        "day": np.array([recordings[n][0] for n in names]),
        "U": np.array([recordings[n][1] for n in names]),
        "T": np.array([recordings[n][2] for n in names]),
        "direction": np.array([recordings[n][3] for n in names]),
        "modes": [],
    }
    for t in sorted(modes):
        m = modes[t]
        m["freq"] = np.array([m["values"].get(n, (np.nan, np.nan))[0] for n in names])
        m["damping"] = np.array([m["values"].get(n, (np.nan, np.nan))[1] for n in names])
        del m["values"]
        table["modes"].append(m)
    return table


def least_squares(X, y):
    """Ordinary least squares y = X beta. Returns beta and the (ordinary) standard errors."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    s2 = np.sum(residuals ** 2) / (len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


def mode_coefficients(table, rows, mode):
    """
    Wind and temperature coefficients of one mode, using only the recordings in 'rows'
    (rows may contain repeated indices, as in a bootstrap sample).
    Returns [b_freq %/10 m/s, c_freq %/degC, b_damping %-pt/10 m/s] and their ordinary standard errors.
    """
    U, T = table["U"][rows], table["T"][rows]
    f, z = mode["freq"][rows], mode["damping"][rows]
    ok = ~np.isnan(U) & ~np.isnan(T) & ~np.isnan(f)
    X = np.column_stack([np.ones(np.sum(ok)), U[ok], T[ok]])
    beta_f, se_f = least_squares(X, f[ok])
    beta_z, se_z = least_squares(X, z[ok])
    scale = 100 / np.mean(f[ok])
    values = np.array([10 * scale * beta_f[1], scale * beta_f[2], 10 * beta_z[1]])
    errors = np.array([10 * scale * se_f[1], scale * se_f[2], 10 * se_z[1]])
    return values, errors


def scaling_fit(freqs, slopes, slope_errors):
    """
    Step 5c: weighted fit of damping slope = K / f, and of the power law slope = C f^p.
    slopes in %-points per 10 m/s. Returns K (1/m), p, and C_heave = 8 pi K m / (rho B).
    """
    freqs, slopes, slope_errors = map(np.asarray, (freqs, slopes, slope_errors))
    x = 1 / freqs
    w = 1 / slope_errors ** 2
    K = np.sum(w * x * slopes) / np.sum(w * x ** 2)          # %-pt Hz per 10 m/s
    ok = slopes > 0
    X = np.column_stack([np.ones(np.sum(ok)), np.log(freqs[ok])])
    W = np.diag((slopes[ok] / slope_errors[ok]) ** 2)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ np.log(slopes[ok]))
    K_si = K / 100 / 10
    C_heave = 8 * np.pi * K_si * MASS / (AIR_DENSITY * B)
    return K_si, beta[1], C_heave

"""
Tools for environmental normalization and novelty detection of tracked natural frequencies.
Used by pipeline/step6_normalization.py and pipeline/step6b_damage_patterns.py.

  - residuals_for_mode: normalization model f [%] = a + sum(coefficients * variables), fitted on
    the reference (training) rows, residuals for all rows
  - hotelling: multivariate novelty index T^2 = (r - mean)^T S^-1 (r - mean)
  - daily_means: average residual vector per day

The table is the one from hardanger.regression.load_table (keys "U", "T", modes with "freq").
This file is imported by the step scripts; it does nothing when run on its own.
"""

import numpy as np

from hardanger.regression import least_squares


def model_matrix(table, rows, variables):
    """Columns [1, ...] for the chosen variables: 'U', 'U2', 'T'."""
    columns = [np.ones(len(rows))]
    for v in variables:
        if v == "U":
            columns.append(table["U"][rows])
        elif v == "U2":
            columns.append(table["U"][rows] ** 2)
        elif v == "T":
            columns.append(table["T"][rows])
    return np.column_stack(columns)


def residuals_for_mode(table, mode, train, variables, mean_freq):
    """
    Fit the normalization model on the training rows, return the residuals [%] for all rows
    (NaN where the mode is missing or U / T are missing).
    """
    y = 100 * (mode["freq"] / mean_freq - 1)
    ok = ~np.isnan(y) & ~np.isnan(table["U"]) & ~np.isnan(table["T"])
    fit_rows = np.where(train & ok)[0]
    if variables:
        beta, _ = least_squares(model_matrix(table, fit_rows, variables), y[fit_rows])
        prediction = model_matrix(table, np.arange(len(y)), variables) @ beta
    else:
        prediction = np.full(len(y), np.mean(y[fit_rows]))
    residual = y - prediction
    residual[~ok] = np.nan
    return residual


def hotelling(residuals, mean, covariance_inverse):
    """T^2 = (r - mean)^T S^-1 (r - mean) for each row of residuals [n, n_modes]."""
    d = residuals - mean
    return np.einsum("ij,jk,ik->i", d, covariance_inverse, d)


def daily_means(residuals, days):
    """Average residual vector per day. Returns the day labels and [n_days, n_modes]."""
    labels = np.unique(days)
    return labels, np.array([np.mean(residuals[days == d], axis=0) for d in labels])


def t2_setup(R_train, days_train, quantile):
    """
    Everything Hotelling T^2 needs from the reference period, for single recordings and for
    daily averages: mean vector, inverse covariance and the alarm limit (quantile of T^2).
    """
    setup = {}
    _, D_train = daily_means(R_train, days_train)
    for key, data in [("single", R_train), ("daily", D_train)]:
        mean = np.mean(data, axis=0)
        S_inv = np.linalg.inv(np.cov(data, rowvar=False))
        setup[key] = (mean, S_inv, np.quantile(hotelling(data, mean, S_inv), quantile))
    return setup


def detection_level(R_test, D_test, weights, setup, levels, target_rate):
    """
    Smallest damage level (from 'levels') detected in at least target_rate of the test recordings
    (single) and test days (daily). A damage level x lowers the residual of mode k by x * weights[k] (%).
    Returns [single, daily]; NaN if not reached within 'levels'.
    """
    found = [np.nan, np.nan]
    for x in levels:
        for i, (key, data) in enumerate([("single", R_test), ("daily", D_test)]):
            if np.isnan(found[i]):
                mean, S_inv, limit = setup[key]
                if np.mean(hotelling(data - x * weights, mean, S_inv) > limit) >= target_rate:
                    found[i] = x
        if not np.isnan(found).any():
            break
    return found

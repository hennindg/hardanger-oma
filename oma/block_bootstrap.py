"""
Block bootstrap: realistic uncertainties for results from correlated recordings.

Why: recordings 30 minutes apart in the same storm are not independent (same wind, same
temperature, same traffic). Ordinary standard errors assume they are, and are far too small.
The block bootstrap resamples whole BLOCKS (here: calendar days) instead of single recordings,
so the correlation inside a block is kept. The analysis is repeated on many resampled data sets;
the spread of the results is the uncertainty.

  1. Split the data into blocks (e.g. one block per day).
  2. Draw as many blocks as there are, at random WITH replacement (some days appear twice,
     some not at all).
  3. Repeat the whole analysis on the recordings of the drawn blocks.
  4. Repeat 2-3 many times (e.g. 1000). Standard deviation of the results = standard error;
     2.5 and 97.5 percentiles = 95 % confidence interval.

General: knows nothing about the bridge. This file is imported by the step scripts.
"""

import numpy as np


def block_bootstrap(block_ids, statistic, n_boot, rng):
    """
    block_ids : one block label per row of the data (e.g. the date of each recording)
    statistic : function that takes an array of row indices and returns a 1-D array of results
    n_boot    : number of bootstrap repetitions
    rng       : numpy random generator (fixed seed for reproducible results)
    Returns an array [n_boot, n_results].
    """
    block_ids = np.asarray(block_ids)
    blocks = np.unique(block_ids)
    rows_of_block = {b: np.where(block_ids == b)[0] for b in blocks}
    results = []
    for _ in range(n_boot):
        drawn = rng.choice(blocks, size=len(blocks), replace=True)
        rows = np.concatenate([rows_of_block[b] for b in drawn])
        results.append(statistic(rows))
    return np.array(results)


def summarize(samples):
    """
    Standard error and 95 % interval (2.5 / 97.5 percentiles) of bootstrap samples, per column.
    Repetitions where a result could not be computed (NaN) are ignored.
    """
    return (np.nanstd(samples, axis=0), np.nanpercentile(samples, 2.5, axis=0),
            np.nanpercentile(samples, 97.5, axis=0))

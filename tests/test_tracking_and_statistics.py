"""Tests of mode tracking, the block bootstrap and the record simulation."""

import numpy as np
from scipy import signal

from oma.mode_tracking import assign_to_reference
from oma.block_bootstrap import block_bootstrap, summarize
from oma.ssi_simulation import modal_acceleration


def test_modes_are_assigned_to_the_closest_reference():
    names = np.array(["H1 y", "H1 z", "H3 y", "H3 z"])
    lateral = np.array([1.0, 0.0, 0.7, 0.0])
    vertical = np.array([0.0, 1.0, 0.0, -0.6])
    references = [{"freq": 0.10, "shape": lateral, "channel_names": names, "type": "lateral"},
                  {"freq": 0.12, "shape": vertical, "channel_names": names, "type": "vertical"}]
    # One recording: the vertical mode slightly shifted, plus a spurious mode far away
    recording = {"freq": np.array([0.121, 0.30]), "damping": np.array([0.02, 0.01]),
                 "shapes": np.array([vertical, lateral]), "channel_names": names}
    tracks = assign_to_reference([recording], references, max_distance=0.2)
    assert np.isnan(tracks[0]["freq"][0])                # no lateral mode near 0.10 Hz
    assert np.isclose(tracks[1]["freq"][0], 0.121)       # vertical mode found
    assert np.isclose(tracks[1]["damping"][0], 0.02)


def test_block_bootstrap_draws_whole_blocks():
    """With a statistic that counts the rows of block 'a', every count must be a multiple of 3."""
    block_ids = np.array(["a", "a", "a", "b", "c", "c"])
    samples = block_bootstrap(block_ids, lambda rows: np.array([np.sum(block_ids[rows] == "a")]),
                              n_boot=200, rng=np.random.default_rng(0))
    assert samples.shape == (200, 1)
    assert np.all(samples % 3 == 0)


def test_bootstrap_summary_ignores_failed_repetitions():
    samples = np.array([[1.0], [2.0], [np.nan], [3.0]])
    standard_error, low, high = summarize(samples)
    assert np.isclose(standard_error[0], np.std([1.0, 2.0, 3.0]))
    assert low[0] >= 1.0 and high[0] <= 3.0


def test_simulated_mode_has_its_spectral_peak_at_the_natural_frequency():
    fs = 2.0
    acc = modal_acceleration(np.random.default_rng(1), 0.25, 0.01, 40000, fs)
    assert np.isclose(np.std(acc), 1.0)
    freqs, psd = signal.welch(acc, fs=fs, nperseg=4096)
    assert abs(freqs[np.argmax(psd)] - 0.25) < 0.002

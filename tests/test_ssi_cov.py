"""
Tests of the SSI-cov building blocks and of the complete identification on simulated records
with KNOWN modal properties. Run all tests from the repository folder with:  python -m pytest
"""

import numpy as np

from oma.ssi_cov import output_covariances, block_toeplitz, ssi_cov, mac, check_stability, normalize_mode_shape
from oma.mode_clustering import identify_modes
from oma.ssi_simulation import simulate_modes


# A small test structure: three well separated modes seen by four channels
FREQS = np.array([0.15, 0.32, 0.61])
DAMPINGS = np.array([0.010, 0.015, 0.008])
SHAPES = np.array([[1.0, 0.8, 0.5, 0.2],
                   [0.6, -0.4, -1.0, 0.3],
                   [0.3, -0.9, 0.7, -1.0]])
FS = 2.0


def simulated_record(seed, duration):
    """A long record of the test structure: white-noise forcing, equal levels, 2 % noise."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * FS)
    return simulate_modes(rng, FREQS, DAMPINGS, SHAPES, np.ones(3), 0.02, n_samples, FS)


def test_zero_lag_covariance_is_the_ordinary_covariance():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((5000, 3))
    R = output_covariances(y, 4)
    assert R.shape == (5, 3, 3)
    assert np.allclose(R[0], np.cov(y, rowvar=False, bias=True))


def test_block_toeplitz_has_the_right_size_and_blocks():
    rng = np.random.default_rng(1)
    R = output_covariances(rng.standard_normal((2000, 2)), 5)
    T = block_toeplitz(R, 3)
    assert T.shape == (6, 6)
    # top-left block is R_s (lag = block rows), see the docstring of block_toeplitz
    assert np.allclose(T[:2, :2], R[3])


def test_mac_is_one_for_the_same_shape_and_zero_for_orthogonal_shapes():
    a = np.array([1.0, 2.0, -1.0])
    assert np.isclose(mac(a, a), 1.0)
    assert np.isclose(mac(a, (2 - 3j) * a), 1.0)          # MAC ignores (complex) scaling
    assert np.isclose(mac(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), 0.0)


def test_normalized_shape_has_largest_component_one():
    shape = np.array([0.2 + 0.1j, -2.0 + 1.0j, 0.5j])
    normalized = normalize_mode_shape(shape)
    assert np.isclose(normalized[np.argmax(np.abs(normalized))], 1.0)


def test_ssi_finds_the_true_poles_at_one_model_order():
    y = simulated_record(seed=2, duration=20000)
    poles, _ = ssi_cov(y, FS, 30, [6], max_damping=0.10)
    found = poles[0]["freq"]
    for f in FREQS:
        assert np.min(np.abs(found - f)) / f < 0.005       # within 0.5 %


def test_truncated_and_full_svd_give_the_same_poles():
    """ssi_cov uses a truncated SVD when only a few singular values are needed."""
    y = simulated_record(seed=3, duration=4000)
    truncated, _ = ssi_cov(y, FS, 30, [6, 10], max_damping=0.10)     # 10 < 4 * 30: truncated SVD
    full, _ = ssi_cov(y, FS, 30, [6, 10, 119], max_damping=0.10)     # 119 = 4 * 30 - 1: full SVD
    for a, b in zip(truncated, full[:2]):
        assert np.allclose(np.sort(a["freq"]), np.sort(b["freq"]))
        assert np.allclose(np.sort(a["damping"]), np.sort(b["damping"]))


def test_stability_check_marks_a_repeated_pole_as_stable():
    shape = np.array([[1.0], [0.5]])
    poles = [{"order": n, "freq": np.array([0.2]), "damping": np.array([0.01]), "shapes": shape}
             for n in [2, 4, 6]]
    poles = check_stability(poles, 0.01, 0.05, 0.95)
    assert poles[2]["stable_all"][0]


def test_automatic_identification_recovers_frequency_and_damping():
    """The complete method (SSI + stability + clustering) on a long record: small errors expected."""
    y = simulated_record(seed=4, duration=20000)
    freq, damping, shapes = identify_modes(y, FS, block_rows=30, orders=list(range(2, 41, 2)), max_damping=0.10,
                                           tol_freq=0.01, tol_damping=0.05, tol_mac=0.95,
                                           max_distance=0.05, min_orders=8)
    for k, f in enumerate(FREQS):
        j = np.argmin(np.abs(freq - f))
        assert abs(freq[j] - f) / f < 0.005                       # frequency within 0.5 %
        assert abs(damping[j] - DAMPINGS[k]) < 0.002               # damping within 0.2 %-points
        assert mac(shapes[j], SHAPES[k]) > 0.99                    # mode shape recovered

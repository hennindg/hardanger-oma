"""Tests of the Hardanger-specific helpers: channel selection, mode classification, C_heave fit,
Hotelling T^2 and the simulated girder."""

import numpy as np

from hardanger.data import girder_channels, section_mode_shape, classify_mode, GRAVITY, PAIR_DISTANCE
from hardanger.regression import scaling_fit, AIR_DENSITY, B, MASS
from hardanger.novelty_detection import hotelling
from hardanger import simulated_girder as girder


def fake_recording():
    """A tiny recording in the format of load_recording(): two girder sensors, a tower top, an anemometer."""
    n = 10
    good = np.ones((n, 3))
    bad = np.ones((n, 3))
    bad[3, 2] = np.nan                                   # "poor" quality component -> NaN
    return {"fs": 2.0, "start_time": "",
            "sensors": {"H1 East": good, "H1 Vest": 2 * good, "H3 East": bad, "H10 East": good,
                        "A1": np.ones((n, 4))},
            "positions": {"H1 East": [480, 6.33, 0], "H1 Vest": [480, -6.64, 0], "H3 East": [240, 6.33, 0],
                          "H10 East": [655, 0, 120.5], "A1": [0, 0, 8]}}


def test_girder_channels_skip_bad_sensors_tower_tops_and_anemometers():
    y, channels = girder_channels(fake_recording())
    names = [c["sensor"] + " " + c["component"] for c in channels]
    assert names == ["H1 East y", "H1 East z", "H1 Vest y", "H1 Vest z"]
    assert np.allclose(y[:, 2], 2 * GRAVITY)             # unit g converted to m/s^2


def test_torsion_shape_is_classified_as_torsion():
    _, channels = girder_channels(fake_recording())
    # east edge up, west edge down, no lateral motion
    shape = np.array([0.0, 1.0, 0.0, -1.0])
    x, lateral, vertical, torsion = section_mode_shape(shape, channels)
    assert np.isclose(vertical[0], 0.0)
    assert np.isclose(torsion[0], 2.0 / PAIR_DISTANCE)
    assert classify_mode(lateral, vertical, torsion) == "torsion"


def test_scaling_fit_recovers_the_heave_coefficient():
    """Damping slopes made with C_heave = 2.5 exactly must give back 2.5."""
    freqs = np.array([0.14, 0.21, 0.28, 0.40, 0.55, 0.71, 0.90])
    K = 2.5 * AIR_DENSITY * B / (8 * np.pi * MASS)           # zeta_aero = K U / f
    slopes = 100 * 10 * K / freqs                            # %-points per 10 m/s
    _, p, C_heave = scaling_fit(freqs, slopes, 0.05 * slopes)
    assert np.isclose(C_heave, 2.5)
    assert np.isclose(p, -1.0)


def test_hotelling_t2_is_zero_at_the_mean_and_grows_with_distance():
    mean = np.array([0.0, 0.0])
    S_inv = np.linalg.inv(np.array([[1.0, 0.3], [0.3, 2.0]]))
    t2 = hotelling(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), mean, S_inv)
    assert np.isclose(t2[0], 0.0)
    assert t2[1] > 0 and np.isclose(t2[2], 4 * t2[1])


def test_simulated_girder_damping_follows_the_quasi_steady_formula():
    U, f = 20.0, 0.2
    expected = girder.ZETA_STRUCTURAL + girder.AIR_DENSITY * girder.B * girder.C_HEAVE_TRUE * U / (8 * np.pi * girder.MASS_VERTICAL * f)
    assert np.isclose(girder.true_damping("vertical", f, U), expected)
    assert girder.true_damping("torsion", f, U) == girder.ZETA_STRUCTURAL
    # mode shapes: one channel pair (y, z) per sensor, vertical modes only in z
    shape = girder.mode_shape("vertical", 1)
    assert shape.shape == (2 * len(girder.SENSORS),)
    assert np.allclose(shape[0::2], 0.0)

"""
A simplified, simulated Hardanger girder with KNOWN modal properties (used by steps 7c and 7d).

ASSUMPTIONS (not an FE model):
  - the real girder sensor layout: 16 accelerometers (H1-H9 east/west, H2 west only, H8 east only),
    y and z components = 32 channels, 2 Hz, 1860 s per record (like the LowFreq data)
  - sinusoidal mode shapes over the main span: lateral modes in y, vertical modes in z (east = west),
    torsion modes in z (east up, west down); frequencies like the tracked modes
  - every mode driven by its own white-noise force, random (lognormal) response level per mode
  - true damping = structural part + quasi-steady aerodynamic part (same formula as step 5c)
  - white measurement noise on every channel

This file is imported by other scripts; it does nothing when run on its own.
"""

import numpy as np

from oma.ssi_cov import mac
from oma.ssi_simulation import modal_acceleration


DURATION = 1860.0           # s, like one LowFreq recording
FS = 2.0                    # Hz, like the LowFreq data
NOISE_RATIO = 0.05          # measurement noise std / channel std
LEVEL_SCATTER = 0.5         # std of ln(response level) between modes and records

# True damping model
ZETA_STRUCTURAL = 0.005     # 0.5 % structural damping in every mode (ASSUMPTION)
C_HEAVE_TRUE = 2.5          # quasi-steady heave coefficient, close to the wind tunnel value 2.53
C_DRAG = 0.70               # drag coefficient referred to D (Oiseth et al. 2010)
AIR_DENSITY = 1.25          # kg/m^3
B = 18.3                    # deck width (m)
D = 3.33                    # deck height (m)
MASS_VERTICAL = 12820.0     # kg/m, Oiseth et al. (2010) Table 2
MASS_LATERAL = 11000.0      # kg/m, inside the 9,400-12,480 range of Oiseth et al. (2010) Table 2
SPAN = 1310.0               # main span (m)

# Modes: (type, frequency [Hz], number of half waves over the span).
# Frequencies like the tracked modes; half-wave numbers as in step 6c for the vertical modes.
MODES = [
    ("lateral", 0.052, 1), ("lateral", 0.105, 2), ("lateral", 0.184, 3), ("lateral", 0.319, 4),
    ("vertical", 0.119, 2), ("vertical", 0.142, 1), ("vertical", 0.206, 3), ("vertical", 0.212, 4),
    ("vertical", 0.276, 5), ("vertical", 0.333, 6), ("vertical", 0.401, 7), ("vertical", 0.470, 8),
    ("vertical", 0.547, 9), ("vertical", 0.626, 10), ("vertical", 0.713, 11), ("vertical", 0.804, 12),
    ("vertical", 0.901, 13),
    ("torsion", 0.371, 1), ("torsion", 0.560, 2),
]

# Sensors: name, x position (m), side (+1 east, -1 west), as in the real system
SENSORS = [("H1", 480, 1), ("H1", 480, -1), ("H2", 360, -1), ("H3", 240, 1), ("H3", 240, -1),
           ("H4", 120, 1), ("H4", 120, -1), ("H5", -7, 1), ("H5", -7, -1), ("H6", -120, 1),
           ("H6", -120, -1), ("H7", -240, 1), ("H7", -240, -1), ("H8", -360, 1), ("H9", -480, 1),
           ("H9", -480, -1)]
Y_EAST, Y_WEST = 6.33, -6.64   # transverse sensor positions (m), for torsion


def true_damping(mode_type, freq, U):
    """True damping ratio of one mode at mean wind speed U (structural + quasi-steady aerodynamic)."""
    if mode_type == "vertical":
        return ZETA_STRUCTURAL + AIR_DENSITY * B * C_HEAVE_TRUE * U / (8 * np.pi * MASS_VERTICAL * freq)
    if mode_type == "lateral":
        return ZETA_STRUCTURAL + AIR_DENSITY * D * C_DRAG * U / (4 * np.pi * MASS_LATERAL * freq)
    return ZETA_STRUCTURAL


def mode_shape(mode_type, n):
    """
    Mode shape at the 32 channels (16 sensors x [y, z]): a sine with n half waves over the span,
    in y for lateral modes, in z for vertical modes, and z = +/- (y position) for torsion.
    """
    shape = np.zeros(2 * len(SENSORS))
    for i, (name, x, side) in enumerate(SENSORS):
        wave = np.sin(n * np.pi * (x + SPAN / 2) / SPAN)
        if mode_type == "lateral":
            shape[2 * i] = wave
        elif mode_type == "vertical":
            shape[2 * i + 1] = wave
        else:
            y_position = Y_EAST if side > 0 else Y_WEST
            shape[2 * i + 1] = wave * y_position / Y_EAST
    return shape


def simulate_record(rng, U, shapes):
    """One record [n_samples, 32] at mean wind U: sum of modes x random levels + measurement noise."""
    n_samples = int(DURATION * FS)
    y = np.zeros((n_samples, shapes.shape[1]))
    for k, (mode_type, freq, n) in enumerate(MODES):
        level = np.exp(LEVEL_SCATTER * rng.standard_normal())
        y += level * np.outer(modal_acceleration(rng, freq, true_damping(mode_type, freq, U), n_samples, FS), shapes[k])
    y += NOISE_RATIO * np.std(y, axis=0) * rng.standard_normal(y.shape)
    return y


def match_modes(freq, shapes, true_freqs, true_shapes, max_distance):
    """For each true mode, the index of the closest identified mode (|df|/f + 1 - MAC), or -1."""
    found = np.full(len(true_freqs), -1)
    for k in range(len(true_freqs)):
        best, best_distance = -1, max_distance
        for j in range(len(freq)):
            distance = abs(freq[j] - true_freqs[k]) / true_freqs[k] + 1 - mac(shapes[j], true_shapes[k])
            if distance < best_distance:
                best, best_distance = j, distance
        found[k] = best
    return found

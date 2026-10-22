"""
Simulation of output-only vibration records with KNOWN modal properties, for testing SSI.
General (no Hardanger file format): modes are given as frequency, damping and a real mode shape.

  - modal_acceleration: acceleration of one mode (SDOF oscillator) under white-noise force
  - simulate_modes: sum of modes x levels + white measurement noise -> [n_samples, n_channels]

Each mode is a single-degree-of-freedom oscillator q'' + 2 zeta omega q' + omega^2 q = F with its
own white-noise force, simulated exactly in discrete time (zero-order hold). The discrete poles
are then exactly exp(lambda dt), so the true frequency and damping are known exactly.
This file is imported by other scripts; it does nothing when run on its own.
"""

import numpy as np
from scipy import signal


def modal_acceleration(rng, freq, zeta, n_samples, fs):
    """Acceleration of one SDOF oscillator with white-noise force, unit standard deviation."""
    omega = 2 * np.pi * freq
    A = np.array([[0.0, 1.0], [-omega ** 2, -2 * zeta * omega]])
    Bc = np.array([[0.0], [1.0]])
    C = np.array([[-omega ** 2, -2 * zeta * omega]])     # acceleration = F - 2 zeta omega v - omega^2 q
    Dc = np.array([[1.0]])
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, Bc, C, Dc), 1 / fs, method="zoh")
    n_warmup = int(20 / (zeta * omega)) + 1                # let the start transient die out
    force = rng.standard_normal(n_samples + n_warmup)
    _, acc, _ = signal.dlsim((Ad, Bd, Cd, Dd, 1 / fs), force)
    acc = acc[n_warmup:, 0]
    return acc / np.std(acc)


def simulate_modes(rng, freqs, dampings, shapes, levels, noise_std, n_samples, fs):
    """
    Record [n_samples, n_channels] = sum over modes of level_k * a_k(t) * shape_k + noise.

    freqs, dampings, levels : one value per mode (levels = modal acceleration std)
    shapes                  : real mode shapes [n_modes, n_channels]
    noise_std               : measurement noise std per channel (scalar or [n_channels])
    """
    y = np.zeros((n_samples, shapes.shape[1]))
    for k in range(len(freqs)):
        y += levels[k] * np.outer(modal_acceleration(rng, freqs[k], dampings[k], n_samples, fs), shapes[k])
    y += np.asarray(noise_std) * rng.standard_normal(y.shape)
    return y

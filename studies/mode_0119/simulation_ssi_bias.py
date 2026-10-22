"""
Hardanger Bridge project - simulation test: does SSI give biased frequency or damping for a
heavily damped mode under turbulent, gusty (non-stationary) wind?

We simulate a system whose modal properties we KNOW, and identify it with the same 2-mode SSI
procedure as studies/mode_0119/band_ssi_antisym.py. Any difference between the identified and the true
values is a bias of the method (or of the assumptions behind it).

The simulated system copies what we found for the two antisymmetric modes:
  - lateral mode:  0.105 Hz, damping 0.8 %, shape [q_y, q_z] = [1, 0.25]
  - vertical mode: 0.12 Hz,  damping 2 % or 6 %, shape [q_y, q_z] = [0.1, 1]
Each mode is a single-degree-of-freedom oscillator driven by its own random force. The modal
equations are integrated step by step (Runge-Kutta 4), so stiffness and damping may change in time:

    q'' + 2 zeta(t) omega(t) q' + omega(t)^2 q = F(t)

Records: 1860 s (like a 30-min recording), simulated at 20 Hz, then anti-alias filtered and
decimated to 2 Hz (like the LowFreq data), with 10 % white measurement noise.

Forcing cases (see CASES below):
  1-2  white noise (SSI's own assumption): the method should be unbiased here
  3    turbulence-shaped (Kaimal) force at constant wind
  4    gusty wind: the force scales with U(t)^2, U(t) varying slowly between ~8 and ~23 m/s
  5    as 4, plus quasi-steady aeroelastic effects: zeta(t) = zeta_s + K U(t) / f (K from step 5c)
       and the frequency follows U(t) with +0.8 % per 10 m/s (measured for the vertical modes)

How to use:
  python -m studies.mode_0119.simulation_ssi_bias
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from oma.ssi_cov import ssi_cov, bandpass_filter
from hardanger.paths import RESULTS_FOLDER


# ============================ SETTINGS ============================

N_REPETITIONS = 20          # random records per case
RANDOM_SEED = 1             # fixed, so the results can be reproduced
DURATION = 1860.0           # s, like one recording
FS_SIM = 20.0               # Hz, integration rate
FS_OUT = 2.0                # Hz, like the LowFreq data
NOISE_RATIO = 0.10          # measurement noise std / channel std

# True modal properties
LATERAL = {"freq": 0.105, "damping": 0.008, "shape": np.array([1.0, 0.25])}
VERTICAL = {"freq": 0.12, "shape": np.array([0.10, 1.0])}
VERTICAL_TO_LATERAL = 0.6   # response std of the vertical mode relative to the lateral mode

# Wind model for the turbulence-shaped and gusty cases
MEAN_WIND = 15.0            # m/s
GUST_STD = 4.0              # m/s, slow variation of U(t) (1-min means of ~8-23 m/s as in the storm evening)
GUST_CUTOFF = 0.01          # Hz, gusts slower than ~100 s (low-pass filter of the gust process)
KAIMAL_A = 20.0             # spectral parameter A_u, about exp(2.67 + 0.0248 U) at 15 m/s (Fenerci & Oiseth 2018)
KAIMAL_Z = 68.0             # m, height in the Kaimal spectrum (Fenerci & Oiseth 2018)
FREQ_SLOPE = 0.0008         # relative frequency change per m/s (+0.8 % per 10 m/s, step 5 vertical modes)

# Aerodynamic damping constant K (zeta_aero = K U / f), measured in step 5c
SCALING_FIT = RESULTS_FOLDER / "step5" / "damping_scaling_fit.csv"

# Identification: same settings as studies/mode_0119/band_ssi_antisym.py
BAND = (0.07, 0.17)
SEARCH_RANGE = (0.095, 0.140)
ORDERS = [4, 6, 8]
BLOCK_ROWS_LIST = [30, 45, 60, 75, 90]
MAX_DAMPING = 0.15
MIXED_LIMIT = 0.6

# Forcing cases: name -> (force type, vertical damping (mean), aeroelastic changes on/off)
CASES = {
    "1 white, zeta 2 %": ("white", 0.02, False),
    "2 white, zeta 6 %": ("white", 0.06, False),
    "3 turbulence spectrum, zeta 6 %": ("kaimal", 0.06, False),
    "4 gusty force, zeta 6 %": ("gusty", 0.06, False),
    "5 gusty + aeroelastic, zeta 6 %": ("gusty", 0.06, True),
}

OUTPUT_FOLDER = RESULTS_FOLDER / "simulation"

# ==================================================================


def gust_wind(rng, n_steps, dt):
    """
    Slowly varying wind speed U(t) for each repetition: white noise, low-pass filtered
    below GUST_CUTOFF, scaled to std GUST_STD around MEAN_WIND. Returns [n_reps, n_steps].
    """
    sos = signal.butter(2, GUST_CUTOFF, btype="lowpass", fs=1 / dt, output="sos")
    g = signal.sosfiltfilt(sos, rng.standard_normal((N_REPETITIONS, n_steps)), axis=1)
    g = g / np.std(g, axis=1, keepdims=True)
    return np.maximum(MEAN_WIND + GUST_STD * g, 2.0)


def kaimal_noise(rng, n_steps, dt):
    """
    Random force with a Kaimal-shaped spectrum S(f) ~ 1 / (1 + 1.5 A f z / U)^(5/3), made by
    shaping white noise in the frequency domain. Unit standard deviation. Returns [n_reps, n_steps].
    """
    white = rng.standard_normal((N_REPETITIONS, n_steps))
    spectrum = np.fft.rfft(white, axis=1)
    f = np.fft.rfftfreq(n_steps, dt)
    shape = (1 + 1.5 * KAIMAL_A * f * KAIMAL_Z / MEAN_WIND) ** (-5 / 6)   # sqrt of S(f)
    x = np.fft.irfft(spectrum * shape, n=n_steps, axis=1)
    return x / np.std(x, axis=1, keepdims=True)


def make_force(rng, force_type, n_steps, dt, U):
    """Force for one mode: white, Kaimal-shaped, or Kaimal-shaped scaled by (U(t)/U_mean)^2."""
    if force_type == "white":
        return rng.standard_normal((N_REPETITIONS, n_steps))
    force = kaimal_noise(rng, n_steps, dt)
    if force_type == "gusty":
        # Quasi-steady buffeting force is proportional to U(t)^2 (times the turbulence)
        force = force * (U / MEAN_WIND) ** 2
    return force


def simulate_mode(force, freq, damping, dt):
    """
    Integrate q'' + 2 zeta omega q' + omega^2 q = F with Runge-Kutta 4 for all repetitions at once.
    freq and damping are scalars or arrays [n_reps, n_steps] (time-varying).
    Force and parameters are held constant within each time step.
    Returns the acceleration q'' [n_reps, n_steps].
    """
    n_reps, n_steps = force.shape
    omega = 2 * np.pi * np.broadcast_to(freq, force.shape)
    zeta = np.broadcast_to(damping, force.shape)
    x = np.zeros(n_reps)
    v = np.zeros(n_reps)
    acc = np.zeros((n_reps, n_steps))

    for k in range(n_steps):
        w, z, F = omega[:, k], zeta[:, k], force[:, k]

        def accel(xx, vv):
            return F - 2 * z * w * vv - w ** 2 * xx

        acc[:, k] = accel(x, v)
        k1x, k1v = v, accel(x, v)
        k2x, k2v = v + 0.5 * dt * k1v, accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = v + 0.5 * dt * k2v, accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = v + dt * k3v, accel(x + dt * k3x, v + dt * k3v)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return acc


def simulate_case(rng, force_type, zeta_vertical, aeroelastic, K):
    """
    Simulate N_REPETITIONS records for one case.
    Returns channels [n_reps, n_samples_out, 2] at FS_OUT, and the true vertical frequency and
    damping at the mean wind speed (the "target" values).
    """
    dt = 1 / FS_SIM
    n_steps = int(DURATION * FS_SIM)
    U = gust_wind(rng, n_steps, dt) if force_type == "gusty" else np.full((N_REPETITIONS, n_steps), MEAN_WIND)

    # Vertical mode parameters, possibly following the wind speed
    f_v = VERTICAL["freq"]
    if aeroelastic:
        zeta_structural = zeta_vertical - K * MEAN_WIND / f_v      # so the value at the mean wind is zeta_vertical
        freq_v = f_v * (1 + FREQ_SLOPE * (U - MEAN_WIND))
        damp_v = zeta_structural + K * U / f_v
    else:
        freq_v, damp_v = f_v, zeta_vertical

    acc_lat = simulate_mode(make_force(rng, force_type, n_steps, dt, U), LATERAL["freq"], LATERAL["damping"], dt)
    acc_ver = simulate_mode(make_force(rng, force_type, n_steps, dt, U), freq_v, damp_v, dt)

    # Same overall response level in every record (only the ratio between the modes matters)
    acc_lat = acc_lat / np.std(acc_lat, axis=1, keepdims=True)
    acc_ver = VERTICAL_TO_LATERAL * acc_ver / np.std(acc_ver, axis=1, keepdims=True)

    # Channels q_y, q_z = sum of mode shape x modal acceleration
    channels = (acc_lat[:, :, None] * LATERAL["shape"][None, None, :]
                + acc_ver[:, :, None] * VERTICAL["shape"][None, None, :])
    channels = channels + NOISE_RATIO * np.std(channels, axis=1, keepdims=True) * rng.standard_normal(channels.shape)

    # Anti-alias filter and decimate to FS_OUT (like the resampling of the LowFreq data)
    factor = int(FS_SIM / FS_OUT)
    channels = signal.decimate(channels, factor, ftype="fir", axis=1, zero_phase=True)
    return channels, f_v, zeta_vertical


def identify(y, fs):
    """
    The 2-mode SSI of studies/mode_0119/band_ssi_antisym.py: bandpass, low orders x several block rows,
    classify each pole as lateral or vertical by its shape, one pole per model (closest to the
    median), then the median. Returns {"lateral": (f, zeta), "vertical": (f, zeta)} (None if not found).
    """
    y = bandpass_filter(y, fs, BAND[0], BAND[1])
    candidates = {"lateral": [], "vertical": []}
    model = 0
    for s in BLOCK_ROWS_LIST:
        poles, _ = ssi_cov(y, fs, s, ORDERS, MAX_DAMPING)
        for P in poles:
            model += 1
            for f, z, shape in zip(P["freq"], P["damping"], P["shapes"].T):
                if not SEARCH_RANGE[0] < f < SEARCH_RANGE[1]:
                    continue
                ratio = abs(shape[1] / shape[0])
                if ratio < MIXED_LIMIT:
                    candidates["lateral"].append((model, f, z))
                elif ratio > 1 / MIXED_LIMIT:
                    candidates["vertical"].append((model, f, z))
    result = {}
    for kind, cands in candidates.items():
        if len(cands) < 3:
            result[kind] = None
            continue
        median_f = np.median([c[1] for c in cands])
        best = {}
        for c in cands:
            if c[0] not in best or abs(c[1] - median_f) < abs(best[c[0]][1] - median_f):
                best[c[0]] = c
        result[kind] = (np.median([c[1] for c in best.values()]), np.median([c[2] for c in best.values()]))
    return result


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)
with open(SCALING_FIT, newline="") as f:
    K = float(next(csv.DictReader(f))["K_per_m"])
print(f"K = {K:.2e} 1/m (step 5c); {N_REPETITIONS} records per case\n")

summary = []
errors = {}
for name, (force_type, zeta_v, aeroelastic) in CASES.items():
    channels, f_true_v, zeta_true_v = simulate_case(rng, force_type, zeta_v, aeroelastic, K)
    truth = {"lateral": (LATERAL["freq"], LATERAL["damping"]), "vertical": (f_true_v, zeta_true_v)}
    errors[name] = {"lateral": [], "vertical": []}
    for r in range(N_REPETITIONS):
        found = identify(channels[r], FS_OUT)
        for kind in ["lateral", "vertical"]:
            if found[kind] is None:
                continue
            f_id, z_id = found[kind]
            f0, z0 = truth[kind]
            errors[name][kind].append((100 * (f_id - f0) / f0, 100 * (z_id - z0)))

    print(name)
    for kind in ["lateral", "vertical"]:
        e = np.array(errors[name][kind])
        f0, z0 = truth[kind]
        print(f"   {kind:8s} (true {f0:.3f} Hz, {100 * z0:.1f} %): found {len(e)}/{N_REPETITIONS};  "
              f"frequency error {np.mean(e[:, 0]):+.2f} +- {np.std(e[:, 0]):.2f} %;  "
              f"damping error {np.mean(e[:, 1]):+.2f} +- {np.std(e[:, 1]):.2f} %-pt")
        summary.append([name, kind, f0, 100 * z0, len(e), np.mean(e[:, 0]), np.std(e[:, 0]),
                        np.mean(e[:, 1]), np.std(e[:, 1])])

with open(OUTPUT_FOLDER / "ssi_bias_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "mode", "true_freq_Hz", "true_damping_percent", "n_found",
                     "freq_error_mean_percent", "freq_error_std_percent",
                     "damping_error_mean_pt", "damping_error_std_pt"])
    for row in summary:
        writer.writerow([row[0], row[1], row[2], f"{row[3]:.2f}", row[4]] + [f"{v:.3f}" for v in row[5:]])

# ---- Plot: errors per case, each dot one simulated record ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, layout="constrained")
names = list(CASES)
for row, kind in enumerate(["vertical", "lateral"]):
    for col, (index, label) in enumerate([(0, "Frequency error [%]"), (1, "Damping error [%-points]")]):
        ax = axes[row][col]
        for c, name in enumerate(names):
            e = np.array(errors[name][kind])
            jitter = 0.08 * rng.standard_normal(len(e))
            ax.plot(c + jitter, e[:, index], "o", color="tab:blue", alpha=0.5, markersize=4)
            ax.errorbar(c + 0.3, np.mean(e[:, index]), yerr=np.std(e[:, index]), fmt="s", color="k", capsize=3)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_ylabel(label)
        ax.set_title(f"{kind} mode ({'0.12 Hz' if kind == 'vertical' else '0.105 Hz, 0.8 %'})")
        ax.grid(True, alpha=0.3)
for ax in axes[1]:
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
fig.suptitle("2-mode SSI on simulated records: identified minus true value "
             "(dots = records, black = mean +- std)")
fig.savefig(OUTPUT_FOLDER / "ssi_bias.png", dpi=100)

print(f"\nResults saved in {OUTPUT_FOLDER}")
plt.show()

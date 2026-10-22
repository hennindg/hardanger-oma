"""
Hardanger Bridge project - Step 7d
Correct the SSI damping bias of every recording with a simulated "twin" of that recording.

Step 7c showed that SSI-cov underestimates damping on 30-min records, more for heavily damped modes
and for more block rows. How large the bias is for the REAL recordings depends on their response
levels, noise and modes, so here every real recording gets its own calibrated twin:

  1. Take the modes identified in the recording (step 7a results: frequency, damping, mode shape).
  2. Estimate from the recording itself
       - the response level of every mode (band-pass filter + modal filter on the nearby modes,
         std of the modal acceleration, as in step 5f)
       - the measurement noise of every channel (residual after removing all identified modes)
  3. Simulate a twin record with exactly these modes, levels and noise (each mode driven by white
     noise, oma/ssi_simulation.py), identify it with the SAME settings, and match the identified modes
     to the input modes. bias = identified - input damping.
  4. Iterate (parametric bootstrap): the bias belongs to the TRUE damping, which is higher than the
     identified one, so the second twin uses input damping = identified - bias of the first twin.
     corrected damping = identified damping - bias of the last twin.

This is done for 80, 120 and 160 block rows. If the correction works, the corrected results of the
three settings must agree (they differ by up to 0.45 in C_heave before correction, step 7b).

Validation first: the same procedure on the step 7c simulated records, where the true damping is
known (true C_heave = 2.5).

Output:
  results/robustness/<variant>_corrected/<file>_modes.npz  same format as step 7a, corrected damping
     -> run pipeline/step7b_robustness_compare.py afterwards for slopes, C_heave and bootstrap
  results/step7d/calibration/<file>.npz   levels, noise, real and twin spectra (calibration check)
  results/step7d/validation/*.npz          validation on simulated records
Resumable: files already done are skipped. Slow (~2 min per recording), start it detached.

How to use:
  python -m pipeline.step7d_bias_correction
"""

import time

import numpy as np

from hardanger.data import load_recording, girder_channels
from oma.ssi_cov import bandpass_filter, first_singular_value_spectrum
from oma.mode_clustering import identify_modes
from oma.ssi_simulation import simulate_modes
from hardanger import simulated_girder as girder
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

ROBUSTNESS_FOLDER = RESULTS_FOLDER / "robustness"
OUTPUT_FOLDER = RESULTS_FOLDER / "step7d"

# Variants to correct and their number of block rows (all other settings = baseline, step 3)
VARIANTS = {"rows80": 80, "baseline": 120, "rows160": 160}
ORDERS = list(range(2, 161, 2))
MAX_DAMPING = 0.10
TOL_FREQ, TOL_DAMPING, TOL_MAC = 0.01, 0.05, 0.95
MAX_DISTANCE = 0.05
MIN_ORDERS = 20

N_ITERATIONS = 2            # twins per mode and variant: 1 per iteration
MIN_INPUT_DAMPING = 0.001   # input damping of a twin never below 0.1 %
MATCH_DISTANCE = 0.20       # identified twin mode <-> input mode: |df|/f + (1 - MAC) below this
RANDOM_SEED = 11

# Response level and noise estimation
BAND_HALF_WIDTH = 0.15          # band-pass f0 +- 15 % (as step 5f)
FILTER_MODES_HALF_WIDTH = 0.30  # modal filter on all modes within +- 30 % (as step 5f)
SPECTRUM_SEGMENT = 256.0        # s, Welch segments for spectra

# Validation on the step 7c simulated records (same random numbers as step 7c)
VALIDATION_SPEEDS = [2, 6, 10, 14, 18, 22]
VALIDATION_RECORDS_PER_SPEED = 8
VALIDATION_SEED = 7             # = RANDOM_SEED of step 7c
VALIDATION_BLOCK_ROWS = 120

# ==================================================================


def modal_levels(y, fs, freqs, shapes):
    """
    Std of the modal acceleration of every mode (band-pass + modal filter on nearby modes).
    shapes: real mode shapes [n_modes, n_channels], in the scaling used for the twin.
    """
    levels = np.zeros(len(freqs))
    for k, f0 in enumerate(freqs):
        # Upper band edge kept just below the Nyquist frequency
        high = min(f0 * (1 + BAND_HALF_WIDTH), 0.98 * fs / 2)
        band = bandpass_filter(y, fs, f0 * (1 - BAND_HALF_WIDTH), high)
        nearby = np.where(np.abs(freqs - f0) <= FILTER_MODES_HALF_WIDTH * f0)[0]
        q = band @ np.linalg.pinv(shapes[nearby].T).T          # modal accelerations [n_samples, n_nearby]
        levels[k] = np.std(q[:, list(nearby).index(k)])
    return levels


def noise_levels(y, shapes):
    """
    Measurement noise std per channel from what the identified modes cannot explain: project the
    record onto the mode shapes (least squares, all modes at once) and take the residual.
    The residual has only (n_channels - n_modes) of the n_channels noise dimensions left, so its
    variance is scaled up by n_channels / (n_channels - n_modes). Modes that were not identified
    end up in the residual too and are then treated as noise.
    """
    n_channels = y.shape[1]
    q = y @ np.linalg.pinv(shapes.T).T                     # modal accelerations [n_samples, n_modes]
    residual = y - q @ shapes
    n_free = max(n_channels - shapes.shape[0], 1)
    return np.std(residual, axis=0) * np.sqrt(n_channels / n_free)


def twin_damping(rng, freqs, input_damping, shapes, levels, noise, n_samples, fs, block_rows):
    """
    Simulate one twin with the given modes, identify it, and return the identified damping of every
    input mode (NaN if not found) and the twin record itself.
    """
    y_twin = simulate_modes(rng, freqs, input_damping, shapes, levels, noise, n_samples, fs)
    f_id, z_id, s_id = identify_modes(y_twin, fs, block_rows, ORDERS, MAX_DAMPING, TOL_FREQ, TOL_DAMPING,
                                      TOL_MAC, MAX_DISTANCE, MIN_ORDERS)
    found = girder.match_modes(f_id, s_id, freqs, shapes, MATCH_DISTANCE)
    damping = np.full(len(freqs), np.nan)
    damping[found >= 0] = z_id[found[found >= 0]]
    return damping, y_twin


def correct_record(rng, y, fs, freqs, damping, complex_shapes, block_rows):
    """
    Bias correction of one identified record (steps 2-4 in the docstring at the top).
    Returns the corrected damping, the bias of the last twin, levels, noise and the last twin record.
    Modes the twin never finds keep their uncorrected damping (bias NaN).
    """
    shapes = np.real(complex_shapes)
    levels = modal_levels(y, fs, freqs, shapes)
    noise = noise_levels(y, shapes)
    input_damping = damping.copy()
    bias = np.full(len(freqs), np.nan)
    for iteration in range(N_ITERATIONS):
        identified, y_twin = twin_damping(rng, freqs, input_damping, shapes, levels, noise, len(y), fs, block_rows)
        bias = identified - input_damping
        corrected = np.where(np.isnan(bias), damping, damping - bias)
        input_damping = np.clip(corrected, MIN_INPUT_DAMPING, None)
    return corrected, bias, levels, noise, y_twin


def run_validation():
    """The correction on the step 7c simulated records (known true damping); results per record."""
    folder = OUTPUT_FOLDER / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    true_shapes = np.array([girder.mode_shape(t, n) for t, f, n in girder.MODES])
    true_freqs = np.array([m[1] for m in girder.MODES])
    rng_records = np.random.default_rng(VALIDATION_SEED)
    rng_twins = np.random.default_rng(RANDOM_SEED)
    for U in VALIDATION_SPEEDS:
        for r in range(VALIDATION_RECORDS_PER_SPEED):
            record_rng = np.random.default_rng(rng_records.integers(1e9))     # same record as step 7c
            twin_rng = np.random.default_rng(rng_twins.integers(1e9))
            path = folder / f"U{U:02d}_record{r}.npz"
            if path.exists():
                continue
            t0 = time.time()
            y = girder.simulate_record(record_rng, U, true_shapes)
            freq, damping, shapes = identify_modes(y, girder.FS, VALIDATION_BLOCK_ROWS, ORDERS, MAX_DAMPING,
                                                   TOL_FREQ, TOL_DAMPING, TOL_MAC, MAX_DISTANCE, MIN_ORDERS)
            corrected, bias, _, _, _ = correct_record(twin_rng, y, girder.FS, freq, damping, shapes,
                                                      VALIDATION_BLOCK_ROWS)
            # Which identified mode is which true mode (for the comparison with the truth)
            found = girder.match_modes(freq, shapes, true_freqs, true_shapes, MATCH_DISTANCE)
            identified_true = np.full(len(true_freqs), np.nan)
            corrected_true = np.full(len(true_freqs), np.nan)
            identified_true[found >= 0] = damping[found[found >= 0]]
            corrected_true[found >= 0] = corrected[found[found >= 0]]
            np.savez(path, U=U, identified=identified_true, corrected=corrected_true)
            print(f"validation U = {U} m/s record {r}: {np.sum(found >= 0)} of {len(true_freqs)} modes "
                  f"({time.time() - t0:.0f} s)", flush=True)


def run_real_recordings():
    """The correction on every recording of step 7a, for the variants in VARIANTS."""
    mat_files = {p.stem: p for p in DATA_FOLDER.rglob("*.mat") if "results" not in p.parts}
    files = sorted(p.name.replace("_modes.npz", "") for p in (ROBUSTNESS_FOLDER / "baseline").glob("*_modes.npz"))
    (OUTPUT_FOLDER / "calibration").mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        (ROBUSTNESS_FOLDER / f"{variant}_corrected").mkdir(exist_ok=True)
    rng_files = np.random.default_rng(RANDOM_SEED + 1)

    for i, file_name in enumerate(files):
        twin_rng = np.random.default_rng(rng_files.integers(1e9))
        outputs = [ROBUSTNESS_FOLDER / f"{v}_corrected" / f"{file_name}_modes.npz" for v in VARIANTS]
        if all(p.exists() for p in outputs):
            continue
        t0 = time.time()
        rec = load_recording(mat_files[file_name])
        y, channels = girder_channels(rec)
        names = [c["sensor"] + " " + c["component"] for c in channels]
        for variant, block_rows in VARIANTS.items():
            data = np.load(ROBUSTNESS_FOLDER / variant / f"{file_name}_modes.npz")
            result = {key: data[key] for key in data.files}
            if list(result["channel_names"]) != names:
                raise ValueError(f"{file_name}: channels differ from step 7a")
            corrected, bias, levels, noise, y_twin = correct_record(
                twin_rng, y, rec["fs"], result["freq"], result["damping"], result["shapes"], block_rows)
            result["damping_uncorrected"] = result["damping"]
            result["damping"] = corrected
            result["bias"] = bias
            np.savez(ROBUSTNESS_FOLDER / f"{variant}_corrected" / f"{file_name}_modes.npz", **result)
            if variant == "baseline":
                # Calibration check: does the twin look like the real recording?
                sv_freqs, sv_real = first_singular_value_spectrum(y, rec["fs"], SPECTRUM_SEGMENT)
                _, sv_twin = first_singular_value_spectrum(y_twin, rec["fs"], SPECTRUM_SEGMENT)
                np.savez(OUTPUT_FOLDER / "calibration" / f"{file_name}.npz", freq=result["freq"],
                         levels=levels, noise=noise, std_real=np.std(y, axis=0), std_twin=np.std(y_twin, axis=0),
                         sv_freqs=sv_freqs, sv_real=sv_real, sv_twin=sv_twin)
        print(f"[{i + 1}/{len(files)}] {file_name} ({time.time() - t0:.0f} s)", flush=True)


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
print("Validation on simulated records (true C_heave = 2.5)", flush=True)
run_validation()
print("\nReal recordings", flush=True)
run_real_recordings()
print(f"\nResults saved in {OUTPUT_FOLDER} and {ROBUSTNESS_FOLDER}/<variant>_corrected")

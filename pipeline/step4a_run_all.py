"""
Hardanger Bridge project - Step 4a
Run SSI-cov + automatic mode picking (steps 2 and 3) on ALL recordings in a folder,
and save the identified modes of each recording to its own .npz file.

This is the slow part (about 20 s per recording). Recordings that already have a result
file are skipped, so you can stop the script and start it again. Delete the files in
results/step4/<data set>/modes (or set RERUN = True) if you change the SSI or clustering settings.

Step 4b (pipeline/step4b_tracking.py) then links the modes between recordings.

How to use:
  python -m pipeline.step4a_run_all
"""

import shutil
import time

import numpy as np

from hardanger.data import load_recording, girder_channels, section_mode_shape, mean_wind, classify_mode
from oma.ssi_cov import ssi_cov, check_stability, normalize_mode_shape
from oma.mode_clustering import cluster_poles
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

# Folders with recordings (.mat files) to analyse, one data set each. The .mat files may be in
# sub-folders (they are searched for). All 27 data sets used in the project (1,766 recordings):
# September 2014, storms, max wind, low-wind events and the full year 2016 (tools/download_data.py).
DATA_FOLDERS = (
    [DATA_FOLDER / "LowFreq_2014_9" / "2014_9", DATA_FOLDER / "Events_StormUrd_LowFreq" / "StormUrd_LowFreq"]
    + [DATA_FOLDER / name for name in ["Events_StormNina_LowFreq", "Events_StormOle_LowFreq",
                                       "Events_StormRoar_LowFreq", "Events_StormTor_LowFreq",
                                       "Events_Max_Wind_LowFreq"]]
    + sorted(DATA_FOLDER.glob("Events_Low_Wind_*_LowFreq"))
    + [DATA_FOLDER / f"LowFreq_2016_{month}" for month in range(1, 13)]
)

# Results go to one folder per data set, named after the data folder:
# results\step4\<data folder name>\modes
RESULTS_ROOT = RESULTS_FOLDER / "step4"

# True = analyse all recordings again, False = skip recordings that already have a result
RERUN = False

# --- SSI and clustering settings: same values as step 3 (see comments there) ---
BLOCK_ROWS = 120
MIN_ORDER = 2
MAX_ORDER = 160
ORDER_STEP = 2
MAX_DAMPING = 0.10
TOL_FREQ = 0.01
TOL_DAMPING = 0.05
TOL_MAC = 0.95
MAX_DISTANCE = 0.05
MIN_ORDERS = 20

# ==================================================================


def identify_modes(path):
    """
    Steps 2 + 3 for one recording: read the file, run SSI-cov, check stability,
    cluster the stable poles. Returns a dictionary with everything that is saved.
    """
    recording = load_recording(path)
    y, channels = girder_channels(recording)
    wind_speed, wind_direction, anemometers_used = mean_wind(recording)

    orders = list(range(MIN_ORDER, MAX_ORDER + 1, ORDER_STEP))
    poles, singular_values = ssi_cov(y, recording["fs"], BLOCK_ROWS, orders, MAX_DAMPING)
    poles = check_stability(poles, TOL_FREQ, TOL_DAMPING, TOL_MAC)
    modes, all_stable = cluster_poles(poles, MAX_DISTANCE, MIN_ORDERS)

    for m in modes:
        m["shape"] = normalize_mode_shape(m["shape"])
        x, lateral, vertical, torsion = section_mode_shape(m["shape"], channels)
        m["type"] = classify_mode(lateral, vertical, torsion)

    return {
        "file_name": path.stem,
        "start_time": recording["start_time"],
        "mean_wind_speed": wind_speed,
        "mean_wind_direction": wind_direction,
        "n_anemometers": len(anemometers_used),
        "freq": np.array([m["freq"] for m in modes]),
        "freq_std": np.array([m["freq_std"] for m in modes]),
        "damping": np.array([m["damping"] for m in modes]),
        "damping_std": np.array([m["damping_std"] for m in modes]),
        "n_orders": np.array([m["n_orders"] for m in modes]),
        "mode_type": np.array([m["type"] for m in modes]),
        "shapes": np.array([m["shape"] for m in modes]),     # [n_modes, n_channels], complex
        "channel_names": np.array([c["sensor"] + " " + c["component"] for c in channels]),
        "channel_x": np.array([c["x"] for c in channels]),
    }


# --------------------------- Main program ---------------------------

for data_folder in DATA_FOLDERS:
    output_folder = RESULTS_ROOT / data_folder.name / "modes"
    output_folder.mkdir(parents=True, exist_ok=True)
    files = sorted(data_folder.rglob("*.mat"))      # also finds .mat files in sub-folders
    print(f"\n{len(files)} recordings in {data_folder.name}")

    for k, path in enumerate(files):
        result_file = output_folder / f"{path.stem}_modes.npz"
        if result_file.exists() and not RERUN:
            print(f"[{k + 1}/{len(files)}] {path.stem}: already done, skipped")
            continue

        # The same recording can be in two data sets (e.g. a storm event and its month).
        # Then copy the earlier result instead of running SSI again.
        earlier = [] if RERUN else list(RESULTS_ROOT.glob(f"*/modes/{path.stem}_modes.npz"))
        if earlier:
            shutil.copy(earlier[0], result_file)
            print(f"[{k + 1}/{len(files)}] {path.stem}: same recording as in "
                  f"{earlier[0].parent.parent.name}, result copied")
            continue

        t0 = time.time()
        try:
            result = identify_modes(path)
        except Exception as error:
            # One bad file should not stop the whole data set. Report it and go on.
            print(f"[{k + 1}/{len(files)}] {path.stem}: FAILED ({error})")
            continue

        np.savez(result_file, **result)
        print(f"[{k + 1}/{len(files)}] {path.stem}: {len(result['freq'])} modes, "
              f"U = {result['mean_wind_speed']:.1f} m/s  ({time.time() - t0:.0f} s)")

    print(f"Results saved in {output_folder}")

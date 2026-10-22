"""
Hardanger Bridge project - Step 7a
Robustness check: run SSI-cov + automatic mode picking with OTHER SETTINGS on a subset of recordings.

Only the number of block rows needs a new SSI (a new SVD). All other settings only change how the
poles are picked: the maximum model order uses a subset of the same poles (every order is computed
from the same SVD), and the stability tolerances and clustering limits are applied afterwards.
So for each recording: SSI with 80, 120 and 160 block rows (orders up to 200), then every
VARIANT below picks its modes from these poles.

Recordings: a subset of the step 5 recordings, stratified by mean wind speed, so all conditions
are represented. Results: one .npz per recording and variant in results/robustness/<variant>/,
plus recording_conditions.csv (wind, temperature, direction, day). Recordings already done are skipped.

Long run (~1 min per recording): start it detached, e.g. with PowerShell Start-Process.

How to use:
  python -m pipeline.step7a_robustness_run
"""

import csv
import time

import numpy as np

from hardanger.data import load_recording, girder_channels, section_mode_shape, mean_wind, mean_temperature, classify_mode
from oma.ssi_cov import ssi_cov, check_stability, normalize_mode_shape
from oma.mode_clustering import cluster_poles
from hardanger.paths import DATA_FOLDER, RESULTS_FOLDER


# ============================ SETTINGS ============================

MODES_VS_WIND = RESULTS_FOLDER / "step5_all" / "modes_vs_wind.csv"
OUTPUT_FOLDER = RESULTS_FOLDER / "robustness"

# Subset: this many recordings from each wind speed bin (m/s)
WIND_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 40)]
PER_BIN = 30
RANDOM_SEED = 9

# SSI runs (one SVD each) and the orders computed
BLOCK_ROWS_RUNS = [80, 120, 160]
ORDERS = list(range(2, 201, 2))
MAX_DAMPING = 0.10

# Baseline settings (= step 3 / 4a) and the variants. min_orders_fraction = minimum share of the
# model orders up to max_order in which a cluster must appear (step 3: 20 of 80 orders = 0.25).
BASELINE = {"block_rows": 120, "max_order": 160, "tol_freq": 0.01, "tol_damping": 0.05, "tol_mac": 0.95,
            "max_distance": 0.05, "min_orders_fraction": 0.25}
VARIANTS = {
    "baseline": {},
    "rows80": {"block_rows": 80},
    "rows160": {"block_rows": 160},
    "order120": {"max_order": 120},
    "order200": {"max_order": 200},
    "tol_relaxed": {"tol_freq": 0.02, "tol_damping": 0.10, "tol_mac": 0.90},
    "tol_strict": {"tol_freq": 0.005, "tol_damping": 0.03, "tol_mac": 0.98},
    "dist0.03": {"max_distance": 0.03},
    "dist0.08": {"max_distance": 0.08},
    "minord15": {"min_orders_fraction": 15 / 80},
    "minord30": {"min_orders_fraction": 30 / 80},
}

# ==================================================================


def choose_subset():
    """Recordings from step 5, PER_BIN at random from each wind speed bin. Returns {file name: data set}."""
    recordings = {}
    with open(MODES_VS_WIND, newline="") as f:
        for row in csv.DictReader(f):
            recordings[row["file_name"]] = (row["data_set"], float(row["mean_wind_speed_ms"]))
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = {}
    for low, high in WIND_BINS:
        names = sorted(n for n, (d, U) in recordings.items() if low <= U < high)
        for n in rng.choice(names, size=min(PER_BIN, len(names)), replace=False):
            chosen[n] = recordings[n][0]
    return chosen


def pick_modes(poles_all, settings, channels):
    """Stability check + clustering of the poles with one set of settings. Returns a dict to save."""
    # Copies, because check_stability adds the stability flags to the dictionaries
    poles = [dict(p) for p in poles_all if p["order"] <= settings["max_order"]]
    poles = check_stability(poles, settings["tol_freq"], settings["tol_damping"], settings["tol_mac"])
    min_orders = int(round(settings["min_orders_fraction"] * len(poles)))
    modes, _ = cluster_poles(poles, settings["max_distance"], min_orders)
    types = []
    for m in modes:
        m["shape"] = normalize_mode_shape(m["shape"])
        x, lateral, vertical, torsion = section_mode_shape(m["shape"], channels)
        types.append(classify_mode(lateral, vertical, torsion))
    return {
        "freq": np.array([m["freq"] for m in modes]),
        "damping": np.array([m["damping"] for m in modes]),
        "n_orders": np.array([m["n_orders"] for m in modes]),
        "mode_type": np.array(types),
        "shapes": np.array([m["shape"] for m in modes]),
        "channel_names": np.array([c["sensor"] + " " + c["component"] for c in channels]),
    }


# --------------------------- Main program ---------------------------

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
for name in VARIANTS:
    (OUTPUT_FOLDER / name).mkdir(exist_ok=True)

subset = choose_subset()
mat_files = {p.stem: p for p in DATA_FOLDER.rglob("*.mat") if "results" not in p.parts}
print(f"{len(subset)} recordings, {len(VARIANTS)} variants, SSI with block rows {BLOCK_ROWS_RUNS}")

conditions_file = OUTPUT_FOLDER / "recording_conditions.csv"
conditions = {}
if conditions_file.exists():
    with open(conditions_file, newline="") as f:
        conditions = {r["file_name"]: r for r in csv.DictReader(f)}

for k, (file_name, data_set) in enumerate(sorted(subset.items())):
    if all((OUTPUT_FOLDER / v / f"{file_name}_modes.npz").exists() for v in VARIANTS):
        print(f"[{k + 1}/{len(subset)}] {file_name}: already done")
        continue
    t0 = time.time()
    try:
        rec = load_recording(mat_files[file_name])
        y, channels = girder_channels(rec)
        U, direction, _ = mean_wind(rec)
        conditions[file_name] = {"file_name": file_name, "data_set": data_set, "day": file_name[7:17],
                                 "mean_wind_speed_ms": f"{U:.3f}", "mean_wind_direction_deg": f"{direction:.1f}",
                                 "mean_temperature_degC": f"{mean_temperature(rec):.2f}"}
        poles_by_rows = {s: ssi_cov(y, rec["fs"], s, ORDERS, MAX_DAMPING)[0] for s in BLOCK_ROWS_RUNS}
        for name, change in VARIANTS.items():
            settings = {**BASELINE, **change}
            result = pick_modes(poles_by_rows[settings["block_rows"]], settings, channels)
            np.savez(OUTPUT_FOLDER / name / f"{file_name}_modes.npz", **result)
    except Exception as error:
        print(f"[{k + 1}/{len(subset)}] {file_name}: FAILED ({error})")
        continue
    with open(conditions_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(conditions.values())).keys()))
        writer.writeheader()
        writer.writerows(conditions.values())
    print(f"[{k + 1}/{len(subset)}] {file_name}: U = {U:.1f} m/s  ({time.time() - t0:.0f} s)")

print(f"\nResults saved in {OUTPUT_FOLDER}")

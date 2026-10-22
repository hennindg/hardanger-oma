"""
Shared code of the robustness steps (7b and 7e): read the modes of the setting variants from
step 7a (results/robustness/<variant>/), link them to the step 5 mode tracks, and recompute the
main conclusions (damping slope with wind, temperature coefficient, C_heave) for a set of recordings.

This file is imported by the step scripts; it does nothing when run on its own.
"""

import csv

import numpy as np

from oma.mode_tracking import assign_to_reference
from hardanger.regression import mode_coefficients, scaling_fit


def load_variant(folder, variant, file_name):
    """Modes of one recording for one variant, as a dictionary (or None if missing)."""
    path = folder / variant / f"{file_name}_modes.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {key: data[key] for key in data.files}


def step5_tracks(modes_vs_wind):
    """Step 5 track frequencies per recording: {file name: {track: frequency}} and track info."""
    per_file, info = {}, {}
    with open(modes_vs_wind, newline="") as f:
        for row in csv.DictReader(f):
            t = int(row["track"])
            per_file.setdefault(row["file_name"], {})[t] = float(row["freq_Hz"])
            info[t] = (float(row["track_median_freq_Hz"]), row["type"])
    return per_file, info


def match_variants(folder, files, variants, track_freqs, track_ids, match_distance):
    """
    1. Link the baseline modes of every recording to the step 5 tracks (identical frequencies,
       because the baseline uses the step 4 settings).
    2. Match the modes of every variant to these baseline modes, per recording
       (distance |df|/f + (1 - MAC) <= match_distance).
    Returns values[variant][track] = (frequency [n_files], damping in % [n_files]) with NaN where
    the mode was not found, and the number of baseline modes linked / expected.
    """
    baseline_index = {}          # (file, track) -> index of the baseline mode
    n_expected, n_matched = 0, 0
    for f in files:
        base = load_variant(folder, "baseline", f)
        for t, freq in track_freqs.get(f, {}).items():
            n_expected += 1
            i = int(np.argmin(np.abs(base["freq"] - freq)))
            # The step 5 table stores frequencies with 5 decimals, so "identical" means within 1e-5 Hz
            if abs(base["freq"][i] - freq) <= 1.01e-5:
                baseline_index[(f, t)] = i
                n_matched += 1

    values = {v: {t: (np.full(len(files), np.nan), np.full(len(files), np.nan)) for t in track_ids} for v in variants}
    for r, f in enumerate(files):
        base = load_variant(folder, "baseline", f)
        references = []
        ref_track = []
        for t in track_ids:
            if (f, t) in baseline_index:
                i = baseline_index[(f, t)]
                references.append({"freq": base["freq"][i], "type": str(base["mode_type"][i]),
                                   "shape": base["shapes"][i], "channel_names": base["channel_names"]})
                ref_track.append(t)
        for v in variants:
            rec = load_variant(folder, v, f)
            if len(rec["freq"]) == 0 or not references:
                continue
            matched = assign_to_reference([rec], references, match_distance)
            for t, m in zip(ref_track, matched):
                values[v][t][0][r] = m["freq"][0]
                values[v][t][1][r] = 100 * m["damping"][0]
    return values, n_matched, n_expected


def build_table(files, conditions, tracks_info, values):
    """Table in the form of hardanger.regression.load_table, from values[track] = (freq array, damping array)."""
    table = {"file_name": np.array(files), "day": np.array([conditions[f]["day"] for f in files]),
             "U": np.array([float(conditions[f]["mean_wind_speed_ms"]) for f in files]),
             "T": np.array([float(conditions[f]["mean_temperature_degC"]) for f in files]),
             "modes": []}
    for t in sorted(values):
        f0, kind = tracks_info[t]
        table["modes"].append({"track": t, "f0": f0, "type": kind, "freq": values[t][0], "damping": values[t][1]})
    return table


def conclusions(table, rows, exclude_from_scaling):
    """[b_z for every mode] + [c_f for every mode] + [C_heave] from the recordings in 'rows'."""
    b_z, c_f, scaling = [], [], ([], [], [])
    for mode in table["modes"]:
        try:
            v, e = mode_coefficients(table, rows, mode)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            v, e = np.full(3, np.nan), np.full(3, np.nan)     # too few recordings with this mode
        if np.isnan(v).any():
            c_f.append(np.nan)
            b_z.append(np.nan)
            continue
        c_f.append(v[1])
        b_z.append(v[2])
        if mode["type"] == "vertical" and all(abs(mode["f0"] - x) > 0.003 for x in exclude_from_scaling):
            scaling[0].append(mode["f0"])
            scaling[1].append(v[2])
            scaling[2].append(e[2])
    # C_heave needs damping slopes of at least 3 vertical modes (fewer in very small bootstrap samples)
    C_heave = np.nan
    if len(scaling[0]) >= 3:
        try:
            C_heave = scaling_fit(*scaling)[2]
        except np.linalg.LinAlgError:
            pass                     # e.g. negative slopes in a tiny sample: no estimate
    return np.array(b_z + c_f + [C_heave])

"""
Functions for reading Hardanger Bridge recordings (LowFreq .mat files).

Everything that depends on the Hardanger file format and sensor layout lives here,
so that the SSI functions in oma/ssi_cov.py stay general.

This file is imported by the step scripts; it does nothing when run on its own.
"""

import numpy as np
from scipy.io import loadmat


# Distance between the east and west accelerometer in a girder pair (m).
# From Table 1 in the data paper: y = +6.33 m and y = -6.64 m.
PAIR_DISTANCE = 6.33 + 6.64

# Acceleration of gravity, to convert the data from g to m/s^2
GRAVITY = 9.81


def load_recording(path):
    """
    Read one recording file. Returns a dictionary with:
      "fs"         sample rate (Hz)
      "start_time" start time as text, e.g. "26/09/2014 18:20:01.440" (UTC)
      "sensors"    {sensor name: data}
      "positions"  {sensor name: [x, y, z] in m}
    Accelerometers: data has one row per time step and 3 columns x, y, z (global axes, unit g).
    Anemometers:    data has 4 columns: angle [deg], U [m/s], w [m/s], T [degC].
    """
    # The files are old-style MATLAB (v5) files, which scipy can read directly.
    # squeeze_me removes MATLAB's extra 1x1 dimensions, and struct_as_record=False
    # lets us write rec.sensor instead of rec["sensor"][0][0].
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    rec = mat["recording"]

    sensors = {}
    positions = {}
    for sensor in rec.sensor:
        sensors[sensor.sensor_name] = np.asarray(sensor.data, dtype=float)
        positions[sensor.sensor_name] = np.asarray(sensor.position, dtype=float)

    return {
        "fs": float(rec.samplerate),
        "start_time": str(rec.starttime),
        "sensors": sensors,
        "positions": positions,
    }


def is_girder_accelerometer(name):
    """True for girder accelerometers H1 ... H9 (not the tower tops H10, H11, not anemometers)."""
    label = name.split()[0]          # e.g. "H1 East" -> "H1"
    if not label.startswith("H"):
        return False
    return int(label[1:]) <= 9


def girder_channels(recording):
    """
    Collect the lateral (y) and vertical (z) accelerations of all girder sensors
    in one matrix, ready for SSI. Sensors with NaN (poor quality) are skipped.

    Returns
      y        : accelerations in m/s^2, [n_samples, n_channels]
      channels : list with one dictionary per column of y, with keys
                 "sensor" (e.g. "H1 East"), "section" (e.g. "H1"), "side" ("East"/"Vest"),
                 "component" ("y" or "z"), "x" (position along the bridge, m)
    """
    columns = []
    channels = []
    for name, data in recording["sensors"].items():
        if not is_girder_accelerometer(name):
            continue
        if np.isnan(data).any():
            print(f"   skipping {name}: contains NaN (poor data quality)")
            continue

        section, side = name.split()
        x = recording["positions"][name][0]

        # Column 1 = y = lateral, column 2 = z = vertical. The x-component (along the
        # bridge) is left out: the girder hardly moves in that direction.
        for component, col in [("y", 1), ("z", 2)]:
            columns.append(data[:, col] * GRAVITY)
            channels.append({"sensor": name, "section": section, "side": side,
                             "component": component, "x": x})

    y = np.column_stack(columns)
    return y, channels


def section_mode_shape(shape, channels):
    """
    Turn a mode shape at the sensor channels into lateral, vertical and torsional
    motion of each girder cross-section (H1 ... H9).

    For a section with both an east and a west sensor:
      lateral  = average of the two y values
      vertical = average of the two z values
      torsion  = (z_east - z_west) / PAIR_DISTANCE    (rotation, rad per unit)
    For a section with only one sensor (H2 west, H8 east), vertical motion and torsion
    cannot be separated, so lateral = y, vertical = z and torsion = NaN.

    Returns arrays sorted by x: x, lateral, vertical, torsion (real parts).
    """
    shape = np.real(shape)

    # Collect the values per section: values[section][side + component] = number
    values = {}
    x_of_section = {}
    for value, ch in zip(shape, channels):
        values.setdefault(ch["section"], {})[ch["side"] + ch["component"]] = value
        x_of_section[ch["section"]] = ch["x"]

    rows = []
    for section, v in values.items():
        if "Easty" in v and "Vesty" in v:
            lateral = (v["Easty"] + v["Vesty"]) / 2
            vertical = (v["Eastz"] + v["Vestz"]) / 2
            torsion = (v["Eastz"] - v["Vestz"]) / PAIR_DISTANCE
        elif "Easty" in v:
            lateral, vertical, torsion = v["Easty"], v["Eastz"], np.nan
        else:
            lateral, vertical, torsion = v["Vesty"], v["Vestz"], np.nan
        rows.append((x_of_section[section], lateral, vertical, torsion))

    rows.sort()      # sort by x position
    rows = np.array(rows)
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]


def classify_mode(lateral, vertical, torsion):
    """
    Call a mode "lateral", "vertical" or "torsion", depending on which motion dominates.
    Torsion is multiplied by half the pair distance, so it becomes the vertical
    displacement of the girder edge and can be compared with the translations.
    """
    half_width = PAIR_DISTANCE / 2
    energy_lateral = np.nansum(lateral ** 2)
    energy_vertical = np.nansum(vertical ** 2)
    energy_torsion = np.nansum((torsion * half_width) ** 2)
    energies = [energy_lateral, energy_vertical, energy_torsion]
    return ["lateral", "vertical", "torsion"][int(np.argmax(energies))]


def mean_temperature(recording):
    """
    Mean temperature [degC] over the span anemometers A1 ... A8 for the whole recording
    (4th anemometer column). This is the "sonic temperature" from the speed of sound,
    which is close to the air temperature. Anemometers with NaN are skipped.
    """
    values = []
    for name, data in recording["sensors"].items():
        if not name.startswith("A") or name == "A9":
            continue
        if np.isnan(data).any():
            continue
        values.append(np.mean(data[:, 3]))
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def mean_wind(recording):
    """
    Mean wind speed and direction over the span anemometers A1 ... A8 for the whole recording.
    A9 (tower top) is left out, and anemometers with NaN (poor data quality) are skipped.

    Wind direction is averaged as a vector (so 350 deg and 10 deg give 0 deg, not 180 deg).
    The angle is measured from the global x-axis (data paper, section on coordinate systems).

    Returns mean speed [m/s], mean direction [deg], list of anemometers used.
    """
    speeds = []
    sin_sum = 0.0
    cos_sum = 0.0
    used = []
    for name, data in recording["sensors"].items():
        if not name.startswith("A") or name == "A9":
            continue
        if np.isnan(data).any():
            continue
        angle = np.radians(data[:, 0])
        speeds.append(np.mean(data[:, 1]))
        sin_sum += np.mean(np.sin(angle))
        cos_sum += np.mean(np.cos(angle))
        used.append(name)

    if len(used) == 0:
        return np.nan, np.nan, used
    direction = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    return float(np.mean(speeds)), float(direction), used

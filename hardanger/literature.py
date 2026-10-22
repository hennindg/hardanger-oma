"""
Reference modes of the Hardanger Bridge from the literature, and matching of our tracked
modes to them.

The reference table is stored as data in literature/petersen_oiseth_2017_modes.csv:
Petersen, O.W. & Oiseth, O. (2017). Finite element model updating of a long span suspension
bridge. Int. Conf. on Earthquake Engineering and Structural Dynamics (ICESD), Reykjavik.
Table 1: 30 modes identified with Cov-SSI from a 120 min recording on 13 Jan 2016 at a mean
wind of 3-6 m/s (close to still-air conditions).

This file is imported by the step scripts; it does nothing when run on its own.
"""

import csv
from hardanger.paths import LITERATURE_FOLDER


REFERENCE_FILE = LITERATURE_FOLDER / "petersen_oiseth_2017_modes.csv"

# Which literature mode types can correspond to our types. Our classification only sees the
# girder, so a cable or pylon mode shows up as whatever girder motion it causes.
COMPATIBLE_TYPES = {
    "lateral": ["Horizontal", "Cable/horizontal", "Pylon"],
    "vertical": ["Vertical"],
    "torsion": ["Torsion", "Cable/horizontal"],
}

# Largest relative frequency difference for a match. 4 % because wind changes the
# frequencies (the torsion mode at 0.560 Hz is ~0.54 Hz in strong wind, -3 %).
MAX_RELATIVE_DIFFERENCE = 0.04


def load_reference_modes(path=REFERENCE_FILE):
    """Read the literature table. Returns a list of dictionaries with "mode", "type", "freq"."""
    modes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            modes.append({"mode": int(row["mode"]), "type": row["type"], "freq": float(row["freq_Hz"])})
    return modes


def match_reference(freq, mode_type, reference):
    """
    Find the literature mode for one of our modes: the closest frequency among the
    compatible types, if it is within MAX_RELATIVE_DIFFERENCE. Returns the literature
    mode (dictionary) or None.
    """
    best = None
    for ref in reference:
        if ref["type"] not in COMPATIBLE_TYPES.get(mode_type, []):
            continue
        difference = abs(freq - ref["freq"]) / ref["freq"]
        if difference > MAX_RELATIVE_DIFFERENCE:
            continue
        if best is None or difference < abs(freq - best["freq"]) / best["freq"]:
            best = ref
    return best


def reference_label(ref):
    """Short text for tables and plot titles, e.g. "P&O 2017 #14: 0.418 Hz cable/horizontal"."""
    if ref is None:
        return "no literature match"
    return f"P&O 2017 #{ref['mode']}: {ref['freq']:.3f} Hz {ref['type'].lower()}"

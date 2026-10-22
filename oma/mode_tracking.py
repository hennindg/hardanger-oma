"""
Mode tracking: link the modes identified in many recordings into "tracks"
(one track = one physical mode followed over time).

Like oma/ssi_cov.py and oma/mode_clustering.py, these functions know nothing about the Hardanger
file format. Each recording is a dictionary with "freq", "damping", "shapes"
[n_modes, n_channels] and "channel_names" (so recordings with different sensors can be compared).

Method (same idea as the pole clustering in oma/mode_clustering.py, one level up):
  1. Put the modes from all recordings together.
  2. Distance between two modes: d = |f_i - f_j| / max(f_i, f_j) + (1 - MAC_ij),
     with the MAC computed only on the channels that both recordings have.
  3. Hierarchical clustering with average linkage. Average (not single) linkage is used
     here because the modes of one physical mode scatter AROUND a mean value from
     recording to recording; single linkage could chain two neighbouring modes together
     through a few in-between estimates.
  4. A cluster is a track if it contains modes from enough recordings.
     If a recording has two modes in the same track, the one closest to the track's
     median frequency is kept.

The pairwise clustering needs a distance matrix of all modes against all modes, which is
only possible for a few hundred recordings. For long-term data (thousands of recordings) the
second method is used: REFERENCE MODES.
  1. Build reference modes from tracks found with track_modes() on a smaller set of recordings
     (build_reference_modes): median frequency and a representative mode shape per track.
  2. Assign every mode of every recording to the closest reference mode, with the same distance
     |f_i - f_ref| / max(f_i, f_ref) + (1 - MAC), if it is below a limit (assign_to_reference).
     Each reference gets at most one mode per recording, and each mode at most one reference.

This file is imported by the step scripts; it does nothing when run on its own.
"""

from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def mac_common_channels(shape_a, names_a, shape_b, names_b):
    """
    MAC between two mode shapes that may come from recordings with different sensors.
    Only the channels present in both recordings (matched by name) are used.
    """
    position_b = {name: k for k, name in enumerate(names_b)}
    a = []
    b = []
    for k, name in enumerate(names_a):
        if name in position_b:
            a.append(shape_a[k])
            b.append(shape_b[position_b[name]])
    a = np.array(a)
    b = np.array(b)
    top = np.abs(np.vdot(a, b)) ** 2
    bottom = np.real(np.vdot(a, a)) * np.real(np.vdot(b, b))
    return top / bottom


def pool_modes(recordings):
    """
    Put the modes of all recordings in one flat list.
    Returns a list of (recording number, mode number) pairs and an array of frequencies.
    """
    index = []
    freq = []
    for r, rec in enumerate(recordings):
        for m in range(len(rec["freq"])):
            index.append((r, m))
            freq.append(rec["freq"][m])
    return index, np.array(freq)


def mode_distances(recordings, index, freq):
    """
    Distance between every pair of modes: relative frequency difference + (1 - MAC).
    Returns a square matrix [n_modes_total, n_modes_total].
    """
    n = len(index)
    distances = np.zeros((n, n))
    for i in range(n):
        r_i, m_i = index[i]
        for j in range(i + 1, n):
            r_j, m_j = index[j]
            freq_difference = abs(freq[i] - freq[j]) / max(freq[i], freq[j])
            if freq_difference > 0.2:
                # Clearly different modes: skip the (slow) MAC, the distance is large anyway
                distances[i, j] = distances[j, i] = 1 + freq_difference
                continue
            mac = mac_common_channels(recordings[r_i]["shapes"][m_i], recordings[r_i]["channel_names"],
                                      recordings[r_j]["shapes"][m_j], recordings[r_j]["channel_names"])
            distances[i, j] = distances[j, i] = freq_difference + (1 - mac)
    return distances


def track_modes(recordings, max_distance, min_recordings):
    """
    Link the modes of all recordings into tracks.

    recordings     : list of dictionaries (one per recording, sorted by time), each with
                     "freq", "damping", "shapes", "channel_names", and optionally
                     "freq_std", "damping_std", "mode_type"
    max_distance   : clusters are cut at this (average) distance
    min_recordings : a cluster is a track only if it has modes from at least this many recordings

    Returns a list of tracks sorted by frequency. Each track is a dictionary with
      "freq", "damping", "freq_std", "damping_std" : arrays [n_recordings], NaN where the
                                                     mode was not found in that recording
      "mode_number" : which mode of each recording belongs to the track (-1 = not found)
      "n_found", "median_freq", "type" (most common lateral/vertical/torsion label)
    and an array telling for each pooled mode which track it ended in (-1 = no track).
    """
    index, freq = pool_modes(recordings)
    distances = mode_distances(recordings, index, freq)
    tree = linkage(squareform(distances, checks=False), method="average")
    cluster = fcluster(tree, t=max_distance, criterion="distance")

    n_rec = len(recordings)
    tracks = []
    for c in np.unique(cluster):
        members = np.where(cluster == c)[0]
        rec_of_member = np.array([index[k][0] for k in members])
        if len(np.unique(rec_of_member)) < min_recordings:
            continue

        # One mode per recording: the one closest to the median frequency of the cluster
        median_freq = np.median(freq[members])
        track = {
            "freq": np.full(n_rec, np.nan), "damping": np.full(n_rec, np.nan),
            "freq_std": np.full(n_rec, np.nan), "damping_std": np.full(n_rec, np.nan),
            "mode_number": np.full(n_rec, -1),
        }
        types = []
        for r in np.unique(rec_of_member):
            candidates = members[rec_of_member == r]
            best = candidates[np.argmin(np.abs(freq[candidates] - median_freq))]
            m = index[best][1]
            rec = recordings[r]
            track["mode_number"][r] = m
            track["freq"][r] = rec["freq"][m]
            track["damping"][r] = rec["damping"][m]
            if "freq_std" in rec:
                track["freq_std"][r] = rec["freq_std"][m]
                track["damping_std"][r] = rec["damping_std"][m]
            if "mode_type" in rec:
                types.append(str(rec["mode_type"][m]))

        track["n_found"] = int(np.sum(track["mode_number"] >= 0))
        track["median_freq"] = float(np.nanmedian(track["freq"]))
        track["type"] = Counter(types).most_common(1)[0][0] if types else ""
        tracks.append(track)

    tracks.sort(key=lambda t: t["median_freq"])

    # For plotting: which track each pooled mode belongs to (-1 = not in any track)
    in_track = np.full(len(index), -1)
    for t, track in enumerate(tracks):
        for r in range(n_rec):
            m = track["mode_number"][r]
            if m >= 0:
                in_track[index.index((r, m))] = t
    return tracks, index, freq, in_track


def build_reference_modes(recordings, tracks):
    """
    Reference modes from tracks made by track_modes(): for each track the median frequency,
    the most common type, and the mode shape of the recording whose frequency is closest to
    the median (with that recording's channel names, so the MAC can use common channels).
    Returns a list of dictionaries with "freq", "type", "shape", "channel_names".
    """
    references = []
    for track in tracks:
        found = np.where(track["mode_number"] >= 0)[0]
        best = found[np.argmin(np.abs(track["freq"][found] - track["median_freq"]))]
        rec = recordings[best]
        references.append({
            "freq": track["median_freq"],
            "type": track["type"],
            "shape": rec["shapes"][track["mode_number"][best]],
            "channel_names": rec["channel_names"],
        })
    return references


def assign_to_reference(recordings, references, max_distance):
    """
    Link the modes of all recordings to reference modes (see build_reference_modes).

    For each recording, the distance d = |f - f_ref| / max(f, f_ref) + (1 - MAC) is computed
    between every mode and every reference (MAC on the common channels). Pairs with
    d <= max_distance are accepted, smallest distance first, so each reference gets at most
    one mode and each mode at most one reference.

    Returns tracks in the same form as track_modes(): a list (one per reference) of dictionaries
    with "freq", "damping", "freq_std", "damping_std" ([n_recordings], NaN if not found),
    "mode_number", "n_found", "median_freq" (median of the assigned frequencies) and "type".
    """
    n_rec = len(recordings)
    tracks = []
    for ref in references:
        tracks.append({"freq": np.full(n_rec, np.nan), "damping": np.full(n_rec, np.nan),
                       "freq_std": np.full(n_rec, np.nan), "damping_std": np.full(n_rec, np.nan),
                       "mode_number": np.full(n_rec, -1), "type": ref["type"],
                       "reference_freq": ref["freq"]})

    for r, rec in enumerate(recordings):
        pairs = []          # (distance, reference number, mode number)
        for t, ref in enumerate(references):
            for m in range(len(rec["freq"])):
                freq_difference = abs(rec["freq"][m] - ref["freq"]) / max(rec["freq"][m], ref["freq"])
                if freq_difference > max_distance:
                    continue            # cannot be below the limit anyway
                mac = mac_common_channels(rec["shapes"][m], rec["channel_names"],
                                          ref["shape"], ref["channel_names"])
                distance = freq_difference + (1 - mac)
                if distance <= max_distance:
                    pairs.append((distance, t, m))

        used_refs, used_modes = set(), set()
        for distance, t, m in sorted(pairs):
            if t in used_refs or m in used_modes:
                continue
            used_refs.add(t)
            used_modes.add(m)
            track = tracks[t]
            track["mode_number"][r] = m
            track["freq"][r] = rec["freq"][m]
            track["damping"][r] = rec["damping"][m]
            if "freq_std" in rec:
                track["freq_std"][r] = rec["freq_std"][m]
                track["damping_std"][r] = rec["damping_std"][m]

    for track in tracks:
        track["n_found"] = int(np.sum(track["mode_number"] >= 0))
        track["median_freq"] = float(np.nanmedian(track["freq"])) if track["n_found"] else np.nan
    return tracks

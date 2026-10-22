"""
Automatic mode picking: cluster the stable poles from SSI over all model orders.

Like oma/ssi_cov.py, these functions know nothing about the Hardanger file format. They take the
"poles" list made by ssi_cov() and check_stability() in oma/ssi_cov.py.

Method (the distance measure follows Magalhaes, Cunha & Caetano 2009, "Online automatic
identification of the modal parameters of a long span arch bridge", Mech. Syst. Signal
Process. 23(2)):
  1. Collect all poles that are stable (frequency, damping and MAC) at any model order.
  2. Distance between two poles:  d = |f_i - f_j| / max(f_i, f_j) + (1 - MAC_ij)
     Poles of the same physical mode have almost the same frequency and MAC close to 1,
     so their distance is small.
  3. Hierarchical clustering with single linkage: a pole joins a cluster if it is closer
     than the distance limit to ANY pole already in the cluster. This follows a column in
     the stabilization diagram from order to order, even if the mode shape estimate slowly
     changes with model order (average linkage split such columns into small pieces).
  4. A physical mode shows up at many model orders, a spurious (noise) pole only at a few.
     So only clusters that contain poles from many different model orders are kept.
  5. Inside each cluster, poles with outlying damping are removed (box plot rule), and
     only one pole per model order is kept (the one closest to the median frequency).
  6. The mode = mean frequency and damping of the cluster, with their standard deviation.

This file is imported by the step scripts; it does nothing when run on its own.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def collect_stable_poles(poles):
    """
    Put all fully stable poles from all model orders in flat arrays.
    Returns freq [n], damping [n], order [n], shapes [n_channels, n].
    """
    freq, damping, order, shapes = [], [], [], []
    for result in poles:
        stable = result["stable_all"]
        freq.extend(result["freq"][stable])
        damping.extend(result["damping"][stable])
        order.extend([result["order"]] * int(stable.sum()))
        shapes.append(result["shapes"][:, stable])
    return np.array(freq), np.array(damping), np.array(order), np.hstack(shapes)


def mac_matrix(shapes):
    """
    MAC between every pair of mode shapes (columns of shapes). Returns [n, n].
    Same formula as mac() in oma/ssi_cov.py, but for all pairs at once with matrix products.
    """
    products = shapes.conj().T @ shapes                 # phi_i^H phi_j for all i, j
    norms = np.real(np.sum(shapes.conj() * shapes, axis=0))   # phi_i^H phi_i
    return np.abs(products) ** 2 / np.outer(norms, norms)


def pole_distances(freq, shapes):
    """
    Distance between every pair of poles: relative frequency difference + (1 - MAC).
    Returns a square matrix [n, n] with zeros on the diagonal.
    """
    f_i = freq[:, None]       # as a column
    f_j = freq[None, :]       # as a row
    freq_difference = np.abs(f_i - f_j) / np.maximum(f_i, f_j)
    distances = freq_difference + (1 - mac_matrix(shapes))
    np.fill_diagonal(distances, 0.0)
    distances = (distances + distances.T) / 2          # remove tiny round-off asymmetry
    return distances


def remove_damping_outliers(damping):
    """
    Box plot rule: a damping value is an outlier if it is more than 1.5 times the
    interquartile range (IQR) below the first quartile or above the third quartile.
    Returns a boolean array: True = keep.
    """
    q1, q3 = np.percentile(damping, [25, 75])
    iqr = q3 - q1
    return (damping >= q1 - 1.5 * iqr) & (damping <= q3 + 1.5 * iqr)


def one_pole_per_order(members, freq, order):
    """
    If a cluster has two or more poles from the same model order, keep only the one
    closest to the median frequency of the cluster. Each model order then counts once
    in the mean and standard deviation.
    Returns the reduced list of pole indices.
    """
    median_freq = np.median(freq[members])
    kept = []
    for n in np.unique(order[members]):
        same_order = members[order[members] == n]
        kept.append(same_order[np.argmin(np.abs(freq[same_order] - median_freq))])
    return np.array(kept)


def cluster_poles(poles, max_distance, min_orders):
    """
    Group the stable poles into physical modes.

    poles        : list from ssi_cov() + check_stability()
    max_distance : clusters are cut at this distance (e.g. 0.05 = 5 % frequency
                   difference with MAC 1, or 2 % frequency difference with MAC 0.97)
    min_orders   : a cluster is accepted as a mode only if it has poles from at least
                   this many different model orders

    Returns
      modes       : list of dictionaries, one per mode, sorted by frequency, with keys
                    "freq", "freq_std", "damping", "damping_std", "n_poles", "n_orders",
                    "shape" (complex, at the sensor channels), "pole_freq", "pole_damping",
                    "pole_order" (the poles in the cluster, after outlier removal)
      all_stable  : dictionary with all stable poles and their cluster number
                    (keys "freq", "damping", "order", "cluster", "accepted")
    """
    freq, damping, order, shapes = collect_stable_poles(poles)

    # Hierarchical clustering. linkage() wants the distances in "condensed" form
    # (only the upper triangle as one long vector), which squareform() makes.
    distances = pole_distances(freq, shapes)
    tree = linkage(squareform(distances, checks=False), method="single")
    cluster = fcluster(tree, t=max_distance, criterion="distance")

    modes = []
    accepted = np.zeros(len(freq), dtype=bool)
    for c in np.unique(cluster):
        members = np.where(cluster == c)[0]

        # A physical mode should appear at many model orders
        if len(np.unique(order[members])) < min_orders:
            continue

        # Remove poles with outlying damping, keep one pole per order, then check the size again
        members = members[remove_damping_outliers(damping[members])]
        members = one_pole_per_order(members, freq, order)
        n_orders = len(np.unique(order[members]))
        if n_orders < min_orders:
            continue
        accepted[members] = True

        # Representative mode shape: the pole closest to the median frequency of the cluster
        median_freq = np.median(freq[members])
        representative = members[np.argmin(np.abs(freq[members] - median_freq))]

        modes.append({
            "freq": np.mean(freq[members]),
            "freq_std": np.std(freq[members]),
            "damping": np.mean(damping[members]),
            "damping_std": np.std(damping[members]),
            "n_poles": len(members),
            "n_orders": n_orders,
            "shape": shapes[:, representative],
            "pole_freq": freq[members],
            "pole_damping": damping[members],
            "pole_order": order[members],
        })

    modes.sort(key=lambda m: m["freq"])
    all_stable = {"freq": freq, "damping": damping, "order": order,
                  "cluster": cluster, "accepted": accepted}
    return modes, all_stable


def identify_modes(y, fs, block_rows, orders, max_damping, tol_freq, tol_damping, tol_mac,
                   max_distance, min_orders):
    """
    The complete automatic identification of one record: SSI-cov at all model orders, stability
    check, clustering. Returns the frequencies, damping ratios and (complex) mode shapes of the modes.
    """
    from oma.ssi_cov import ssi_cov, check_stability
    poles, _ = ssi_cov(y, fs, block_rows, orders, max_damping)
    poles = check_stability(poles, tol_freq, tol_damping, tol_mac)
    modes, _ = cluster_poles(poles, max_distance, min_orders)
    freq = np.array([m["freq"] for m in modes])
    damping = np.array([m["damping"] for m in modes])
    shapes = np.array([m["shape"] for m in modes])
    return freq, damping, shapes

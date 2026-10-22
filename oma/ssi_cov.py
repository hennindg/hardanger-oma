"""
Covariance-driven stochastic subspace identification (SSI-cov).

These functions only need a plain numpy array of measurements [n_samples, n_channels]
and the sample rate. They know nothing about the Hardanger file format, so they can be
reused on other bridges (e.g. the Z24 benchmark).

Theory in short (Peeters & De Roeck 1999, "Reference-based stochastic subspace
identification for output-only modal analysis", Mech. Syst. Signal Process. 13(6)):
  - For a linear system driven by white noise, the output covariances
    R_i = E[ y(k+i) y(k)^T ] can be written as R_i = C A^(i-1) G.
  - Stacking the covariances in a block Toeplitz matrix gives T = O * Gamma, where
    O = [C; CA; CA^2; ...] is the observability matrix.
  - An SVD of T gives O (up to a change of basis). The system order n is the number
    of singular values we keep.
  - A follows from the shift structure of O, and C is the first block row of O.
  - Eigenvalues of A give natural frequencies and damping ratios.
    C times the eigenvectors of A gives the mode shapes at the sensors.

This file is imported by the step scripts; it does nothing when run on its own.
"""

import numpy as np
from scipy import signal
from scipy.sparse.linalg import svds


def bandpass_filter(y, fs, low, high, filter_order=4):
    """
    Zero-phase Butterworth bandpass filter of every channel (column) of y.
    Used to look at a few modes in a narrow band with a low model order.
    Zero-phase (filtering forwards and backwards) means the filter does not shift the
    signals in time, so the phase between channels (the mode shape) is not changed.
    """
    sos = signal.butter(filter_order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, y, axis=0)


def output_covariances(y, max_lag):
    """
    Estimate the output covariance matrices R_i = E[ y(k+i) y(k)^T ] for i = 0 ... max_lag.

    y       : measurements, [n_samples, n_channels]
    max_lag : largest time lag (in samples)
    Returns an array [max_lag + 1, n_channels, n_channels].
    """
    y = y - np.mean(y, axis=0)   # remove the mean: we only want the dynamic response
    n_samples, n_channels = y.shape

    R = np.zeros((max_lag + 1, n_channels, n_channels))
    for i in range(max_lag + 1):
        later = y[i:, :]                    # y(k+i) for all k
        earlier = y[:n_samples - i, :]      # y(k)   for all k
        R[i] = later.T @ earlier / (n_samples - i)
    return R


def block_toeplitz(R, block_rows):
    """
    Build the block Toeplitz matrix of output covariances, with s = block_rows:

        [ R_s       R_(s-1)  ...  R_1 ]
        [ R_(s+1)   R_s      ...  R_2 ]
        [ ...                         ]
        [ R_(2s-1)  ...           R_s ]

    Size: (s * n_channels) x (s * n_channels). Needs R up to lag 2s-1.
    """
    s = block_rows
    n_channels = R.shape[1]
    T = np.zeros((s * n_channels, s * n_channels))
    for row in range(s):
        for col in range(s):
            lag = s + row - col
            T[row * n_channels:(row + 1) * n_channels,
              col * n_channels:(col + 1) * n_channels] = R[lag]
    return T


def poles_for_one_order(U, S, order, n_channels, fs, max_damping):
    """
    Compute the poles (frequency, damping, mode shape) for ONE model order,
    from the SVD of the block Toeplitz matrix (T = U S V^T).

    Only physical poles are returned:
      - one pole of each complex conjugate pair (positive imaginary part),
      - damping ratio between 0 and max_damping.
    Returns freq [Hz], damping ratio [-], mode shapes [n_channels, n_poles] (complex).
    """
    # Observability matrix O = U_n * sqrt(S_n), size (s * n_channels) x order
    O = U[:, :order] * np.sqrt(S[:order])

    # Output matrix C = first block row of O
    C = O[:n_channels, :]

    # System matrix A from the shift structure of O:
    # (O without its last block row) * A = (O without its first block row)
    O_top = O[:-n_channels, :]
    O_bottom = O[n_channels:, :]
    A = np.linalg.pinv(O_top) @ O_bottom

    # Discrete-time eigenvalues mu and eigenvectors psi of A
    mu, psi = np.linalg.eig(A)

    # Keep only eigenvalues with positive imaginary part. This removes one of each
    # complex conjugate pair (they describe the same mode) and the real eigenvalues,
    # which are not vibration modes.
    keep = np.imag(mu) > 0
    mu = mu[keep]
    psi = psi[:, keep]

    # Continuous-time eigenvalues: lambda = ln(mu) * fs  (rad/s)
    lam = np.log(mu) * fs
    freq = np.abs(lam) / (2 * np.pi)
    damping = -np.real(lam) / np.abs(lam)

    # Mode shapes at the sensors
    shapes = C @ psi

    # Throw away non-physical poles: negative damping, or damping above the limit
    keep = (damping > 0) & (damping < max_damping)
    freq = freq[keep]
    damping = damping[keep]
    shapes = shapes[:, keep]

    # Sort by frequency so the output is easier to read
    order_of_freq = np.argsort(freq)
    return freq[order_of_freq], damping[order_of_freq], shapes[:, order_of_freq]


def ssi_cov(y, fs, block_rows, orders, max_damping=0.10):
    """
    Run SSI-cov on the measurements y [n_samples, n_channels] with sample rate fs.

    block_rows  : number of block rows s in the Toeplitz matrix (time lags used: 1 ... 2s-1)
    orders      : list of model orders to compute (each order must be <= s * n_channels)
    max_damping : poles with higher damping ratio are thrown away

    Returns
      poles : a list with one dictionary per model order, with keys
              "order", "freq", "damping", "shapes"
      S     : singular values of the block Toeplitz matrix (useful to judge the model order)
    """
    n_channels = y.shape[1]
    R = output_covariances(y, 2 * block_rows - 1)
    T = block_toeplitz(R, block_rows)

    # One SVD is enough: each model order only uses the first columns of U and S.
    # Only the largest max(orders) singular values are needed, so a truncated SVD (Lanczos method,
    # scipy svds) is used: same result to ~1e-13, but 3-7 times faster for large Toeplitz matrices.
    n_needed = max(orders)
    if n_needed < min(T.shape) - 1:
        U, S, Vt = svds(T, k=n_needed)
        largest_first = np.argsort(S)[::-1]       # svds returns them in increasing order
        U, S = U[:, largest_first], S[largest_first]
    else:
        U, S, Vt = np.linalg.svd(T, full_matrices=False)

    poles = []
    for order in orders:
        freq, damping, shapes = poles_for_one_order(U, S, order, n_channels, fs, max_damping)
        poles.append({"order": order, "freq": freq, "damping": damping, "shapes": shapes})
    return poles, S


def mac(shape_1, shape_2):
    """
    Modal Assurance Criterion between two (complex) mode shape vectors.
    1 = same shape, 0 = completely different (orthogonal) shapes.
    """
    top = np.abs(np.vdot(shape_1, shape_2)) ** 2
    bottom = np.real(np.vdot(shape_1, shape_1)) * np.real(np.vdot(shape_2, shape_2))
    return top / bottom


def check_stability(poles, tol_freq, tol_damping, tol_mac):
    """
    Mark which poles are stable, by comparing each pole with the closest-frequency pole
    at the previous model order.

    A pole is stable in frequency if |f - f_prev| / f_prev < tol_freq,
    in damping if |zeta - zeta_prev| / zeta_prev < tol_damping,
    and in mode shape if MAC(phi, phi_prev) > tol_mac.

    Adds three boolean arrays to each dictionary in poles:
    "stable_freq", "stable_freq_damping", "stable_all" (frequency, damping and MAC).
    """
    for k in range(len(poles)):
        current = poles[k]
        n_poles = len(current["freq"])
        current["stable_freq"] = np.zeros(n_poles, dtype=bool)
        current["stable_freq_damping"] = np.zeros(n_poles, dtype=bool)
        current["stable_all"] = np.zeros(n_poles, dtype=bool)

        if k == 0:
            continue                       # nothing to compare the lowest order with
        previous = poles[k - 1]
        if len(previous["freq"]) == 0:
            continue

        for p in range(n_poles):
            f = current["freq"][p]
            # closest pole in frequency at the previous order
            q = np.argmin(np.abs(previous["freq"] - f))
            f_prev = previous["freq"][q]
            z = current["damping"][p]
            z_prev = previous["damping"][q]

            ok_freq = abs(f - f_prev) / f_prev < tol_freq
            ok_damping = abs(z - z_prev) / z_prev < tol_damping
            ok_mac = mac(current["shapes"][:, p], previous["shapes"][:, q]) > tol_mac

            current["stable_freq"][p] = ok_freq
            current["stable_freq_damping"][p] = ok_freq and ok_damping
            current["stable_all"][p] = ok_freq and ok_damping and ok_mac
    return poles


def normalize_mode_shape(shape):
    """
    Rotate a complex mode shape so its largest component is real and positive,
    and scale it so that component equals 1.
    For a classically damped structure the result is (almost) real.
    """
    biggest = shape[np.argmax(np.abs(shape))]
    return shape / biggest


def first_singular_value_spectrum(y, fs, segment_seconds):
    """
    First singular value of the cross spectral density matrix at each frequency
    (as in Frequency Domain Decomposition). Peaks show where the modes are, using
    all channels at once. Used as background in the stabilization diagram.
    Returns freqs [Hz] and the first singular value at each frequency.
    """
    y = y - np.mean(y, axis=0)
    n_channels = y.shape[1]
    block = min(y.shape[0], int(segment_seconds * fs))

    # Cross spectral density between every pair of channels
    freqs, _ = signal.csd(y[:, 0], y[:, 0], fs=fs, nperseg=block)
    G = np.zeros((len(freqs), n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, G[:, i, j] = signal.csd(y[:, i], y[:, j], fs=fs, nperseg=block)

    first_sv = np.zeros(len(freqs))
    for k in range(len(freqs)):
        first_sv[k] = np.linalg.svd(G[k], compute_uv=False)[0]
    return freqs, first_sv

"""
Plot functions shared by the step scripts (stabilization diagram, mode shapes, clusters).

This file is imported by the step scripts; it does nothing when run on its own.
"""

import numpy as np
import matplotlib.pyplot as plt

from hardanger.data import PAIR_DISTANCE


def plot_stabilization_diagram(poles, freqs_sv, first_sv, picked_freqs, title,
                               freq_max, order_line=None):
    """
    Model order vs. frequency for all poles, with the first singular value spectrum behind it.
    picked_freqs : frequencies of the identified modes (drawn as green dashed lines)
    order_line   : if given, a dotted horizontal line at this model order
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    max_order = poles[-1]["order"]

    for result in poles:
        order = result["order"]
        f = result["freq"]
        not_stable = ~result["stable_freq"]
        only_freq = result["stable_freq"] & ~result["stable_freq_damping"]
        freq_damp = result["stable_freq_damping"] & ~result["stable_all"]
        all_ok = result["stable_all"]
        ax.plot(f[not_stable], np.full(not_stable.sum(), order), ".", color="0.75", markersize=3)
        ax.plot(f[only_freq], np.full(only_freq.sum(), order), "x", color="tab:orange", markersize=4)
        ax.plot(f[freq_damp], np.full(freq_damp.sum(), order), "+", color="tab:blue", markersize=5)
        ax.plot(f[all_ok], np.full(all_ok.sum(), order), "o", color="tab:red",
                markersize=3, markerfacecolor="none")

    for f in picked_freqs:
        ax.axvline(f, color="tab:green", linewidth=0.6, linestyle="--")
    if order_line is not None:
        ax.axhline(order_line, color="k", linewidth=0.6, linestyle=":")

    # Empty plots, only to get the legend entries
    ax.plot([], [], ".", color="0.75", label="new / not stable")
    ax.plot([], [], "x", color="tab:orange", label="stable frequency")
    ax.plot([], [], "+", color="tab:blue", label="stable frequency + damping")
    ax.plot([], [], "o", color="tab:red", markerfacecolor="none", label="stable freq + damping + MAC")
    ax.plot([], [], "--", color="tab:green", label="picked modes")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Model order")
    ax.set_xlim(0, freq_max)
    ax.set_ylim(0, max_order + 4)
    # Legend below the plot, so it does not hide any poles
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8)
    ax.set_title(title)

    # First singular value of the spectral matrix on a second y-axis (log scale)
    ax2 = ax.twinx()
    ax2.semilogy(freqs_sv, first_sv, color="k", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("1st singular value of spectral matrix")
    fig.tight_layout()
    return fig


def plot_mode_shapes(modes, title):
    """
    One small plot per mode: lateral, vertical and torsion along the girder.
    Each mode needs the keys "x", "lateral", "vertical", "torsion", "freq", "damping", "type".
    """
    n_modes = len(modes)
    n_cols = 5
    n_rows = int(np.ceil(n_modes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.6 * n_rows), sharex=True, squeeze=False)
    half_width = PAIR_DISTANCE / 2

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols][k % n_cols]
        if k >= n_modes:
            ax.axis("off")
            continue
        m = modes[k]
        ax.plot(m["x"], m["lateral"], "o-", color="tab:blue", markersize=3, label="lateral (y)")
        ax.plot(m["x"], m["vertical"], "s-", color="tab:red", markersize=3, label="vertical (z)")
        ax.plot(m["x"], m["torsion"] * half_width, "^-", color="tab:green", markersize=3,
                label="torsion x B/2")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"{k + 1}: {m['freq']:.4f} Hz, {100 * m['damping']:.2f} %, {m['type']}", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0][0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("x along bridge [m]", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])    # leave room at the top for the title
    return fig


def plot_clusters(all_stable, modes, freqs_sv, first_sv, title, freq_max):
    """
    Two panels sharing the frequency axis:
      top    : stable poles, model order vs. frequency. Grey = rejected (not in an accepted
               cluster), coloured = accepted, one colour per mode.
      bottom : damping of the accepted poles vs. frequency, with the mean +- std of each mode.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    rejected = ~all_stable["accepted"]
    ax1.plot(all_stable["freq"][rejected], all_stable["order"][rejected], ".",
             color="0.7", markersize=3, label="stable pole, rejected")

    colors = plt.cm.tab10(np.arange(len(modes)) % 10)
    for k, m in enumerate(modes):
        ax1.plot(m["pole_freq"], m["pole_order"], "o", color=colors[k], markersize=3,
                 markerfacecolor="none")
        ax1.text(m["freq"], all_stable["order"].max() + 6, str(k + 1), ha="center",
                 fontsize=7, color=colors[k])
        ax2.plot(m["pole_freq"], 100 * m["pole_damping"], "o", color=colors[k], markersize=2,
                 alpha=0.5)
        ax2.errorbar(m["freq"], 100 * m["damping"], yerr=100 * m["damping_std"], fmt="s",
                     color="k", markersize=4, capsize=3)

    ax1.set_ylabel("Model order")
    ax1.set_ylim(0, all_stable["order"].max() + 12)
    ax1.set_title(title)
    ax1.legend(loc="lower right", fontsize=8)
    ax1b = ax1.twinx()
    ax1b.semilogy(freqs_sv, first_sv, color="k", linewidth=0.8, alpha=0.5)
    ax1b.set_yticks([])

    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Damping ratio [%]")
    ax2.set_xlim(0, freq_max)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

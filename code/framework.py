"""
    Figure 1 panels a and b: the estimation framework.
        (a) two Ornstein-Uhlenbeck measurement traces M(t) with unit variance
            whose means differ by 0.5
        (b) running average V_hat(t) of each trace, with shaded bands of
            mean +- BAND_SCALE * sqrt(2 VAR TAU_C / t)
    Panel c is drawn by hand.
-------------------------------------------------------------------------------
created on:
    Wed 4 Nov 2026
-------------------------------------------------------------------------------
last change:
    Wed 4 Nov 2026
-------------------------------------------------------------------------------
notes:
    Schematic figure: all quantities are dimensionless.
    Tweak the constants below to rescale the traces and bands.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from params import figure_path

plt.rcParams.update(
    {
        "text.usetex": True,  # real LaTeX typesetting
        "font.family": "serif",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

mpl.rcParams["figure.dpi"] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# ── Figure sizes and margins ──────────────────────────────────────────────────
fig_dir = "framework"  # figures/<fig_dir>/raw/

# Reference size (Nature single-column width in mm) for scaling panel dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from the mm reference), width / height = aspect
aspect = 1.4
width_md = 0.35 * pg_width / mm_to_in
height_md = width_md / aspect

# Fixed margins for consistent axes size across both single-panel figures
fig_margins_md = dict(left=0.12, right=0.8, bottom=0.15, top=0.9)

con_colors = (
    np.array(
        [
            [0, 77, 128],
            [181, 23, 0],
            [1, 113, 0],
            [242, 112, 0],
            [120, 0, 150],
            [0, 168, 157],
            [203, 41, 123],
            [0, 0, 0],
        ]
    )
    / 255
)
c_high, c_low = con_colors[0], con_colors[1]  # blue (top), red (bottom)

# ── Tunable constants ─────────────────────────────────────────────────────────
M_HIGH, M_LOW = 0.5, -0.5  # trace means
VAR = 0.5  # stationary variance of each OU trace
TAU_C = 1.0  # OU correlation time
T_MAX = 50.0  # total simulated time
DT = 0.02  # time step
SEED = 1467

BAND_SCALE = 2.0  # shaded band is mean +- BAND_SCALE * sqrt(2 VAR TAU_C / t)
T_MIN_BAND = 0.5  # band starts here to avoid the 1/sqrt(t) divergence at t -> 0
N_TRACES = 20  # estimator traces in panel b (the first is the one shown in panel a)
ALPHA_TRACES = 0.1  # opacity of the extra traces
LABELPAD = 8  # distance of axis labels from the axes (pt)
YLIM_A = (-3.2, 3.2)
YLIM_B = (-1.5, 1.5)

# ── Simulation ────────────────────────────────────────────────────────────────
t = np.arange(0.0, T_MAX + DT, DT)


def ou_trace(mean, rng):
    """Exact discretization of an OU process started in its stationary state."""
    rho = np.exp(-DT / TAU_C)
    noise = np.sqrt(VAR * (1 - rho**2)) * rng.standard_normal(len(t))
    x = np.empty(len(t))
    x[0] = mean + np.sqrt(VAR) * rng.standard_normal()
    for k in range(len(t) - 1):
        x[k + 1] = mean + rho * (x[k] - mean) + noise[k]
    return x


def running_average(x):
    """V_hat(t) = (1/t) int_0^t M(t') dt' (trapezoid rule); V_hat(0) = M(0)."""
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * DT)])
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = integral[1:] / t[1:]
    return out


rng = np.random.default_rng(SEED)
M_hi = ou_trace(M_HIGH, rng)
M_lo = ou_trace(M_LOW, rng)
V_hi = running_average(M_hi)
V_lo = running_average(M_lo)
# extra realizations drawn after the panel-a traces, so those are unchanged
V_extra_hi = [running_average(ou_trace(M_HIGH, rng)) for _ in range(N_TRACES - 1)]
V_extra_lo = [running_average(ou_trace(M_LOW, rng)) for _ in range(N_TRACES - 1)]


def new_panel():
    fig, ax = plt.subplots(figsize=(width_md, height_md))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks([])  # cartoon: no tick marks or labels
    ax.set_yticks([])
    plt.subplots_adjust(**fig_margins_md)
    return fig, ax


def mean_lines(ax, labels):
    """Dashed lines at the two means with labels just right of the axes."""
    for m, lab, col in zip((M_HIGH, M_LOW), labels, (c_high, c_low)):
        ax.axhline(m, lw=1, ls="--", color=col, zorder=2)
        ax.text(
            1.02,
            m,
            lab,
            color=col,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=10,
            clip_on=False,
        )


if __name__ == "__main__":
    print(
        f"a: empirical variance of traces: high = {M_hi.var():.2f}, "
        f"low = {M_lo.var():.2f} (target {VAR})"
    )

    # panel a
    fig_a, axa = new_panel()
    axa.plot(t, M_hi, "-", lw=0.6, color=c_high, zorder=3)
    axa.plot(t, M_lo, "-", lw=0.6, color=c_low, zorder=3)
    mean_lines(axa, [r"$\xi_{\mathrm{high}}$", r"$\xi_{\mathrm{low}}$"])
    axa.set_xlim(0, T_MAX)
    axa.set_ylim(*YLIM_A)
    axa.set_xlabel(r"Time $t$", labelpad=LABELPAD)
    axa.set_ylabel(r"Measurement $\xi(t)$", labelpad=LABELPAD)
    fig_a.savefig(figure_path(fig_dir, "framework_a.svg"), dpi=600)

    # panel b
    band = t >= T_MIN_BAND
    sd = BAND_SCALE * np.sqrt(2 * VAR * TAU_C / t[band])  # long-time Var[V_hat]
    fig_b, axb = new_panel()
    for V, extra, m, col in [
        (V_hi, V_extra_hi, M_HIGH, c_high),
        (V_lo, V_extra_lo, M_LOW, c_low),
    ]:
        axb.fill_between(t[band], m - sd, m + sd, color=col, alpha=0.2, lw=0, zorder=1)
        for Vk in extra:
            axb.plot(t, Vk, "-", lw=0.8, color=col, alpha=ALPHA_TRACES, zorder=2)
        axb.plot(t, V, "-", lw=1.2, color=col, zorder=3)
    mean_lines(axb, [r"$V_{\mathrm{high}}$", r"$V_{\mathrm{low}}$"])
    axb.set_xlim(0, T_MAX)
    axb.set_ylim(*YLIM_B)
    axb.set_xlabel(r"Time $t$", labelpad=LABELPAD)
    axb.set_ylabel(r"Estimator $\hat{V}(t)$", labelpad=LABELPAD)
    fig_b.savefig(figure_path(fig_dir, "framework_b.svg"), dpi=600)
    plt.show()

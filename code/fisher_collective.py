"""
    SI figure: Fisher information about V vs. measurement time t_meas
    (supplement Sec. "Collective sensing", subsections "The perfect instrument"
    and "Multiple sensors").
        black   perfect instrument, I_cell(t) = Idot_cell (t + 2 tau_leak)
        colors  N sensors, each with Idot_ind = 1 / (dV^2 t) (dV = 10 mV in
                t = 10 us), correlated only through the zeroth harmonic
-------------------------------------------------------------------------------
created on:
    Fri 13 Nov 2026
-------------------------------------------------------------------------------
last change:
    Fri 13 Nov 2026
-------------------------------------------------------------------------------
notes:
    I_N(t) = I_cell(t) * ratio_to_perfect(t / tau_leak, N* / N), with
    ratio_to_perfect from collective_sensing.py (handoff
    calculations/markdowns/fisher_info_vs_density_and_time.md). Equivalently
        I_N = (N Idot_ind / k^2) [t + 2 tau (k^2 - 1) / (k (k + coth(k t / 2 tau)))],
        k^2 = 1 + N r,  r = Idot_ind / Idot_cell,
    which has the limits N Idot_ind t (short t) and
    (Idot_cell^-1 + (N Idot_ind)^-1)^-1 t (long t).
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
import matplotlib.colors as mcolors

from params import figure_path, tau
from collective_sensing import I_cell, I_ind, N_star, ratio_to_perfect

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
fig_dir = "si_fisher_collective"  # figures/<fig_dir>/raw/

pg_width = 165  # mm
mm_to_in = 25.4

aspect = 1.4
width_md = 0.5 * pg_width / mm_to_in
height_md = width_md / aspect

# Same margins as charged_plate.py
fig_margins_md = dict(left=0.18, right=0.96, bottom=0.2, top=0.96)

# Magma, truncated: skip the near-black low end (clashes with the cell curve)
# and the pale-yellow high end (low contrast on white)
CMAP_RANGE = (0.3, 0.85)

# ── Tunable constants ─────────────────────────────────────────────────────────
N_VALUES = (1e1, 1e2, 1e3, 1e4, 1e5)  # number of sensors per curve
T_MEAS_RANGE = (1e-6, 1.0)  # measurement time range (s)
N_T = 300
LW_CELL = 2
LW_N = 1.5
V2_TO_MV2 = 1e-6  # V^-2 -> mV^-2


def fisher_cell(t):
    """Perfect instrument: I_cell(t) = Idot_cell (t + 2 tau_leak), in V^-2."""
    return I_cell * (t + 2 * tau)


def fisher_N(t, N):
    """N sensors sharing the zeroth-harmonic noise, in V^-2."""
    return ratio_to_perfect(t / tau, N_star / N) * fisher_cell(t)


if __name__ == "__main__":
    t = np.logspace(np.log10(T_MEAS_RANGE[0]), np.log10(T_MEAS_RANGE[1]), N_T)
    t_us = 1e6 * t
    cmap = plt.get_cmap("magma")
    norm = mcolors.LogNorm(min(N_VALUES), max(N_VALUES))

    fig, ax = plt.subplots(figsize=(width_md, height_md))
    ax.spines[["right", "top"]].set_visible(False)
    plt.subplots_adjust(**fig_margins_md)

    ax.plot(
        t_us,
        V2_TO_MV2 * fisher_cell(t),
        "-",
        lw=LW_CELL,
        color="k",
        label="Cell",
        zorder=4,
    )
    for N in N_VALUES:
        ax.plot(
            t_us,
            V2_TO_MV2 * fisher_N(t, N),
            "-",
            lw=LW_N,
            color=cmap(CMAP_RANGE[0] + (CMAP_RANGE[1] - CMAP_RANGE[0]) * norm(N)),
            label=rf"$N = 10^{{{int(round(np.log10(N)))}}}$",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(t_us[0], t_us[-1])
    ax.set_xticks([1e0, 1e2, 1e4, 1e6])
    ax.set_xlabel(r"Measurement time $t_{\rm meas}$ ($\mu$s)")
    ax.set_ylabel(r"Fisher information ${\cal I}$ (mV$^{-2}$)")
    ax.set_ylim(1e-2, 1e8)
    ax.set_yticks([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
    ax.legend(
        frameon=False, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.5
    )
    fig.savefig(figure_path(fig_dir, "fisher_collective_vs_t.pdf"))

    print(f"Idot_cell = {I_cell:.4g} V^-2 s^-1")
    print(f"Idot_ind  = {I_ind:.4g} V^-2 s^-1")
    print(f"N*        = {N_star:.4g}")
    print(f"I_cell(0) = {V2_TO_MV2 * fisher_cell(0.0):.4g} mV^-2")
    plt.show()

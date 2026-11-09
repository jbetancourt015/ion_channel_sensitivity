"""
    Saturation of the N-channel Fisher information rate: I_N / I_cell vs.
    channel density nu = N / 4 pi R^2.
-------------------------------------------------------------------------------
created on:
    Fri 6 Nov 2026
-------------------------------------------------------------------------------
last change:
    Fri 6 Nov 2026
-------------------------------------------------------------------------------
notes:
    Fisher-information-rate version of the old SNR-saturation figure
    (old_code/snr_scaling.py). See calculations/markdowns/
    fisher_info_rate_handoff.md (Secs. 1-2) and supplement Secs. S4.4, S5.
    All rates at omega = 0, voltage-referred. Channel j reads V_j = V_0 + local
    noise (S5.28: unit gain in voltage units); channels are correlated only
    through the zeroth harmonic (S5.51), with S_V,cell(0) = 2 k_B T R_m,
    R_m = tau_leak / (4 pi R^2 c) (S5.39). The local noise is set by an assumed
    single-channel rate I_ind = 1 / (dV^2 t), i.e. resolving dV within t
    (S4.26). Sherman-Morrison on S_jk = I_ind^-1 delta_jk + S_V,cell(0) gives
        I_N / I_cell = N / (N* + N),   N* = I_cell / I_ind.
    The ratio is the same whether the estimated parameter is V_0 or the input
    current (the common gain cancels).
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

from params import figure_path, kT, c, tau, R, n_points, N_log_min, N_log_max

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
fig_dir = "collective_sensing"  # figures/<fig_dir>/raw/

# Reference size (Nature single-column width in mm) for scaling panel dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from the mm reference), width / height = aspect
aspect = 1.4
width_md = 0.45 * pg_width / mm_to_in
height_md = width_md / aspect

fig_margins_md = dict(left=0.22, right=0.95, bottom=0.24, top=0.9)

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

# ── Tunable constants ─────────────────────────────────────────────────────────
DV_IND = 10e-3  # voltage a single channel resolves (V) ...
T_IND = 10e-6  # ... within this time (s), from local fluctuations
LW = 1.5  # linewidth of the curve
LW_DASH = 1  # linewidth of the nu* marker
FONT_ANNOT = 9  # fontsize of in-plot annotations
YLIM = (1e-3, 1.5)

# ── Fisher information rates (V^-2 s^-1) ──────────────────────────────────────
area = 4 * np.pi * R**2  # cell area (m^2)
R_m = tau / (area * c)  # membrane resistance (Ohm)
S_V_cell = 2 * kT * R_m  # zeroth-harmonic voltage noise at omega = 0 (V^2 s)
I_cell = 1 / S_V_cell  # whole-cell readout
I_ind = 1 / (DV_IND**2 * T_IND)  # single channel, local noise only
N_star = I_cell / I_ind  # crossover channel number
nu_star = N_star / area  # crossover density (m^-2)


def I_ratio(N):
    """I_N / I_cell for N channels sharing the zeroth-harmonic noise."""
    return N / (N_star + N)


if __name__ == "__main__":
    N = np.logspace(N_log_min, N_log_max, n_points)
    nu_um = 1e-12 * N / area  # channels / um^2
    nu_star_um = 1e-12 * nu_star

    fig, ax = plt.subplots(figsize=(width_md, height_md))
    ax.spines[["right", "top"]].set_visible(False)
    plt.subplots_adjust(**fig_margins_md)

    ax.plot(nu_um, I_ratio(N), "-", lw=LW, color=con_colors[0], zorder=3)
    ax.plot(
        [nu_star_um, nu_star_um],
        [YLIM[0], 1.0],
        "--",
        lw=LW_DASH,
        color=con_colors[1],
        zorder=2,
    )
    ax.text(
        1.5 * nu_star_um,
        3 * YLIM[0],
        r"$\nu^*$",
        color=con_colors[1],
        fontsize=FONT_ANNOT,
        va="center",
        ha="left",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(nu_um[0], nu_um[-1])
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Channel density $\nu$ (channels/$\mu$m$^2$)")
    ax.set_ylabel(r"$\dot{\cal I}_N / \dot{\cal I}_{\rm cell}$")

    fig.savefig(
        figure_path(fig_dir, "collective_sensing.pdf"), dpi=600, bbox_inches="tight"
    )
    fig.savefig(
        figure_path(fig_dir, "collective_sensing.svg"), dpi=600, bbox_inches="tight"
    )

    print(
        f"I_ind  = {I_ind:.4g} V^-2 s^-1  ({DV_IND * 1e3:g} mV in {T_IND * 1e6:g} us)"
    )
    print(f"R_m    = {R_m * 1e-6:.4g} MOhm")
    print(
        f"I_cell = {I_cell:.4g} V^-2 s^-1  (resolves {DV_IND * 1e3:g} mV in "
        f"{1 / (I_cell * DV_IND**2) * 1e9:.4g} ns)"
    )
    print(f"N*     = {N_star:.4g}")
    print(f"nu*    = {nu_star_um:.4g} channels/um^2")
    plt.show()

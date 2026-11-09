"""
    SI figure: capacitive and counting fluctuations of the effective
    charge-counting measurement of the charged plate (supplement Sec. S4.3,
    kernel Psi_p of Eq. S4.13 with the Gaussian Gamma of Eq. S4.15) vs. plate
    width sigma, at omega = 0.
    Both are normalized by the large-sigma capacitive noise
    S_ref = 16 pi^(3/2) sigma^3 c_eff^2 (code units), i.e. Eq. (S4.20) with
    U -> 1 and eps_m/d -> c_eff (Debye layers included).
-------------------------------------------------------------------------------
created on:
    Fri 6 Nov 2026
-------------------------------------------------------------------------------
last change:
    Fri 6 Nov 2026
-------------------------------------------------------------------------------
notes:
    Dimensionless units: lengths in l_D, times in tau_D, energies in kT.
    Without the Debye layers (leading order in l_D, as in Eq. S4.20) the
    reference is 4 pi^(3/2) sigma^3 / l_th^2, larger by ((1 + l_th)/l_th)^2.
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
from matplotlib.lines import Line2D
from scipy import integrate

from params import l_D, d, e_m, figure_path
from noise_reference import A_plate_cap, A_plate_count, c_eff

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
fig_dir = "si_charged_plate"  # figures/<fig_dir>/raw/

# Reference size (Nature single-column width in mm) for scaling panel dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from the mm reference), width / height = aspect
aspect = 1.4
width_md = 0.5 * pg_width / mm_to_in
height_md = width_md / aspect

# Same margins as asymptotic_numerics.py
fig_margins_md = dict(left=0.18, right=0.96, bottom=0.2, top=0.96)

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

SIG_MIN_NM, SIG_MAX_NM, N_SIG = 1.0, 1000.0, 60
YLIM = (1e-4, 2e0)

# ── Dimensionless parameters ──────────────────────────────────────────────────
nm = 1e-9 / l_D  # 1 nm in units of l_D
r_eps = 1.0 / e_m  # eps / eps_m
d_norm = d / l_D  # membrane thickness
c_e = c_eff(r_eps, d_norm)


def S_plate(s, A, epsrel):
    """4 pi s^4 int_0^inf dk k exp(-k^2 s^2) A(k): same normalization as S_cap/S_count."""
    f = lambda k: k * np.exp(-((k * s) ** 2)) * A(k, r_eps, d_norm)
    return (
        4.0
        * np.pi
        * s**4
        * integrate.quad(f, 1e-12, 8.0 / s, limit=300, epsrel=epsrel)[0]
    )


def S_ref(s):
    """Large-sigma capacitive noise of the plate kernel, 16 pi^(3/2) s^3 c_eff^2."""
    return 16.0 * np.pi**1.5 * s**3 * c_e**2


if __name__ == "__main__":
    sig_nm = np.logspace(np.log10(SIG_MIN_NM), np.log10(SIG_MAX_NM), N_SIG)
    s = sig_nm * nm
    cap = np.array([S_plate(si, A_plate_cap, 1e-9) for si in s]) / S_ref(s)
    count = np.array([S_plate(si, A_plate_count, 1e-6) for si in s]) / S_ref(s)

    print(f"eps/eps_m = {r_eps:.2f}, d/l_D = {d_norm:.2f}, c_eff = {c_e:.4f}")
    print(f"S_cap/S_ref at sigma = {sig_nm[-1]:g} nm: {cap[-1]:.4f}")
    i = np.argmax(count)
    print(f"max S_count/S_ref = {count[i]:.4f} at sigma = {sig_nm[i]:.2f} nm")

    fig, ax = plt.subplots(figsize=(width_md, height_md))
    ax.spines[["right", "top"]].set_visible(False)
    plt.subplots_adjust(**fig_margins_md)
    ax.plot(sig_nm, cap, "-", lw=2, color=con_colors[3], label="Capacitive", zorder=3)
    ax.plot(sig_nm, count, "-", lw=2, color=con_colors[0], label="Counting", zorder=3)
    ax.axhline(1.0, ls="--", lw=1, color=con_colors[3])
    ax.set_xscale("log")
    ax.set_xlim(SIG_MIN_NM, SIG_MAX_NM)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Plate width $\sigma$ (nm)")
    ax.set_ylabel(r"$S^Q\,/\,S^Q_{\mathrm{cap},\infty}$")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], ls="--", lw=1, color="k", label=r"Large-$\sigma$ limit")
    )
    ax.legend(handles=handles, frameon=False, loc="lower left")
    fig.savefig(figure_path(fig_dir, "charged_plate_noise_vs_sigma.pdf"))
    plt.show()

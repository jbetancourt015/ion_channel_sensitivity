"""
    Figure 2 panels a and b: the mode structure of the membrane.
        (a) cartoon of one bound mode and one oscillatory mode vs. distance z
            from the bulk
        (b) relaxation time vs. wavenumber: bound mode (solid blue) with its
            low-k asymptote 2 c_eff / k (dashed), and the maximum relaxation
            time of the oscillatory modes 1 / (1 + k^2) (solid red) with the
            oscillatory band shaded below it
        (c) capacitive and counting contributions to the inverse Fisher
            information rate of the charged-plate (field sensor) vs. plate
            width sigma
-------------------------------------------------------------------------------
created on:
    Thu 5 Nov 2026
-------------------------------------------------------------------------------
last change:
    Fri 6 Nov 2026
-------------------------------------------------------------------------------
notes:
    Panel a is a cartoon with placeholder shapes (from
    old_code/exploratory/mode_structure.py).
    Panels b and c use noise_reference.py (units: l_D = tau_D = 1) with the
    Table I parameters from params.py.
    Panel c: plate kernel of supplement Eq. (S4.11) with a Gaussian lateral
    profile Gamma(x) = exp(-|x|^2 / 2 sigma^2), at omega = 0;
    I_x^-1 = S^Q_x(0) / (4 c_eff^2 A_p^2) (Eq. S4.28), computed in units
    k_B T / (alpha l_D) and plotted in mV^2 ns. Numerics only.
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

from params import figure_path, l_D, d, e_m, kT, a
from noise_reference import bound_mode, c_eff, SV_plate_cap, SV_plate_count

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
fig_dir = "modes"  # figures/<fig_dir>/raw/

# Reference size (Nature single-column width in mm) for scaling panel dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from the mm reference), width / height = aspect
aspect = 1.4
width_md = 0.35 * pg_width / mm_to_in
height_md = width_md / aspect

# Fixed margins for consistent axes size across both single-panel figures
fig_margins_md = dict(left=0.29, right=0.95, bottom=0.24, top=0.9)

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
c_bound, c_osc = con_colors[0], con_colors[1]  # blue (bound), red (oscillatory)

# ── Tunable constants ─────────────────────────────────────────────────────────
XLABEL_Y = -0.2  # x-label height in axes fractions, identical in both panels
# (independent of tick-label size, so the x labels line up)
LABELPAD = 4  # y-label distance from the tick labels (pt)
LW = 1.5  # linewidth of the curves (both panels)
LW_DASH = 1  # linewidth of dashed guide lines (both panels)
FONT_ANNOT = 9  # fontsize of in-plot annotations (both panels)
YLABEL_GAP_FRAC = (
    0.5  # y-label distance from the left edge, relative to panel b's default
)

# panel a (cartoon)
Z_MAX = 2.0
OSC_AMP, OSC_FREQ = 0.2, 10.0  # oscillatory mode 0.2 cos(5 z)
YLIM_A = (-0.25, 1.05)
N_Z = 400

# panel b
K_MIN, K_MAX, N_K = 1e-4, 1e1, 600
YLIM_B = (1e-2, 1e3)
ALPHA_FILL = 0.15  # opacity of the oscillatory band

# panel c
SIG_MIN_NM, SIG_MAX_NM, N_SIG = 0.1, 100.0, 60
YLIM_C = (1.0, 1e4)  # mV^2 ns
YLABEL_GAP_C = "3pt"  # extra space between the two lines of the y label
I_UNIT = kT / (a * l_D) * 1e6 * 1e9  # k_B T/(alpha l_D) in mV^2 ns

# Table I parameters (params.py) in units of l_D
R_EPS = 1.0 / e_m  # eps / eps_m
D_NORM = d / l_D  # d / l_D


def align_labels(ax):
    ax.xaxis.set_label_coords(0.5, XLABEL_Y)


def ylabel_gap(fig, ax):
    """Distance (in) from the left edge of the figure to the left edge of the y label."""
    fig.canvas.draw()
    box = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
    return box.x0 / fig.dpi


def trim_left(fig, ax, gap):
    """Narrow the figure from the left until ylabel_gap == gap; axes size is kept."""
    shift = ylabel_gap(fig, ax) - gap
    w, h = fig.get_size_inches()
    pos = ax.get_position()
    left_in, width_in = pos.x0 * w - shift, pos.width * w
    fig.set_size_inches(w - shift, h)
    ax.set_position([left_in / (w - shift), pos.y0, width_in / (w - shift), pos.height])


def new_panel():
    fig, ax = plt.subplots(figsize=(width_md, height_md))
    ax.spines[["right", "top"]].set_visible(False)
    plt.subplots_adjust(**fig_margins_md)
    return fig, ax


if __name__ == "__main__":
    print(f"Table I: eps/eps_m = {R_EPS:.2f}, d/l_D = {D_NORM:.2f}")

    # panel a
    z = np.linspace(0.0, Z_MAX, N_Z)
    fig_a, axa = new_panel()
    axa.set_xticks([0.0, 1.0])  # cartoon: only z = 0 and z = l_D are marked
    axa.set_xticklabels(["0", r"$l_D$"])
    axa.set_yticks([0.0])  # only the zero level is marked
    axa.set_yticklabels(["0"])
    axa.axhline(0.0, lw=LW_DASH, ls="--", color="k", zorder=1)
    axa.plot(z, np.exp(-z), "-", lw=LW, color=c_bound, zorder=3)
    axa.plot(z, OSC_AMP * np.cos(OSC_FREQ * z), "-", lw=LW, color=c_osc, zorder=3)
    for x, y, lab, col in [
        (0.3, 0.76, "bound", c_bound),
        (0.06, 0.46, "oscillatory", c_osc),
    ]:  # positions in axes fractions, so they survive changes to Z_MAX / YLIM_A
        axa.text(
            x,
            y,
            lab,
            color=col,
            transform=axa.transAxes,
            fontsize=FONT_ANNOT,
            va="center",
            ha="left",
        )
    axa.set_xlim(0, Z_MAX)
    axa.set_ylim(*YLIM_A)
    axa.set_xlabel(r"Distance from membrane $z$")
    axa.set_ylabel(r"Charge density $\rho(\mathbf{k},z)$", labelpad=LABELPAD)
    align_labels(axa)

    # panel b
    k = np.logspace(np.log10(K_MIN), np.log10(K_MAX), N_K)
    tau_b = bound_mode(k, R_EPS, D_NORM)[1]
    tau_edge = 1.0 / (1.0 + k**2)  # max relaxation time of oscillatory modes
    tau_low = 2.0 * c_eff(R_EPS, D_NORM) / k  # k tau_b -> 2 c_eff as k -> 0
    low = tau_low >= tau_edge  # draw the asymptote only above the red line
    k_low, tau_low = k[low], tau_low[low]

    fig_b, axb = new_panel()
    axb.fill_between(
        k, YLIM_B[0], tau_edge, color=c_osc, alpha=ALPHA_FILL, lw=0, zorder=1
    )
    axb.plot(k, tau_edge, "-", lw=LW, color=c_osc, zorder=4)  # red on top of blue
    axb.plot(k, tau_b, "-", lw=LW, color=c_bound, zorder=3)
    axb.plot(k_low, tau_low, "--", lw=LW_DASH, color=c_bound, zorder=2)
    axb.text(
        3e-4,
        1e-1,
        "oscillatory modes",
        color=c_osc,
        fontsize=FONT_ANNOT,
        va="center",
        ha="left",
    )
    axb.text(
        3e-4,
        3e2,
        "bound mode",
        color=c_bound,
        fontsize=FONT_ANNOT,
        va="center",
        ha="left",
    )
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(K_MIN, K_MAX)
    axb.set_ylim(*YLIM_B)
    axb.set_xlabel(r"Wavenumber $k l_D$")
    axb.set_ylabel(r"Relaxation time $\tau/\tau_D$", labelpad=LABELPAD)
    align_labels(axb)

    # panel c
    sig_nm = np.logspace(np.log10(SIG_MIN_NM), np.log10(SIG_MAX_NM), N_SIG)
    s_c = sig_nm * 1e-9 / l_D
    cap = I_UNIT * np.array([SV_plate_cap(s, R_EPS, D_NORM) for s in s_c])
    count = I_UNIT * np.array([SV_plate_count(s, R_EPS, D_NORM) for s in s_c])

    fig_c, axc = new_panel()
    axc.plot(sig_nm, cap, "-", lw=LW, color=c_bound, zorder=3)
    axc.plot(sig_nm, count, "-", lw=LW, color=c_osc, zorder=3)
    for x, y, lab, col in [
        (0.3, 2e3, "capacitive", c_bound),
        (0.5, 4.0, "counting", c_osc),
    ]:  # data coordinates
        axc.text(x, y, lab, color=col, fontsize=FONT_ANNOT, va="center", ha="left")
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(SIG_MIN_NM, SIG_MAX_NM)
    axc.set_ylim(*YLIM_C)
    axc.set_xlabel(r"Plate width $\sigma$ (nm)")
    axc.set_ylabel(
        rf"\begin{{tabular}}{{c}}Inverse info rate\\[{YLABEL_GAP_C}]"
        r"$\dot{\cal I}^{-1}$ (mV$^2$ns)\end{tabular}",
        labelpad=LABELPAD,
    )
    align_labels(axc)

    # trim the left margins (same axes size) so all y labels are the same
    # distance from the left edge: YLABEL_GAP_FRAC of panel b's original gap
    gap = YLABEL_GAP_FRAC * ylabel_gap(fig_b, axb)
    trim_left(fig_a, axa, gap)
    trim_left(fig_b, axb, gap)
    trim_left(fig_c, axc, gap)
    fig_a.savefig(figure_path(fig_dir, "modes_a.svg"), dpi=600)
    fig_b.savefig(figure_path(fig_dir, "modes_b.svg"), dpi=600)
    fig_c.savefig(figure_path(fig_dir, "modes_c.svg"), dpi=600)

    i = np.argmin(np.abs(tau_b - 1.0))
    print(f"tau_b = 1 at k l_D = {k[i]:.3f}")
    print(
        f"k tau_b at k_min = {k[0] * tau_b[0]:.4f}  (2 c_eff = {2 * c_eff(R_EPS, D_NORM):.4f})"
    )
    for j in (0, -1):
        print(
            f"sigma = {sig_nm[j]:g} nm: I_cap^-1 = {cap[j]:.4g}, "
            f"I_count^-1 = {count[j]:.4g}  (mV^2 ns)"
        )
    plt.show()

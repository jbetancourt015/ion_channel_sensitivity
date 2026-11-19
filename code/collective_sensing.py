"""
    Collective sensing figure (one row, 178 x 35 mm), panels saved separately:
        a  hand-drawn, 43 mm wide (not produced here)
        b  cartoon of N measurement traces xi_j(t)
        c  long-time saturation I_N / I_cell vs. channel density nu
        d  critical density nu*(t_obs) / nu*_0 vs. observation time
-------------------------------------------------------------------------------
created on:
    Fri 6 Nov 2026
-------------------------------------------------------------------------------
last change:
    Mon 9 Nov 2026
-------------------------------------------------------------------------------
notes:
    Panel c: Fisher-information-rate version of the old SNR-saturation figure
    (old_code/snr_scaling.py). See calculations/markdowns/
    fisher_info_rate_handoff.md (Secs. 1-2) and supplement Secs. S4.4, S5.
    All rates at omega = 0, voltage-referred. Channel j reads V_j = V_0 + local
    noise (S5.28: unit gain in voltage units); channels are correlated only
    through the zeroth harmonic (S5.51), with S_V,cell(0) = 2 k_B T R_m,
    R_m = tau_leak / (4 pi R^2 c) (S5.39). The local noise is set by an assumed
    single-channel rate I_ind = 1 / (dV^2 t), i.e. resolving dV = 10 mV within
    t = 10 us (S4.26), so I_ind^-1 = 1e3 mV^2 us. Sherman-Morrison on
    S_jk = I_ind^-1 delta_jk + S_V,cell(0) gives
        I_N / I_cell = N / (N* + N),   N* = I_cell / I_ind.
    This is the t -> infinity limit of the finite-time result used in panel d.

    Panel d: calculations/markdowns/fisher_info_vs_density_and_time.md.
    nu*(t_obs) is the density at which the exact finite-time Fisher
    information reaches half the perfect-instrument value,
    I(t, nu*) = I_cell(t) / 2 (the handoff's nu_1/2). Writing
    nu*(t) = nu*_0 f(t / tau_leak), f -> 1 at long times (nu*_0 = nu* of
    panel c) and f ~ 2 tau / t at short times. Same I_ind as panel c.
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
import matplotlib.transforms as mtransforms
from scipy.optimize import brentq

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

# ── Figure layout (all in mm) ─────────────────────────────────────────────────
# Each panel is saved as its own file, to be assembled by hand into a 178 x 35 mm
# row (PRL page width): panel a (hand-drawn) 43 mm wide, then b, c, d (widths
# sum to 135 mm).
fig_dir = "collective_sensing"  # figures/<fig_dir>/raw/
mm_to_in = 25.4
PANEL_W = {"b": 40, "c": 48.5, "d": 46.5}  # canvas width of each panel (mm)
PANEL_H = 35
AX_BOTTOM, AX_TOP = 10.5, 33  # vertical extent of each axes box
# Panel b has no tick labels: its axes box extends down toward its x-label
# (which stays aligned with those of c and d)
AX_BOTTOM_B = 5
XLABEL_Y = 1.0  # baseline of every x-label above the canvas bottom (mm)
# (left edge, right edge) of each axes box within its own panel canvas;
# tick/axis labels sit outside
AX_X = {
    "b": (6, 31),
    "c": (15, 46.5),
    "d": (12.5, 43.5),
}

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
LW = 1.5  # linewidth of the curves
LW_DASH = 1  # linewidth of the nu* markers
FONT_ANNOT = 9  # fontsize of in-plot annotations
YLIM = (1e-3, 1.5)  # panel c
T_OBS_RANGE = (1e-6, 1.0)  # panel d, observation time range (s)

# ── Traces cartoon panel (xi_j(t) for many channels) ──────────────────────────
TR_OFFSETS = (3.2, 2.0, 0.0)  # vertical offsets of xi_1, xi_2, xi_N
TR_VDOTS_Y = 1.0  # height of the vdots between xi_2 and xi_N
TR_STD = 0.22  # stationary std of each OU trace
TR_TAU = 1.0  # OU correlation time
TR_TMAX = 25.0  # total time
TR_DT = 0.02  # time step
TR_LW = 0.6  # trace linewidth
TR_SEED = 2718
TR_LABELS = (r"$\xi_1(t)$", r"$\xi_2(t)$", r"$\xi_N(t)$")

# ── Fisher information rates (V^-2 s^-1) ──────────────────────────────────────
area = 4 * np.pi * R**2  # cell area (m^2)
R_m = tau / (area * c)  # membrane resistance (Ohm)
S_V_cell = 2 * kT * R_m  # zeroth-harmonic voltage noise at omega = 0 (V^2 s) = S_L
I_cell = 1 / S_V_cell  # whole-cell readout
I_ind = 1 / (DV_IND**2 * T_IND)  # single channel, local noise only (= I_loc)
N_star = I_cell / I_ind  # crossover channel number
nu_star = N_star / area  # crossover density (m^-2) = handoff's nu*_0


def panel_axes(panel):
    """New PANEL_W x PANEL_H figure with the axes box of panel b, c or d."""
    W = PANEL_W[panel]
    fig = plt.figure(figsize=(W / mm_to_in, PANEL_H / mm_to_in))
    x0, x1 = AX_X[panel]
    y0 = AX_BOTTOM_B if panel == "b" else AX_BOTTOM
    ax = fig.add_axes(
        [
            x0 / W,
            y0 / PANEL_H,
            (x1 - x0) / W,
            (AX_TOP - y0) / PANEL_H,
        ]
    )
    # Same x-label baseline in every panel, so they line up when assembled
    ax.xaxis.set_label_coords(
        0.5,
        XLABEL_Y / mm_to_in,
        transform=mtransforms.blended_transform_factory(
            ax.transAxes, fig.dpi_scale_trans
        ),
    )
    ax.xaxis.label.set_va("baseline")
    return fig, ax


def save_panel(fig, panel):
    """Fixed canvas size (no bbox_inches='tight') so panels tile exactly."""
    fig.savefig(figure_path(fig_dir, f"collective_sensing_{panel}.svg"), dpi=600)


def ou_trace(t, mean, std, tau, rng):
    """Exact discretization of a stationary OU process."""
    dt = t[1] - t[0]
    rho = np.exp(-dt / tau)
    noise = std * np.sqrt(1 - rho**2) * rng.standard_normal(len(t))
    x = np.empty(len(t))
    x[0] = mean + std * rng.standard_normal()
    for k in range(len(t) - 1):
        x[k + 1] = mean + rho * (x[k] - mean) + noise[k]
    return x


def traces_panel(ax):
    """Panel b: cartoon of N measurement traces xi_j(t) stacked vertically."""
    t = np.arange(0.0, TR_TMAX + TR_DT, TR_DT)
    rng = np.random.default_rng(TR_SEED)

    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for off, lab in zip(TR_OFFSETS, TR_LABELS):
        ax.plot(t, ou_trace(t, off, TR_STD, TR_TAU, rng), "-", lw=TR_LW, color="k")
        ax.text(
            1.03,
            off,
            lab,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=FONT_ANNOT,
            clip_on=False,
        )
    ax.text(
        0.5,
        TR_VDOTS_Y,
        # \vdots hard-codes a 4pt dot spacing; the SVG backend typesets TeX at
        # 100pt and rescales, which squashes the dots. Use em units instead.
        r"$\vbox{\baselineskip=0.39em\lineskiplimit=0pt"
        r"\hbox{.}\hbox{.}\hbox{.}}$",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="center",
        fontsize=FONT_ANNOT + 2,
    )

    pad = 4 * TR_STD
    ax.set_xlim(0, TR_TMAX)
    ax.set_ylim(min(TR_OFFSETS) - pad, max(TR_OFFSETS) + pad)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Measurements")


def I_ratio(N):
    """I_N / I_cell for N channels sharing the zeroth-harmonic noise (t -> inf)."""
    return N / (N_star + N)


def saturation_panel(ax):
    """Panel c: I_N / I_cell vs. channel density, long observation times."""
    N = np.logspace(N_log_min, N_log_max, n_points)
    nu_um = 1e-12 * N / area  # channels / um^2
    nu_star_um = 1e-12 * nu_star

    ax.spines[["right", "top"]].set_visible(False)
    ax.plot(nu_um, I_ratio(N), "-", lw=LW, color=con_colors[0], zorder=3)
    ax.plot(
        [nu_star_um, nu_star_um],
        [YLIM[0], 1.0],
        "--",
        lw=LW_DASH,
        color=con_colors[3],
        zorder=2,
    )
    ax.text(
        1.5 * nu_star_um,
        3 * YLIM[0],
        r"$\nu_0^*$",
        color=con_colors[3],
        fontsize=FONT_ANNOT,
        va="center",
        ha="left",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(nu_um[0], nu_um[-1])
    ax.set_xticks([1e-3, 1e-1, 1e1])
    ax.set_ylim(*YLIM)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xlabel(r"Density $\nu$ ($\mu$m$^{-2}$)")
    ax.set_ylabel(r"Info rate $\dot{\cal I}_N / \dot{\cal I}_{\rm cell}$")


def ratio_to_perfect(x, rho):
    """I(t, nu) / I_cell(t), x = t / tau_leak, rho = nu*_0 / nu (handoff Sec. 5)."""
    k = np.sqrt(1 + 1 / rho)
    th = np.tanh(k * x / 2)
    g = (1 + k) * th / (1 + k * th)
    return (x + 2 * (1 - 1 / k) * g) / ((1 + rho) * (x + 2))


def nu_star_factor(x):
    """f(x) = nu*(t_obs) / nu*_0, where I(t, nu*) = I_cell(t) / 2 and x = t / tau."""
    h = lambda lr: ratio_to_perfect(x, np.exp(lr)) - 0.5
    return 1 / np.exp(brentq(h, np.log(1e-14), np.log(1e8), xtol=1e-14))


def nu_star_panel(ax):
    """Panel d: nu*(t_obs) / nu*_0 vs. observation time."""
    t = np.logspace(np.log10(T_OBS_RANGE[0]), np.log10(T_OBS_RANGE[1]), n_points)
    f = np.array([nu_star_factor(x) for x in t / tau])  # nu*(t_obs) / nu*_0

    ax.spines[["right", "top"]].set_visible(False)
    ax.plot(1e6 * t, f, "-", lw=LW, color=con_colors[3], zorder=3)
    # Short-time asymptote f ~ 2 tau_leak / t_obs (handoff Sec. 6)
    ax.plot(1e6 * t, 2 * tau / t, "--", lw=LW_DASH, color=con_colors[3], zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(f.min() / 1.4, 1.4 * f.max())  # asymptote clipped below
    ax.set_xlim(1e6 * T_OBS_RANGE[0], 1e6 * T_OBS_RANGE[1])  # us
    ax.set_xticks([1e0, 1e2, 1e4, 1e6])
    ax.set_yticks([1e0, 1e1, 1e2, 1e3])
    ax.set_xlabel(r"Time $t_{\rm obs}$ ($\mu$s)")
    ax.set_ylabel(r"Density $\nu^* / \nu^*_0$")
    return t, f


if __name__ == "__main__":
    for panel, plot in (
        ("b", traces_panel),
        ("c", saturation_panel),
        ("d", nu_star_panel),
    ):
        fig, ax = panel_axes(panel)
        plot(ax)
        save_panel(fig, panel)

    print(
        f"I_ind  = {I_ind:.4g} V^-2 s^-1  ({DV_IND * 1e3:g} mV in {T_IND * 1e6:g} us;"
        f" I_ind^-1 = {1e12 / I_ind:.4g} mV^2 us)"
    )
    print(f"R_m    = {R_m * 1e-6:.4g} MOhm")
    print(f"S_L    = {S_V_cell * 1e12:.4g} mV^2 us")
    print(
        f"I_cell = {I_cell:.4g} V^-2 s^-1  (resolves {DV_IND * 1e3:g} mV in "
        f"{1 / (I_cell * DV_IND**2) * 1e9:.4g} ns)"
    )
    print(f"N*     = {N_star:.4g}")
    print(f"nu*_0  = {1e-12 * nu_star:.4g} channels/um^2")
    xs = (0.01, 0.1, 1, 10, 100)
    print(
        "f(t/tau) at",
        xs,
        "=",
        [round(float(nu_star_factor(x)), 3) for x in xs],
        " (handoff: 201.334, 21.337, 3.329, 1.283, 1.028)",
    )

    plt.show()

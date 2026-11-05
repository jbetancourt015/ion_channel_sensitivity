"""
    This script plots the scaling of the signal-to-noise ratio with the number
    of ion channels.
-------------------------------------------------------------------------------
created on:
    Wed 3 Dec 2025
-------------------------------------------------------------------------------
last change:
    Wed 3 Dec 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

plt.rcParams.update(
    {
        "text.usetex": False,  # keep LaTeX off globally
        "mathtext.fontset": "cm",  # or 'stixsans' for sans-serif
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "font.family": "Helvetica",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,  # thin axis lines for publication
        "pdf.fonttype": 42,  # ensures text stays as text in Illustrator
        "ps.fonttype": 42,  # same for PostScript
    }
)

mpl.rcParams["figure.dpi"] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 3.5
height = 3.2

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
latex_gray = np.array([240, 240, 240]) / 255

colormap = "cool_r"

import os
from params import (
    kT,
    eps,
    sig,
    sig_lo,
    sig_hi,
    a,
    c,
    l_D,
    tau,
    R,
    n_points,
    N_log_min,
    N_log_max,
    FIGURES_DIR,
)

# Derived quantities
tau_D = eps / a
om = 1 / tau

# Crossover point
N0 = (1 / 8) * (a * (tau_D**2) / (tau * c * l_D)) * ((R / sig) ** 2)
N0_lo, N0_hi = (
    N0 * (sig / sig_hi) ** 2,
    N0 * (sig / sig_lo) ** 2,
)  # bounds for N0 based on σ range
N_s = N0 * (1 + (om * tau) ** 2)

# N values and corresponding channel densities (ν = N / 4πR²)
N_vals = np.logspace(N_log_min, N_log_max, n_points)
_area = 4 * np.pi * R**2
nu_vals = N_vals / _area
nu_s_hi = N0_hi * (1 + (om * tau) ** 2) / _area
nu_s_lo = N0_lo * (1 + (om * tau) ** 2) / _area


# SNR functions
def SNR_ratio(N, N_star):
    return N / (N_star + N)


# FIGURE: Signal-to-noise as a function of channel density---------------------
def make_snr_saturation_fig(gray_bg=False, large=False):
    if large:
        width = 5.0
        height = 4.0
    else:
        width = 3.5
        height = 3.2

    fig, ax = plt.subplots(figsize=(0.9 * width, 0.7 * height))

    if gray_bg:
        fig.patch.set_facecolor(latex_gray)
        ax.set_facecolor(latex_gray)

    ax.plot(
        1e-12 * nu_vals,
        SNR_ratio(N_vals, N0_hi * (1 + (om * tau) ** 2)),
        c=con_colors[0],
        lw=2,
        label="$\sigma = 1$ nm",
    )
    ax.plot(
        1e-12 * nu_vals,
        SNR_ratio(N_vals, N0_lo * (1 + (om * tau) ** 2)),
        c=con_colors[0],
        alpha=0.7,
        lw=2,
        label="$\sigma = 5$ nm",
    )
    ax.plot(
        [1e-12 * nu_s_hi, 1e-12 * nu_s_hi],
        [SNR_ratio(1.0, N0_hi * (1 + (om * tau) ** 2)), 1.0],
        c=con_colors[1],
        ls="--",
        lw=1,
    )
    ax.plot(
        [1e-12 * nu_s_lo, 1e-12 * nu_s_lo],
        [SNR_ratio(1.0, N0_hi * (1 + (om * tau) ** 2)), 1.0],
        c=con_colors[1],
        alpha=0.7,
        ls="--",
        lw=1,
    )

    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    # ax.tick_params(axis='y')
    ax.text(1.5 * 1e-12 * nu_s_hi, 0.003, r"$\nu^*$", c=con_colors[1])
    ax.text(1.5 * 1e-12 * nu_s_lo, 0.003, r"$\nu^*$", c=con_colors[1], alpha=0.5)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("$\\text{SNR}_N(\omega)/\\text{SNR}_{\\text{cell}}(\omega)$")
    ax.set_xlabel(r"Channel density $\nu$ (channels/$\mu$m$^2$)")

    return fig, ax


# PRL Fig 3 -> figures/snr_saturation/raw/
main_dir = os.path.join(FIGURES_DIR, "snr_saturation", "raw")
os.makedirs(main_dir, exist_ok=True)
fig, ax = make_snr_saturation_fig(gray_bg=False)
plt.savefig(os.path.join(main_dir, "snr_saturation.pdf"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(main_dir, "snr_saturation.svg"), dpi=600, bbox_inches="tight")
plt.show()

# Grey-background (poster) figure -> figures/poster/
poster_dir = os.path.join(FIGURES_DIR, "poster")
os.makedirs(poster_dir, exist_ok=True)
fig_gray, ax_gray = make_snr_saturation_fig(gray_bg=True, large=True)
plt.savefig(
    os.path.join(poster_dir, "snr_saturation.pdf"), dpi=600, bbox_inches="tight"
)
plt.savefig(
    os.path.join(poster_dir, "snr_saturation.svg"), dpi=600, bbox_inches="tight"
)
plt.show()

# Numerical results used in the text
S_shot_lo = (np.pi / 4) * ((sig_lo**2) * (tau_D**2) * a * kT) / l_D
S_shot_hi = (np.pi / 4) * ((sig_hi**2) * (tau_D**2) * a * kT) / l_D
S_JN_lo = 4 * (np.pi**1.5) * (sig_lo**3) * (c**2) * kT / a
S_JN_hi = 4 * (np.pi**1.5) * (sig_hi**3) * (c**2) * kT / a
V0_lo = np.sqrt((1 / (16 * np.pi)) * tau_D * a * kT / ((c**2) * (sig_lo**2) * l_D))
V0_hi = np.sqrt((1 / (16 * np.pi)) * tau_D * a * kT / ((c**2) * (sig_hi**2) * l_D))
density_lo, density_hi = (
    N0_lo / (4 * np.pi * (R**2)),
    N0_hi / (4 * np.pi * (R**2)),
)  # channel densities at crossover
om_hi = 1e5
S_shot_voltage_lo = S_shot_lo / (2 * np.pi * (sig_lo**2) * c) ** 2
S_shot_voltage_hi = S_shot_hi / (2 * np.pi * (sig_hi**2) * c) ** 2
t_meas = 1e-5  # measurement time (s)
dV_shot_lo = np.sqrt(S_shot_voltage_lo / t_meas)
dV_shot_hi = np.sqrt(S_shot_voltage_hi / t_meas)
dV_ref = 1e-2  # reference voltage noise for comparison (10 mV)
t_meas_lo = S_shot_voltage_lo / (dV_ref**2)
t_meas_hi = S_shot_voltage_hi / (dV_ref**2)

# Cell measurements
S_cell = 8 * np.pi * (R**2) * c * tau * kT
S_cell_voltage = S_cell / (4 * np.pi * (R**2) * c) ** 2
dV_cell = np.sqrt(S_cell_voltage / t_meas)


def sf(x, n=4):
    return f"{x:.{n}g}"


print(
    f"The ratio of shot noise to JN noise is: {sf(S_shot_lo / S_JN_lo)} - {sf(S_shot_hi / S_JN_hi)}"
)
print(f"The resolution per Debye time is: {sf(V0_lo * 1e3)} - {sf(V0_hi * 1e3)} mV")
print(
    f"The measurement time to reach 10 mV noise is: {sf(t_meas_lo * 1e6)} - {sf(t_meas_hi * 1e6)} μs"
)
print(f"The crossover number of channels is: {sf(N0_lo)} - {sf(N0_hi)}")
print(
    f"The low-frequency channel density at crossover is: {sf(density_lo * 1e-12)} - {sf(density_hi * 1e-12)} channels/μm²"
)
print(
    f"The high-frequency channel density at crossover is: {sf(density_lo * (1 + (om_hi * tau) ** 2) * 1e-12)} - {sf(density_hi * (1 + (om_hi * tau) ** 2) * 1e-12)} channels/μm²"
)
print(f"The voltage noise for a cell measurement is: {sf(dV_cell * 1e3)} mV")
print(
    f"The voltage noise for a sensor measurement is: {sf(dV_shot_lo * 1e3)} - {sf(dV_shot_hi * 1e3)} mV"
)

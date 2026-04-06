"""
    This script numerically integrates measurement noise and compares it to
    shot noise and Johnson-Nyquist noise.
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
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from tqdm import tqdm

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
width = 2.5
height = 2.3

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

# Model parameters
import os
from params import (
    e_m,
    n_herm,
    N_params,
    s_log_min,
    s_log_max,
    d_log_min,
    d_log_max,
    h2d_log_min,
    h2d_log_max,
    FIGURES_DIR,
)


# ------------------------------------------------------------------------------
# DEFINE INTEGRAND
# ------------------------------------------------------------------------------
# The integral defined here is the dimensionless integral obtained at the end
# of section S3.5
def u(x, d):
    return (1.0 + (1 / e_m) * np.tanh(x * d / 2)) ** (-1)


def G(x, q, h, d):
    # Components of B
    B1 = (h**2) / ((q * h) ** 2 - 1.0)
    B2 = ((1.0 + q * h) * (u(x, d) / (x + q) - q)) ** (-1)
    B3 = (q**2) - (x**2) - u(x, d) * (1 - x * h)

    # Components of C
    C1 = -2 * B1 * (q * (h**2)) / ((q * h) ** 2 - 1.0)
    C2 = -(B2**2) * (
        h * (u(x, d) / (x + q) - q) - (1 + q * h) * (u(x, d) / ((x + q) ** 2) + 1)
    )
    C3 = 2 * q

    # Derivative of components
    G1 = C1 * (2 * B2 * B3 + (1 - (x * h) ** 2) / h)
    G2 = 2 * B1 * (C2 * B3 + B2 * C3)

    return (G1 + G2) * x / q


def unweighted_integrand(x, h, d):
    q = np.sqrt(1 + x**2)
    G_int = G(x, q, h, d)
    return G_int


# ------------------------------------------------------------------------------
# INTEGRATION FUNCTIONS
# ------------------------------------------------------------------------------
# Define range of h to search
h_vals = np.logspace(h2d_log_min, h2d_log_max, N_params)


def hermite_half_range_integral(F, h, s, d, n_herm=n_herm):
    """
    Uses n-point Gauss–Hermite on (-inf, inf) with weight exp(-t^2)
    to compute: ∫_0^∞ exp(-(s x)^2) F(x,h) dx.

    Change of variable: t = s x  => dx = dt / s
    Integral = (1/s) ∫_0^∞ exp(-t^2) F(t/s, h) dt

    We approximate ∫_0^∞ exp(-t^2) G(t) dt = 0.5 ∫_{-∞}^{∞} exp(-t^2) G(|t|) dt
    and apply Hermite to the even extension G(|t|).
    """
    # Hermite nodes and weights for ∫ e^{-t^2} g(t) dt
    x, w = np.polynomial.hermite.hermgauss(n_herm)
    # Even extension via |x| to cover only t >= 0
    G = F(np.abs(x) / s, h, d)
    return (1.0 / s) * 0.5 * np.sum(w * G)


def noise_scaling(s, d):
    # Get raw noise
    S_vals = np.array(
        [hermite_half_range_integral(unweighted_integrand, h, s, d) for h in h_vals]
    )
    S_0 = hermite_half_range_integral(unweighted_integrand, 1.0, s, d)
    # Get SNR values
    chi_vals = h_vals / (1 + h_vals)
    SNR_vals = (chi_vals**2) / S_vals
    SNR_vals /= 1.0 / (4 * S_0)
    # Get optimal height
    idx_opt = np.argmax(SNR_vals)
    h_opt = h_vals[idx_opt]
    S_opt = S_vals[idx_opt]
    return S_vals, SNR_vals, h_opt, S_opt


# ------------------------------------------------------------------------------
# PARAMETER SWEEPS
# ------------------------------------------------------------------------------
# Parameter range
s_vals = np.logspace(s_log_min, s_log_max, N_params)
d_vals = np.logspace(d_log_min, d_log_max, N_params)

# Get optimal noise and individual components
h_opt_ar = np.zeros((N_params, N_params))
S_ar = np.zeros((N_params, N_params))
S_shot_ar = np.zeros((N_params, N_params))
S_JN_ar = np.zeros((N_params, N_params))

for i, s in enumerate(tqdm(s_vals)):
    for j, d in enumerate(d_vals):
        S_vals, SNR_vals, h_opt, S_opt = noise_scaling(s, d)
        h_opt_ar[i, j] = h_opt
        S_ar[i, j] = S_opt
        S_shot_ar[i, j] = 1.0 / (8 * (s**2))
        S_JN_ar[i, j] = (np.sqrt(np.pi) / (2 * s)) * ((1.0 + d / (2 * e_m)) ** (-2))


# ------------------------------------------------------------------------------
# BUILDING THE HEATMAPS
# ------------------------------------------------------------------------------
def geom_edges(centers):
    centers = np.asarray(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


s_edges = geom_edges(s_vals)
d_edges = geom_edges(d_vals)

# FIGURE: Optimal height heatmap-----------------------------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

pcm = ax.pcolormesh(
    s_edges,  # 1D, length 21
    d_edges,  # 1D, length 21
    h_opt_ar.T,  # shape (20, 20)
    norm=LogNorm(),
    cmap="viridis",
    shading="auto",
    rasterized=True,
)

ax.set_xscale("log")
ax.set_yscale("log")

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r"Optimal height $h^*/l_D$", rotation=270, labelpad=15)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

plt.savefig(
    os.path.join(FIGURES_DIR, "optimal_height_sweep.svg"), dpi=600, bbox_inches="tight"
)
plt.savefig(
    os.path.join(FIGURES_DIR, "optimal_height_sweep.pdf"), dpi=600, bbox_inches="tight"
)
plt.show()

# FIGURE: Optimal noise vs shot noise------------------------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

# Center colormap around 1
S_min = np.nanmin(S_ar / S_shot_ar)
S_max = np.nanmax(S_ar / S_shot_ar)

R = max(S_max / 1, 1 / S_min)

vmin = 1 / R
vmax = 1 * R

# Step 2: choose a diverging colormap
cmap = plt.get_cmap("seismic")  # or seismic, etc.


# Step 3: remap colormap so midpoint is 1
class LogMidpointNormalize(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)

    def __call__(self, value, clip=None):
        # log-normalization
        log_value = np.log(value)
        log_min = np.log(self.vmin)
        log_max = np.log(self.vmax)
        log_mid = np.log(self.midpoint)

        # Scale log values so midpoint → 0.5
        result = (log_value - log_min) / (log_max - log_min)

        # Compress/stretch so midpoint becomes exactly 0.5
        mid = (log_mid - log_min) / (log_max - log_min)
        result = np.where(
            value < self.midpoint,
            0.5 * result / mid,
            0.5 + 0.5 * (result - mid) / (1 - mid),
        )
        return result


norm = LogMidpointNormalize(vmin=vmin, vmax=vmax, midpoint=1)

pcm = ax.pcolormesh(
    s_edges,  # 1D, length 21
    d_edges,  # 1D, length 21
    (S_ar / S_shot_ar).T,  # shape (20, 20)
    norm=norm,
    cmap=cmap,
    shading="auto",
    rasterized=True,
)

d_crit = 2 * e_m * (-1.0 + np.sqrt(s_vals * np.sqrt(np.pi) / 2))
ax.plot(s_vals, d_crit, c="k", lw=1, ls="--", label="$\sigma^*(d)$")
ax.legend()

ax.set_ylim(min(d_vals), max(d_vals))
ax.set_xscale("log")
ax.set_yscale("log")

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fluctuations $S^M(\omega)/S^M_{\text{shot}}(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

plt.savefig(
    os.path.join(FIGURES_DIR, "shot_noise_sweep.svg"), dpi=600, bbox_inches="tight"
)
plt.savefig(
    os.path.join(FIGURES_DIR, "shot_noise_sweep.pdf"), dpi=600, bbox_inches="tight"
)
plt.show()

# FIGURE: Optimal noise vs JN noise------------------------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

# Center colormap around 1
S_min = np.nanmin(S_ar / S_JN_ar)
S_max = np.nanmax(S_ar / S_JN_ar)

R = max(S_max / 1, 1 / S_min)

vmin = 1 / R
vmax = 1 * R

# Step 2: choose a diverging colormap
cmap = plt.get_cmap("seismic")  # or seismic, etc.


# Step 3: remap colormap so midpoint is 1
class LogMidpointNormalize(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)

    def __call__(self, value, clip=None):
        # log-normalization
        log_value = np.log(value)
        log_min = np.log(self.vmin)
        log_max = np.log(self.vmax)
        log_mid = np.log(self.midpoint)

        # Scale log values so midpoint → 0.5
        result = (log_value - log_min) / (log_max - log_min)

        # Compress/stretch so midpoint becomes exactly 0.5
        mid = (log_mid - log_min) / (log_max - log_min)
        result = np.where(
            value < self.midpoint,
            0.5 * result / mid,
            0.5 + 0.5 * (result - mid) / (1 - mid),
        )
        return result


norm = LogMidpointNormalize(vmin=vmin, vmax=vmax, midpoint=1)

pcm = ax.pcolormesh(
    s_edges,  # 1D, length 21
    d_edges,  # 1D, length 21
    (S_ar / S_JN_ar).T,  # shape (20, 20)
    norm=norm,
    cmap=cmap,
    shading="auto",
    rasterized=True,
)

ax.plot(s_vals, d_crit, c="k", lw=1, ls="--", label="$\sigma^*(d)$")
ax.legend()

ax.set_ylim(min(d_vals), max(d_vals))
ax.set_xscale("log")
ax.set_yscale("log")

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fluctuations $S^M(\omega)/S^M_{\text{JN}}(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

plt.savefig(
    os.path.join(FIGURES_DIR, "JN_noise_sweep.svg"), dpi=600, bbox_inches="tight"
)
plt.savefig(
    os.path.join(FIGURES_DIR, "JN_noise_sweep.pdf"), dpi=600, bbox_inches="tight"
)
plt.show()

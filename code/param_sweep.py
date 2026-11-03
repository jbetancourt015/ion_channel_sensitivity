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
from noise_integral import noise_scaling

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
    s_log_min_r,
    s_log_max_r,
    d_log_min_r,
    d_log_max_r,
    FIGURES_DIR,
    figure_path,
)

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
SNR_rel_ar = np.zeros((N_params, N_params))

for i, s in enumerate(tqdm(s_vals)):
    for j, d in enumerate(d_vals):
        S_vals, SNR_vals, h_opt, S_opt = noise_scaling(s, d)
        h_opt_ar[i, j] = h_opt
        S_ar[i, j] = S_opt
        SNR_rel_ar[i, j] = np.max(SNR_vals)  # SNR(h*)/SNR(l_D); SNR_vals normalized to 1 at h=l_D
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
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r"Optimal height $h^*/l_D$", rotation=270, labelpad=15)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# PRL Fig S2(b) -> figures/si_optimal_height/raw/
plt.savefig(
    figure_path("si_optimal_height", "optimal_height_sweep.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("si_optimal_height", "optimal_height_sweep.pdf"),
    dpi=600,
    bbox_inches="tight",
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
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fluctuations $S^M(\omega)/S^M_{\text{shot}}(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# PRL Fig 1(b) -> figures/noise_numerics/raw/  (also reused by SI Fig S3(a))
plt.savefig(
    figure_path("noise_numerics", "shot_noise_sweep.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("noise_numerics", "shot_noise_sweep.pdf"),
    dpi=600,
    bbox_inches="tight",
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
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fluctuations $S^M(\omega)/S^M_{\text{JN}}(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# PRL Fig 1(c) -> figures/noise_numerics/raw/  (also reused by SI Fig S3(b))
plt.savefig(
    figure_path("noise_numerics", "JN_noise_sweep.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("noise_numerics", "JN_noise_sweep.pdf"),
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# FIGURE: Shot noise vs optimal noise (flipped) ----------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

pcm = ax.pcolormesh(
    s_edges,
    d_edges,
    (S_shot_ar / S_ar).T,
    norm=LogNorm(),  # colorbar range set automatically
    cmap="viridis",
    shading="auto",
    rasterized=True,
)

ax.plot(s_vals, d_crit, c="k", lw=1, ls="--", label="$\sigma^*(d)$")
ax.legend()

ax.set_ylim(min(d_vals), max(d_vals))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"$S^M_{\text{shot}}(\omega)/S^M(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# PRL Fig 1(b), flipped ratio -> figures/noise_numerics/raw/
plt.savefig(
    figure_path("noise_numerics", "shot_noise_sweep_flipped.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("noise_numerics", "shot_noise_sweep_flipped.pdf"),
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# FIGURE: JN noise vs optimal noise (flipped) -----------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

pcm = ax.pcolormesh(
    s_edges,
    d_edges,
    (S_JN_ar / S_ar).T,
    norm=LogNorm(),  # colorbar range set automatically
    cmap="viridis",
    shading="auto",
    rasterized=True,
)

ax.plot(s_vals, d_crit, c="k", lw=1, ls="--", label="$\sigma^*(d)$")
ax.legend()

ax.set_ylim(min(d_vals), max(d_vals))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"$S^M_{\text{JN}}(\omega)/S^M(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# PRL Fig 1(c), flipped ratio -> figures/noise_numerics/raw/
plt.savefig(
    figure_path("noise_numerics", "JN_noise_sweep_flipped.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("noise_numerics", "JN_noise_sweep_flipped.pdf"),
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# FIGURE: Relative SNR SNR(h*)/SNR(l_D)---------------------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

pcm = ax.pcolormesh(
    s_edges,
    d_edges,
    SNR_rel_ar.T,
    norm=LogNorm(),
    cmap="viridis",
    shading="auto",
    rasterized=True,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r"Relative SNR $\mathrm{SNR}(h^*)/\mathrm{SNR}(l_D)$", rotation=270, labelpad=15)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# not in the manuscript -> figures/si_relative_snr/raw/
plt.savefig(
    figure_path("si_relative_snr", "relative_SNR.svg"), dpi=600, bbox_inches="tight"
)
plt.savefig(
    figure_path("si_relative_snr", "relative_SNR.pdf"), dpi=600, bbox_inches="tight"
)
plt.show()

# ------------------------------------------------------------------------------
# RESTRICTED SWEEP: σ/l_D and d/l_D from 1 to 10
# ------------------------------------------------------------------------------
s_vals_r = np.logspace(s_log_min_r, s_log_max_r, N_params)
d_vals_r = np.logspace(d_log_min_r, d_log_max_r, N_params)

S_ar_r = np.zeros((N_params, N_params))
S_shot_ar_r = np.zeros((N_params, N_params))
S_JN_ar_r = np.zeros((N_params, N_params))

for i, s in enumerate(tqdm(s_vals_r)):
    for j, d in enumerate(d_vals_r):
        S_vals, SNR_vals, h_opt, S_opt = noise_scaling(s, d)
        S_ar_r[i, j] = S_opt
        S_shot_ar_r[i, j] = 1.0 / (8 * (s**2))
        S_JN_ar_r[i, j] = (np.sqrt(np.pi) / (2 * s)) * ((1.0 + d / (2 * e_m)) ** (-2))

s_edges_r = geom_edges(s_vals_r)
d_edges_r = geom_edges(d_vals_r)

# FIGURE: Restricted shot noise sweep (no dashed line) ------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

ratio_shot_r = S_ar_r / S_shot_ar_r
S_min = np.nanmin(ratio_shot_r)
S_max = np.nanmax(ratio_shot_r)
R = max(S_max / 1, 1 / S_min)
norm = LogMidpointNormalize(vmin=1 / R, vmax=1 * R, midpoint=1)

pcm = ax.pcolormesh(
    s_edges_r,
    d_edges_r,
    ratio_shot_r.T,
    norm=norm,
    cmap=plt.get_cmap("seismic"),
    shading="auto",
    rasterized=True,
)

ax.set_ylim(min(d_vals_r), max(d_vals_r))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fluctuations $S^M(\omega)/S^M_{\text{shot}}(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# not in the manuscript -> figures/si_noise_param_sweep_restricted/raw/
plt.savefig(
    figure_path("si_noise_param_sweep_restricted", "shot_noise_sweep_restricted.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("si_noise_param_sweep_restricted", "shot_noise_sweep_restricted.pdf"),
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# FIGURE: Restricted JN noise sweep (no dashed line) --------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

ratio_JN_r = S_ar_r / S_JN_ar_r
S_min = np.nanmin(ratio_JN_r)
S_max = np.nanmax(ratio_JN_r)
R = max(S_max / 1, 1 / S_min)
norm = LogMidpointNormalize(vmin=1 / R, vmax=1 * R, midpoint=1)

pcm = ax.pcolormesh(
    s_edges_r,
    d_edges_r,
    ratio_JN_r.T,
    norm=norm,
    cmap=plt.get_cmap("seismic"),
    shading="auto",
    rasterized=True,
)

ax.set_ylim(min(d_vals_r), max(d_vals_r))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fluctuations $S^M(\omega)/S^M_{\text{JN}}(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# not in the manuscript -> figures/si_noise_param_sweep_restricted/raw/
plt.savefig(
    figure_path("si_noise_param_sweep_restricted", "JN_noise_sweep_restricted.svg"),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path("si_noise_param_sweep_restricted", "JN_noise_sweep_restricted.pdf"),
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# FIGURE: Restricted shot noise sweep, flipped ratio -------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

pcm = ax.pcolormesh(
    s_edges_r,
    d_edges_r,
    (S_shot_ar_r / S_ar_r).T,
    norm=LogNorm(),  # colorbar range set automatically
    cmap="viridis",
    shading="auto",
    rasterized=True,
)

ax.set_ylim(min(d_vals_r), max(d_vals_r))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"$S^M_{\text{shot}}(\omega)/S^M(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# not in the manuscript -> figures/si_noise_param_sweep_restricted/raw/
plt.savefig(
    figure_path(
        "si_noise_param_sweep_restricted",
        "shot_noise_sweep_restricted_flipped.svg",
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path(
        "si_noise_param_sweep_restricted",
        "shot_noise_sweep_restricted_flipped.pdf",
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# FIGURE: Restricted JN noise sweep, flipped ratio --------------------------
fig, ax = plt.subplots(figsize=(0.9 * width, 0.9 * height))

pcm = ax.pcolormesh(
    s_edges_r,
    d_edges_r,
    (S_JN_ar_r / S_ar_r).T,
    norm=LogNorm(),  # colorbar range set automatically
    cmap="viridis",
    shading="auto",
    rasterized=True,
)

ax.set_ylim(min(d_vals_r), max(d_vals_r))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"$S^M_{\text{JN}}(\omega)/S^M(\omega)$", rotation=270, labelpad=15
)

ax.set_xlabel(r"Sensor width $\sigma/l_D$")
ax.set_ylabel(r"Membrane thickness $d/l_D$")

# not in the manuscript -> figures/si_noise_param_sweep_restricted/raw/
plt.savefig(
    figure_path(
        "si_noise_param_sweep_restricted",
        "JN_noise_sweep_restricted_flipped.svg",
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    figure_path(
        "si_noise_param_sweep_restricted",
        "JN_noise_sweep_restricted_flipped.pdf",
    ),
    dpi=600,
    bbox_inches="tight",
)
plt.show()
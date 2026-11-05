"""
    Exact numerics vs. asymptotics for the equilibrium noise of a localized
    charge sensor next to a membrane.
        (a) counting noise vs. sensor height h at fixed sigma
        (b) capacitive and counting noise vs. sigma at h = l_D
    All noise is normalized by S_ref = (1/8) 2 pi eps tau_D kT sigma^2 / l_D.
    Numerics follow calculations/markdowns/noise_numerics_guide.md and are
    cached in repo/numerical_results/.
-------------------------------------------------------------------------------
created on:
    Tue 3 Nov 2026
-------------------------------------------------------------------------------
last change:
    Wed 4 Nov 2026
-------------------------------------------------------------------------------
notes:
    Dimensionless units: lengths in l_D, times in tau_D, energies in kT.
    Set RECOMPUTE = True to ignore the cached results.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import os
import logging
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from tqdm import tqdm

from params import sig, l_D, d, e_m, figure_path, results_path
from noise_reference import (
    S_cap,
    S_count,
    S_count_shallow,
    S_count_deep,
    c_eff,
)

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
fig_dir = "si_asymptotic_numerics"  # figures/<fig_dir>/raw/

# Reference size (Nature single-column width in mm) for scaling panel dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from the mm reference), width / height = aspect
aspect = 1.4
width_md = 0.5 * pg_width / mm_to_in
height_md = width_md / aspect

# Fixed margins for consistent axes size across both single-panel figures
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

RECOMPUTE = False

# ── Dimensionless parameters ──────────────────────────────────────────────────
nm = 1e-9 / l_D  # 1 nm in units of l_D
s0 = sig / l_D  # sensor width for panel a
r_eps = 1.0 / e_m  # eps / eps_m
d_norm = d / l_D  # membrane thickness
l_th = 0.5 * r_eps * d_norm  # threshold length
c_e = c_eff(r_eps, d_norm)


def S_bulk(s):
    """(1/8) 2 pi eps tau_D kT sigma^2 / l_D in code units."""
    return 0.25 * np.pi * s**2


def _count(args):
    s, h = args
    return S_count(s, h, r_eps, d_norm)


def _cap_count(args):
    s, h = args
    return S_cap(s, h, r_eps, d_norm), S_count(s, h, r_eps, d_norm)


def load_or_compute(name, build):
    """Return the cached npz dict at repo/numerical_results/<name>, or build it."""
    path = results_path(name)
    if os.path.exists(path) and not RECOMPUTE:
        return dict(np.load(path))
    out = build()
    np.savez(path, **out)
    return out


# ── Panel a: counting noise vs. h at fixed sigma ──────────────────────────────
def build_height_scan():
    h_nm = np.unique(
        np.concatenate(
            [
                np.logspace(-1, np.log10(500), 62),
                np.logspace(np.log10(3), np.log10(15), 34),
            ]
        )
    )
    h = h_nm * nm
    with Pool() as pool:
        S = np.array(
            list(
                tqdm(
                    pool.imap(_count, [(s0, hi) for hi in h]),
                    total=len(h),
                    desc="a: S_count(h)",
                )
            )
        )
    return dict(h_nm=h_nm, h=h, S_count=S, s=s0, r_eps=r_eps, d_norm=d_norm)


# ── Panel b: noise vs. sigma at h = l_D ───────────────────────────────────────
def build_sigma_scan():
    sig_nm = np.logspace(0, np.log10(3000), 38)
    s = sig_nm * nm
    with Pool() as pool:
        res = np.array(
            list(
                tqdm(
                    pool.imap(_cap_count, [(si, 1.0) for si in s]),
                    total=len(s),
                    desc="b: S(sigma)",
                )
            )
        )
    return dict(
        sig_nm=sig_nm,
        s=s,
        S_cap=res[:, 0],
        S_count=res[:, 1],
        h=1.0,
        r_eps=r_eps,
        d_norm=d_norm,
    )


if __name__ == "__main__":
    ha = load_or_compute("asymptotic_counting_vs_h.npz", build_height_scan)
    sb = load_or_compute("asymptotic_noise_vs_sigma.npz", build_sigma_scan)

    # ── Sanity checks ─────────────────────────────────────────────────────
    # local minimum at h ~ sigma (S_count -> 0 as h -> 0 is a separate, trivial limit)
    deep = ha["h"] > 2.0
    i_min = np.flatnonzero(deep)[np.argmin(ha["S_count"][deep])]
    print(
        f"a: sigma = {s0 / nm:.2f} nm; local minimum at h = {ha['h_nm'][i_min]:.2f} nm, "
        f"S_count/S_bulk = {ha['S_count'][i_min] / S_bulk(s0):.4f}"
    )
    print(
        f"b: l_th = {l_th / nm:.1f} nm; S_count/S_bulk at largest sigma = "
        f"{sb['S_count'][-1] / S_bulk(sb['s'][-1]):.4f}"
    )
    print(
        f"b: S_cap/S_bulk at largest sigma = {sb['S_cap'][-1] / S_bulk(sb['s'][-1]):.3f} "
        f"vs asymptote {16 * np.sqrt(np.pi) * sb['s'][-1] * c_e**2:.3f}"
    )

    def new_panel():
        fig, ax = plt.subplots(figsize=(width_md, height_md))
        ax.spines[["right", "top"]].set_visible(False)
        plt.subplots_adjust(**fig_margins_md)
        return fig, ax

    # panel a
    norm = S_bulk(s0)
    hh = np.logspace(np.log10(ha["h"].min()), np.log10(ha["h"].max()), 400)
    fig_a, axa = new_panel()
    axa.plot(
        ha["h_nm"],
        ha["S_count"] / norm,
        "-",
        lw=2,
        color="k",
        label="Numerical",
        zorder=3,
    )
    axa.plot(
        hh / nm,
        S_count_shallow(s0, hh) / norm,
        "--",
        lw=1,
        color=con_colors[2],
        label="Shallow expansion",
    )
    axa.plot(
        hh / nm,
        S_count_deep(s0, hh) / norm,
        "-.",
        lw=1,
        color=con_colors[4],
        label="Deep expansion",
    )
    for x, lab in [(1 / nm, r"$l_D$"), (s0 / nm, r"$\sigma$")]:  # positions in nm
        # stop below the legend (ymax in axes fraction)
        axa.axvline(x, ymax=0.65, lw=0.5, color="0.5", ls="--", zorder=0)
        axa.text(x * 1.08, 0.015, lab, color="0.4", fontsize=8, va="bottom", ha="left")
    axa.set_xscale("log")
    axa.set_xlim(ha["h_nm"].min(), ha["h_nm"].max())
    axa.set_yscale("log")
    axa.set_ylim(1e-2, 1e3)
    axa.set_xlabel(r"Sensor height $h$ (nm)")
    axa.set_ylabel(r"$S^Q_{\mathrm{count}}\,/\,S^Q_{\mathrm{ref}}$")
    axa.legend(frameon=False)
    fig_a.savefig(figure_path(fig_dir, "asymptotic_counting_vs_h.svg"), dpi=600)

    # panel b
    s = sb["s"]
    sn = sb["sig_nm"]
    ss = np.logspace(np.log10(s.min()), np.log10(s.max()), 400)
    fig_b, axb = new_panel()
    axb.plot(
        sn,
        sb["S_cap"] / S_bulk(s),
        "-",
        lw=2,
        color=con_colors[3],
        label="Capacitive",
        zorder=3,
    )
    axb.plot(
        sn,
        sb["S_count"] / S_bulk(s),
        "-",
        lw=2,
        color=con_colors[0],
        label="Counting",
        zorder=3,
    )
    axb.plot(ss / nm, np.ones_like(ss), "--", lw=1, color=con_colors[0])
    axb.plot(
        ss / nm,
        16 * np.sqrt(np.pi) * ss * c_e**2,
        "--",
        lw=1,
        color=con_colors[3],
    )
    axb.set_xscale("log")
    axb.set_xlim(sn.min(), sn.max())
    axb.set_yscale("log")
    axb.set_ylim(1e-2, 1e2)
    axb.set_xlabel(r"Sensor width $\sigma$ (nm)")
    axb.set_ylabel(r"$S^Q\,/\,S^Q_{\mathrm{ref}}$")
    handles, _ = axb.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="--", lw=1, color="k", label="Asymptotics"))
    axb.legend(handles=handles, frameon=False, loc="upper left")
    fig_b.savefig(figure_path(fig_dir, "asymptotic_noise_vs_sigma.svg"), dpi=600)
    plt.show()

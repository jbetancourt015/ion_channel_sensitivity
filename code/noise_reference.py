"""
Reference implementation: equilibrium noise of charge-sensing kernels next to a membrane
(linearized Poisson-Nernst-Planck model, odd-parity sector).

Units (dimensionless):
    lengths in l_D, times in tau_D, permittivity in eps (bulk), energies in k_B T.
    Then alpha = eps/tau_D = 1 and D/alpha = l_D^2/eps = 1.
    Charge-counting noise S^M is in units of eps * tau_D * k_B T * l_D.
    Voltage-referred noise S^V is in units of k_B T * tau_D / (eps * l_D).

Membrane parameters:
    r_eps = eps/eps_m   (bulk / membrane permittivity)
    d     = membrane thickness (in l_D)
    l_th  = r_eps * d / 2  (threshold length l^th_o, in l_D)

Localized kernel: Psi(x, z) = exp(-|x|^2 / 2 sigma^2) * exp(-|z|/h) * sgn(z),  s = sigma/l_D.
"""
import numpy as np
from scipy import integrate, linalg

SQPI = np.sqrt(np.pi)

# ------------------------------------------------------------------ membrane and modes
def w_o(k, r_eps, d):
    """Odd-parity membrane factor w_o(k) = [1 + (eps/eps_m) tanh(k d / 2)]^(-1)."""
    return 1.0 / (1.0 + r_eps * np.tanh(0.5 * k * d))


def bound_mode(k, r_eps, d):
    """Odd bound mode at lateral wavenumber k > 0.
    Returns decay length l_o, relaxation time tau_b, and normalization N_b.
    Profile (z > 0, measured from the membrane face): N_b^(-1/2) exp(-z / l_o)."""
    w = w_o(k, r_eps, d)
    u = 0.5 * (-k + np.sqrt(k * k + 4.0 * w))        # u = l_D / l_o
    l_o = 1.0 / u
    tau_b = 1.0 / (1.0 - w + k * (k + u))
    N_b = l_o**2 * (2.0 + k * l_o) / (k * tau_b * (1.0 + k * l_o)**2)
    return l_o, tau_b, N_b


def osc_mode(k, kz, r_eps, d):
    """Odd oscillatory mode (k, kz). Profile (z > 0): N_o^(-1/2) sin(kz z + phi), phi in (-pi/2, 0].
    Returns phi, relaxation time tau_o, and normalization N_o (Dirac-delta normalized in kz)."""
    w = w_o(k, r_eps, d)
    K2 = k * k + kz * kz
    phi = -np.arctan((kz / k) * (1.0 + K2 / w))
    tau_o = 1.0 / (1.0 + K2)
    N_o = np.pi / (tau_o * K2)                       # note the K^2 = k^2 + kz^2 factor
    return phi, tau_o, N_o


# ------------------------------------------------------------------ vertical overlaps (localized kernel)
def V_bound(k, h, r_eps, d):
    """Vertical overlap of exp(-|z|/h) sgn(z) with the normalized odd bound mode (both half-spaces)."""
    l_o, tau_b, N_b = bound_mode(k, r_eps, d)
    return 2.0 * h * l_o / (h + l_o) / np.sqrt(N_b), tau_b


def V_osc(k, kz, h, r_eps, d):
    """Vertical overlap of exp(-|z|/h) sgn(z) with the normalized odd oscillatory mode."""
    phi, tau_o, N_o = osc_mode(k, kz, r_eps, d)
    return 2.0 * h * (np.sin(phi) + kz * h * np.cos(phi)) / (1.0 + (kz * h)**2) / np.sqrt(N_o), tau_o


# ------------------------------------------------------------------ per-k integrands
def A_cap(k, h, r_eps, d, omega=0.0, static=False):
    """Bound-mode contribution at wavenumber k: |V_b|^2 tau_b/(1 + omega^2 tau_b^2)  (or |V_b|^2 if static)."""
    V, tau = V_bound(k, h, r_eps, d)
    return V * V if static else V * V * tau / (1.0 + (omega * tau)**2)


def kz_breakpoints(h):
    return sorted({0.0, 0.1 / h, 1.0 / h, 10.0 / h, 1.0, 10.0})


def A_count(k, h, r_eps, d, omega=0.0, static=False):
    """Oscillatory-mode contribution at wavenumber k: integral over kz of |V_o|^2 tau_o/(1 + omega^2 tau_o^2)."""
    def f(kz):
        V, tau = V_osc(k, kz, h, r_eps, d)
        return V * V if static else V * V * tau / (1.0 + (omega * tau)**2)
    pts = kz_breakpoints(h)
    tot = sum(integrate.quad(f, a, b, limit=300)[0] for a, b in zip(pts[:-1], pts[1:]))
    return tot + integrate.quad(f, pts[-1], np.inf, limit=300)[0]


# ------------------------------------------------------------------ noise spectra (Gaussian lateral profile)
def S_cap(s, h, r_eps, d, omega=0.0):
    """Capacitive noise S^M_cap(omega) = 4 pi s^4 int_0^inf dk k exp(-k^2 s^2) A_cap(k)."""
    f = lambda k: k * np.exp(-(k * s)**2) * A_cap(k, h, r_eps, d, omega)
    return 4.0 * np.pi * s**4 * integrate.quad(f, 1e-12, 8.0 / s, limit=300, epsrel=1e-8)[0]


def S_count(s, h, r_eps, d, omega=0.0):
    """Counting noise S^M_count(omega) = 4 pi s^4 int_0^inf dk k exp(-k^2 s^2) A_count(k)."""
    f = lambda k: k * np.exp(-(k * s)**2) * A_count(k, h, r_eps, d, omega)
    return 4.0 * np.pi * s**4 * integrate.quad(f, 1e-12, 8.0 / s, limit=200, epsrel=1e-6)[0]


# ------------------------------------------------------------------ charged plate (field sensor in the membrane)
def plate_prefactor(k, r_eps, d):
    """w_o(k)/cosh(k d/2) = 1/[cosh(k d/2) + (eps/eps_m) sinh(k d/2)]."""
    x = 0.5 * k * d
    return 1.0 / (np.cosh(x) + r_eps * np.sinh(x))


def A_plate_cap(k, r_eps, d):
    l_o, tau_b, N_b = bound_mode(k, r_eps, d)
    V = plate_prefactor(k, r_eps, d) * 2.0 * l_o / (1.0 + k * l_o) / np.sqrt(N_b)
    return V * V * tau_b


def A_plate_count(k, r_eps, d):
    pref = plate_prefactor(k, r_eps, d)
    def f(kz):
        phi, tau_o, N_o = osc_mode(k, kz, r_eps, d)
        V = pref * 2.0 * (k * np.sin(phi) + kz * np.cos(phi)) / (k * k + kz * kz) / np.sqrt(N_o)
        return V * V * tau_o
    return integrate.quad(f, 0.0, 10.0, limit=300)[0] + integrate.quad(f, 10.0, np.inf, limit=300)[0]


def c_eff(r_eps, d):
    """Effective capacitance per area of membrane + two Debye layers, in units of eps/l_D: 1/[2 (1 + l_th)]."""
    return 1.0 / (2.0 * (1.0 + 0.5 * r_eps * d))


def SV_plate_cap(s, r_eps, d):
    """Capacitive voltage-referred noise of the plate, (eps_m/(Q c_eff))^2 S^H_cap, units k_B T tau_D/(eps l_D)."""
    f = lambda k: k * np.exp(-(k * s)**2) * A_plate_cap(k, r_eps, d)
    return integrate.quad(f, 1e-12, 8.0 / s, limit=300, epsrel=1e-9)[0] / (4.0 * np.pi * c_eff(r_eps, d)**2)


def SV_plate_count(s, r_eps, d):
    """Counting voltage-referred noise of the plate, (eps_m/(Q c_eff))^2 S^H_count, units k_B T tau_D/(eps l_D)."""
    f = lambda k: k * np.exp(-(k * s)**2) * A_plate_count(k, r_eps, d)
    return integrate.quad(f, 1e-12, 8.0 / s, limit=200, epsrel=1e-6)[0] / (4.0 * np.pi * c_eff(r_eps, d)**2)


# ------------------------------------------------------------------ gain, voltage noise, closed forms
def gain(s, h, r_eps, d):
    """Gain g = d<M>/dV = 4 pi sigma^2 c_eff eta/(1 + eta), eta = h/l_D."""
    return 4.0 * np.pi * s**2 * c_eff(r_eps, d) * h / (1.0 + h)

def S_shot(s):                 return 0.25 * np.pi * s**2
def S_JN(s, r_eps, d):         return 4.0 * np.pi**1.5 * s**3 * c_eff(r_eps, d)**2
def S_count_shallow(s, h):     return 2.0 * np.pi * h**2 / (1.0 + h)**3 * s**2
def S_count_deep(s, h):        return 2.0 * np.pi * (s**2 / h - SQPI * s + h)
def SV_JN(s):                  return 1.0 / (SQPI * s)        # k_B T/(pi^(1/2) alpha sigma)


def optimal_height(s, r_eps, d, n=26):
    """Height minimizing S^V = (S_cap + S_count)/g^2 at omega -> 0. Returns (h*, SNR(h*)/SNR(h = l_D))."""
    hs = np.logspace(np.log10(0.05), np.log10(max(20.0 * s, 20.0)), n)
    SV = lambda h: (S_cap(s, h, r_eps, d) + S_count(s, h, r_eps, d)) / gain(s, h, r_eps, d)**2
    v = np.array([SV(h) for h in hs])
    i = int(np.argmin(v))
    c = np.polyfit(np.log(hs[i - 1:i + 2]), np.log(v[i - 1:i + 2]), 2)
    hstar = float(np.exp(-c[1] / (2.0 * c[0])))
    return hstar, SV(1.0) / SV(hstar)


# ------------------------------------------------------------------ independent check: finite differences at fixed k
def fd_check(k, u_fun, r_eps, d, L=40.0, N=3000):
    """Returns (A, B) = (<u, C G^-1 C u>, <u, C u>) for the odd vertical profile u(z), z > 0,
    from a direct discretization (no mode decomposition). Compare with the mode sums
    A = A_cap + A_count and B = A_cap(static) + A_count(static)."""
    dz = L / N
    z = (np.arange(N) + 0.5) * dz
    w = w_o(k, r_eps, d)
    Z, Zp = np.meshgrid(z, z, indexing='ij')
    G = (np.exp(-k * np.abs(Z - Zp)) + (1.0 - 2.0 * w) * np.exp(-k * (Z + Zp))) / (2.0 * k)   # Poisson, odd sector
    Cinv = G * dz + np.eye(N)                                   # C^-1 = Phi[.] + (D/alpha) 1
    u = u_fun(z)
    rho = linalg.solve(Cinv, u, assume_a='sym')                 # rho = C u
    B = 2.0 * np.sum(u * rho) * dz
    r = (1.0 - 0.5 * k * dz) / (1.0 + 0.5 * k * dz)             # Robin condition dv/dz = -k v at z = L
    main = np.full(N, 2.0 / dz**2 + k * k)
    main[0] = 1.0 / dz**2 + k * k                               # Neumann (no flux) at z = 0
    main[-1] = (2.0 - r) / dz**2 + k * k
    off = np.full(N - 1, -1.0 / dz**2)
    ab = np.zeros((3, N)); ab[0, 1:] = off; ab[1] = main; ab[2, :-1] = off
    v = linalg.solve_banded((1, 1), ab, rho)                    # v = G^-1 C u   (G = -(d^2/dz^2 - k^2))
    A = 2.0 * np.sum(rho * v) * dz
    return A, B

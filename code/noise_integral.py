"""
    This script contains the numerical implementation of the noise integral.
-------------------------------------------------------------------------------
created on:
    Fri 24 Apr 2026
-------------------------------------------------------------------------------
last change:
    Fri 24 Apr 2026
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
from params import e_m, n_herm, h_log_min, h_log_max, n_points, h_vals


# ------------------------------------------------------------------------------
# DEFINE INTEGRAND
# ------------------------------------------------------------------------------
def gamma(x, d):
    return (1.0 + (1 / e_m) * np.tanh(x * d / 2)) ** (-1)


def G(x, q, h, d):
    # Components of B
    B1 = (h**2) / ((q * h) ** 2 - 1.0)
    B2 = ((1.0 + q * h) * (gamma(x, d) / (x + q) - q)) ** (-1)
    B3 = (q**2) - (x**2) - gamma(x, d) * (1 - x * h)

    # Components of C
    C1 = -2 * B1 * (q * (h**2)) / ((q * h) ** 2 - 1.0)
    C2 = -(B2**2) * (
        h * (gamma(x, d) / (x + q) - q)
        - (1 + q * h) * (gamma(x, d) / ((x + q) ** 2) + 1)
    )
    C3 = 2 * q

    # Derivative of components
    G1 = C1 * (2 * B2 * B3 + (1 - (x * h) ** 2) / h)
    G2 = 2 * B1 * (C2 * B3 + B2 * C3)

    return (G1 + G2) * x / q

def G_stable(x, q, h, d):
    # G_func = h * N_hat / E  where N_hat = N_raw/(qh-1), E = (1+qh)^2 * V.
    # Only (qh-1) divides N_raw, so this removes the singular factor at h=1, x->0.
    u  = gamma(x, d)
    A  = x + q
    V       = u / A - q
    V_prime = -u / A**2 - 1.0

    N_hat       = 2*q + 2*x**2*h + (1 - (x*h)**2)*V - 2*(1 - x*h)*u/A
    N_hat_prime = 1 + (x*h)**2 + u*(1 - x*h)**2 / A**2

    E       = (1.0 + q*h)**2 * V
    E_prime = (1.0 + q*h) * (2*h*V + (1.0 + q*h)*V_prime)

    G_prime = h * (N_hat_prime*E - N_hat*E_prime) / E**2
    return G_prime * x / q

def unweighted_integrand(x, h, d):
    q = np.sqrt(1 + x**2)
    G_int = G(x, q, h, d)
    return G_int


def unweighted_integrand_stable(x, h, d):
    q = np.sqrt(1 + x**2)
    return G_stable(x, q, h, d)


# ------------------------------------------------------------------------------
# INTEGRATION FUNCTION
# ------------------------------------------------------------------------------
# Define range of h to search
h_vals = np.logspace(h_log_min, h_log_max, n_points)

# -----------------------------
# (B) Gauss–Hermite (recommended if you can factor the Gaussian)
# -----------------------------
def fluctuation_integral(h, s, d, n_herm=n_herm):
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
    G = unweighted_integrand(np.abs(x) / s, h, d)
    return (1.0 / s) * 0.5 * np.sum(w * G)

# ------------------------------------------------------------------------------
# HEIGHT OPTIMIZATION
# ------------------------------------------------------------------------------
def noise_scaling(s, d):
    S_vals = np.array(
        [fluctuation_integral(h, s, d) for h in h_vals]
    )
    S_0 = fluctuation_integral(1.0, s, d)
    chi_vals = h_vals / (1 + h_vals)
    SNR_vals = (chi_vals**2) / S_vals
    SNR_vals /= 1.0 / (4 * S_0)
    idx_opt = np.argmax(SNR_vals)
    h_opt = h_vals[idx_opt]
    SNR_opt = SNR_vals[idx_opt]
    return S_vals, SNR_vals, h_opt, SNR_opt
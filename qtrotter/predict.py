"""Optimal Trotter step count r*: fitted to simulated error curves, and predicted analytically.

Both analytic papers minimize a Trotter term a / r^q plus a noise term b r, giving
    r* = (q a / b)^(1 / (q + 1)).

Knee & Munro (PRA 91, 052327, 2015), first order, whole-register depolarizing noise with
probability p per step, distance = diamond norm, d >> 1:
    D(r) = (1 - p) A t^2 / (2 r) + 2 p r,
where A is the diamond norm of rho -> [C, rho] with C = sum_{j<l} [H_j, H_l]; for C = [A, B]
that is the spectral spread of i[A, B]. Their general form D = C / r^(s-1) + D r also covers
higher orders; for p = 2 we feed it our commutator bound (doubled: diamond <= 2 x spectral).

Avtandilyan & Pogosov (arXiv:2405.01131), TFIM with K = 2N - 1 exponentials per step, each
failing with probability P0 (total gate error b r, b = K P0 at first order):
    k = 1:  a = t^2 J h (N - 1)
    k = 2:  a = 4 t^3 J h (J + h) (N - 1)
Their estimates are order-of-magnitude (written with ~). We use their constants as published.

For b we use the calibrated gate error per Trotter step of the actual transpiled circuit
(slope of the summed gate error versus r), not one uniform rate.
"""
import numpy as np
from scipy.optimize import least_squares


def optimum(a, q, b):
    return (q * a / b) ** (1 / (q + 1))


def knee_munro(order, t, p_step, norms):
    if order == 1:
        return optimum((1 - p_step) * norms["spread_AB"] * t**2 / 2, 1, 2 * p_step)
    if order == 2:
        c2 = norms["comm_BBA"] / 12 + norms["comm_AAB"] / 24
        return optimum(2 * t**3 * c2, 2, 2 * p_step)
    return np.nan


def avtandilyan_pogosov(order, t, p_step, n, J=1.0, h=1.0):
    if order == 1:
        return optimum(t**2 * J * h * (n - 1), 1, p_step)
    if order == 2:
        return optimum(4 * t**3 * J * h * (J + h) * (n - 1), 2, p_step)
    return np.nan


def per_step_error(r, gate_error_sum):
    """Slope of the summed calibrated gate error versus r (removes state-preparation cost)."""
    slope, _ = np.polyfit(np.asarray(r, float), np.asarray(gate_error_sum, float), 1)
    return slope


def fit_tradeoff(r, err, q=None):
    """Fit err ~ c r^-q + kappa r in log space. Returns dict with c, q, kappa, r_star.

    With q=None the Trotter exponent is fitted too.
    """
    r, err = np.asarray(r, float), np.asarray(err, float)
    keep = err > 0
    r, err = r[keep], err[keep]

    def model(x):
        c, kappa = np.exp(x[0]), np.exp(x[1])
        qq = x[2] if q is None else q
        return c * r ** (-qq) + kappa * r

    def resid(x):
        return np.log(model(x)) - np.log(err)

    x0 = [np.log(err[0] * r[0]), np.log(err[-1] / r[-1])] + ([1.0] if q is None else [])
    lb = [-np.inf, -np.inf] + ([0.1] if q is None else [])
    ub = [np.inf, np.inf] + ([8.0] if q is None else [])
    sol = least_squares(resid, x0, bounds=(lb, ub))
    c, kappa = np.exp(sol.x[0]), np.exp(sol.x[1])
    qq = sol.x[2] if q is None else q
    return {"c": c, "q": qq, "kappa": kappa, "r_star": optimum(c, qq, kappa),
            "rms_log_resid": np.sqrt(np.mean(sol.fun**2))}

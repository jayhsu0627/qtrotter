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
import pandas as pd
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


def semi_empirical(r, trotter_err, p_step):
    """r minimizing 1 - (1 - E_T(r)) exp(-p_step r).

    E_T is the noiseless Trotter error of this initial state (a classical simulation), and gate
    failures are treated as independent of it: the output is right only if neither happened.
    """
    r, trotter_err = np.asarray(r, float), np.asarray(trotter_err, float)
    total = 1 - (1 - trotter_err) * np.exp(-p_step * r)
    return r[np.argmin(total)]


def summarize(noisy, noiseless, metric="trace_dist"):
    """Measured and predicted r* for every (model, n, t, order) in the noisy sweep."""
    norm_cols = ["norm_A", "norm_B", "comm_AB", "spread_AB", "comm_AAB", "comm_BBA"]
    norms = noiseless.groupby(["model", "n"])[norm_cols].first()
    rows = []
    for (model, n, t, order), g in noisy.groupby(["model", "n", "t", "order"]):
        g = g.sort_values("r")
        p_step = per_step_error(g.r, g.gate_error_sum)
        best = g.loc[g[metric].idxmin()]
        rows.append({
            "model": model, "n": n, "t": t, "order": order,
            "p_step": p_step, "twoq_per_step": np.polyfit(g.r, g.two_qubit_gates, 1)[0],
            "r_measured": best.r, "err_at_optimum": best[metric],
            "r_semi_empirical": semi_empirical(g.r, g[f"trotter_{metric}"], p_step),
            "r_knee_munro": knee_munro(order, t, p_step, norms.loc[(model, n)].to_dict()),
            "r_avtandilyan_pogosov": (avtandilyan_pogosov(order, t, p_step, n)
                                      if model == "tfim" else np.nan),
        })
    return pd.DataFrame(rows)


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

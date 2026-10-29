"""Optimal Trotter step count r*: fitted to simulated error curves, and predicted analytically.

Both analytic papers minimize a Trotter term a / r^q plus a noise term b r, giving
    r* = (q a / b)^(1 / (q + 1)).

Knee & Munro (PRA 91, 052327, 2015), first order, whole-register depolarizing noise with
probability p per step, distance = diamond norm, d >> 1:
    D(r) = (1 - p) A t^2 / (2 r) + 2 p r,
where A is the diamond norm of rho -> [C, rho] with C = sum_{j<l} [H_j, H_l]; for C = [A, B]
that is the spectral spread of i[A, B]. Their general form D = C / r^(s-1) + D r also covers
higher orders, but they evaluate the constants only at first order. Our second-order extension
feeds it Childs et al.'s commutator bound (doubled: diamond <= 2 x spectral). Below, both terms are
halved to trace-distance units, which leaves r* unchanged.

Avtandilyan & Pogosov (arXiv:2405.01131), TFIM with K = 2N - 1 exponentials per step, each
failing with probability P0 (total gate error b r, b = K P0 at first order):
    k = 1:  a = t^2 J h (N - 1)
    k = 2:  a = 4 t^3 J h (J + h) (N - 1)
Their estimates are order-of-magnitude (written with ~). We use their constants as published.

Xu, Zhao, Fan & Zhao (arXiv:2504.10247) keep the worst-case constant B_p (the nested-commutator
norm: ||[A, B]|| / 2 at first order, Childs et al.'s at second) but let every step's error decay
as the noise contracts the state toward the maximally mixed one, their Eq. (14):
    sum_{d=1..r} (C gamma Upsilon + B_p (t/r)^(p+1)) exp(-c gamma Upsilon d),   c = 1/2.
Without the decay this is their closed form, Eq. (15), the same power law as above.

For b we use the calibrated gate error per Trotter step of the actual transpiled circuit
(slope of the summed gate error versus r), not one uniform rate.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def optimum(a, q, b):
    return (q * a / b) ** (1 / (q + 1))


def knee_munro_constant(order, t, p_step, norms):
    """Knee & Munro's Trotter constant a (term a / r^order), in trace-distance units."""
    if order == 1:
        return (1 - p_step) * norms["spread_AB"] * t**2 / 4
    if order == 2:
        return t**3 * (norms["comm_BBA"] / 12 + norms["comm_AAB"] / 24)
    return np.nan


def knee_munro(order, t, p_step, norms):
    return optimum(knee_munro_constant(order, t, p_step, norms), order, p_step)


def avtandilyan_pogosov(order, t, p_step, n, J=1.0, h=1.0):
    if order == 1:
        return optimum(t**2 * J * h * (n - 1), 1, p_step)
    if order == 2:
        return optimum(4 * t**3 * J * h * (J + h) * (n - 1), 2, p_step)
    return np.nan


def xu_et_al(order, t, p_step, n, norms, r_max=400):
    """r minimizing Xu et al.'s Eq. (14) with calibrated noise: C gamma Upsilon is the per-step
    error p_step, and the per-qubit rate in the decay is gamma Upsilon = p_step / n."""
    b_p = {1: norms["comm_AB"] / 2, 2: norms["comm_BBA"] / 12 + norms["comm_AAB"] / 24}.get(order)
    if b_p is None:
        return np.nan
    r = np.linspace(1, r_max, 100 * r_max)
    k = p_step / (2 * n)
    steps = np.exp(-k) * -np.expm1(-k * r) / -np.expm1(-k)  # sum_{d=1..r} exp(-k d)
    return r[np.argmin((p_step + b_p * (t / r) ** (order + 1)) * steps)]


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
    total = 1 - (1 - np.minimum(trotter_err, 1)) * np.exp(-p_step * r)
    return r[np.argmin(total)]


def fitted_trotter_constant(r, trotter_err, order, below=0.5):
    """C in E_T(r) ~ C / r^order, fitted in log space to the points below saturation."""
    r, trotter_err = np.asarray(r, float), np.asarray(trotter_err, float)
    keep = (trotter_err < below) & (trotter_err > 1e-10)
    return float(np.exp(np.mean(np.log(trotter_err[keep]) + order * np.log(r[keep]))))


def min_rise(err):
    """How far the error rises above its minimum on the flatter side (inf on a side with no points).

    A small value means the curve is flat around r* and the measured optimum is not resolved.
    """
    err = np.asarray(err, float)
    i = np.argmin(err)
    left = err[:i].max() - err[i] if i > 0 else np.inf
    right = err[i + 1:].max() - err[i] if i < len(err) - 1 else np.inf
    return min(left, right)


def ablation(noisy, noiseless, metric="trace_dist", r_max=400):
    """Which change fixes the analytic r*: the Trotter term, or how it is combined with noise?

    Three Trotter terms (Knee & Munro's a / r^q; this state's C / r^q, with C fitted to its
    noiseless error; that error E_T(r) itself) times two combination rules (additive, term + p r,
    as in both papers; saturating, 1 - (1 - term) exp(-p r)). The power-law terms are minimized
    over a fine r axis, E_T over the simulated grid.
    """
    norm_cols = ["spread_AB", "comm_AAB", "comm_BBA"]
    norms = noiseless.groupby(["model", "n"])[norm_cols].first()
    fine = np.linspace(1, r_max, 100 * r_max)
    rows = []
    for (model, n, t, order), g in noisy.groupby(["model", "n", "t", "order"]):
        g = g.sort_values("r")
        r, err, et = g.r.to_numpy(float), g[metric].to_numpy(), g[f"trotter_{metric}"].to_numpy()
        p_step = per_step_error(r, g.gate_error_sum)
        c_fit = fitted_trotter_constant(r, et, order)
        a_km = knee_munro_constant(order, t, p_step, norms.loc[(model, n)].to_dict())
        row = {"model": model, "n": n, "t": t, "order": order, "r_measured": r[np.argmin(err)],
               "a_knee_munro": a_km, "c_fitted": c_fit}
        for name, c in [("knee_munro", a_km), ("fitted", c_fit)]:
            row[f"{name} additive"] = optimum(c, order, p_step)
            row[f"{name} saturating"] = (np.nan if np.isnan(c) else
                                         semi_empirical(fine, c / fine**order, p_step))
        row["E_T additive"] = r[np.argmin(et + p_step * r)]
        row["E_T saturating"] = semi_empirical(r, et, p_step)
        rows.append(row)
    return pd.DataFrame(rows)


def damped_rms(trotter_values, exact_values, p_step, r):
    """RMS local-observable error if noise shrinks every expectation value by exp(-p_step r).

    The observable version of semi_empirical: the same global-depolarizing assumption.
    """
    damped = np.exp(-p_step * r) * np.asarray(trotter_values, float)
    return float(np.sqrt(np.mean((damped - np.asarray(exact_values, float)) ** 2)))


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
            "r_measured": best.r, "err_at_optimum": best[metric], "min_rise": min_rise(g[metric]),
            "r_semi_empirical": semi_empirical(g.r, g[f"trotter_{metric}"], p_step),
            "r_knee_munro": knee_munro(order, t, p_step, norms.loc[(model, n)].to_dict()),
            "r_xu_et_al": xu_et_al(order, t, p_step, n, norms.loc[(model, n)].to_dict()),
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


VALUE_INDEX = ["model", "n", "t", "order", "r", "noise_scale", "source"]


def local_errors(values):
    """RMS error over every local Pauli, per state, from a local_values table (run_noisy.py):
    the index columns plus `rms`, for the noisy and noiseless-Trotter rows, against exact evolution."""
    out = []
    for _, g in values.groupby(["model", "n", "t"]):
        obs = [c for c in g.columns if c not in VALUE_INDEX and g[c].notna().all()]
        exact = g.loc[g.source == "exact", obs].iloc[0].to_numpy()
        h = g[g.source != "exact"]
        out.append(h[VALUE_INDEX].assign(rms=np.sqrt(((h[obs].to_numpy() - exact) ** 2).mean(axis=1))))
    return pd.concat(out, ignore_index=True)


def refined_optimum(r, err):
    """argmin of err over the r grid, refined by a parabola in log r through the minimum and its two
    neighbours. Grid points alone are too coarse to fit how r* scales."""
    r, err = np.asarray(r, float), np.asarray(err, float)
    i = int(np.argmin(err))
    if i in (0, len(r) - 1):
        return r[i]
    x, y = np.log(r[i - 1:i + 2]), err[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    return float(np.exp(np.clip(-b / (2 * a), x[0], x[2]))) if a > 0 else r[i]

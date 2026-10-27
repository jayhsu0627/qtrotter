"""Worst-case Trotter error bounds on ||U(t) - S(t/r)^r|| (spectral norm), for two-group models.

Commutator bounds (Childs et al., PRX 11, 011020, 2021), with A = group 0 (outer), B = group 1:
    p = 1:  (t^2 / 2r)   ||[A, B]||
    p = 2:  (t^3 / r^2) (||[B, [B, A]]|| / 12 + ||[A, [A, B]]|| / 24)

Norm bound (Taylor remainder; no commutators). With G = sum over stages |a_s| ||H_{j_s}||, the
Taylor coefficients of S and U agree up to order p and are bounded by G^m / m!, so per step
    ||S(dt) - U(dt)|| <= 2 (G dt)^(p+1) / (p+1)! * exp(G dt),
and r steps add at most r times that. This is the bound to which the textbook r = O(t^2/eps)
for first order belongs; it grows with the total norm (~n) rather than with commutators.
"""
from math import factorial

import numpy as np
from scipy.sparse.linalg import eigsh

from .formulas import stages


def _hermitian_extremes(m):
    """(smallest, largest) eigenvalue of a sparse Hermitian matrix."""
    if m.shape[0] <= 512:
        w = np.linalg.eigvalsh(m.toarray())
        return w[0], w[-1]
    lo = eigsh(m, k=1, which="SA", return_eigenvectors=False)[0]
    hi = eigsh(m, k=1, which="LA", return_eigenvectors=False)[0]
    return lo, hi


def _norm(m):
    lo, hi = _hermitian_extremes(m)
    return max(abs(lo), abs(hi))


def commutator_norms(model):
    """Spectral norms of the commutators the bounds need, plus ||A|| and ||B||."""
    if len(model.groups) != 2:
        raise ValueError("bounds are implemented for two-group models")
    a, b = (g.to_matrix(sparse=True) for g in model.groups)
    ab = a @ b - b @ a  # anti-Hermitian
    aab = a @ ab - ab @ a  # Hermitian
    bba = b @ (-ab) - (-ab) @ b  # [B, [B, A]], Hermitian
    lo, hi = _hermitian_extremes(1j * ab)
    return {
        "norm_A": _norm(a),
        "norm_B": _norm(b),
        "comm_AB": max(abs(lo), abs(hi)),
        "spread_AB": hi - lo,  # used by Knee & Munro's constant (see predict.py)
        "comm_AAB": _norm(aab),
        "comm_BBA": _norm(bba),
    }


def commutator_bound(order, t, reps, norms):
    if order == 1:
        return t**2 / (2 * reps) * norms["comm_AB"]
    if order == 2:
        return t**3 / reps**2 * (norms["comm_BBA"] / 12 + norms["comm_AAB"] / 24)
    return np.nan


def norm_bound(order, t, reps, norms):
    group_norm = [norms["norm_A"], norms["norm_B"]]
    g = sum(abs(a) * group_norm[j] for j, a in stages(order, 2))
    x = g * t / reps
    return reps * 2 * x ** (order + 1) / factorial(order + 1) * np.exp(x)

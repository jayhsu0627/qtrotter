"""Follow-up: why the worst-case Trotter constant overshoots, factor by factor.

To leading order, r steps of a product formula give U_r psi0 = e^{-iHt} (psi0 + sum_k E(s_k) psi0),
E(s) = e^{iHs} E e^{-iHs}, with E the formula's per-step BCH error. So the noiseless trace distance is
E_T(r) ~ a_psi / r^q, with
  first order  (step e^{-iB dt} e^{-iA dt}):      a_psi = (t/2) || P_perp int_0^t C(s) ds psi0 ||,  C = [B, A]
  second order (step e^{-iA dt/2} e^{-iB dt} e^{-iA dt/2}):
                                                  a_psi = t^2 || P_perp int_0^t D(s) ds psi0 ||,
                                                  D = [B,[B,A]]/12 - [A,[A,B]]/24
(P_perp removes the component along the exact state: a phase). The worst-case constant replaces the
vector norm by an operator norm and the time integral by t times it, so
  a_worst / a_psi = (||K|| / ||P_perp K psi0||) * (t ||P_perp K psi0|| / ||P_perp int K(s) ds psi0||),
a state factor (locality: ||K psi0|| grows as sqrt(n) for a sum of n local terms on a short-range
correlated state, ||K|| as n) and a cancellation-in-time factor. K = C or D. Uses the operator norms
of K's parts as bounds.commutator_bound does (||C||; ||[B,[B,A]]||/12 + ||[A,[A,B]]||/24), and checks
a_psi against the constant fitted to the exact noiseless error (predict.fitted_trotter_constant).

    python scripts/trotter_constants.py [config.yaml]      # writes results/trotter_constants.csv
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build  # noqa: E402
from qtrotter.bounds import commutator_norms  # noqa: E402
from qtrotter.predict import fitted_trotter_constant  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
models, sizes, times = ["tfim", "xxz", "mfim"], [4, 6, 8, 10, 12], [1.0, 2.0, 4.0]
r_fit = np.array([16, 24, 32, 48, 64, 96, 128])
nodes, weights = np.polynomial.legendre.leggauss(120)
rows = []
for name in models:
    for n in sizes:
        model = build(name, n, **cfg["models"][name])
        prop = Propagator(model)
        a, b = (g.to_matrix(sparse=True) for g in model.groups)
        c = b @ a - a @ b                                     # [B, A]
        aab, bba = a @ (a @ b - b @ a) - (a @ b - b @ a) @ a, b @ c - c @ b
        d = bba / 12 - aab / 24
        norms = commutator_norms(model)
        worst = {1: norms["comm_AB"], 2: norms["comm_BBA"] / 12 + norms["comm_AAB"] / 24}
        w, v = prop.full
        psi0 = model.initial_state()
        evolve = lambda s, x: v @ (np.exp(-1j * s * w) * (v.conj().T @ x))
        for t in times:
            exact = prop.exact_state(t)
            for order, k in ((1, c), (2, d)):
                perp = lambda x: x - np.vdot(psi0, x) * psi0  # in the interaction frame, psi0 is the reference
                s_k, w_k = 0.5 * t * (nodes + 1), 0.5 * t * weights
                integral = sum(wk * evolve(-s, k @ evolve(s, psi0)) for s, wk in zip(s_k, w_k))
                static = np.linalg.norm(perp(k @ psi0))
                moving = np.linalg.norm(perp(integral))
                pref = (t / 2) if order == 1 else t**2
                a_psi = pref * moving
                a_worst = pref * t * worst[order]
                e_t = np.array([np.sqrt(max(0.0, 1 - abs(np.vdot(exact, prop.trotter_state(order, r, t))) ** 2))
                                for r in r_fit])
                rows.append({"model": name, "n": n, "t": t, "order": order, "a_worst": a_worst,
                             "a_psi": a_psi, "c_fit": fitted_trotter_constant(r_fit, e_t, order, below=0.2),
                             "state_factor": worst[order] / static, "time_factor": t * static / moving})
        print(f"{name} n={n} done", flush=True)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "trotter_constants.csv", index=False)
out["a_psi/c_fit"] = out.a_psi / out.c_fit
out["worst/c_fit"] = out.a_worst / out.c_fit
print(out.round(3).to_string(index=False))

"""Follow-up: when does fourth order beat second? Predictions made before any noisy run of the
three-stage fourth order ("4y"), for the noise_scaling_4y sweep.

Two predictions per (t, fourth-order formula), as the noise scale s at which the best fourth-order
error equals the best second-order one:
  closed form: E(r) = a / r^q + s p r gives r* = (q a / (s p))^(1/(q+1)) and
      E* = (1 + 1/q) (q a)^(1/(q+1)) (s p)^(q/(q+1)); equating q = 2 and q = 4 gives
      s_x = [ (3/2) (2 a_2)^(1/3) p_2^(2/3) / ((5/4) (4 a_4)^(1/5) p_4^(4/5)) ]^(15/2),
      so s_x ~ a_2^(5/2) a_4^(-3/2) kappa^(-6) with kappa = p_4 / p_2.
  semi-empirical: the saturating predictor 1 - (1 - E_T(r)) exp(-s p r), minimized over r for each
      order, on a fine grid of s.
a_q is fitted to the noiseless trace distance below 0.2; p is the slope of the summed calibrated
gate error in r (orders 2 and 4 from results/noisy.csv, "4y" from its transpiled circuits). Suzuki's
fourth order is included as a check against the measured crossover of noise_scaling.csv. Also writes
the semi-empirical r* and error at every scale of the sweep, as noise_scaling_predicted.csv does.

    python scripts/predict_crossover.py [config.yaml]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import yaml
from qiskit_ibm_runtime import fake_provider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build  # noqa: E402
from qtrotter.circuits import choose_layout, device_circuit, gate_error_sum, trotter_circuit  # noqa: E402
from qtrotter.predict import fitted_trotter_constant, per_step_error  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
run = cfg["noise_scaling_4y"]
backend = getattr(fake_provider, run["backend"])()
noisy = pd.read_csv(ROOT / "results" / "noisy.csv")
r_max = {2: 384, 4: 96, "4y": 128}
q = {2: 2, 4: 4, "4y": 4}
scales = np.geomspace(1e-3, 1.0, 600)
crossings, rows = [], []
for name in run["models"]:
    for n in run["sizes"]:
        model = build(name, n, **cfg["models"][name])
        prop = Propagator(model)
        layout = choose_layout(model, backend, run["seed"], run["optimization_level"])
        for t in run["times"]:
            exact = prop.exact_state(t)
            curve, a, p = {}, {}, {}
            for order in (2, 4, "4y"):
                r = np.unique(np.round(np.geomspace(1, r_max[order], 80)).astype(int))
                e_t = np.array([np.sqrt(max(0.0, 1 - abs(np.vdot(exact, prop.trotter_state(order, k, t))) ** 2))
                                for k in r])
                if order == "4y":
                    steps = [1, 2, 3, 4, 6, 8]
                    cost = [gate_error_sum(device_circuit(trotter_circuit(model, order, k, t), backend, layout,
                                                          seed=run["seed"], optimization_level=run["optimization_level"],
                                                          schedule=run["schedule"]), backend) for k in steps]
                    p[order] = per_step_error(steps, cost)
                else:
                    g = noisy[(noisy.model == name) & (noisy.n == n) & (noisy.t == t) & (noisy.order == order)]
                    p[order] = per_step_error(g.r, g.gate_error_sum)
                a[order] = fitted_trotter_constant(r, e_t, q[order], below=0.2)
                curve[order] = (r, e_t)
            best = {}
            for order, (r, e_t) in curve.items():
                total = 1 - (1 - np.minimum(e_t, 1))[None, :] * np.exp(-np.outer(scales, p[order] * r))
                best[order] = total.min(axis=1)
                if order == "4y":
                    for s in run["noise_scales"]:
                        tot = 1 - (1 - np.minimum(e_t, 1)) * np.exp(-s * p[order] * r)
                        i = np.argmin(tot)
                        rows.append({"model": name, "n": n, "t": t, "order": order, "noise_scale": s,
                                     "p_step": s * p[order], "r_predicted": r[i], "err_predicted": tot[i]})
            for four in (4, "4y"):
                closed = ((1.5 * (2 * a[2]) ** (1 / 3) * p[2] ** (2 / 3))
                          / (1.25 * (4 * a[four]) ** (1 / 5) * p[four] ** (4 / 5))) ** 7.5
                d, ls = np.log(best[four] / best[2]), np.log(scales)
                i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
                semi = (float(np.exp(ls[i[0]] - d[i[0]] * (ls[i[0] + 1] - ls[i[0]]) / (d[i[0] + 1] - d[i[0]])))
                        if len(i) else np.nan)
                crossings.append({"model": name, "n": n, "t": t, "fourth_order": four, "a_2": a[2],
                                  "a_4": a[four], "p_2": p[2], "p_4": p[four], "kappa": p[four] / p[2],
                                  "s_cross_closed_form": closed, "s_cross_semi_empirical": semi})
            print(f"{name} n={n} t={t}: kappa(4y) {p['4y'] / p[2]:.2f}, kappa(4) {p[4] / p[2]:.2f}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "noise_scaling_4y_predicted.csv", index=False)
pd.DataFrame(crossings).to_csv(ROOT / "results" / "crossover_predicted.csv", index=False)
print(pd.DataFrame(crossings).round(4).to_string(index=False))
print(pd.DataFrame(rows).round(4).to_string(index=False))

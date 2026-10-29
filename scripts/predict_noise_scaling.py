"""Semi-empirical r* predictions for the noise_scaling sweep, made without that sweep's data.

Inputs: exact Trotter states (no noise) on a fine r grid, and the per-step calibrated gate error
of Question 2 (results/noisy.csv) times each scale. Writes results/noise_scaling_predicted.csv.

    python scripts/predict_noise_scaling.py [config.yaml]
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
from qtrotter.predict import per_step_error  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
run = cfg["noise_scaling"]
noisy = pd.read_csv(ROOT / "results" / "noisy.csv")
r_max = {1: 512, 2: 384, 4: 96}  # well past the largest r* at the smallest scale
rows = []
for name in run["models"]:
    for n in run["sizes"]:
        prop = Propagator(build(name, n, **cfg["models"][name]))
        for t in run["times"]:
            exact = prop.exact_state(t)
            for order in run["orders"]:
                r = np.unique(np.round(np.geomspace(1, r_max[order], 80)).astype(int))
                e_t = np.array([np.sqrt(max(0.0, 1 - abs(np.vdot(exact, prop.trotter_state(order, k, t))) ** 2))
                                for k in r])
                g = noisy[(noisy.model == name) & (noisy.n == n) & (noisy.t == t) & (noisy.order == order)]
                p1 = per_step_error(g.r, g.gate_error_sum)
                for s in run["noise_scales"]:
                    total = 1 - (1 - np.minimum(e_t, 1)) * np.exp(-s * p1 * r)
                    i = np.argmin(total)
                    rows.append({"model": name, "n": n, "t": t, "order": order, "noise_scale": s,
                                 "p_step": s * p1, "r_predicted": r[i], "err_predicted": total[i]})
                print(f"{name} n={n} t={t} p={order} done", flush=True)
out = ROOT / "results" / "noise_scaling_predicted.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)

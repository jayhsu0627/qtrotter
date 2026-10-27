"""Question 1: noiseless Trotter error versus r, with commutator and norm bounds.

Writes results/noiseless.csv, one row per (model, n, t, order, r).

    python scripts/run_noiseless.py [config.yaml]
"""
import os
import sys
import time
from pathlib import Path

# Small dense matrices: many BLAS threads slow this down, and two sweeps at once thrash.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build, state_metrics  # noqa: E402
from qtrotter.bounds import commutator_bound, commutator_norms, norm_bound  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
run = cfg["noiseless"]
rows = []
for name in run["models"]:
    for n in run["sizes"]:
        start = time.perf_counter()
        model = build(name, n, **cfg["models"][name])
        prop, norms, obs = Propagator(model), commutator_norms(model), model.observables()
        for t in run["times"]:
            exact = prop.exact_state(t)
            u_exact = prop.exact_unitary(t) if n <= run["spectral_norm_max_n"] else None
            for order in run["orders"]:
                for reps in run["steps"]:
                    row = {"model": name, "n": n, "t": t, "order": order, "r": reps}
                    row.update(state_metrics(exact, prop.trotter_state(order, reps, t), obs))
                    if u_exact is not None:
                        diff = u_exact - prop.trotter_unitary(order, reps, t)
                        row["spectral_err"] = np.linalg.norm(diff, 2)
                    row["commutator_bound"] = commutator_bound(order, t, reps, norms)
                    row["norm_bound"] = norm_bound(order, t, reps, norms)
                    rows.append(row | norms)
        print(f"{name} n={n}: {time.perf_counter() - start:.1f} s", flush=True)

out = ROOT / "results" / "noiseless.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, len(rows), "rows")

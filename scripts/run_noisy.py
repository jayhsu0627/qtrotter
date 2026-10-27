"""Question 2: total error versus r under a fake-backend noise model (density matrix, Aer).

Appends to results/noisy.csv, one row per (model, n, t, order, r), and skips rows already
there, so an interrupted sweep resumes where it stopped.

    python scripts/run_noisy.py [config.yaml]
"""
import os
import sys
import time
from pathlib import Path

# Small dense matrices: many BLAS threads slow this down, and two sweeps at once thrash.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import pandas as pd
import yaml
from qiskit_ibm_runtime import fake_provider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build, state_metrics  # noqa: E402
from qtrotter.circuits import (choose_layout, device_circuit, gate_error_sum,  # noqa: E402
                               trotter_circuit, two_qubit_count)
from qtrotter.noisy import density_matrix, noisy_simulator, pick_device  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
run = cfg["noisy"]
backend = getattr(fake_provider, run["backend"])()
device = pick_device(run["device"])
sim = noisy_simulator(backend, device=device, precision=run["precision"])
opts = {"seed": run["seed"], "optimization_level": run["optimization_level"],
        "schedule": run["schedule"]}
print(f"{run['backend']} noise model, density matrix on {device}", flush=True)

out = ROOT / "results" / run.get("output", "noisy.csv")
key = ["model", "n", "t", "order", "r"]
done = set(map(tuple, pd.read_csv(out)[key].values.tolist())) if out.exists() else set()

for name in run["models"]:
    for n in run["sizes"]:
        model = build(name, n, **cfg["models"][name])
        prop, obs = Propagator(model), model.observables()
        layout = choose_layout(model, backend, run["seed"], run["optimization_level"])
        for t in run["times"]:
            exact = prop.exact_state(t)
            for order in run["orders"]:
                start, rows = time.perf_counter(), []
                for reps in run["steps"]:
                    if (name, n, t, order, reps) in done:
                        continue
                    tqc = device_circuit(trotter_circuit(model, order, reps, t), backend,
                                         layout, **opts)
                    row = {"model": name, "n": n, "t": t, "order": order, "r": reps,
                           "two_qubit_gates": two_qubit_count(tqc), "depth": tqc.depth(),
                           "gate_error_sum": gate_error_sum(tqc, backend),
                           "layout": " ".join(map(str, layout))}
                    row.update(state_metrics(exact, density_matrix(tqc, sim), obs))
                    trotter_only = state_metrics(exact, prop.trotter_state(order, reps, t), obs)
                    row.update({f"trotter_{k}": v for k, v in trotter_only.items()
                                if not k.endswith("_exact")})
                    rows.append(row)
                if rows:
                    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{name} n={n} t={t} p={order}: {len(rows)} runs, "
                      f"{time.perf_counter() - start:.1f} s", flush=True)
print("wrote", out)

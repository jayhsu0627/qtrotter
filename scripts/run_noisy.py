"""Question 2 and its follow-ups: total error versus r under a fake-backend noise model
(density matrix, Aer).

    python scripts/run_noisy.py [config.yaml] [section]     # section defaults to "noisy"

A section with `noise_scales` runs every circuit again on uniformly better devices
(qtrotter.noisy.scaled_backend) and records the scale in each row; `max_steps` caps r per order.
Appends to results/<output>, one row per (model, n, t, order, r[, noise_scale]), and skips rows
already there, so an interrupted sweep resumes where it stopped.
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
from qtrotter.noisy import density_matrix, noisy_simulator, pick_device, scaled_backend  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
section = sys.argv[2] if len(sys.argv) > 2 else "noisy"
run = cfg[section]
backend = getattr(fake_provider, run["backend"])()
device = pick_device(run["device"])
scaling = "noise_scales" in run
devices = {s: scaled_backend(backend, s) for s in run.get("noise_scales", [1.0])}
sims = {s: noisy_simulator(b, device=device, precision=run["precision"]) for s, b in devices.items()}
max_steps = run.get("max_steps", {})
opts = {"seed": run["seed"], "optimization_level": run["optimization_level"],
        "schedule": run["schedule"]}
print(f"[{section}] {run['backend']} noise model, scales {list(devices)}, density matrix on {device}",
      flush=True)

out = ROOT / "results" / run.get("output", "noisy.csv")
key = ["model", "n", "t", "order", "r"] + (["noise_scale"] if scaling else [])
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
                    todo = [s for s in devices
                            if (name, n, t, order, reps, s)[:len(key)] not in done]
                    if reps > max_steps.get(order, reps) or not todo:
                        continue
                    # One transpiled circuit for every scale: only the noise changes.
                    tqc = device_circuit(trotter_circuit(model, order, reps, t), backend,
                                         layout, **opts)
                    trotter_only = state_metrics(exact, prop.trotter_state(order, reps, t), obs)
                    for s in todo:
                        row = {"model": name, "n": n, "t": t, "order": order, "r": reps}
                        if scaling:
                            row["noise_scale"] = s
                        row.update({"two_qubit_gates": two_qubit_count(tqc), "depth": tqc.depth(),
                                    "gate_error_sum": gate_error_sum(tqc, devices[s]),
                                    "layout": " ".join(map(str, layout))})
                        row.update(state_metrics(exact, density_matrix(tqc, sims[s]), obs))
                        row.update({f"trotter_{k}": v for k, v in trotter_only.items()
                                    if not k.endswith("_exact")})
                        rows.append(row)
                if rows:
                    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{name} n={n} t={t} p={order}: {len(rows)} runs, "
                      f"{time.perf_counter() - start:.1f} s", flush=True)
print("wrote", out)

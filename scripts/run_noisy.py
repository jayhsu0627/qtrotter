"""Question 2 and its follow-ups: total error versus r under a fake-backend noise model
(density matrix, Aer).

    python scripts/run_noisy.py [config.yaml] [section]     # section defaults to "noisy"

A section with `noise_scales` runs every circuit again on uniformly better devices
(qtrotter.noisy.scaled_backend) and records the scale in each row; `max_steps` caps r per order,
and `cap_exponent` lowers that cap on worse devices, to max_steps * (smallest scale / s)^exponent.
With `local_values`, every local Pauli (Model.local_paulis) is also written to that file, one row
per state: exact evolution, the noiseless Trotter state, and each noisy run.
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

import numpy as np
import pandas as pd
import yaml
from qiskit_ibm_runtime import fake_provider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build, state_metrics  # noqa: E402
from qtrotter.circuits import (choose_layout, device_circuit, gate_error_sum,  # noqa: E402
                               trotter_circuit, two_qubit_count)
from qtrotter.hardware import expectations  # noqa: E402
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


def step_cap(order, scale):
    cap = max_steps.get(order, np.inf)
    return cap * (min(devices) / scale) ** run.get("cap_exponent", 0)


opts = {"seed": run["seed"], "optimization_level": run["optimization_level"],
        "schedule": run["schedule"]}
print(f"[{section}] {run['backend']} noise model, scales {list(devices)}, density matrix on {device}",
      flush=True)

out = ROOT / "results" / run.get("output", "noisy.csv")
key = ["model", "n", "t", "order", "r"] + (["noise_scale"] if scaling else [])
done = set(map(tuple, pd.read_csv(out)[key].values.tolist())) if out.exists() else set()
values_out = ROOT / "results" / run["local_values"] if "local_values" in run else None
if values_out:  # one fixed column set, the largest chain's
    value_cols = ["model", "n", "t", "order", "r", "noise_scale", "source"] + list(
        build(run["models"][0], max(run["sizes"]), **cfg["models"][run["models"][0]]).local_paulis())


def value_row(name, n, t, order, reps, scale, source, state, paulis):
    row = {"model": name, "n": n, "t": t, "order": order, "r": reps, "noise_scale": scale,
           "source": source}
    return row | expectations(state, paulis)


def write_values(values):
    pd.DataFrame(values).reindex(columns=value_cols).to_csv(
        values_out, mode="a", header=not values_out.exists(), index=False, float_format="%.8g")


for name in run["models"]:
    for n in run["sizes"]:
        model = build(name, n, **cfg["models"][name])
        prop, obs = Propagator(model), model.observables()
        paulis = model.local_paulis() if values_out else None
        layout = choose_layout(model, backend, run["seed"], run["optimization_level"])
        for t in run["times"]:
            exact = prop.exact_state(t)
            if values_out and not any(k[:3] == (name, n, t) for k in done):
                write_values([value_row(name, n, t, 0, 0, 0.0, "exact", exact, paulis)])
            for order in run["orders"]:
                start, rows, values = time.perf_counter(), [], []
                for reps in run["steps"]:
                    todo = [s for s in devices
                            if (name, n, t, order, reps, s)[:len(key)] not in done
                            and reps <= step_cap(order, s)]
                    if not todo:
                        continue
                    # One transpiled circuit for every scale: only the noise changes.
                    tqc = device_circuit(trotter_circuit(model, order, reps, t), backend,
                                         layout, **opts)
                    trotter_state = prop.trotter_state(order, reps, t)
                    trotter_only = state_metrics(exact, trotter_state, obs)
                    if values_out and not any(k[:5] == (name, n, t, order, reps) for k in done):
                        values.append(value_row(name, n, t, order, reps, 0.0, "trotter",
                                                trotter_state, paulis))
                    for s in todo:
                        row = {"model": name, "n": n, "t": t, "order": order, "r": reps}
                        if scaling:
                            row["noise_scale"] = s
                        row.update({"two_qubit_gates": two_qubit_count(tqc), "depth": tqc.depth(),
                                    "gate_error_sum": gate_error_sum(tqc, devices[s]),
                                    "layout": " ".join(map(str, layout))})
                        rho = density_matrix(tqc, sims[s])
                        row.update(state_metrics(exact, rho, obs))
                        row.update({f"trotter_{k}": v for k, v in trotter_only.items()
                                    if not k.endswith("_exact")})
                        rows.append(row)
                        if values_out:
                            values.append(value_row(name, n, t, order, reps, s, "noisy", rho, paulis))
                if values:  # written first: a row in `out` marks its values as saved
                    write_values(values)
                if rows:
                    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{name} n={n} t={t} p={order}: {len(rows)} runs, "
                      f"{time.perf_counter() - start:.1f} s", flush=True)
print("wrote", out)

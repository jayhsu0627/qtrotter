"""Follow-up: an independent check of the patches on a chain too long for a density matrix.

    python scripts/run_trajectories.py [config.yaml] [section]   # section defaults to "trajectories"

Noisy statevector trajectories (Aer, device noise model) of the full chain, averaged in independent
batches so that their statistical error can be estimated. Writes every local Pauli per batch, plus
the exact and noiseless Trotter values (sparse evolution of the full chain), in the long format
of hardware_values.csv: source, order, r, noise_scale, batch, observable, value.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import yaml
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import build  # noqa: E402
from qtrotter.circuits import choose_layout, device_circuit, trotter_circuit  # noqa: E402
from qtrotter.formulas import sequence  # noqa: E402
from qtrotter.hardware import expectations  # noqa: E402
from qtrotter.noisy import drop_idle_qubits, restricted_noise_model, scaled_backend  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
section = sys.argv[2] if len(sys.argv) > 2 else "trajectories"
run = cfg[section]
backend = getattr(fake_provider, run["backend"])()
opts = {"seed": run["seed"], "optimization_level": run["optimization_level"],
        "schedule": run["schedule"]}
print(f"[{section}] {run['backend']} noise model, statevector trajectories on {run['device']}",
      flush=True)


def evolve(groups, steps, psi):
    """Apply exp(-i a H_j) for each (group, a) in `steps`, by sparse Krylov evolution."""
    mats = [g.to_matrix(sparse=True) for g in groups]
    for j, a in steps:
        psi = expm_multiply(-1j * a * mats[j], psi)
    return psi


def trajectory_values(tqc, sim, shots, seed, n):
    """Batch-averaged single-site and nearest-neighbour reduced density matrices -> local Paulis.
    The noise model keeps only the circuit's own qubits, as in qtrotter.noisy.density_matrix."""
    tqc = drop_idle_qubits(tqc)
    used = {tqc.find_bit(q).index for ins in tqc.data for q in ins.qubits}
    noise = restricted_noise_model(sim.options.noise_model, used)
    phys = tqc.layout.final_index_layout()
    for i in range(n):
        tqc.save_density_matrix(qubits=[phys[i]], label=f"s{i}")
    for i in range(n - 1):
        tqc.save_density_matrix(qubits=[phys[i], phys[i + 1]], label=f"p{i}")
    data = sim.run(tqc, shots=shots, noise_model=noise, seed_simulator=seed).result().data()
    x, z = np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
    out = {}
    for i in range(n):
        rho = np.asarray(data[f"s{i}"])
        out[f"Z{i}"], out[f"X{i}"] = np.real(np.trace(z @ rho)), np.real(np.trace(x @ rho))
    for i in range(n - 1):
        rho = np.asarray(data[f"p{i}"])  # qubit order (site i, site i+1), little-endian
        out[f"Z{i}Z{i + 1}"] = np.real(np.trace(np.kron(z, z) @ rho))
        out[f"X{i}X{i + 1}"] = np.real(np.trace(np.kron(x, x) @ rho))
    return out


out = ROOT / "results" / run["output"]
rows = []
for name in run["models"]:
    for n in run["sizes"]:
        model = build(name, n, **cfg["models"][name])
        paulis = model.local_paulis()
        layout = choose_layout(model, backend, run["seed"], run["optimization_level"])
        h = model.hamiltonian.to_matrix(sparse=True)
        for t in run["times"]:
            exact = expm_multiply(-1j * t * h, model.initial_state())
            rows += [{"model": name, "n": n, "t": t, "source": "exact", "order": 0, "r": 0,
                      "noise_scale": 0.0, "batch": 0, "observable": k, "value": v}
                     for k, v in expectations(exact, paulis).items()]
            for order in run["orders"]:
                for scale, steps in run["steps"].items():
                    device = scaled_backend(backend, scale)
                    sim = AerSimulator.from_backend(device, method="statevector", device=run["device"])
                    for reps in steps:
                        start = time.perf_counter()
                        trotter = evolve(model.groups,
                                         [(j, a * t / reps) for j, a in sequence(order, 2, reps)],
                                         model.initial_state())
                        rows += [{"model": name, "n": n, "t": t, "source": "trotter", "order": order,
                                  "r": reps, "noise_scale": 0.0, "batch": 0, "observable": k,
                                  "value": v} for k, v in expectations(trotter, paulis).items()]
                        tqc = device_circuit(trotter_circuit(model, order, reps, t), backend,
                                             layout, **opts)
                        for batch in range(run["batches"]):
                            vals = trajectory_values(tqc.copy(), sim, run["shots"],
                                                     run["seed"] + 1000 * batch, n)
                            rows += [{"model": name, "n": n, "t": t, "source": "noisy",
                                      "order": order, "r": reps, "noise_scale": scale,
                                      "batch": batch, "observable": k, "value": v}
                                     for k, v in vals.items()]
                        print(f"{name} n={n} t={t} p={order} scale={scale} r={reps}: "
                              f"{time.perf_counter() - start:.1f} s", flush=True)
                        pd.DataFrame(rows).to_csv(out, index=False, float_format="%.8g")
print("wrote", out)

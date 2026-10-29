"""Follow-up: where the noise that reaches a local observable comes from.

    python scripts/run_noise_response.py [config.yaml] [section]   # section defaults to "noise_response"

Noise on one qubit only: a depolarizing channel of probability p on site j after every Trotter
step, on otherwise ideal gates (density matrix, Aer). For every j, r, t and order, writes how far
each local Pauli moves from its noiseless Trotter value. If noise reached an observable through
the circuit's light cone, which grows by a site or two per step, every site would matter once
r is large enough; through the dynamics' light cone, only sites within about v t of it would.
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
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer.noise import depolarizing_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build  # noqa: E402
from qtrotter.circuits import logical_circuit  # noqa: E402
from qtrotter.hardware import expectations  # noqa: E402
from qtrotter.noisy import noisy_simulator, pick_device  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
section = sys.argv[2] if len(sys.argv) > 2 else "noise_response"
run = cfg[section]
device = pick_device(run["device"])
sim = noisy_simulator(None, device=device, precision=run["precision"], noise=False)
p = run["p"]
print(f"[{section}] depolarizing p = {p} per step on one site, density matrix on {device}", flush=True)


def noisy_circuit(model, order, reps, t, site):
    """`reps` Trotter steps, each followed by depolarizing noise on `site` (no half-step merging)."""
    synth = LieTrotter(reps=1) if order == 1 else SuzukiTrotter(order=order, reps=1)
    step = QuantumCircuit(model.n)
    step.append(PauliEvolutionGate(list(model.groups), time=t / reps, synthesis=synth),
                range(model.n))
    step = logical_circuit(step)
    qc = QuantumCircuit(model.n)
    for i, bit in enumerate(model.init_bits):
        if bit == "1":
            qc.x(i)
    noise = depolarizing_error(p, 1).to_instruction()
    for _ in range(reps):
        qc.compose(step, inplace=True)
        qc.append(noise, [site])
    qc.save_density_matrix()
    return qc


out = ROOT / "results" / run["output"]
key = ["model", "n", "t", "order", "r", "site"]
done = set(map(tuple, pd.read_csv(out)[key].drop_duplicates().values.tolist())) if out.exists() else set()

for name in run["models"]:
    for n in run["sizes"]:
        model = build(name, n, **cfg["models"][name])
        prop, paulis = Propagator(model), model.local_paulis()
        for t in run["times"]:
            for order in run["orders"]:
                start, rows = time.perf_counter(), []
                for reps in run["steps"]:
                    trotter = expectations(prop.trotter_state(order, reps, t), paulis)
                    for site in range(n):
                        if (name, n, t, order, reps, site) in done:
                            continue
                        result = sim.run(noisy_circuit(model, order, reps, t, site)).result()
                        noisy = expectations(np.asarray(result.data()["density_matrix"]), paulis)
                        rows += [{"model": name, "n": n, "t": t, "order": order, "r": reps,
                                  "p": p, "site": site, "observable": k, "trotter": trotter[k],
                                  "delta": noisy[k] - trotter[k]} for k in paulis]
                if rows:
                    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False,
                                              float_format="%.8g")
                print(f"{name} n={n} t={t} p={order}: {len(rows) // len(paulis)} runs, "
                      f"{time.perf_counter() - start:.1f} s", flush=True)
print("wrote", out)

"""Question 3: the same Trotter circuits on an IBM device and on that device's noise model.

Hardware returns expectation values, not states, so the error metric here is the RMS deviation
from exact evolution over all local Paulis (Model.local_paulis). Squaring each observable's error
before averaging stops errors in different observables from cancelling.
"""
import numpy as np
import pandas as pd
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

from .circuits import device_circuit, gate_error_sum, trotter_circuit, two_qubit_count
from .noisy import density_matrix


def build(model, orders, steps, t, backend, layout, seed=1, optimization_level=2):
    """ISA circuits for the device (unscheduled: the device schedules them itself)."""
    jobs = []
    for order in orders:
        for reps in steps:
            tqc = device_circuit(trotter_circuit(model, order, reps, t), backend, layout, seed=seed,
                                 optimization_level=optimization_level, schedule=False)
            jobs.append({"order": order, "r": reps, "circuit": tqc,
                         "two_qubit_gates": two_qubit_count(tqc), "depth": tqc.depth(),
                         "gate_error_sum": gate_error_sum(tqc, backend)})
    return jobs


def scheduled(tqc, backend):
    """Same circuit with ALAP delays, so the simulator applies idle-qubit relaxation."""
    pm = PassManager([ALAPScheduleAnalysis(target=backend.target), PadDelay(target=backend.target)])
    out = pm.run(tqc)
    out._layout = tqc.layout  # PassManager.run drops it; the readout needs it
    return out


def expectations(state, paulis):
    """<P> for each Pauli, from a state vector or a density matrix."""
    vals = {}
    for name, op in paulis.items():
        m = op.to_matrix(sparse=True)
        if state.ndim == 1:
            vals[name] = np.real(np.vdot(state, m @ state))
        else:
            vals[name] = np.real(m.T.multiply(state).sum())
    return vals


def noise_model_values(jobs, sim, backend, paulis):
    return [expectations(density_matrix(scheduled(j["circuit"], backend), sim), paulis) for j in jobs]


def estimator_pubs(jobs, paulis):
    names = list(paulis)
    return names, [(j["circuit"], [paulis[k].apply_layout(j["circuit"].layout) for k in names])
                   for j in jobs]


def rms_error(values, exact):
    keys = list(exact)
    return float(np.sqrt(np.mean([(values[k] - exact[k]) ** 2 for k in keys])))


def rms_table(values, draws=2000, seed=0):
    """RMS error per (source, order, r) from the long-format values table.

    Hardware values carry shot noise, which inflates a mean squared error by the mean variance;
    that bias is subtracted. The spread comes from a parametric bootstrap over the reported stds.
    """
    exact = values[values.source == "exact"].set_index("observable").value
    rng = np.random.default_rng(seed)
    rows = []
    for (source, order, reps), g in values[values.source != "exact"].groupby(["source", "order", "r"]):
        g = g.set_index("observable").loc[exact.index]
        err = g.value.to_numpy() - exact.to_numpy()
        if source != "hardware":
            rows.append({"source": source, "order": order, "r": reps,
                         "rms": float(np.sqrt(np.mean(err**2))), "rms_std": 0.0})
            continue
        std = g["std"].to_numpy(float)
        debiased = lambda e: np.sqrt(np.maximum(np.mean(e**2 - std**2, axis=-1), 0.0))
        boot = debiased(err + rng.normal(0.0, std, size=(draws, len(std))))
        rows.append({"source": source, "order": order, "r": reps,
                     "rms": float(debiased(err)), "rms_std": float(boot.std())})
    return pd.DataFrame(rows)

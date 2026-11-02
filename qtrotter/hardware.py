"""Question 3: the same Trotter circuits on an IBM device and on that device's noise model.

Hardware returns expectation values, not states, so the error metric here is the RMS deviation
from exact evolution over all local Paulis (Model.local_paulis). Squaring each observable's error
before averaging stops errors in different observables from cancelling.
"""
import re

import numpy as np
import pandas as pd
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

from .circuits import device_circuit, gate_error_sum, trotter_circuit, two_qubit_count
from .noisy import density_matrix


def build(model, orders, steps, t, backend, layout, seed=1, optimization_level=2, layer=True):
    """ISA circuits for the device (unscheduled: the device schedules them itself)."""
    jobs = []
    for order in orders:
        for reps in steps:
            tqc = device_circuit(trotter_circuit(model, order, reps, t, layer), backend, layout, seed=seed,
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


def wide(values, source, column="value"):
    """One row per (order, r), one column per observable (in the exact rows' order)."""
    names = values[values.source == "exact"].observable.tolist()
    g = values[values.source == source]
    return g.pivot_table(index=["order", "r"], columns="observable", values=column)[names]


def sites(name):
    """Chain sites a local Pauli acts on: 'X3X4' -> {3, 4}."""
    return {int(s) for s in re.findall(r"\d+", name)}


def gap_table(values, prediction="patches"):
    """Where a device run departs from its prediction, per (order, r), over all local Paulis.

    `dev` and `pred` are RMS errors against exact evolution; `noise_dev` and `noise_pred` the RMS
    departure from the noiseless Trotter values, i.e. the noise alone; `gap` the RMS of device
    minus prediction; `damp_dev` and `damp_pred` the least-squares factor by which the values are a
    shrunken copy of the noiseless Trotter values. Device terms have the shot-noise bias removed.
    """
    exact = values[values.source == "exact"].set_index("observable").value
    dev, var = wide(values, "hardware"), wide(values, "hardware", "std") ** 2
    pred, trot = wide(values, prediction), wide(values, "trotter")
    e = exact[dev.columns].to_numpy()
    rms = lambda d, v=0: np.sqrt(np.maximum((d**2 - v).mean(axis=1), 0.0))
    shrink = lambda a: (a * trot).sum(axis=1) / (trot**2).sum(axis=1)
    return pd.DataFrame({"dev": rms(dev - e, var), "pred": rms(pred - e),
                         "noise_dev": rms(dev - trot, var), "noise_pred": rms(pred - trot),
                         "gap": rms(dev - pred, var), "damp_dev": shrink(dev), "damp_pred": shrink(pred)},
                        index=dev.index)


def leave_out(values, drop, prediction="patches"):
    """r* and errors per order over the local Paulis that touch none of the sites in `drop`:
    the device's r*, the predicted r*, and both errors at the predicted r*."""
    exact = values[values.source == "exact"].set_index("observable").value
    keep = [k for k in exact.index if not sites(k) & set(drop)]
    dev, var = wide(values, "hardware")[keep], wide(values, "hardware", "std")[keep] ** 2
    pred, e = wide(values, prediction)[keep], exact[keep].to_numpy()
    err_dev = pd.Series(np.sqrt(np.maximum(((dev - e) ** 2 - var).mean(axis=1), 0.0)), index=dev.index)
    err_pred = pd.Series(np.sqrt(((pred - e) ** 2).mean(axis=1)), index=dev.index)
    rows = []
    for order in sorted(dev.index.get_level_values("order").unique()):
        d, p = err_dev.loc[order], err_pred.loc[order]
        r_pred = p.idxmin()
        rows.append({"order": order, "observables": len(keep), "r_dev": d.idxmin(), "r_pred": r_pred,
                     "dev_at_r_pred": d[r_pred], "pred_at_r_pred": p[r_pred]})
    return pd.DataFrame(rows)


def reproducibility(values, pubs, prediction="patches"):
    """Does a circuit's deviation from the prediction come back when the same circuit runs again?

    `pubs`: the job record's pub list in execution order, each with its `copy` (results of copy c
    under source hardware_c). Every run's deviation is its device values minus the prediction, over
    all local Paulis. Returns (pairs, verdict): pairs has the correlation of every two runs' deviations
    and the kind of pair, 'same' (two copies of one circuit), 'same_r' (same r, other order),
    'neighbour' (ran one after the other, different r) or 'other'; verdict is the median per kind
    and the pre-registered call: "circuit" if same >= 0.8 and same >= neighbour + 0.3, "drift" if
    neighbour >= same or same < 0.5, "mixed" otherwise.
    """
    pred = wide(values, prediction)
    dev = [(wide(values, f"hardware_{p['copy']}") - pred).loc[(p["order"], p["r"])].to_numpy() for p in pubs]
    rows = []
    for i in range(len(pubs)):
        for j in range(i + 1, len(pubs)):
            a, b = pubs[i], pubs[j]
            kind = ("same" if (a["order"], a["r"]) == (b["order"], b["r"]) else "same_r" if a["r"] == b["r"]
                    else "neighbour" if j == i + 1 else "other")
            rows.append({"first": i, "second": j, "kind": kind, "order": (a["order"], b["order"]),
                         "r": (a["r"], b["r"]), "corr": float(np.corrcoef(dev[i], dev[j])[0, 1])})
    pairs = pd.DataFrame(rows)
    med = pairs.groupby("kind")["corr"].median()
    same, neighbour = med["same"], med["neighbour"]
    call = ("circuit" if same >= 0.8 and same >= neighbour + 0.3
            else "drift" if neighbour >= same or same < 0.5 else "mixed")
    return pairs, {**med.to_dict(), "verdict": call}


def clean_graph(backend, max_2q_error=0.01, max_readout_error=0.05):
    """Couplers of `backend` with two-qubit error at most `max_2q_error`, between qubits whose readout
    error is at most `max_readout_error`: {qubit: set of neighbours}."""
    target = backend.target
    gate = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    readout = {q[0]: p.error for q, p in target["measure"].items() if p is not None and p.error is not None}
    good = {q for q, e in readout.items() if e <= max_readout_error}
    graph = {q: set() for q in good}
    for (a, b), p in target[gate].items():
        if p is not None and p.error is not None and p.error <= max_2q_error and a in good and b in good:
            graph[a].add(b)
            graph[b].add(a)
    return graph


def longest_path(graph, restarts=200, budget=20000, seed=0):
    """A long simple path in `graph` by depth-first search from random starts, each limited to
    `budget` expansions (finding the longest one is NP-hard; heavy-hex graphs have degree <= 3)."""
    rng = np.random.default_rng(seed)
    nodes = [q for q in graph if graph[q]]
    best = []
    for _ in range(restarts):
        start = nodes[rng.integers(len(nodes))]
        path, on_path, left = [start], {start}, [budget]

        def extend(q):
            nonlocal best
            if len(path) > len(best):
                best = list(path)
            if left[0] <= 0:
                return
            left[0] -= 1
            for nxt in rng.permutation(sorted(graph[q] - on_path)):
                nxt = int(nxt)
                path.append(nxt)
                on_path.add(nxt)
                extend(nxt)
                path.pop()
                on_path.discard(nxt)

        extend(start)
    return best

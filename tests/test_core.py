"""Consistency checks. Run with `python tests/test_core.py` (or pytest)."""
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtrotter import Model, Propagator, state_metrics, tfim, xxz  # noqa: E402
from qtrotter.bounds import commutator_bound, commutator_norms, norm_bound  # noqa: E402
from qtrotter.circuits import (choose_layout, device_circuit, logical_circuit,  # noqa: E402
                               trotter_circuit)


def overlap(a, b):
    return abs(np.vdot(a, b)) ** 2


def test_numpy_formulas_match_qiskit_synthesis():
    for model in (tfim(5), xxz(5)):
        prop = Propagator(model)
        for order in (1, 2, 4):
            sv = Statevector(logical_circuit(trotter_circuit(model, order, 3, 0.7))).data
            assert overlap(prop.trotter_state(order, 3, 0.7), sv) > 1 - 1e-10, (model.name, order)


def test_layered_groups():
    """Layering reorders commuting terms only: same unitary, and the TFIM's ZZ bonds go from a
    staircase as deep as the chain to two layers."""
    from qiskit.quantum_info import Operator

    for model in (tfim(6), xxz(6)):
        for order in (1, 2):
            a, b = (Operator(logical_circuit(trotter_circuit(model, order, 3, 0.7, layer))).data
                    for layer in (True, False))
            assert np.abs(a - b).max() < 1e-12, (model.name, order)
    model = tfim(10)
    depth = {layer: logical_circuit(trotter_circuit(model, 1, 1, 0.5, layer)).depth(
                 lambda ins: ins.operation.num_qubits == 2) for layer in (True, False)}
    assert depth[True] == 4 and depth[False] == 2 * (model.n - 1), depth


def test_exact_state_and_unitaries():
    model = xxz(6)
    prop = Propagator(model)
    ref = expm_multiply(-1.3j * model.hamiltonian.to_matrix(sparse=True), model.initial_state())
    assert overlap(prop.exact_state(1.3), ref) > 1 - 1e-12
    psi0 = model.initial_state()
    for order in (1, 2, 4):
        u = prop.trotter_unitary(order, 5, 1.3)
        assert overlap(u @ psi0, prop.trotter_state(order, 5, 1.3)) > 1 - 1e-12


def test_bounds_hold():
    for model in (tfim(6), xxz(6)):
        prop, norms = Propagator(model), commutator_norms(model)
        for t in (1.0, 3.0):
            exact = prop.exact_unitary(t)
            for order in (1, 2, 4):
                for reps in (1, 2, 4, 8, 16, 32):
                    err = np.linalg.norm(exact - prop.trotter_unitary(order, reps, t), 2)
                    nb = norm_bound(order, t, reps, norms)
                    assert err <= nb * (1 + 1e-9), (model.name, t, order, reps, err, nb)
                    if order <= 2:
                        cb = commutator_bound(order, t, reps, norms)
                        assert err <= cb * (1 + 1e-9), (model.name, t, order, reps, err, cb)


def test_first_order_equals_zz_outer_second_order_on_z_observables():
    # S2^r = a S1^r a^-1 with a = exp(-i dt/2 H_ZZ): from |0...0>, Z-type observables can't tell
    # first order from ZZ-outer second order. Group order [X, ZZ] avoids this.
    base = tfim(6)
    zz_outer = Model("tfim_zz_outer", 6, base.groups[::-1], base.init_bits, base.params)
    prop = Propagator(zz_outer)
    obs = {k: v for k, v in base.observables().items() if k in ("mz", "zz_mid")}
    exact = prop.exact_state(1.0)
    m1 = state_metrics(exact, prop.trotter_state(1, 8, 1.0), obs)
    m2 = state_metrics(exact, prop.trotter_state(2, 8, 1.0), obs)
    for k in ("mz_err", "zz_mid_err"):
        assert abs(m1[k] - m2[k]) < 1e-12, k
    assert abs(m1["infidelity"] - m2["infidelity"]) > 1e-6  # the states themselves differ


def test_device_pipeline_without_noise_reproduces_numpy():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qtrotter.noisy import density_matrix, noisy_simulator

    backend = FakeTorino()
    for model in (tfim(4), xxz(4)):
        layout = choose_layout(model, backend)
        tqc = device_circuit(trotter_circuit(model, 2, 3, 0.9), backend, layout)
        rho = density_matrix(tqc, noisy_simulator(backend, noise=False))
        psi = Propagator(model).trotter_state(2, 3, 0.9)
        assert np.real(np.vdot(psi, rho @ psi)) > 1 - 1e-9, model.name
        noisy = density_matrix(tqc, noisy_simulator(backend))
        # Aer's relaxation channels leave the trace ~1e-8 short of 1.
        assert abs(np.trace(noisy) - 1) < 1e-6 and np.real(np.vdot(psi, noisy @ psi)) < 1 - 1e-4


def test_scaled_backend():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qtrotter.circuits import gate_error_sum
    from qtrotter.noisy import density_matrix, noisy_simulator, scaled_backend

    backend = FakeTorino()
    t1 = backend.target.qubit_properties[0].t1
    model = tfim(4)
    tqc = device_circuit(trotter_circuit(model, 2, 3, 0.9), backend, choose_layout(model, backend))
    psi = Propagator(model).trotter_state(2, 3, 0.9)
    infidelity = {}
    for scale in (1, 0.1, 1e-6):
        scaled = scaled_backend(backend, scale)
        assert np.isclose(gate_error_sum(tqc, scaled), scale * gate_error_sum(tqc, backend))
        assert np.isclose(scaled.target.qubit_properties[0].t1, t1 / scale)
        rho = density_matrix(tqc, noisy_simulator(scaled))
        infidelity[scale] = 1 - np.real(np.vdot(psi, rho @ psi))
    assert backend.target.qubit_properties[0].t1 == t1  # the original is not modified
    assert infidelity[1e-6] < 1e-5 and 0.05 < infidelity[0.1] / infidelity[1] < 0.2, infidelity


def test_restricted_noise_model():
    """density_matrix keeps only the noise on the circuit's qubits; the state must not change."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qtrotter.noisy import density_matrix, drop_idle_qubits, noisy_simulator

    backend = FakeTorino()
    model = tfim(5)
    tqc = device_circuit(trotter_circuit(model, 2, 3, 0.9), backend, choose_layout(model, backend))
    sim = noisy_simulator(backend)
    full = drop_idle_qubits(tqc)
    full.save_density_matrix(qubits=full.layout.final_index_layout())
    reference = np.asarray(sim.run(full).result().data()["density_matrix"])
    assert np.abs(density_matrix(tqc, sim) - reference).max() < 1e-12


def test_patches():
    """Every local Pauli has one owning patch, and a patch as long as the chain is the chain."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qtrotter.hardware import expectations
    from qtrotter.noisy import density_matrix, noisy_simulator
    from qtrotter.patches import Patches, owners

    model = tfim(12)
    for w in (6, 8, 10):
        names = [k for owned in owners(model, w).values() for k in owned]
        assert sorted(names) == sorted(model.local_paulis()), w
    backend = FakeTorino()
    model = xxz(5)
    layout = choose_layout(model, backend)
    sim = noisy_simulator(backend)
    whole = Patches(model, layout, 5)
    noisy = whole.noisy(whole.circuits(2, 3, 0.9, backend), sim)
    full = expectations(density_matrix(device_circuit(trotter_circuit(model, 2, 3, 0.9), backend,
                                                      layout), sim), model.local_paulis())
    assert max(abs(noisy[k] - full[k]) for k in full) < 1e-12
    exact = expectations(Propagator(model).exact_state(0.9), model.local_paulis())
    assert max(abs(whole.exact(0.9)[k] - exact[k]) for k in exact) < 1e-12


def test_longest_path():
    """A 7-site line with a 2-site branch off site 3: the longest simple path is the line."""
    from qtrotter.hardware import longest_path

    edges = [(i, i + 1) for i in range(6)] + [(3, 7), (7, 8)]
    graph = {q: set() for q in range(9)}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    path = longest_path(graph, restarts=20)
    assert len(path) == 7 and all(b in graph[a] for a, b in zip(path, path[1:])), path


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)

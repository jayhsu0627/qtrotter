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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)

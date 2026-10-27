"""Trotter circuits in Qiskit, and transpilation onto a device with a fixed qubit layout."""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter


def trotter_circuit(model, order, reps, t):
    """Initial-state preparation followed by `reps` steps of the order-`order` formula."""
    qc = QuantumCircuit(model.n)
    for i, bit in enumerate(model.init_bits):
        if bit == "1":
            qc.x(i)
    synth = LieTrotter(reps=reps) if order == 1 else SuzukiTrotter(order=order, reps=reps)
    qc.append(PauliEvolutionGate(list(model.groups), time=t, synthesis=synth), range(model.n))
    return qc


def logical_circuit(qc):
    """Synthesize to basis gates with no device mapping, so qubit order is kept.

    Needed before Statevector/Operator: on an unsynthesized PauliEvolutionGate they use the
    gate's exact matrix exponential, which silently reports zero Trotter error.
    """
    return transpile(qc, basis_gates=["rz", "rx", "h", "cx"], optimization_level=0)


def choose_layout(model, backend, seed=1, optimization_level=2):
    """Physical qubits for the chain, picked once by the transpiler and reused for every r."""
    tqc = transpile(trotter_circuit(model, 1, 1, 1.0), backend,
                    optimization_level=optimization_level, seed_transpiler=seed)
    return tqc.layout.initial_index_layout(filter_ancillas=True)


def device_circuit(qc, backend, layout, seed=1, optimization_level=2, schedule=True):
    """Transpile onto `layout`; with `schedule`, idle periods become delays, so idle qubits
    pick up T1/T2 noise in the simulator."""
    return transpile(qc, backend, initial_layout=layout, optimization_level=optimization_level,
                     seed_transpiler=seed, scheduling_method="alap" if schedule else None)


def two_qubit_count(tqc):
    return sum(1 for ins in tqc.data if ins.operation.num_qubits == 2)


def gate_error_sum(tqc, backend):
    """Sum of calibrated gate errors over the circuit (delays and readout excluded)."""
    target, total = backend.target, 0.0
    for ins in tqc.data:
        name = ins.operation.name
        qargs = tuple(tqc.find_bit(q).index for q in ins.qubits)
        props = target[name].get(qargs) if name in target.operation_names else None
        if props is not None and props.error is not None:
            total += props.error
    return total

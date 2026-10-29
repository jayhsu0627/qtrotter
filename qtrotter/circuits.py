"""Trotter circuits in Qiskit, and transpilation onto a device with a fixed qubit layout."""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter


def layered(group):
    """The same group with its terms reordered into layers of disjoint support (first fit).

    The terms commute, so the unitary is unchanged. Listed in chain order, the TFIM's ZZ bonds
    each share a qubit with the next, and the synthesized gates run as one serial staircase down
    the chain: about 0.2 us per bond on ibm_fez, 20 us at 100 sites. Layered, they run as two
    layers (even bonds, then odd bonds).
    """
    layers = []  # (qubits used, term indices)
    for k, p in enumerate(group.paulis):
        support = p.z | p.x
        for used, terms in layers:
            if not (used & support).any():
                used |= support
                terms.append(k)
                break
        else:
            layers.append((support.copy(), [k]))
    return group[[k for _, terms in layers for k in terms]]


def trotter_circuit(model, order, reps, t, layer=True):
    """Initial-state preparation followed by `reps` steps of the order-`order` formula.

    `layer=False` keeps each group's terms in model order, as every circuit in results/ before
    the layered fix was built (the 100-qubit ibm_fez run included).
    """
    qc = QuantumCircuit(model.n)
    for i, bit in enumerate(model.init_bits):
        if bit == "1":
            qc.x(i)
    synth = LieTrotter(reps=reps) if order == 1 else SuzukiTrotter(order=order, reps=reps)
    groups = [layered(g) for g in model.groups] if layer else list(model.groups)
    qc.append(PauliEvolutionGate(groups, time=t, synthesis=synth), range(model.n))
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

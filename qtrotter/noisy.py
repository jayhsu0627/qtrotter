"""Noisy density-matrix simulation under a fake-backend noise model (Qiskit Aer)."""
import copy

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import QubitProperties
from qiskit.transpiler import InstructionProperties
from qiskit_aer import AerSimulator


def pick_device(device="auto"):
    """'GPU' if this Aer build can run a density-matrix kernel on one, else 'CPU'."""
    if device != "auto":
        return device
    if "GPU" not in AerSimulator().available_devices():
        return "CPU"
    probe = QuantumCircuit(2)
    probe.h(0)
    probe.cx(0, 1)
    probe.save_density_matrix()
    try:
        ok = AerSimulator(method="density_matrix", device="GPU").run(probe).result().success
    except Exception:
        ok = False
    return "GPU" if ok else "CPU"


def scaled_backend(backend, scale):
    """Copy of `backend` with every calibrated error rate times `scale`: gate and readout errors
    multiplied by it, T1 and T2 divided by it. Gates, durations and the coupling map are unchanged,
    so the same transpiled circuit runs on a uniformly better (scale < 1) device.

    Aer splits each gate error into thermal relaxation (from duration, T1, T2) and depolarizing
    noise; both parts scale, and so does the relaxation on idle qubits.
    """
    if scale == 1:
        return backend
    out = copy.deepcopy(backend)
    target = out.target
    for name in target.operation_names:
        for qargs, props in target[name].items():
            if props is not None and props.error is not None:
                target.update_instruction_properties(
                    name, qargs, InstructionProperties(duration=props.duration, error=props.error * scale))
    # Reading target.qubit_properties returns copies, so the list is replaced, not edited.
    if target.qubit_properties:
        target.qubit_properties = [
            QubitProperties(t1=None if q.t1 is None else q.t1 / scale,
                            t2=None if q.t2 is None else q.t2 / scale, frequency=q.frequency)
            for q in target.qubit_properties]
    return out


def noisy_simulator(backend, device="CPU", precision="double", noise=True):
    if noise:
        return AerSimulator.from_backend(backend, method="density_matrix", device=device,
                                         precision=precision)
    return AerSimulator(method="density_matrix", device=device, precision=precision)


def restricted_noise_model(noise_model, qubits):
    """The errors of `noise_model` that can act on `qubits`: gate and readout errors that touch any
    other qubit are dropped, which changes nothing for a circuit using only `qubits`.

    Aer deep-copies the noise model for every circuit with delays (it adds their relaxation as
    channels). With all 133 qubits of a Heron device that costs about 2 s per run, 25 times the
    simulation itself at 10 qubits. NoiseModel cannot drop errors through its public API, so this
    edits a copy's tables (qiskit-aer 0.18).
    """
    keep = set(qubits)
    out = copy.copy(noise_model)
    out._local_quantum_errors = {}
    for name, errors in noise_model._local_quantum_errors.items():
        kept = {q: e for q, e in errors.items() if keep.issuperset(q)}
        if kept:
            out._local_quantum_errors[name] = kept
    out._local_readout_errors = {q: e for q, e in noise_model._local_readout_errors.items()
                                 if keep.issuperset(q)}
    out._noise_qubits = noise_model._noise_qubits & keep
    out._custom_noise_passes = list(noise_model._custom_noise_passes)
    return out


def drop_idle_qubits(tqc):
    """Remove delays on wires that carry no gates, so Aer can truncate them away."""
    active = {q for ins in tqc.data if ins.operation.name != "delay" for q in ins.qubits}
    out = tqc.copy_empty_like()
    for ins in tqc.data:
        if ins.operation.name == "delay" and not set(ins.qubits) & active:
            continue
        out.append(ins)
    return out


def density_matrix(tqc, sim):
    """Final state of the chain qubits, in chain order (site 0 = least significant bit)."""
    tqc = drop_idle_qubits(tqc)
    tqc.save_density_matrix(qubits=tqc.layout.final_index_layout())
    noise = sim.options.noise_model
    if noise is not None:
        used = {tqc.find_bit(q).index for ins in tqc.data for q in ins.qubits}
        noise = restricted_noise_model(noise, used)
    result = sim.run(tqc, noise_model=noise).result()
    if not result.success:
        raise RuntimeError(result.status)
    return np.asarray(result.data()["density_matrix"])

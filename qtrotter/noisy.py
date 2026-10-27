"""Noisy density-matrix simulation under a fake-backend noise model (Qiskit Aer)."""
import numpy as np
from qiskit import QuantumCircuit
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


def noisy_simulator(backend, device="CPU", precision="double", noise=True):
    if noise:
        return AerSimulator.from_backend(backend, method="density_matrix", device=device,
                                         precision=precision)
    return AerSimulator(method="density_matrix", device=device, precision=precision)


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
    result = sim.run(tqc).result()
    if not result.success:
        raise RuntimeError(result.status)
    return np.asarray(result.data()["density_matrix"])

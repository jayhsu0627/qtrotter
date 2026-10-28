"""GPU check for the Trotter-under-noise project.

Setup (x86_64 Linux with an NVIDIA driver; the GPU build replaces qiskit-aer):
    pip uninstall -y qiskit-aer
    pip install "qiskit[all]~=2.5" qiskit-ibm-runtime qiskit-aer-gpu scipy

Run:
    python gpu_check.py          # 10-qubit chain
    python gpu_check.py 16       # 16 qubits: ~69 GB density matrix in double precision
"""
import sys
import time

import numpy as np
from scipy.sparse.linalg import expm_multiply
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # chain length
STEPS, T = 16, 1.0                                  # Trotter steps, evolution time
PRECISION = "double"                                # "single" halves memory


def tfim(n, J=1.0, h=1.0):
    """Open transverse-field Ising chain, -J sum Z_i Z_{i+1} - h sum X_i, as [X field, ZZ bonds].

    Passed as a list, the X field sits outermost and its half-steps merge between Trotter steps.
    Passed as one operator, the ZZ bonds end up outermost and second order costs ~1.7x the gates.
    """
    x = SparsePauliOp.from_sparse_list([("X", [i], -h) for i in range(n)], num_qubits=n)
    zz = SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], -J) for i in range(n - 1)], num_qubits=n)
    return [x, zz]


# 1. Does this Aer build see a GPU, and can it run a kernel on it?
#    Blackwell cards (sm_120) need a build that includes them; a missing one fails here.
devices = AerSimulator().available_devices()
print("Aer devices:", devices)
device = "CPU"
if "GPU" in devices:
    try:
        probe = QuantumCircuit(2)
        probe.h(0)
        probe.cx(0, 1)
        probe.save_density_matrix()
        res = AerSimulator(method="density_matrix", device="GPU").run(probe).result()
        if res.success:
            device = "GPU"
        else:
            print("GPU run failed:", res.status)
    except Exception as err:
        print("GPU run failed:", err)
print("Using:", device)

# 2. Memory for the noisy density matrix: 4^n complex numbers.
gb = 4**N * (16 if PRECISION == "double" else 8) / 1e9
print(f"{N} qubits, {PRECISION} precision: density matrix needs {gb:.3g} GB (96 GB on the card)")

# 3. One noisy run: second-order Trotter, IBM Torino noise model.
groups = tfim(N)
H = groups[0] + groups[1]
qc = QuantumCircuit(N)
qc.append(PauliEvolutionGate(groups, time=T, synthesis=SuzukiTrotter(order=2, reps=STEPS)), range(N))
backend = FakeTorino()
tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=1)
q0 = tqc.layout.final_index_layout()[0]  # physical qubit holding chain site 0
tqc.save_expectation_value(SparsePauliOp("Z"), [q0])
sim = AerSimulator.from_backend(backend, method="density_matrix", device=device, precision=PRECISION)
start = time.perf_counter()
noisy = sim.run(tqc).result().data()["expectation_value"]
print(f"Noisy <Z_0> after {STEPS} steps: {noisy:+.4f}  ({time.perf_counter() - start:.1f} s on {device})")

# 4. Exact reference from sparse evolution of |0...0>.
psi = np.zeros(2**N, dtype=complex)
psi[0] = 1.0
psi = expm_multiply(-1j * T * H.to_matrix(sparse=True), psi)
z0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=N).to_matrix(sparse=True)
exact = np.real(psi.conj() @ (z0 @ psi))
print(f"Exact <Z_0>: {exact:+.4f}   total error (Trotter + noise): {abs(noisy - exact):.4f}")
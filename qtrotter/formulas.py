"""Product formulas, evaluated exactly in NumPy (no circuit, no noise).

A formula is a time-ordered list of stages (group index, fraction of the step dt). The same
recursion is what Qiskit's LieTrotter / SuzukiTrotter synthesize; tests check the two agree.
"""
import numpy as np


def stages(order, n_groups):
    """One Trotter step as a time-ordered list of (group, fraction of dt)."""
    if order == 1:
        return [(j, 1.0) for j in range(n_groups)]
    if order == 2:
        half = [(j, 0.5) for j in range(n_groups - 1)]
        return half + [(n_groups - 1, 1.0)] + half[::-1]
    if order >= 4 and order % 2 == 0:
        # Suzuki: S_2k(dt) = S_2k-2(u dt)^2 S_2k-2((1-4u) dt) S_2k-2(u dt)^2
        u = 1 / (4 - 4 ** (1 / (order - 1)))
        inner = stages(order - 2, n_groups)
        outer = [(j, a * u) for j, a in inner] * 2
        return outer + [(j, a * (1 - 4 * u)) for j, a in inner] + outer
    raise ValueError(f"order must be 1 or even, got {order}")


def sequence(order, n_groups, reps):
    """All stages of `reps` steps, with adjacent stages of the same group merged."""
    merged = []
    for j, a in stages(order, n_groups) * reps:
        if merged and merged[-1][0] == j:
            merged[-1] = (j, merged[-1][1] + a)
        else:
            merged.append((j, a))
    return merged


class Propagator:
    """Exact exponentials of each group and of H, from one dense eigendecomposition each."""

    def __init__(self, model):
        self.model = model
        self.groups = [np.linalg.eigh(g.to_matrix()) for g in model.groups]
        self.full = np.linalg.eigh(model.hamiltonian.to_matrix())

    @staticmethod
    def _expm(eig, theta, x):
        """exp(-i theta G) @ x, for a vector or a matrix x."""
        w, v = eig
        phase = np.exp(-1j * theta * w)
        y = v.conj().T @ x
        y = phase * y if y.ndim == 1 else phase[:, None] * y
        return v @ y

    def exact_state(self, t, psi=None):
        psi = self.model.initial_state() if psi is None else psi
        return self._expm(self.full, t, psi)

    def trotter_state(self, order, reps, t, psi=None):
        psi = self.model.initial_state() if psi is None else psi
        dt = t / reps
        for j, a in sequence(order, len(self.groups), reps):
            psi = self._expm(self.groups[j], a * dt, psi)
        return psi

    def exact_unitary(self, t):
        w, v = self.full
        return (v * np.exp(-1j * t * w)) @ v.conj().T

    def trotter_unitary(self, order, reps, t):
        step = np.eye(2**self.model.n, dtype=complex)
        for j, a in stages(order, len(self.groups)):
            step = self._expm(self.groups[j], a * t / reps, step)
        return np.linalg.matrix_power(step, reps)

    def group_norms(self):
        return [np.max(np.abs(w)) for w, _ in self.groups]


def state_metrics(psi_exact, approx, observables):
    """Infidelity, trace distance and observable errors of `approx` (vector or density matrix)."""
    if approx.ndim == 1:
        fid = abs(np.vdot(psi_exact, approx)) ** 2
        td = np.sqrt(max(0.0, 1.0 - fid))  # pure states: trace distance = sqrt(1 - F)
        expect = lambda m: np.real(np.vdot(approx, m @ approx))
    else:
        fid = np.real(np.vdot(psi_exact, approx @ psi_exact))
        diff = approx - np.outer(psi_exact, psi_exact.conj())
        td = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff)))
        expect = lambda m: np.real(m.T.multiply(approx).sum())  # Tr(m @ rho), m sparse
    out = {"infidelity": 1.0 - fid, "trace_dist": td}
    for name, op in observables.items():
        m = op.to_matrix(sparse=True)
        ref = np.real(np.vdot(psi_exact, m @ psi_exact))
        out[f"{name}_exact"] = ref
        out[f"{name}_err"] = abs(expect(m) - ref)
    return out

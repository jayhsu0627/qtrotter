"""Open 1D spin chains, split into groups of mutually commuting Pauli terms.

H = sum of groups. Terms inside a group commute, so exp(-i a H_j) is exact, and a product
formula over groups is a product formula over the term list with a fixed ordering. Group 0 is
the outermost group of the second-order formula.

Qubit 0 is the first chain site, and states use Qiskit's little-endian ordering.
"""
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass(frozen=True)
class Model:
    name: str
    n: int
    groups: tuple  # SparsePauliOp per group, in circuit order
    init_bits: str  # initial computational-basis state, site 0 first
    params: dict

    @property
    def hamiltonian(self):
        return sum(self.groups[1:], self.groups[0]).simplify()

    def initial_state(self):
        psi = np.zeros(2**self.n, dtype=complex)
        psi[int(self.init_bits[::-1], 2)] = 1.0
        return psi

    def observables(self):
        """Magnetizations and a mid-chain correlator, as SparsePauliOps."""
        n, c = self.n, self.n // 2 - 1
        z = [("Z", [i], 1.0 / n) for i in range(n)]
        return {
            "mz": SparsePauliOp.from_sparse_list(z, n),
            "mx": SparsePauliOp.from_sparse_list([("X", [i], 1.0 / n) for i in range(n)], n),
            "mstag": SparsePauliOp.from_sparse_list([("Z", [i], (-1) ** i / n) for i in range(n)], n),
            "zz_mid": SparsePauliOp.from_sparse_list([("ZZ", [c, c + 1], 1.0)], n),
            "xx_mid": SparsePauliOp.from_sparse_list([("XX", [c, c + 1], 1.0)], n),
        }


    def local_paulis(self):
        """Every Z_i, X_i, Z_iZ_{i+1}, X_iX_{i+1}: two measurement bases on hardware."""
        n, ops = self.n, {}
        for p in "ZX":
            ops.update({f"{p}{i}": SparsePauliOp.from_sparse_list([(p, [i], 1.0)], n)
                        for i in range(n)})
            ops.update({f"{p}{i}{p}{i + 1}": SparsePauliOp.from_sparse_list([(p * 2, [i, i + 1], 1.0)], n)
                        for i in range(n - 1)})
        return ops


def tfim(n, J=1.0, h=1.0):
    """-J sum Z_i Z_{i+1} - h sum X_i, from |0...0>. Groups: [X field, ZZ bonds].

    Putting the single-qubit group outermost makes the second-order formula cost one ZZ layer
    per step, the same as first order.
    """
    x = SparsePauliOp.from_sparse_list([("X", [i], -h) for i in range(n)], n)
    zz = SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], -J) for i in range(n - 1)], n)
    return Model("tfim", n, (x, zz), "0" * n, {"J": J, "h": h})


def xxz(n, delta=0.5, J=1.0):
    """J sum (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1}), from the Neel state |0101...>.

    Groups: [even bonds, odd bonds]; bonds in one group share no site, so they commute.
    (|0...0> is an eigenstate of this model, hence the Neel start.)
    """
    def bonds(start):
        terms = []
        for i in range(start, n - 1, 2):
            terms += [("XX", [i, i + 1], J), ("YY", [i, i + 1], J), ("ZZ", [i, i + 1], J * delta)]
        return SparsePauliOp.from_sparse_list(terms, n)

    return Model("xxz", n, (bonds(0), bonds(1)), ("01" * n)[:n], {"J": J, "delta": delta})


def build(name, n, **params):
    return {"tfim": tfim, "xxz": xxz}[name](n, **params)

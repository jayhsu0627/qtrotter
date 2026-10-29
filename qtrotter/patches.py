"""Light-cone patches: the local observables of a long chain from short sub-chains.

Everything that moves a local observable comes from inside its light cone: the Trotter error from
the dynamics nearby, and the noise from gates whose effect can spread to it within the evolution
time. A sub-chain (patch) of w sites around the observable, run on the same physical qubits under
the same noise model, then reproduces its exact, Trotterized and noisy values once w exceeds the
light cone, at the cost of a w-qubit simulation whatever the chain length.

Each observable is read from the patch whose cut edges are farthest from it. The chain's own ends
are not cuts, so observables near them see the true boundary.
"""
from dataclasses import replace

import numpy as np

from .circuits import device_circuit, trotter_circuit
from .formulas import Propagator
from .hardware import expectations
from .models import build
from .noisy import density_matrix


def patch_starts(n, w, stride=2):
    """First sites of the patches covering an n-site chain. Even starts keep each bond in the same
    group as in the full chain (the XXZ model splits even and odd bonds)."""
    if w >= n:
        return [0]
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] != n - w:
        starts.append(n - w)
    if any(a % 2 for a in starts):
        raise ValueError("use an even stride and an even n - w")
    return starts


def owners(model, w, stride=2):
    """{patch start: {chain observable name: patch observable name}} over Model.local_paulis()."""
    n, starts = model.n, patch_starts(model.n, w, stride)

    def margin(a, sites):
        left = min(sites) - a if a > 0 else np.inf
        right = a + w - 1 - max(sites) if a + w < n else np.inf
        return min(left, right)

    out = {a: {} for a in starts}
    for p in "ZX":
        for i in range(n):
            for sites, name in [((i,), f"{p}{i}"), ((i, i + 1), f"{p}{i}{p}{i + 1}")]:
                if max(sites) >= n:
                    continue
                a = max(starts, key=lambda a: margin(a, sites))
                out[a][name] = "".join(f"{p}{s - a}" for s in sites)
    return out


class Patches:
    """Patch models and physical qubits for one chain (`layout`: physical qubit of each site)."""

    def __init__(self, model, layout, w, stride=2):
        self.model, self.layout, self.w = model, list(layout), min(w, model.n)
        self.owners = owners(model, self.w, stride)
        self.models = {a: replace(build(model.name, self.w, **model.params),
                                  init_bits=model.init_bits[a:a + self.w]) for a in self.owners}
        # Patches with the same initial bits are the same model: one propagator each.
        self.props = {}
        for sub in self.models.values():
            if sub.init_bits not in self.props:
                self.props[sub.init_bits] = Propagator(sub)

    def _collect(self, state_of):
        """Chain observable values, from a function giving each patch start's state."""
        values = {}
        for a, sub in self.models.items():
            paulis = sub.local_paulis()
            local = expectations(state_of(a), {k: paulis[k] for k in self.owners[a].values()})
            values.update({name: local[k] for name, k in self.owners[a].items()})
        return values

    def _noiseless(self, evolve):
        states = {bits: evolve(prop) for bits, prop in self.props.items()}
        return self._collect(lambda a: states[self.models[a].init_bits])

    def exact(self, t):
        return self._noiseless(lambda prop: prop.exact_state(t))

    def trotter(self, order, reps, t):
        return self._noiseless(lambda prop: prop.trotter_state(order, reps, t))

    def circuits(self, order, reps, t, backend, **opts):
        """Transpiled patch circuits, one per patch, on the patch's own physical qubits."""
        return {a: device_circuit(trotter_circuit(sub, order, reps, t), backend,
                                  self.layout[a:a + self.w], **opts)
                for a, sub in self.models.items()}

    def noisy(self, circuits, sim):
        return self._collect(lambda a: density_matrix(circuits[a], sim))

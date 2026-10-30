"""Follow-up: could fixed coherent errors explain what the layered ibm_fez run has beyond its
prediction? TFIM only, noiseless statevector, the same layered first- and second-order steps as the
circuits ([X field, ZZ bonds], half-steps of the field merged at second order).

    python scripts/coherent_errors.py [config.yaml] [section]   # section defaults to "coherent_errors"

Each draw fixes an over-rotation of every bond's ZZ angle per step, normal with std sigma_zz, and a
Z phase on every site per step (residual detuning), std sigma_z; both stay the same at every r and
order, as a miscalibration would. For every draw, order and r, writes the RMS over all local Paulis
of (with errors - without), and the correlation of that difference between the two orders at the
same r. The infidelity a ZZ over-rotation eps adds to a gate is about eps^2/5, so eps = 0.05 rad
costs 5e-4, well below the CZ errors that randomized benchmarking reports.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
section = sys.argv[2] if len(sys.argv) > 2 else "coherent_errors"
run = cfg[section]
J, h = cfg["models"]["tfim"]["J"], cfg["models"]["tfim"]["h"]
n, t = run["n"], run["t"]
z = 1 - 2 * ((np.arange(2**n)[None, :] >> np.arange(n)[:, None]) & 1)  # z[i, k] = <k|Z_i|k>
zz = z[:-1] * z[1:]


def rx_layer(psi, theta):
    """RX(theta) on every qubit (qubit q is axis n - 1 - q: little-endian)."""
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    psi = psi.reshape([2] * n)
    for q in range(n):
        a0, a1 = np.take(psi, 0, axis=n - 1 - q), np.take(psi, 1, axis=n - 1 - q)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=n - 1 - q)
    return psi.reshape(-1)


def local_paulis(psi):
    """Z_i, Z_iZ_i+1, X_i, X_iX_i+1, in that order."""
    p, k = np.abs(psi) ** 2, np.arange(2**n)
    x = [np.real(np.vdot(psi, psi[k ^ (1 << i)])) for i in range(n)]
    xx = [np.real(np.vdot(psi, psi[k ^ (3 << i)])) for i in range(n - 1)]
    return np.concatenate([z @ p, zz @ p, x, xx])


def trotter(order, reps, eps_zz, phi_z):
    """H = -J sum ZZ - h sum X: the field step is RX(-2 h dt), the bond step RZZ(-2 J dt)."""
    dt = t / reps
    bonds = np.exp(-0.5j * ((-2 * J * dt + eps_zz)[:, None] * zz).sum(0))
    phase = np.exp(-0.5j * (phi_z[:, None] * z).sum(0))
    psi = np.zeros(2**n, complex)
    psi[0] = 1.0
    for k in range(reps):
        psi = rx_layer(psi, -2 * h * dt if order == 1 or k > 0 else -h * dt)
        psi = phase * (bonds * psi)
    return local_paulis(rx_layer(psi, -h * dt) if order == 2 else psi)


rng = np.random.default_rng(run["seed"])
ideal = {(o, r): trotter(o, r, np.zeros(n - 1), np.zeros(n)) for o in run["orders"] for r in run["steps"]}
rows = []
for sigma_zz, sigma_z in run["sigmas"]:
    for draw in range(run["draws"]):
        eps, phi = rng.normal(0, sigma_zz, n - 1), rng.normal(0, sigma_z, n)
        for r in run["steps"]:
            d = {o: trotter(o, r, eps, phi) - ideal[(o, r)] for o in run["orders"]}
            corr = np.corrcoef(d[1], d[2])[0, 1] if {1, 2} <= set(d) else np.nan
            rows += [{"sigma_zz": sigma_zz, "sigma_z": sigma_z, "draw": draw, "order": o, "r": r,
                      "gap": float(np.sqrt(np.mean(d[o] ** 2))), "corr_orders": corr} for o in d]
    print(f"sigma_zz={sigma_zz} sigma_z={sigma_z}: {run['draws']} draws", flush=True)
out = ROOT / "results" / run["output"]
pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
print("wrote", out)

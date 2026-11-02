"""Follow-up: do fixed coherent gate errors explain what the layered ibm_fez run has beyond its
prediction? Fits per-step over-rotations to the device's local Paulis at small r and tests them at
the larger r the fit never sees.

    python scripts/fit_coherent.py [config.yaml] [section]   # section defaults to "coherent_fit"

Model: every Trotter step carries the same unknown errors, as a miscalibration would: an
over-rotation eps_b of each bond's ZZ angle, a Z phase phi_i on each site (residual detuning), and
an over-rotation delta_i of each site's X-field angle (at every field layer, so r + 1 times at
second order). The device's values are modelled as the patch prediction plus the change these
errors make to the noiseless Trotter values, shrunk by the prediction's own damping factor per
(order, r). The noiseless values with errors come from the same light-cone patches as the
prediction (statevector, all patches at once), in the step order of coherent_errors.py.

Fit: least squares weighted by the device's shot-noise std, on train_steps and both orders, with a
Gaussian prior of std prior_rad on every angle. Also fitted, for comparison only: ZZ angles alone
(what twirling the two-qubit gates would turn stochastic) and single-qubit angles alone. Baselines:
the prediction itself; the deviation at the last training r frozen for every larger r (an error
that does not grow with r); a straight line in r through each observable's training deviations.

Writes results/<output>.csv (excess RMS over shot noise, and RMS error against exact evolution,
per model, order and r) and results/<output>_angles.csv (fitted angles per site and bond).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import build  # noqa: E402
from qtrotter import hardware as hw  # noqa: E402
from qtrotter.patches import owners  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
section = sys.argv[2] if len(sys.argv) > 2 else "coherent_fit"
run = cfg[section]
J, h = cfg["models"]["tfim"]["J"], cfg["models"]["tfim"]["h"]
values = pd.read_csv(ROOT / "results" / run["values"])
(t,) = values[values.source == "hardware"].t.unique()
dev, std = hw.wide(values, "hardware"), hw.wide(values, "hardware", "std")
pred, trot = hw.wide(values, "patches"), hw.wide(values, "trotter")
exact = values[values.source == "exact"].set_index("observable").value[dev.columns].to_numpy()
names = list(dev.columns)
n = 1 + max(max(hw.sites(k)) for k in names)
orders = sorted(dev.index.get_level_values("order").unique())
train, test = run["train_steps"], run["test_steps"]

# Patches: every observable is read from its owner patch, as in qtrotter.patches.
w = run["width"]
own = owners(build("tfim", n, J=J, h=h), w, run["stride"])
starts = sorted(own)
P = len(starts)
sites = np.array([[a + q for q in range(w)] for a in starts])  # chain site of each patch qubit
z = 1 - 2 * ((np.arange(2**w)[None, :] >> np.arange(w)[:, None]) & 1)  # z[q, k] = <k|Z_q|k>
zz = z[:-1] * z[1:]
k_all = np.arange(2**w)
lookup = {}  # chain observable -> (patch, kind, local qubit)
for p, a in enumerate(starts):
    for name, local in own[a].items():
        q = min(hw.sites(local))
        lookup[name] = (p, ("ZZ" if local.count("Z") == 2 else "XX" if local.count("X") == 2 else local[0]), q)
owner_patch = np.array([lookup[k][0] for k in names])


def rx_layer(psi, theta):
    """RX(theta[p, q]) on qubit q of every patch p (qubit q is axis w - q: little-endian)."""
    psi = psi.reshape([P] + [2] * w)
    for q in range(w):
        c = np.cos(theta[:, q] / 2).reshape([P] + [1] * (w - 1))
        s = -1j * np.sin(theta[:, q] / 2).reshape([P] + [1] * (w - 1))
        a0, a1 = np.take(psi, 0, axis=w - q), np.take(psi, 1, axis=w - q)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=w - q)
    return psi.reshape(P, -1)


def observables(psi):
    """Every chain observable, read from its owner patch."""
    prob = np.abs(psi) ** 2
    zv, zzv = prob @ z.T, prob @ zz.T
    xv = np.stack([np.real((psi.conj() * psi[:, k_all ^ (1 << q)]).sum(1)) for q in range(w)], 1)
    xxv = np.stack([np.real((psi.conj() * psi[:, k_all ^ (3 << q)]).sum(1)) for q in range(w - 1)], 1)
    table = {"Z": zv, "ZZ": zzv, "X": xv, "XX": xxv}
    return np.array([table[kind][p, q] for p, kind, q in (lookup[k] for k in names)])


def split(theta):
    """Chain angles -> per-patch arrays: eps (P, w-1), phi (P, w), delta (P, w)."""
    eps, phi, delta = theta[:n - 1], theta[n - 1:2 * n - 1], theta[2 * n - 1:]
    return eps[sites[:, :-1]], phi[sites], delta[sites]


def trotter(order, reps, theta):
    """H = -J sum ZZ - h sum X: field step RX(-2 h dt), bond step RZZ(-2 J dt), plus the errors."""
    eps, phi, delta = split(theta)
    dt = t / reps
    bonds = np.exp(-0.5j * ((-2 * J * dt + eps)[:, :, None] * zz[None]).sum(1))
    phase = np.exp(-0.5j * (phi[:, :, None] * z[None]).sum(1))
    psi = np.zeros((P, 2**w), complex)
    psi[:, 0] = 1.0
    for k in range(reps):
        field = -2 * h * dt if order == 1 or k > 0 else -h * dt
        psi = phase * (bonds * rx_layer(psi, field + delta))
    if order == 2:
        psi = rx_layer(psi, -h * dt + delta)
    return observables(psi)


n_par = 3 * n - 1
zero = np.zeros(n_par)
steps = sorted(set(train) | set(test))
base = {(o, r): trotter(o, r, zero) for o in orders for r in steps}
check = max(np.abs(base[key] - trot.loc[key].to_numpy()).max() for key in base)
print(f"patch statevector vs the run's noiseless Trotter values: max diff {check:.1e}", flush=True)
assert check < 1e-5, "step conventions differ from the prediction's"
shrink = {key: float((pred.loc[key] * trot.loc[key]).sum() / (trot.loc[key] ** 2).sum()) for key in base}


def model(theta, key):
    return pred.loc[key].to_numpy() + shrink[key] * (trotter(*key, theta) - base[key])


def fit(free):
    """MAP angles with only the parameters in `free` (a boolean mask) allowed to move."""
    keys = [(o, r) for o in orders for r in train]
    m = len(names)
    sparsity = lil_matrix((m * len(keys) + int(free.sum()), int(free.sum())), dtype=int)
    touches = [set() for _ in range(n_par)]  # patches each parameter enters
    for p in range(P):
        for i in sites[p]:
            touches[n - 1 + i].add(p)
            touches[2 * n - 1 + i].add(p)
        for i in sites[p][:-1]:
            touches[i].add(p)
    cols = np.flatnonzero(free)
    for j, c in enumerate(cols):
        rows = np.flatnonzero(np.isin(owner_patch, list(touches[c])))
        for b in range(len(keys)):
            sparsity[b * m + rows, j] = 1
        sparsity[m * len(keys) + j, j] = 1

    def residuals(x):
        theta = zero.copy()
        theta[cols] = x
        res = [(dev.loc[key].to_numpy() - model(theta, key)) / std.loc[key].to_numpy() for key in keys]
        return np.concatenate(res + [x / run["prior_rad"]])

    sol = least_squares(residuals, np.zeros(len(cols)), jac_sparsity=sparsity, x_scale=0.05, verbose=0)
    theta = zero.copy()
    theta[cols] = sol.x
    print(f"  {int(free.sum())} angles, cost {sol.cost:.0f}, {sol.nfev} evaluations, {sol.message}", flush=True)
    return theta


kinds = np.array(["zz"] * (n - 1) + ["phase"] * n + ["field"] * n)
fits = {}
for label, free in [("coherent", kinds != ""), ("coherent_zz", kinds == "zz"),
                    ("coherent_1q", kinds != "zz")]:
    print(f"fitting {label}", flush=True)
    fits[label] = fit(free)

last = max(train)
rows = []
for o in orders:
    d_train = np.array([dev.loc[(o, r)].to_numpy() - pred.loc[(o, r)].to_numpy() for r in train])
    slope, icpt = np.polyfit(np.array(train, float), d_train, 1)
    for r in steps:
        key = (o, r)
        guesses = {"prediction": pred.loc[key].to_numpy(),
                   "frozen": pred.loc[key].to_numpy() + (d_train[-1] if r > last else d_train[train.index(r)]),
                   "linear": pred.loc[key].to_numpy() + icpt + slope * r}
        guesses.update({label: model(theta, key) for label, theta in fits.items()})
        d, var = dev.loc[key].to_numpy(), std.loc[key].to_numpy() ** 2
        for label, g in guesses.items():
            rows.append({"model": label, "order": o, "r": r, "split": "train" if r in train else "test",
                         "excess": np.sqrt(max(((d - g) ** 2 - var).mean(), 0.0)),
                         "error": np.sqrt(((g - exact) ** 2).mean())})
        rows.append({"model": "device", "order": o, "r": r, "split": "train" if r in train else "test",
                     "excess": 0.0, "error": np.sqrt(max(((d - exact) ** 2 - var).mean(), 0.0))})
table = pd.DataFrame(rows)
angles = pd.DataFrame([{"fit": label, "kind": kinds[j], "index": j - {"zz": 0, "phase": n - 1, "field": 2 * n - 1}[kinds[j]],
                        "angle": theta[j]} for label, theta in fits.items() for j in range(n_par)])
out = ROOT / "results" / run["output"]
table.to_csv(f"{out}.csv", index=False, float_format="%.6g")
angles.to_csv(f"{out}_angles.csv", index=False, float_format="%.6g")

# The pre-stated criteria.
test_rms = table[table.split == "test"].groupby("model").excess.apply(lambda e: np.sqrt((e**2).mean()))
rms_angle = float(np.sqrt((fits["coherent"] ** 2).mean()))
print("\nexcess RMS over shot noise on the test steps:", test_rms.round(4).to_dict())
print(f"(1) halves the prediction's: {test_rms.coherent:.4f} vs {test_rms.prediction:.4f} ->",
      test_rms.coherent <= 0.5 * test_rms.prediction)
print(f"(2) beats frozen {test_rms.frozen:.4f} and linear {test_rms.linear:.4f} ->",
      test_rms.coherent < min(test_rms.frozen, test_rms.linear))
print(f"(3) RMS angle {rms_angle:.4f} rad < {run['max_rms_rad']} ->", rms_angle < run["max_rms_rad"])
print("wrote", f"{out}.csv", f"{out}_angles.csv")

"""Follow-up: local observables of long chains from light-cone patches (qtrotter.patches).

    python scripts/run_patches.py [config.yaml] [section]     # section defaults to "patches"

For each chain length, patch width, t, order, r and noise scale, writes three RMS errors over all
local Paulis of the chain (Model.local_paulis), each read from its own patch under the device's
noise model (density matrix, Aer): noisy against exact (`rms`), the noiseless Trotter state
against exact (`trotter_rms`), and noisy against noiseless Trotter (`noise_rms`).
Appends to results/<output> and skips rows already there.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import yaml
from qiskit_ibm_runtime import fake_provider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import build  # noqa: E402
from qtrotter.circuits import choose_layout  # noqa: E402
from qtrotter.hardware import rms_error  # noqa: E402
from qtrotter.noisy import noisy_simulator, pick_device, scaled_backend  # noqa: E402
from qtrotter.patches import Patches  # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.yaml"))
section = sys.argv[2] if len(sys.argv) > 2 else "patches"
run = cfg[section]
backend = getattr(fake_provider, run["backend"])()
device = pick_device(run["device"])
devices = {s: scaled_backend(backend, s) for s in run["noise_scales"]}
sims = {s: noisy_simulator(b, device=device, precision=run["precision"]) for s, b in devices.items()}
max_steps = run.get("max_steps", {})


def step_cap(order, scale):
    """As in run_noisy.py: fewer steps on worse devices."""
    cap = max_steps.get(order, np.inf)
    return cap * (min(devices) / scale) ** run.get("cap_exponent", 0)


opts = {"seed": run["seed"], "optimization_level": run["optimization_level"],
        "schedule": run["schedule"]}
print(f"[{section}] {run['backend']} noise model, scales {list(devices)}, density matrix on {device}",
      flush=True)

out = ROOT / "results" / run["output"]
key = ["model", "n", "width", "t", "order", "r", "noise_scale"]
done = set(map(tuple, pd.read_csv(out)[key].values.tolist())) if out.exists() else set()

for name in run["models"]:
    for n in run["chains"]:
        model = build(name, n, **cfg["models"][name])
        layout = choose_layout(model, backend, run["seed"], run["optimization_level"])
        for w in run["widths"]:
            if w > n:
                continue
            patches = Patches(model, layout, w, run["stride"])
            for t in run["times"]:
                exact = patches.exact(t)
                for order in run["orders"]:
                    start, rows = time.perf_counter(), []
                    for reps in run["steps"]:
                        todo = [s for s in devices if (name, n, w, t, order, reps, s) not in done
                                and reps <= step_cap(order, s)]
                        if not todo:
                            continue
                        circuits = patches.circuits(order, reps, t, backend, **opts)
                        trotter = patches.trotter(order, reps, t)
                        for s in todo:
                            noisy = patches.noisy(circuits, sims[s])
                            rows.append({"model": name, "n": n, "width": w, "t": t, "order": order,
                                         "r": reps, "noise_scale": s, "patches": len(circuits),
                                         "rms": rms_error(noisy, exact),
                                         "trotter_rms": rms_error(trotter, exact),
                                         "noise_rms": rms_error(noisy, trotter)})
                    if rows:
                        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                    print(f"{name} n={n} w={w} t={t} p={order}: {len(rows)} runs, "
                          f"{time.perf_counter() - start:.1f} s", flush=True)
print("wrote", out)

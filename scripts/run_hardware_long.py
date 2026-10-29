"""Follow-up: r* for local observables on a ~100-qubit chain of an IBM device, predicted from
light-cone patches (qtrotter.patches) before the run.

    python scripts/run_hardware_long.py dry-run   # layout and predictions; uses no QPU time
    python scripts/run_hardware_long.py submit    # submit one Estimator job (commit the predictions first)
    python scripts/run_hardware_long.py fetch     # collect the submitted job's results

Layout: the longest chain found whose couplers and readout pass the thresholds in config.yaml
(hardware.clean_graph, hardware.longest_path), cut to the `max_sites` sites with the lowest summed
two-qubit error. Predictions, from the device's calibration at dry-run time: every local Pauli from
patches (exact, noiseless Trotter and noisy values), and the global-damping estimate of question 3
(every value shrunk by exp(-p r), p the per-step gate error of the whole chain) for comparison.
Outputs in results/: hardware_long_values.csv (long format, as hardware_values.csv),
hardware_long_circuits.csv, hardware_long_job.json.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import yaml
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import build  # noqa: E402
from qtrotter import hardware as hw  # noqa: E402
from qtrotter.noisy import noisy_simulator, pick_device  # noqa: E402
from qtrotter.patches import Patches  # noqa: E402
from qtrotter.predict import per_step_error  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "dry-run"
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
run = cfg["hardware_long"]
out = ROOT / "results"
job_file = out / "hardware_long_job.json"
service = QiskitRuntimeService()


def chain_layout(backend):
    """Longest clean chain, cut to the max_sites window with the lowest summed two-qubit error."""
    graph = hw.clean_graph(backend, run["max_2q_error"], run["max_readout_error"])
    path = hw.longest_path(graph, seed=run["seed"])
    n = min(len(path), run["max_sites"])
    n -= n % 2  # patches start on even sites
    target = backend.target
    gate = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    err = [(target[gate].get((a, b)) or target[gate].get((b, a))).error for a, b in zip(path[:-1], path[1:])]
    start = min(range(len(path) - n + 1), key=lambda i: sum(err[i:i + n - 1]))
    return path[start:start + n]


def long_rows(source, t, order, reps, values, stds=None):
    return [{"source": source, "t": t, "order": order, "r": reps, "observable": k, "value": v,
             "std": None if stds is None else stds[k]} for k, v in values.items()]


if mode in ("dry-run", "submit"):
    backend = service.backend(run["backend"])
    calibrated = backend.properties().last_update_date.isoformat()
    if mode == "dry-run":
        layout = chain_layout(backend)
    else:  # the layout the predictions were made for
        layout = json.load(open(job_file))["layout"]
    model = build(run["model"], len(layout), **cfg["models"][run["model"]])
    paulis = model.local_paulis()
    opts = {"seed": run["seed"], "optimization_level": run["optimization_level"]}
    jobs = [dict(j, t=t) for t in run["times"]
            for j in hw.build(model, run["orders"], run["steps"], t, backend, layout, **opts)]
    print(f"{backend.name} (calibrated {calibrated}): {len(layout)}-site chain, {len(jobs)} circuits",
          flush=True)

    if mode == "dry-run":
        start = time.perf_counter()
        patches = Patches(model, layout, run["width"], run["stride"])
        sim = noisy_simulator(backend, device=pick_device("auto"))
        rows = []
        for t in run["times"]:
            rows += long_rows("exact", t, 0, 0, patches.exact(t))
            for order in run["orders"]:
                for reps in run["steps"]:
                    rows += long_rows("trotter", t, order, reps, patches.trotter(order, reps, t))
                    noisy = patches.noisy(patches.circuits(order, reps, t, backend, **opts), sim)
                    rows += long_rows("patches", t, order, reps, noisy)
                print(f"t={t} order {order}: patches done, {time.perf_counter() - start:.0f} s", flush=True)
        values = pd.DataFrame(rows)

        # Global damping (question 3's observable predictor): every value shrunk by exp(-p r).
        meta = pd.DataFrame([{k: v for k, v in j.items() if k != "circuit"} for j in jobs])
        for (t, order), g in meta.groupby(["t", "order"]):
            p_step = per_step_error(g.r, g.gate_error_sum)
            for reps in g.r:
                tro = values[(values.source == "trotter") & (values.t == t) & (values.order == order)
                             & (values.r == reps)].set_index("observable").value
                rows += long_rows("global_damping", t, order, reps,
                                  (np.exp(-p_step * reps) * tro).to_dict())
        pd.DataFrame(rows).to_csv(out / "hardware_long_values.csv", index=False, float_format="%.8g")
        meta["backend"], meta["calibrated"] = backend.name, calibrated
        meta.to_csv(out / "hardware_long_circuits.csv", index=False)
        json.dump({"backend": backend.name, "calibrated": calibrated, "layout": layout,
                   "config": run}, open(job_file, "w"), indent=1)
        print("wrote predictions; commit them before submitting", flush=True)

    else:
        record = json.load(open(job_file))
        names, pubs = hw.estimator_pubs(jobs, paulis)
        estimator = EstimatorV2(mode=backend)
        estimator.options.default_shots = run["shots"]
        estimator.options.resilience_level = run["resilience_level"]
        estimator.options.max_execution_time = run["max_execution_time"]
        job = estimator.run(pubs)
        record.update({"job_id": job.job_id(), "submitted": datetime.now(timezone.utc).isoformat(),
                       "calibrated_at_submission": calibrated, "observables": names,
                       "pubs": [{"t": j["t"], "order": j["order"], "r": j["r"]} for j in jobs]})
        json.dump(record, open(job_file, "w"), indent=1)
        print("submitted job", job.job_id())

elif mode == "fetch":
    record = json.load(open(job_file))
    job = service.job(record["job_id"])
    print("status:", job.status())
    result = job.result()
    rows = []
    for pub, res in zip(record["pubs"], result):
        rows += long_rows("hardware", pub["t"], pub["order"], pub["r"],
                          dict(zip(record["observables"], res.data.evs)),
                          dict(zip(record["observables"], res.data.stds)))
    values = pd.read_csv(out / "hardware_long_values.csv")
    values = values[values.source != "hardware"]
    pd.concat([values, pd.DataFrame(rows)]).to_csv(out / "hardware_long_values.csv", index=False,
                                                    float_format="%.8g")
    record["usage_seconds"] = job.usage()
    record["finished"] = datetime.now(timezone.utc).isoformat()
    json.dump(record, open(job_file, "w"), indent=1)
    print(f"fetched {len(rows)} values; QPU usage {record['usage_seconds']} s")

"""Question 3: one TFIM chain on a real IBM device, against that device's own noise model.

    python scripts/run_hardware.py dry-run   # noise-model simulation only; uses no QPU time
    python scripts/run_hardware.py submit    # simulate, then submit one Estimator job
    python scripts/run_hardware.py fetch     # collect the submitted job's results

Needs saved IBM Quantum credentials (QiskitRuntimeService.save_account). Outputs, in results/:
hardware_circuits.csv (cost per circuit), hardware_values.csv (every local Pauli from exact
evolution, noiseless Trotter, the noise model, and the device), hardware_job.json.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Small dense matrices: many BLAS threads slow this down.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import pandas as pd
import yaml
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import Propagator, build  # noqa: E402
from qtrotter import hardware as hw  # noqa: E402
from qtrotter.circuits import choose_layout  # noqa: E402
from qtrotter.noisy import noisy_simulator, pick_device  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "dry-run"
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
run = cfg["hardware"]
out = ROOT / "results"
job_file = out / "hardware_job.json"
service = QiskitRuntimeService()


def long_rows(source, order, reps, values, stds=None):
    return [{"source": source, "order": order, "r": reps, "observable": k, "value": v,
             "std": None if stds is None else stds[k]} for k, v in values.items()]


if mode in ("dry-run", "submit"):
    backend = service.backend(run["backend"])
    model = build(run["model"], run["n"], **cfg["models"][run["model"]])
    paulis = model.local_paulis()
    layout = choose_layout(model, backend, run["seed"], run["optimization_level"])
    jobs = hw.build(model, run["orders"], run["steps"], run["t"], backend, layout,
                    run["seed"], run["optimization_level"])
    print(f"{backend.name}: layout {layout}, {len(jobs)} circuits", flush=True)

    start = time.perf_counter()
    prop = Propagator(model)
    rows = long_rows("exact", 0, 0, hw.expectations(prop.exact_state(run["t"]), paulis))
    sim = noisy_simulator(backend, device=pick_device("auto"))
    for j, noisy in zip(jobs, hw.noise_model_values(jobs, sim, backend, paulis)):
        trotter = hw.expectations(prop.trotter_state(j["order"], j["r"], run["t"]), paulis)
        rows += long_rows("trotter", j["order"], j["r"], trotter)
        rows += long_rows("noise_model", j["order"], j["r"], noisy)
    print(f"noise-model simulation: {time.perf_counter() - start:.0f} s", flush=True)

    meta = pd.DataFrame([{k: v for k, v in j.items() if k != "circuit"} for j in jobs])
    meta["backend"], meta["layout"] = backend.name, " ".join(map(str, layout))
    meta.to_csv(out / "hardware_circuits.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "hardware_values.csv", index=False)

    if mode == "submit":
        names, pubs = hw.estimator_pubs(jobs, paulis)
        estimator = EstimatorV2(mode=backend)
        estimator.options.default_shots = run["shots"]
        estimator.options.resilience_level = run["resilience_level"]
        job = estimator.run(pubs)
        record = {"job_id": job.job_id(), "backend": backend.name, "layout": layout,
                  "submitted": datetime.now(timezone.utc).isoformat(), "observables": names,
                  "pubs": [{"order": j["order"], "r": j["r"]} for j in jobs], "config": run}
        json.dump(record, open(job_file, "w"), indent=1)
        print("submitted job", job.job_id())

elif mode == "fetch":
    record = json.load(open(job_file))
    job = service.job(record["job_id"])
    print("status:", job.status())
    result = job.result()
    rows = []
    for pub, res in zip(record["pubs"], result):
        evs, stds = res.data.evs, res.data.stds
        rows += long_rows("hardware", pub["order"], pub["r"], dict(zip(record["observables"], evs)),
                          dict(zip(record["observables"], stds)))
    values = pd.read_csv(out / "hardware_values.csv")
    values = values[values.source != "hardware"]
    pd.concat([values, pd.DataFrame(rows)]).to_csv(out / "hardware_values.csv", index=False)
    record["usage_seconds"] = job.usage()
    record["finished"] = datetime.now(timezone.utc).isoformat()
    json.dump(record, open(job_file, "w"), indent=1)
    print(f"fetched {len(rows)} values; QPU usage {record['usage_seconds']} s")

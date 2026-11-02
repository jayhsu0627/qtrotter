"""Follow-up: what the device actually ran. Fetches the executed (ISA) circuits of a finished job
and writes, per circuit, its gate counts on the chain's qubits and how many sites got an X gate in
every Trotter step (the transpiler turns each single-qubit block into virtual RZ rotations and SX or
X pulses, choosing by the rotation angles, and that choice changed with r in the layered ibm_fez
run). With --recompile, also compiles the same circuits again under the device's current
calibration, for comparison. Uses no QPU time.

    python scripts/compiled_structure.py results/hardware_long_layered_job.json [--recompile]

Writes results/<job stem>_compiled.csv (one row per circuit, `circuits` executed or recompiled)
and results/<job stem>_compiled_sites.csv (X gates per site in each executed circuit).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qiskit_ibm_runtime import QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtrotter import build  # noqa: E402
from qtrotter import hardware as hw  # noqa: E402

job_file = Path(sys.argv[1])
record = json.load(open(job_file))
site_of = {q: i for i, q in enumerate(record["layout"])}
service = QiskitRuntimeService()


def structure(qc, reps):
    """Gate counts on the chain's qubits, pulse layers, and X gates per site."""
    x = np.zeros(len(site_of), int)
    counts = {"sx": 0, "x": 0, "rz": 0, "cz": 0}
    for ins in qc.data:
        qubits = [qc.find_bit(q).index for q in ins.qubits]
        if ins.operation.name in counts and all(q in site_of for q in qubits):
            counts[ins.operation.name] += 1
            if ins.operation.name == "x":
                x[site_of[qubits[0]]] += 1
    pulses = qc.depth(lambda ins: ins.operation.name in ("sx", "x", "cz"))  # rz is virtual
    return {**counts, "cz_layers": qc.depth(lambda ins: ins.operation.num_qubits == 2), "pulse_layers": pulses,
            "pulse_layers_per_step": pulses / reps, "sites_x_every_step": int((x >= max(reps - 1, 1)).sum()),
            "sites_any_x": int((x > 0).sum())}, x


rows, site_rows = [], []
for pub, qc in zip(record["pubs"], [p[0] for p in service.job(record["job_id"]).inputs["pubs"]]):
    row, x = structure(qc, pub["r"])
    rows.append({"circuits": "executed", "order": pub["order"], "r": pub["r"], **row})
    site_rows += [{"order": pub["order"], "r": pub["r"], "site": i, "x": int(n)} for i, n in enumerate(x)]
if "--recompile" in sys.argv:
    cfg, run = yaml.safe_load(open(ROOT / "config.yaml")), record["config"]
    backend = service.backend(record["backend"])
    label = f"recompiled {backend.properties().last_update_date.date()}"
    model = build(run["model"], len(record["layout"]), **cfg["models"][run["model"]])
    for t in run["times"]:
        for j in hw.build(model, run["orders"], run["steps"], t, backend, record["layout"], layer=run.get("layer", True),
                          seed=run["seed"], optimization_level=run["optimization_level"]):
            rows.append({"circuits": label, "order": j["order"], "r": j["r"], **structure(j["circuit"], j["r"])[0]})
stem = ROOT / "results" / job_file.stem.removesuffix("_job")
pd.DataFrame(rows).to_csv(f"{stem}_compiled.csv", index=False, float_format="%.4g")
pd.DataFrame(site_rows).to_csv(f"{stem}_compiled_sites.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("wrote", f"{stem}_compiled.csv", f"{stem}_compiled_sites.csv")

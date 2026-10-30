"""Per-site calibration of a device run's chain, as published at the job's submission time.

    python scripts/fetch_calibration.py results/hardware_long_layered_job.json

Writes results/<job file stem without _job>_calibration.csv: for every chain site its physical
qubit, T1 and T2 (us), readout error and its two asymmetries, sx error, and the CZ error of the
bond to the next site. Reading properties uses no QPU time.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1])
rec = json.load(open(path))
layout = rec["layout"]
backend = QiskitRuntimeService().backend(rec["backend"])
props = backend.properties(datetime=datetime.fromisoformat(rec["submitted"]))
rows = []
for i, q in enumerate(layout):
    qp = {p.name: p.value for p in props.qubits[q]}
    rows.append({"site": i, "qubit": q, "T1": qp.get("T1"), "T2": qp.get("T2"),
                 "readout_error": qp.get("readout_error"), "p01": qp.get("prob_meas0_prep1"),
                 "p10": qp.get("prob_meas1_prep0"), "sx_error": props.gate_error("sx", q),
                 "cz_error_next": props.gate_error("cz", [q, layout[i + 1]]) if i + 1 < len(layout) else None})
out = ROOT / "results" / (path.stem.removesuffix("_job") + "_calibration.csv")
pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6g")
print(f"{rec['backend']} properties of {props.last_update_date}, submitted {rec['submitted']}; wrote {out}")

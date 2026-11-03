"""Follow-up: the schedule of the executed 100-qubit circuits, chain order against layered.

For every executed t = 1 circuit of the first ibm_fez run (hardware_long, ZZ bonds in chain order)
and of the layered rerun (hardware_long_layered), schedules it as late as possible on ibm_fez's
timing (as the device's own scheduler does) and records, per chain site, the time the qubit sits
idle after its first pulse (in a superposition, exposed to detuning and always-on couplings), and,
per bond, when its first CZ starts. Uses the job inputs and the backend's timing: no QPU time.

    python scripts/staircase.py      # writes results/staircase_idle.csv, results/staircase_bonds.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit_ibm_runtime import QiskitRuntimeService

ROOT = Path(__file__).resolve().parents[1]
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
dt = backend.dt
idle_rows, bond_rows = [], []
for run in ("hardware_long", "hardware_long_layered"):
    record = json.load(open(ROOT / "results" / f"{run}_job.json"))
    layout = record["layout"]
    site = {q: i for i, q in enumerate(layout)}
    circuits = [p[0] for p in service.job(record["job_id"]).inputs["pubs"]]
    for pub, qc in zip(record["pubs"], circuits):
        if pub["t"] != 1.0:
            continue
        s = PassManager([ALAPScheduleAnalysis(target=backend.target), PadDelay(target=backend.target)]).run(qc)
        start = s.op_start_times
        idle, pulsed, first_cz = {i: 0.0 for i in range(len(layout))}, set(), {}
        for ins, t0 in zip(s.data, start):
            qubits = [s.find_bit(q).index for q in ins.qubits]
            name = ins.operation.name
            if name == "delay" and qubits[0] in site and qubits[0] in pulsed:
                idle[site[qubits[0]]] += ins.operation.duration * dt * 1e6
            elif name not in ("rz", "barrier", "delay", "measure"):
                pulsed.update(qubits)
            if name == "cz" and all(q in site for q in qubits):
                bond = min(site[q] for q in qubits)
                first_cz.setdefault(bond, t0 * dt * 1e6)
        meta = {"run": run, "order": pub["order"], "r": pub["r"], "duration_us": s.duration * dt * 1e6}
        idle_rows += [meta | {"site": i, "idle_us": v} for i, v in idle.items()]
        bond_rows += [meta | {"bond": b, "first_cz_us": v} for b, v in sorted(first_cz.items())]
        print(run, pub["order"], pub["r"], f"{s.duration * dt * 1e6:.2f} us", flush=True)
pd.DataFrame(idle_rows).to_csv(ROOT / "results" / "staircase_idle.csv", index=False, float_format="%.4g")
pd.DataFrame(bond_rows).to_csv(ROOT / "results" / "staircase_bonds.csv", index=False, float_format="%.4g")
print("wrote results/staircase_idle.csv, results/staircase_bonds.csv")

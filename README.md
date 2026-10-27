# Trotter error under hardware noise

On noisy hardware there is an optimal number of Trotter steps r\*: more steps shrink the Trotter
error but add gates, and gate noise grows with them. This project measures r\* for 1D spin
chains (4 to 12 qubits) under an IBM device noise model and compares it with two analytic
predictions, [Knee & Munro (2015)](https://arxiv.org/abs/1502.04536) and
[Avtandilyan & Pogosov (2024)](https://arxiv.org/abs/2405.01131), which assume one uniform
gate error rate. It also checks how loose standard Trotter error bounds are
([Childs et al., 2021](https://arxiv.org/abs/1912.08854)).

**Status:** work in progress. The main deliverable is the notebook in `notebooks/`.

## Layout

| Path | Contents |
| --- | --- |
| `qtrotter/` | Library: models, exact product formulas, bounds, circuits, noisy simulation, r\* fits and predictions |
| `scripts/` | Sweeps that write `results/*.csv`, plus `gpu_check.py` |
| `config.yaml` | Every sweep setting; the notebook rebuilds all figures from the CSVs |
| `tests/` | Consistency checks (`python tests/test_core.py`) |
| `notebooks/` | The write-up as a notebook |

## Reproduce

```bash
conda create -n qtrotter python=3.12 -y && conda activate qtrotter
pip install -r requirements.txt     # GPU: swap qiskit-aer for qiskit-aer-gpu
python tests/test_core.py
python scripts/run_noiseless.py     # question 1, ~20 min on a workstation CPU
python scripts/run_noisy.py         # question 2, ~40 min with a GPU
```

## Design notes

- **Term grouping.** Each Hamiltonian is passed to `PauliEvolutionGate` as groups of commuting
  terms, `[X field, ZZ bonds]` for the TFIM. Passed as one operator, Qiskit's second-order
  formula costs about twice the two-qubit gates of first order and about 5 to 10 times the depth.
- **A hidden identity.** Second-order Trotter with the ZZ group outermost equals first-order
  Trotter conjugated by a half-step of ZZ. From |0...0>, Z-type observables then cannot tell the
  two orders apart (`tests/test_core.py` checks this). The X-outer grouping avoids it.
- **Synthesize before simulating.** `Statevector` on an unsynthesized `PauliEvolutionGate` uses
  the exact exponential, so it reports zero Trotter error.
- **Noise model.** FakeTorino calibration through Qiskit Aer, density-matrix method,
  optimization level 2, ALAP scheduling so idle qubits decohere, one fixed qubit layout per chain.

# Trotter error under hardware noise

On noisy hardware there is an optimal number of Trotter steps r\*: more steps shrink the Trotter
error but add gates, and gate noise grows with them. This project measures r\* for 1D spin
chains (4 to 12 qubits) under an IBM device noise model and compares it with two analytic
predictions, [Knee & Munro (2015)](https://arxiv.org/abs/1502.04536) and
[Avtandilyan & Pogosov (2024)](https://arxiv.org/abs/2405.01131), which assume one uniform
gate error rate. It also checks how loose standard Trotter error bounds are
([Childs et al., 2021](https://arxiv.org/abs/1912.08854)).

**Read the results:** the 4-page write-up ([PDF](writeup/writeup.pdf), [Markdown](writeup/writeup.md)) or the
full notebook, [`notebooks/trotter_under_noise.ipynb`](notebooks/trotter_under_noise.ipynb).
Questions 1 to 3 are complete, including one run on an IBM device.

## Results so far

- **Bounds.** The commutator bound overestimates the worst-case Trotter error by about 6–7× at
  first order and 4× at second, at every chain length; the norm bound by 10²–10³× at orders 1–2
  and 10⁵–10⁸× at order 4, growing with n. Both reproduce Childs et al. on TFIM and XXZ chains.
- **Optimal step count.** Under the FakeTorino noise model, the measured r\* ranges from 1 to 16
  and grows with evolution time. Knee & Munro's prediction overshoots it by a median 2.4×,
  Avtandilyan & Pogosov's by 3.7× (orders 1 and 2; Knee & Munro's second order is our extension
  of their general form).
- **Why.** An ablation pins it on the Trotter term: the analytic constant exceeds the initial
  state's actual one by a median 15× (first order) and 9× (second), and swapping only that
  constant removes the overshoot, while changing only how the errors combine does not. Keeping
  the calibrated gate errors but using the state's noiseless Trotter error predicts r\* within one
  grid step in all 42 configurations with a clear minimum.
- **Order.** With two groups of terms, second order's outer half-steps merge (standard Strang
  splitting), so it costs the same two-qubit gates as first and wins everywhere. That overturns
  Avtandilyan & Pogosov's preference for first order, whose cost model charges second order double.
  Fourth order (5× the gates) loses everywhere at today's error rates.
- **Hardware.** On `ibm_kingston` (12-qubit chain, 18 circuits, 92 s of device time) the optimum
  falls where the device's own noise model puts it, r\* = 8 at first order and 6–8 at second, and
  the model tracks the device's error to within 0.01 at every r. The semi-empirical predictor,
  adapted to observables, lands one grid step low (6 and 4), because local observables decay
  several times more slowly than the global damping it assumes.

![Total error versus Trotter steps under device noise](figures/fig2_noisy_tradeoff.png)
![Predicted versus measured optimal step count](figures/fig3_predictions.png)
![The same trade-off on ibm_kingston](figures/fig4_hardware.png)

## Layout

| Path | Contents |
| --- | --- |
| `qtrotter/` | Library: models, exact product formulas, bounds, circuits, noisy simulation, r\* fits and predictions |
| `scripts/` | Sweeps that write `results/*.csv`, plus `gpu_check.py` |
| `config.yaml` | Every sweep setting; the notebook rebuilds all figures from the CSVs |
| `tests/` | Consistency checks (`python tests/test_core.py`) |
| `notebooks/` | The full analysis as a notebook |
| `writeup/` | 4-page write-up (Markdown source and PDF; `build.sh` rebuilds the PDF) |

## Reproduce

```bash
conda create -n qtrotter python=3.12 -y && conda activate qtrotter
pip install -r requirements.txt     # GPU: swap qiskit-aer for qiskit-aer-gpu
python tests/test_core.py
python scripts/run_noiseless.py     # question 1, ~20 min on a workstation CPU
python scripts/run_noisy.py         # question 2, ~40 min with a GPU
python scripts/run_hardware.py dry-run   # question 3 on a device's noise model (needs IBM credentials)
python scripts/run_hardware.py submit    # then `fetch`; uses about 1.5 min of device time
```

## Design notes

- **Term grouping.** Each Hamiltonian is passed to `PauliEvolutionGate` as groups of commuting
  terms, `[X field, ZZ bonds]` for the TFIM. Passed as one operator, Qiskit's second-order
  formula costs 1.7 to 1.9 times the two-qubit gates of first order and about 5 times the depth.
- **A hidden identity.** Second-order Trotter with the ZZ group outermost equals first-order
  Trotter conjugated by a half-step of ZZ. From |0...0>, Z-type observables then cannot tell the
  two orders apart (`tests/test_core.py` checks this). The X-outer grouping avoids it.
- **Synthesize before simulating.** `Statevector` on an unsynthesized `PauliEvolutionGate` uses
  the exact exponential, so it reports zero Trotter error.
- **Noise model.** FakeTorino calibration through Qiskit Aer, density-matrix method,
  optimization level 2, ALAP scheduling so idle qubits decohere, one fixed qubit layout per chain.

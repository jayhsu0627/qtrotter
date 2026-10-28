---
title: "Trotter Error Under Hardware Noise: Testing Predictions of the Optimal Step Count on IBM Devices"
author: "Shengjie Xu"
date: "October 2026"
---

*Code, data and notebook: [github.com/jayhsu0627/qtrotter](https://github.com/jayhsu0627/qtrotter)*

## Abstract

On noisy hardware the number of Trotter steps $r$ has an optimum $r^*$, where shrinking Trotter error and growing gate
noise balance. Two analytic formulas predict $r^*$ from a single uniform gate error rate (Knee & Munro 2015;
Avtandilyan & Pogosov 2024). I test them on transverse-field Ising chains of 4 to 12 qubits simulated under the
calibrated noise model of IBM's Torino processor, and on a 12-qubit chain run on `ibm_kingston`. Both formulas get the
scaling right but overshoot $r^*$, by a median factor of 2.4 and 3.7. An ablation traces the overshoot to their
worst-case Trotter constant, which exceeds the initial state's actual one by a median 9 to 15 times: swapping only that
constant removes it. Combining the state's actual Trotter error with the calibrated gate errors predicts $r^*$ within
one grid step in all 42 configurations with a clear minimum. On hardware the measured optimum ($r^* = 8$ at first
order, 6 to 8 at second) matches the device's own noise model; the same predictor, adapted to observables, lands one
grid step low, because local observables decay several times more slowly than it assumes. I also reproduce Childs et
al.'s finding that commutator bounds are tight to a small constant while norm bounds are loose by orders of magnitude,
and note that with the standard merge of half-steps second order costs no more than first, which overturns
Avtandilyan & Pogosov's preference for first order at today's error rates.

## 1. Introduction

IBM's utility experiment (Kim et al. 2023) simulated a 2D Ising model on a 127-qubit processor with first-order Trotter
steps and error mitigation. Every such experiment must choose the step count $r$. More steps reduce the Trotter error
of a $p$-th order formula roughly as $r^{-p}$, but each step adds two-qubit gates, so accumulated gate noise grows with
$r$. The best $r$ is therefore a trade-off, not a precision setting.

Knee & Munro (2015) derived the optimal Trotter number under faulty control (their general form covers higher orders,
but they evaluate it only at first order), and Avtandilyan &
Pogosov (2024) extended the analysis to higher orders, concluding that higher-order formulas pay off only once gate
errors fall about tenfold. Both assume every gate fails independently at one shared rate. This study asks how close
their predicted $r^*$ is to the optimum measured under a realistic device noise model, with separate one- and
two-qubit error rates, idle decoherence and a real qubit layout, and on a real device.

## 2. Setup

**Models.** Open chains with $H_\mathrm{TFIM} = -J\sum_i Z_iZ_{i+1} - h\sum_i X_i$ ($J = h = 1$, initial state
$|0\cdots0\rangle$) and $H_\mathrm{XXZ} = \sum_i (X_iX_{i+1} + Y_iY_{i+1} + \Delta Z_iZ_{i+1})$ ($\Delta = 0.5$, Néel
initial state). Each Hamiltonian is split into two groups of commuting terms, $H = A + B$: the X field and the ZZ bonds
for the TFIM, even and odd bonds for XXZ. XXZ is used only for the bound comparison; everything noisy uses the TFIM.

**Formulas.** Lie–Trotter $S_1 = e^{-iB\delta}e^{-iA\delta}$, Strang
$S_2 = e^{-iA\delta/2}e^{-iB\delta}e^{-iA\delta/2}$, and Suzuki's fourth order (five $S_2$ stages), with
$\delta = t/r$. Circuits come from Qiskit's `PauliEvolutionGate` with the groups passed as a list. This matters: passed
as a single operator, Qiskit's second-order TFIM circuit puts the ZZ terms outermost and needs 1.7 times the two-qubit
gates and 5 times the depth of the grouped version (8 qubits, $r = 16$). Worse, since $S_2^r = a\,S_1^r\,a^{-1}$ with
$a$ a diagonal ZZ half-step, that ungrouped second-order circuit gives exactly the same Z-type observables as first
order from $|0\cdots0\rangle$. With the X field outermost, its half-steps merge between steps (standard Strang
splitting), so second order costs the same gates as first; this works because there are only two groups. An exact NumPy
implementation of the same formulas, checked against Qiskit's synthesized circuits, supplies noiseless errors.

**Bounds.** The commutator bounds of Childs et al. (2021), $\frac{t^2}{2r}\lVert[A,B]\rVert$ at first order and
$\frac{t^3}{r^2}\big(\frac{1}{12}\lVert[B,[B,A]]\rVert + \frac{1}{24}\lVert[A,[A,B]]\rVert\big)$ at second, and a norm
bound from the Taylor remainder, $r\cdot 2(\Gamma t/r)^{p+1}/(p+1)!\,e^{\Gamma t/r}$, with $\Gamma$ the sum of the
group norms weighted by the formula's coefficients.

**Noise.** Circuits are transpiled onto `FakeTorino` (optimization level 2, ALAP scheduling so idle qubits decohere,
one fixed layout per chain) and simulated as density matrices with Qiskit Aer under the backend's calibrated
depolarizing and thermal-relaxation errors (median CZ error $4.2\times10^{-3}$). Readout error is excluded. The
per-step error $p$ is the slope of the summed calibrated gate error of the transpiled circuit against $r$. The sweep
covers $n = 4$ to 12, $t \in \{1, 2, 4\}$, orders 1, 2 and 4, and $r$ from 1 to 32 (540 runs). The main metric is the
trace distance to the exact state $e^{-iHt}|\psi_0\rangle$.

**Hardware.** One Estimator job on `ibm_kingston` (Heron r2, Open plan): a 12-qubit TFIM chain at $t = 2$, orders 1
and 2 at nine values of $r$, 8192 shots per circuit, readout mitigation (TREX) only. The device returns expectation
values, so the metric is the RMS error over all 46 local Paulis ($Z_i$, $X_i$, $Z_iZ_{i+1}$, $X_iX_{i+1}$), with the
shot-noise bias subtracted and bootstrap error bars. The same circuits were simulated under the device's own
calibration at submission time. The job used 92 s of device time.

## 3. Results

**Bounds are loose in the way Childs et al. found.** At $t = 1$ the measured worst-case error falls as $r^{-1}$,
$r^{-2}$ and $r^{-4}$. Over $r \ge 16$, the commutator bound overestimates it by a median 6.5 (TFIM) and 7.2 (XXZ) at
first order and 4.0 and 3.8 at second, independent of chain length, since both grow linearly in $n$. The norm bound
overestimates by $10^2$ to $10^3$ at orders 1 and 2 and $10^5$ to $10^8$ at order 4, growing with $n$ because it scales
as $n^{p+1}$.

**The optimum under device noise.** Figure 1 shows the trade-off for an 8-qubit chain: the noisy error falls with the
Trotter error, then rises as gates accumulate. The optimum grows with evolution time: at second order,
$r^* = 3$–4 at $t = 1$, 5–6 at $t = 2$ and 10–12 at $t = 4$, across all chain lengths. At the optimum the error is
already large: 0.77 to 0.90 at $t = 4$ for 10 to 12 qubits. I call a minimum clear when the error climbs by at least
0.05 on both sides of $r^*$. All 30 first- and second-order minima are clear; three fourth-order ones are not, and one
of those ($n = 12$, $t = 4$) is flat at trace distance 0.99, close to maximally mixed for every $r$.

![Trace distance to the exact state versus $r$ for an 8-qubit TFIM chain under the Torino noise model. Solid: with noise; dashed: noiseless Trotter error; rings: measured optimum.](../figures/fig2_noisy_tradeoff.png){width=86%}

**Analytic predictions overshoot.** Both papers minimize $a/r^q + b\,r$, giving $r^* = (qa/b)^{1/(q+1)}$. Each is fed
its own Trotter constant $a$ and the calibrated per-step error $b$. Knee & Munro do not evaluate second order, so there
I insert Childs et al.'s second-order commutator bound into their general form $\mathcal{C}/r^{s-1} + \mathcal{D}r$;
those numbers are my extension. Knee & Munro's formula overshoots the measured
$r^*$ by a median 3.4 at first order and 2.3 at second; Avtandilyan & Pogosov's by 2.8 and 4.2 (Figure 2).

![Predicted against measured $r^*$ over ($n$, $t$, order). Left and centre: the two analytic formulas, fed calibrated per-step errors; they cover orders 1 and 2 only (30 configurations each). Right: the semi-empirical predictor, all 45.](../figures/fig3_predictions.png){width=86%}

**The cause is the Trotter term.** A semi-empirical predictor that keeps the calibrated $p$ but replaces the analytic
term with the state's actual noiseless Trotter error $E_T(r)$, minimizing $1 - (1 - E_T(r))\,e^{-pr}$, lands within
one grid step of the measured $r^*$ in all 42 configurations with a clear minimum (and in the other three). It changes
two things at once, though: the Trotter term, and the combination rule, which saturates at 1 and could move $r^*$ by
itself when errors are this large. Table 1 separates them. Swapping only the constant, for $C$ fitted to the state's
noiseless error as $C/r^q$, inside the additive formula removes the overshoot; swapping only the rule does not. Knee &
Munro's constant exceeds the fitted one by a median 15 times at first order and 9 at second (never less than 6): partly
the worst-case looseness above, partly the gap between the worst input state and this one. Since
$r^* \propto a^{1/(q+1)}$, that alone predicts overshoots of about 3.9 and 2.1. The saturating rule adds the last two
hits.

| Trotter term | Rule | Order 1 | Order 2 | Hits |
|:--|:--|--:|--:|--:|
| Knee & Munro $a/r^q$ | additive (published) | 3.38 | 2.30 | 0/30 |
| Knee & Munro $a/r^q$ | saturating | 6.21 | 2.55 | 0/30 |
| fitted $C/r^q$ | additive | 0.87 | 1.09 | 28/30 |
| fitted $C/r^q$ | saturating | 1.02 | 1.19 | 30/30 |
| actual $E_T(r)$ | additive | 1.00 | 1.00 | 23/30 |
| actual $E_T(r)$ | saturating (predictor) | 1.00 | 1.20 | 30/30 |

Table: Ablation over the 30 first- and second-order configurations: median predicted ÷ measured $r^*$ per order, and hits within one grid step. Additive: term $+\,p\,r$; saturating: $1 - (1 - \text{term})\,e^{-pr}$.

**Second order wins, at no extra cost.** With the standard merge of outer half-steps, a two-group second-order step
costs the same $2(n-1)$ two-qubit gates as a first-order one (with three or more groups the inner ones would appear
twice). Second order then reaches a lower error at its optimum in all 15 combinations of $n$ and $t$; fourth order
(five times the gates) is worst everywhere. This overturns Avtandilyan & Pogosov's finding that first order is optimal
at error rates like these, which rests on a cost model that charges second order twice the gates.

**Hardware agrees with its noise model.** On `ibm_kingston` the RMS observable error falls with $r$, reaches a minimum
and rises again, exactly as simulated (Figure 3). The device's optimum is $r^* = 8$ at first order (error 0.062 against
0.060 predicted) and a tie between 6 and 8 at second order (0.041 against 0.033). Across all 18 circuits the device
stays within 0.01 of its noise model, and is measurably worse only near the second-order optimum (about four standard
errors). The analytic formulas again overshoot, predicting $r^* = 34$ and 17 (Knee & Munro, the second by my
extension) and 30 and 31 (Avtandilyan & Pogosov). Second order beats first on the device too, at two thirds of the
error for the same gates.

This run is also the semi-empirical predictor's only independent test, since in simulation $p$ comes from the noise
model that generated the data. Adapted to observables under the same global-depolarizing assumption (each Trotterized
value damped by $e^{-pr}$), it predicts $r^* = 6$ and 4: one grid step low at both orders. At first order that costs
0.003 in RMS error; at second it nearly doubles it (0.075 against 0.041), although its own curve is nearly level
between 4 and 6. The damping is what fails: at $r = 8$ the local Paulis shrink by about 0.9 on the device and in its
model, not by $e^{-pr} = 0.68$.

![RMS error of the 46 local Paulis for a 12-qubit TFIM chain at $t = 2$ on `ibm_kingston` (black, with bootstrap error bars), on the device's noise model (coloured) and without noise (dashed).](../figures/fig4_hardware.png){width=64%}

## 4. Related work

Beyond the two formulas tested here, Heyl, Hauke & Zoller (2019) showed that Trotter errors in local observables stay
bounded below a threshold step size. Pastori et al. (2022) proposed learning the effective Floquet Hamiltonian of a
Trotterized device to characterize its errors and find its optimal operating regime, demonstrated numerically. Xu et al.
(2025) derived an improved error analysis with an optimal number of Trotter steps for noisy devices, supported by
numerical simulations. Trivedi & Cirac (2025) proved that local observables in lattice models can be obtained to
system-size-independent precision under noise. Murota, Todo & Endo (2026) analysed the optimal depth with probabilistic
error cancellation. This study is empirical: it tests two published formulas on calibrated noise models and a device.

## 5. Limitations and next steps

The sweeps use one noise snapshot and Aer's noise model (no crosstalk, leakage or drift); the hardware check is one
chain, one evolution time and one day's calibration; $r$ is sampled on a grid. The semi-empirical predictor needs the
noiseless Trotter error, which is classically cheap only for small systems, and its observable version assumes a global
damping rate that overstates the noise on local observables. Next steps: compute local-observable Trotter errors on a
light-cone-sized patch (cheap at any size), pair them with a local noise rate, and validate on large heavy-hex circuits
across several devices, with and without error mitigation.

## References

1. G. C. Knee, W. J. Munro, Optimal Trotterization in universal quantum simulators under faulty control, Phys. Rev. A 91, 052327 (2015).
2. A. A. Avtandilyan, W. V. Pogosov, Optimal-order Trotter–Suzuki decomposition for quantum simulation on noisy quantum computers, arXiv:2405.01131 (2024).
3. A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, Theory of Trotter error with commutator scaling, Phys. Rev. X 11, 011020 (2021).
4. Y. Kim et al., Evidence for the utility of quantum computing before fault tolerance, Nature 618, 500 (2023).
5. M. Heyl, P. Hauke, P. Zoller, Quantum localization bounds Trotter errors in digital quantum simulation, Sci. Adv. 5, eaau8342 (2019).
6. L. Pastori, T. Olsacher, C. Kokail, P. Zoller, Characterization and verification of Trotterized digital quantum simulation via Hamiltonian and Liouvillian learning, PRX Quantum 3, 030324 (2022).
7. J. Xu, C. Zhao, J. Fan, Q. Zhao, Exponentially decaying quantum simulation error with noisy devices, arXiv:2504.10247 (2025).
8. R. Trivedi, J. I. Cirac, Noise robustness of problem-to-simulator mappings for quantum many-body physics, arXiv:2509.17579 (2025).
9. K. Murota, S. Todo, S. Endo, Error-mitigated Hamiltonian simulation: complexity analysis and optimization for near-term and early-fault-tolerant quantum computers, arXiv:2603.11527 (2026).

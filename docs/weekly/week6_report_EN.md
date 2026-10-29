# Weekly Report 6 — Quantum experiments and test (Sprint 3, Week 6)

12 October 2026 · Aryan Faghirollahy · Claude Doc (English + Persian): https://claude.ai/code/artifact/74858056-ada3-49b3-a8c4-ccd1fce7d1a5

All Week 6 items of the plan are done, and with them Sprint 3. The quantum models went through the data-fraction, qubit, entanglement and noise experiments and one frozen test run. The main finding: no hybrid model beats its matched classical control consistently.

## 1. Work done

| Item (plan) | What was done | Where |
| --- | --- | --- |
| Data fraction [main] | 1 / 5 / 20 / 100% of training (most recent block) for hybrid QNN, MLP control, QRC, random-feature control, LightGBM, PatchTST, univariate GRU; 3 seeds; fixed optimiser steps per epoch | notebook 21, Figure A |
| Parameter efficiency [main] | Skill against trainable parameters for all models | notebook 21, Figure B |
| Qubit count [main] | 2 / 4 / 6 / 8 qubits for the hybrid family and the QRC family with their controls | notebook 21, Figure C |
| Entanglement ablation [main] | Every quantum model run with and without entangling gates, on validation and test | notebooks 18, 19, 21, 22, 23 |
| Noise [main] | 1,024 shots, four depolarizing levels, two IBM Heron-like points; inference only; noise-aware QRC readout as a recovery attempt | notebook 22, Figure D |
| Quantum kernel [supplementary] | IQP fidelity kernel ridge vs the same map without entanglement and an RBF kernel; 4 and 8 qubits; 1% and 5% blocks; concentration diagnostic | notebook 22b, Figure E |
| Test run [protocol] | ONE frozen test run of 9 quantum models (validation reproduced exactly, SHA-256 of 25 files recorded) | notebook 23 |
| Outputs [main] | Three main figures, ablation and noise table, one-sentence finding, Methods draft in the paper document | `reports/week6/`, paper draft Section IV |
| Reports | Plain-language project report and a technical report for the professors | Claude Docs |

Code: `src/models/week6.py`, `src/models/qkernel.py`, notebooks 21–23 and 22b · W&B: groups `week6_fraction`, `week6_qkernel` [add links]

## 2. Key results (provisional)

On the test split the hybrid QNN beats persistence and, at 15 minutes, is statistically equal to LightGBM, PatchTST and the GRU, with 287 parameters. It does not beat its matched classical control consistently, and the quantum reservoir equals its classical random-feature control.

| Model | Params | Test skill h1 / h5 / h15 | Validation skill h1 / h5 / h15 |
| --- | --- | --- | --- |
| LightGBM + covariates (ref.) | — | +0.114 / +0.090 / +0.062 | +0.140 / +0.104 / +0.081 |
| LightGBM (ref.) | — | +0.064 / +0.046 / +0.033 | +0.131 / +0.085 / +0.065 |
| PatchTST (ref.) | 75,215 | +0.048 / +0.039 / +0.029 | +0.112 / +0.072 / +0.059 |
| GRU univariate (ref.) | 38,019 | +0.036 / +0.041 / +0.031 | +0.058 / +0.066 / +0.054 |
| **Hybrid QNN (4 qubits)** | 287 (24) | **+0.020 / +0.034 / +0.029** | +0.074 / +0.055 / +0.043 |
| Hybrid + Heron r2-like noise | 287 | +0.019 / +0.033 / +0.027 | +0.069 / +0.052 / +0.040 |
| Hybrid, no entanglement | 287 | +0.020 / +0.028 / +0.025 | +0.086 / +0.054 / +0.041 |
| Matched MLP control | 285 | +0.009 / +0.029 / +0.026 | +0.056 / +0.053 / +0.041 |
| Bottleneck control | 263 | +0.031 / +0.032 / +0.024 | +0.066 / +0.045 / +0.033 |
| **QRC (8 qubits)** | 483 | **+0.016 / +0.022 / +0.015** | +0.075 / +0.041 / +0.031 |
| QRC, no entanglement | 483 | +0.017 / +0.029 / +0.021 | +0.076 / +0.041 / +0.029 |
| Random-feature control | 483 | +0.015 / +0.019 / +0.012 | +0.077 / +0.042 / +0.030 |
| Raw-input control | 99 | +0.004 / +0.024 / +0.014 | +0.058 / +0.032 / +0.021 |

Skill = 1 − MAE / MAE of persistence on the same origins; mean over 3 seeds or reservoirs. Persistence test MAE 0.241 / 0.400 / 0.622 kW. Test opened once, 12 October 2026, 13:48.

- **Test vs validation:** every model loses skill on test, classical models included; the test month is further from training (regime change, as in Week 4).
- **Hybrid vs controls (Diebold–Mariano, Holm):** at h = 1 the hybrid beats the MLP (p 1e-9) and the no-entanglement variant (p 0.003), but the sign was reversed on validation (no-entanglement better, p 8e-22). At h = 5 and 15 the hybrid, MLP and no-entanglement variant are equal on both splits.
- **Hybrid vs strong references:** at h = 15 equal to LightGBM (p 0.8), PatchTST (p 1) and the GRU (p 1); at h = 1 not better than persistence (p 0.17).
- **QRC:** equal to the random-feature control at every horizon (p 0.21–0.58); the weakest quantum family on test.
- **Data fraction (validation):** at 1% only fixed-feature ridge models (QRC, random features) and LightGBM stay positive, and QRC = random features. From 5% on, the hybrid equals the MLP. LightGBM is best at every fraction.
- **Qubits (validation):** QRC gains at h = 1 with more qubits exactly like its classical controls. The entangled hybrid drops beyond 4 qubits while the no-entanglement variant improves, which fits the gradient-variance decay (hypothesis, 2–8 qubits only).
- **Noise:** the hybrid loses only 0.003–0.012 skill under Heron-like noise. QRC collapses under 1,024 shots (skill between −20 and −600): the near-linear input scale chosen in exact simulation makes the features almost constant. A noise-aware readout recovers only +0.01 to +0.02.
- **Quantum kernel (validation, supplementary):** at 1% all kernels give about +0.03 / +0.03 / +0.016; quantum vs RBF differences are about 0.001 kW with inconsistent sign; at 5% RBF is better. The selected bandwidths lie where the quantum kernel behaves like a smooth classical kernel.

**One-sentence finding:** On 211 days of AI-cluster GPU power, hybrid quantum–classical forecasters beat persistence and, at 15 minutes, match classical networks with 100–250× more parameters, but none outperforms its matched classical control consistently across validation and test, data fractions, qubit counts or noise; entanglement gives no reliable benefit, and the quantum reservoir collapses under hardware-like shot noise.

## 3. Blockers and how they were solved

| Blocker | Cause | Solution |
| --- | --- | --- |
| Long runs paused overnight | Closing the lid sleeps the Mac even with `caffeinate` (the `-s` flag works only on mains power) | Lid kept open; notebooks are resumable and append results row by row, so progress can be checked from file times |
| Notebook 21 export missed the tables | Export ran before the last save | Results read from the saved notebook; rule: save after the last cell, then export |
| 8-qubit QRC too slow under gate noise | Density-matrix simulation of all minutes takes hours per setting | Shot and readout noise for 8 qubits; gate noise on a 4-qubit reservoir |
| QRC collapses under shot noise | Near-linear input scale gives almost constant features; the weakly regularised readout amplifies noise | Reported as a finding; noise-aware readout tried as a recovery |
| Freeze record shows the Week 4 commit | Week 5–6 code not committed before notebook 23 | SHA-256 hashes of the 25 input files are the binding record; stated in Methods |

## 4. Decisions (provisional)

1. QRC input-scale grid extended once to π/32 and no further; the readout's preference for the near-linear regime is reported as a finding.
2. Data fraction = most recent contiguous block of training origins, with a fixed number of optimiser steps per epoch; the univariate GRU replaces the plan's LSTM as the recurrent reference.
3. Noise is applied at inference only, with PennyLane density-matrix simulation instead of a Qiskit noise model; Heron error rates from published device data, one-qubit rate assumed to be p₂/10.
4. Quantum kernel done as a supplementary item, validation only, not in the test run.
5. Test set of the quantum group frozen before opening: hybrid, no-entanglement, MLP, bottleneck, QRC, QRC without entanglement, random features, raw inputs, and the hybrid with Heron r2-like noise (secondary). Fraction, qubit-sweep and kernel models stay validation-only.
6. Main finding written after the test run (Section 2).

## 5. Plan for Sprint 4 (Weeks 7–8)

- [ ] Conformal intervals at 90% target coverage: split, rolling and adaptive conformal inference with delayed feedback at 5 and 15 minutes; coverage and width on normal and ramp steps (LightGBM + covariates and the best network)
- [ ] SHAP on LightGBM: ten most important features
- [ ] Battery model and MPC with cvxpy (15-minute horizon), four scenarios: oracle, best model, persistence, no forecast; metrics: maximum ramp rate, ramp-limit violations, battery size needed
- [ ] Paper: Results, Discussion, Introduction, Abstract; publication-quality figures
- [ ] Thesis chapters 1–3 and the results chapter
- [ ] Repository: README for Weeks 3–8, licence, version tag v1.0
- [ ] Meeting 4: full paper draft and remaining work until the defence

**Needed from the supervisor (Meeting 3):** is a well-controlled negative quantum result enough for the paper; which venue and deadline; full control phase (RQ4) or a smaller version with more time for writing.

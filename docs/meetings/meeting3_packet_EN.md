# Meeting 3 packet — End of Sprint 3 (hybrid quantum models)

12 October 2026 · Aryan Faghirollahy · Claude Doc (English + Persian, with figures and an Outcome dropdown): https://claude.ai/code/artifact/6721f722-a655-463d-8755-00c0afb0b288 · Slides: https://claude.ai/artifact/NbQbS5pfgAsYnSds8RvUit

This meeting closes Sprint 3 and decides three things: whether the quantum finding carries a paper, which venue to aim at, and how large Sprint 4 should be. All Week 6 work is done, including one frozen test run of the quantum models.

## 1. Agenda (about 45 minutes)

1. **Status of Sprint 3** (3 min): every Week 5 and Week 6 plan item is done (Section 2).
2. **The three main figures** (10 min): accuracy against data fraction (A), against trainable parameters (B) and against qubit count (C).
3. **Noise and quantum kernel** (5 min): Figure D and Figure E.
4. **Test result and main finding** (7 min): the one frozen test run, the test table and the one-sentence finding.
5. **Methods draft** (5 min): Section IV of the paper draft and how the test protocol is documented.
6. **Decisions and questions** (10 min): paper or not, venue and deadline, scope of Sprint 4 (Sections 4–5).
7. **Next steps** (5 min): Sprint 4 plan, date of Meeting 4, who writes the minutes.

## 2. Status of Sprint 3

| Plan item | Status | Evidence |
| --- | --- | --- |
| Hybrid QNN with matched classical control (Week 5) | Done | notebooks 17–18 |
| Quantum reservoir computing with controls (Week 5) | Done | notebook 19 |
| Epoch timing and sample size; sanity check (Week 5) | Done: full training split, memorisation check passed | notebooks 17–18 |
| Data fraction 1 / 5 / 20 / 100% (Week 6) | Done, 3 seeds | notebook 21, Figure A |
| Parameter efficiency (Week 6) | Done | Figure B |
| Qubits 2 / 4 / 6 / 8 (Week 6) | Done, 3 seeds | Figure C |
| Entanglement ablation (Week 6) | Done on validation and test | notebooks 18–23 |
| Noise: 1,024 shots + hardware noise model (Week 6) | Done (PennyLane instead of Qiskit) | notebook 22, Figure D |
| Ablation and noise table; one-sentence finding (Week 6) | Done | Section 3 |
| Methods draft (Week 6) | Done | paper draft, Section IV |
| Quantum kernel [supplementary] | Done, validation only | notebook 22b, Figure E |
| One frozen test run of the quantum models | Done, 12 October 2026 | notebook 23 |

## 3. Results to present

**One-sentence finding (draft):** On 211 days of AI-cluster GPU power, hybrid quantum–classical forecasters beat persistence and, at 15 minutes, match classical networks with 100–250× more parameters, but none outperforms its matched classical control consistently across validation and test, data fractions, qubit counts or noise; entanglement gives no reliable benefit, and the quantum reservoir collapses under hardware-like shot noise.

- **Figure A** (`reports/figures/week6/A_skill_vs_data_fraction.png`): at 1% only LightGBM and the fixed-feature ridge models (QRC, random features) stay positive, and QRC equals random features. From 5% on, the hybrid equals its MLP control. LightGBM is best at every fraction.
- **Figure B** (`B_skill_vs_params.png`): quantum models and their controls sit at 10²–10³ parameters and reach part of the skill of 10⁴–10⁶-parameter networks at h = 5 and 15; the same-size classical controls reach the same skill.
- **Figure C** (`C_skill_vs_qubits.png`): QRC gains at h = 1 with more qubits exactly like its random-feature control. The entangled hybrid drops beyond 4 qubits while the version without entanglement improves (fits the gradient-variance decay; hypothesis, 2–8 qubits only).
- **Figure D** (`D_noise_val.png`): the hybrid is robust (Heron-like noise costs 0.003–0.012 skill); the reservoir collapses even at the lowest noise level.
- **Figure E** (`E_kernel_concentration.png`): at bandwidth 1 the quantum kernel's off-diagonal variance falls about tenfold from 2 to 8 qubits; the chosen bandwidths (0.03–0.1) avoid this regime, where the quantum kernel behaves like a smooth classical kernel and equals the RBF control within about 0.001 kW.

### Test result (opened once, 12 October 2026)

| Model | Params | Test skill h1 / h5 / h15 |
| --- | --- | --- |
| LightGBM + covariates (ref.) | — | +0.114 / +0.090 / +0.062 |
| PatchTST (ref.) | 75,215 | +0.048 / +0.039 / +0.029 |
| GRU univariate (ref.) | 38,019 | +0.036 / +0.041 / +0.031 |
| **Hybrid QNN** | 287 | **+0.020 / +0.034 / +0.029** |
| Hybrid + Heron r2-like noise | 287 | +0.019 / +0.033 / +0.027 |
| Hybrid, no entanglement | 287 | +0.020 / +0.028 / +0.025 |
| Matched MLP control | 285 | +0.009 / +0.029 / +0.026 |
| **QRC** | 483 | **+0.016 / +0.022 / +0.015** |
| Random-feature control | 483 | +0.015 / +0.019 / +0.012 |

### Ablation and noise table (validation, skill h1 / h5 / h15)

| Model | Exact | 1,024 shots | Heron r2-like | Depol. 10⁻² / 10⁻¹ |
| --- | --- | --- | --- | --- |
| Hybrid QNN (3 seeds) | +0.074 / +0.055 / +0.043 | +0.066 / +0.051 / +0.041 | +0.069 / +0.052 / +0.040 | fails (h15 −0.006) |
| Hybrid, no entanglement (seed 0) | +0.079 / +0.057 / +0.044 | +0.077 / +0.056 / +0.043 | +0.079 / +0.057 / +0.043 | +0.082 / +0.058 / +0.044 |
| QRC, 4 qubits | +0.064 / +0.041 / +0.032 | −115 / −213 / −87 | −125 / −229 / −96 | collapses |
| QRC, 8 qubits | +0.075 / +0.041 / +0.031 | −253 / −97 / −23 | (shots + readout) −280 to −600 | — |
| QRC, 8 qubits, noise-aware readout | — | +0.019 / +0.009 / +0.007 | — | — |

## 4. Decisions for approval

| # | Decision | Proposal | Outcome |
| --- | --- | --- | --- |
| 1 | Main quantum message of the paper | The one-sentence finding of Section 3: a well-controlled negative result, no advantage claim | |
| 2 | Quantum test results | Final as run once on 12 October; no re-tuning after the test; any later change reported as such | |
| 3 | Noise model | PennyLane density-matrix noise with Heron error rates instead of a Qiskit fake backend; stated as a deviation from the plan | |
| 4 | Recurrent reference in the data-fraction study | Univariate GRU instead of LSTM (same input as the quantum models) | |
| 5 | Quantum kernel | Supplementary, validation only, reported in one paragraph | |
| 6 | Methods section | Accept the structure of Section IV of the paper draft | |
| 7 | Venue | IEEE PES General Meeting 2027, 5 pages, deadline 10 November 2026; fallback ACM e-Energy 2027 | |
| 8 | Scope of Sprint 4 | B if PES GM is the target, because the paper is due in 4 weeks; A otherwise | |

## 5. Questions for the supervisor

1. Is a well-controlled negative quantum result, together with the classical benchmark, enough for a paper?
2. Which venue and deadline? Co-authorship and author order?
3. Full control phase (conformal intervals and battery MPC, RQ3–RQ4) in Sprint 4, or a smaller version with more time for writing?
4. Should the thesis use the same structure as the paper (Data, Methods, Results per research question)?

### Venue options (checked 12 October 2026)

| Venue | Format | Deadline | Fit |
| --- | --- | --- | --- |
| IEEE PES General Meeting 2027, San Francisco, 25–29 July 2027 (https://pes-gm.org/2027-san-francisco/2027-call-for-papers/) | 5-page conference paper | 10 November 2026 | Power-engineering audience; RQ1 + RQ2 fit; tight (4 weeks) |
| ACM e-Energy 2027 | full paper | not yet announced; the 2026 deadline was 22 January 2026 | Energy systems + ML; room for the MPC part |
| Tackling Climate Change with ML workshop (next edition) | workshop paper | NeurIPS 2026 edition closed 29 August 2026; next call not yet out | ML audience; short paper |
| IEEE Quantum Week (QCE) 2027 posters | poster abstract | 2027 call not yet out | Quantum audience; benchmarking message |
| arXiv preprint | full paper | any time, after the supervisor agrees | Visibility for applications |

### Sprint 4 scope options

| Option | Content | Risk |
| --- | --- | --- |
| A. Full | Split, rolling and adaptive conformal intervals; SHAP; battery model and MPC in four scenarios; complete paper and thesis chapters | Tight if the paper must be ready by 10 November |
| B. Reduced | Split and adaptive conformal intervals on two models; MPC with oracle vs best model vs persistence only; more time for the paper | RQ4 answered in a smaller form |

## 6. Talking points and after the meeting

- The project never depended on a quantum win: the benchmark, the controls and the frozen test protocol are the contribution.
- Every quantum model was matched with a classical model of the same size and with a version without entanglement; every choice was made on validation; the test split was opened once.
- The strongest honest statement: with 24 circuit parameters the hybrid matches large networks at 15 minutes, but an MLP of the same size does too.
- The reservoir result is a warning for hardware: the setting that is best in exact simulation is the one shot noise destroys.
- All circuits are small enough to simulate exactly on a laptop, so no computational advantage is claimed.

After the meeting (same day): fill the Outcome column; write `docs/meetings/meeting3_minutes.md`; fill the Meeting 1 and 2 minutes; update the plan and start Sprint 4 with the chosen scope.

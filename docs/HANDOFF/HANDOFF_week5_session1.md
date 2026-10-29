# HANDOFF — Week 5, session 1 (Saturday 10 October 2026, 02:30–16:35 London time)

Bachelor project: forecasting AI data center GPU power (MIT Supercloud), classical / foundation / hybrid quantum models, conformal intervals, battery MPC.
This file continues `claude/HANDOFF_week4_session1.md` (read it first; it continues `claude/HANDOFF_week3_complete.md`, the Week 2 handoffs and `Project_Context_FA.txt`). Later files override earlier ones. Everything below is new in this session.

**STATUS:** Week 4 / Sprint 2 confirmed complete (test outputs of notebook 16 re-read and verified; `requirements.lock` refreshed: 204 packages). **Week 5 (Sprint 3) is in progress:** notebooks 17 and 18 are run (results in Section 4); notebooks 19–23 and modules for the rest of Weeks 5–6 are written, tested end to end on synthetic data and are in the repo, **not yet run on real data**. The test split is still closed for all quantum models. **Week 5 is NOT complete** (QRC notebook 19 and completion notebook 20 not run; Week 5 documents not written).

---

## 0. Working setup (unchanged rules; what worked)

- Chat linked to the Mac through the desktop app; folder `~/Desktop/Uni Project/datacenter-power-forecasting` granted. The assistant writes files with `device_commit_files` and reads `*_outputs.md` / reports with `device_stage_files`. No device shell. The link drops now and then (Mac asleep / app closed): then fall back to one zip with only new files + one `unzip -o ...` command.
- **Lesson:** a zip `week5_nb18_20.zip` was sent during a disconnect; later the same files were committed directly in a newer version. The student was told NOT to unzip the old zip (it would overwrite the newer notebook 18). Checked: notebooks 18–20 on the Mac are the newer versions.
- The assistant tests every notebook in its own cloud workspace (same PennyLane 0.45.1 / torch 2.14.1 / polars 1.44.2 / neuralforecast 3.3.0 / lightgbm 4.7.0) on a **fake repo with synthetic data** (copy of the repo's `src/`, a synthetic `model_dataset.parquet`, fake reference predictions) and runs it with nbclient before delivery. This caught several bugs (cell-title order, legend overlap, Elastic Net non-convergence, a too-strict overfit threshold).
- Rules unchanged: Persian in chat; English in code/notebooks/docs; whole notebooks; outputs via `src/tools/nb_outputs_to_md.py` (save first); Terminal one command per block, no `#`; no Git instructions; assistant takes data-science decisions (stated, provisional); cautious claims, never quantum advantage; circuits are classically simulable.

## 1. Environment

- dcpower: pennylane 0.45.1, pennylane_lightning 0.45.0 (`pennylane_lightning` has no `__version__`; use `importlib.metadata.version`), torch 2.14.1. Qiskit NOT installed (not needed: noise is simulated with PennyLane `default.mixed`).
- `requirements.lock` refreshed (204 packages incl. neuralforecast 3.3.0, chronos-forecasting 2.3.2, timesfm 3.0.2, transformers 5.19.0).
- M1 timing (notebook 17): `default.qubit` (backprop, parameter broadcasting) is **113x faster** than `lightning.qubit` (adjoint, one circuit per sample) and both agree to ~1e-7. Hybrid QNN epoch on the full train split (196,700 origins): 4 qubits 2 layers ≈ 0.09 min, 8 qubits 4 layers ≈ 2.3 min. **Plan decision point (> ~10 min/epoch) not reached → full train split, no subsample.**

## 2. New code (all in the repo)

| File | Content |
|---|---|
| `src/models/quantum.py` | Hybrid QNN and controls. `HybridForecaster`: `Linear(60→n) → a·tanh (a trainable, init π/2) → middle → Linear(n→3)`. Middle: `quantum` (`QuantumBlock`: n qubits, L layers, AngleEmbedding (data re-uploading optional), Rot per qubit, CNOT ring with range (l mod (n−1))+1 = exactly `qml.StronglyEntanglingLayers`; `<Z_i>` readout; `entangle=False` = ablation), `mlp` (classical control n→k→n tanh, k matched to 3nL), `bottleneck` (cos of the angles, no trainable middle). `train` (same loop as nf_models, CPU), `predict`, `time_steps`, `grad_variance`, `log_variant` (appends every tried variant to `reports/week5/quantum_variants_log.csv`), `run_and_score` (same file layout as all runs: config.yaml, history.csv, model.pt, predictions_val.parquet, metrics.json last; W&B via tracking.py; `epoch_log=False` default). Week 6 additions: `NoisyQuantumBlock`, `noisy_copy`, `load_trained` |
| `src/models/qrc.py` | Quantum reservoir computing. Per-minute input u(t)=tanh([dP(t)…dP(t−n+1)]/s) (s = train std of 1-min changes); reservoir computed once per minute: fixed random brickwork (RY(a·u_i), depth layers of random Rot + CZ on alternating pairs), readout X,Y,Z,ZZ,XX (5n features); kinds `qrc`, `qrc_noent`, `rf` (size-matched cos(aWu+b)), `raw`; exponential-kernel aggregation over 60 min (λ ∈ {0, 0.5, 0.8, 0.95}); **ridge readout** per horizon, closed form via one eigendecomposition, penalty chosen on validation from `RIDGE_GRID` (logspace 1e-7…10 × n_train). `run_and_score` (saves readout.npz). Week 6 additions: `min_origin` (fraction), `predict_split`, `load_readout`, `reservoir_cached` (`experiments/_cache_qrc/*.npy`), `quantum_features_noisy` |
| `src/models/week6.py` | Data-fraction experiment: `fraction_cutoff` (most recent contiguous block), `train_network` with a **fixed number of optimiser steps per epoch** (ceil(N_full/512)), `run_network_fraction` (hqnn / ctrl_mlp / ctrl_bottleneck / patchtst / gru_uni), `run_lightgbm_fraction` (nb-08 tuned params, rounds by early stopping on validation, min_data_in_leaf capped at n/20), `run_qrc_fraction` (cached reservoir, readout on the fraction), `score_save` |

Not committed to Git by the assistant (student handles Git).

## 3. Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `17_week5_qnn_smoke` | unit checks, synthetic dry run, timing, overfit 1%, gradient variance, first validation run seed 0 | **run** (Section 4.1) |
| `18_week5_qnn_val` | memorisation check, design screen (4 qubits, L∈{2,4} × re-upload), 3 seeds main + no-entanglement + MLP + bottleneck, table merged with Week 4 benchmark, DM (Holm) + ramps | **run** (Section 4.2); `18_week5_qnn_val_outputs.md` exported 16:20 |
| `19_week5_qrc` | QRC + controls: unit checks, synthetic dry run, timing, input-scale screen {π/8, π/4, π/2, π} (seed 0), 3 reservoirs at chosen scale, table, DM vs ridge AR / PatchTST / persistence | written, tested, **not run** (~30–45 min) |
| `20_week5_completion` | one validation table (Week 4 + Week 5), costs, skill-vs-parameters figure, Week 5 checklist from the plan | written, tested, **not run** (<1 min) |
| `21_week6_fraction_qubits` | data fraction 1/5/20% (+100% reuse) for hybrid, MLP, QRC, RF, LightGBM, PatchTST, gru_uni × 3 seeds; qubit sweep 2/4/6/8 (hybrid, no-ent, MLP; QRC, no-ent, RF, raw) × 3 seeds; Figures A (skill vs data), B (skill vs parameters), C (skill vs qubits) | written, tested, **not run** (2–3 h + 8-qubit hybrid runs; overnight; resumable) |
| `22_week6_noise` | inference-only noise on trained hybrid (3 seeds; no-ent seed 0) and 4-qubit QRC (density matrix), 8-qubit QRC shots/readout only; noise-aware QRC readout; noise table, Figure D, ablation table | written, tested, **not run** |
| `23_week6_quantum_test` | ONE test run for the quantum set with the nb-16 protocol (validation reproduction → freeze record with SHA-256 of ~25 files → test → metrics merged with Week 4 test → DM Holm vs controls/references → ramps → weekly skill) | written, tested, **run last, after 18–22 are final** |

Order: 18 ✓ → 19 → 20 → 21 → 22 → 23.

## 4. Results (validation; provisional)

### 4.1 Notebook 17
- Unit checks passed (backends agree; equals StronglyEntanglingLayers exactly; no-entanglement = product state; MLP control 22 vs 24 middle params).
- Overfit on 1% (1,967 origins, 60 epochs, lr 1e-2): all training skills positive, but loss fell only 9–18% (< the 20% threshold chosen) → "CHECK". Judged a threshold-design error (1-minute changes mostly noise). Replaced by a memorisation check in nb 18 (passed).
- **Gradient variance** (20 inits, real batch 256): with entanglement log10 Var drops −0.38 (L=2) / −0.46 (L=4) per added qubit (≈10³ from 2 to 8 qubits); without entanglement −0.15. Consistent with barren-plateau-type concentration at 2–8 qubits only; no asymptotic claim. Figure `reports/figures/week5/01_grad_variance.png`.
- Seed-0 first run (4 qubits, L2, re-upload): hybrid +0.042/+0.051/+0.038; MLP +0.036/+0.055/+0.043; bottleneck +0.083/+0.044/+0.031; no-ent +0.078/+0.053/+0.041. No run hit the epoch cap.

### 4.2 Notebook 18
- Memorisation (64 origins, 500 epochs): loss ratio hybrid 0.208, MLP 0.122 → PASS (bottleneck 0.094, report only).
- Design screen (seed 0, mean val MAE): **no re-upload, L=2 best (0.41744)**, L4 re-upload 0.41986, L2 re-upload 0.4215, L4 no re-upload 0.42158. Margin 0.0024 kW ≈ seed noise. **Main hybrid (provisional): `hqnn_noreup_q4_l2`** (287 params, 24 in the circuit).
- Three seeds (skill h1 / h5 / h15, mean; sd ≤ 0.019 at h1, ≤ 0.004 at h5/h15):

| Model | Params (middle) | h1 | h5 | h15 |
|---|---|---|---|---|
| hybrid QNN `hqnn_noreup_q4_l2` | 287 (24) | +0.074 | **+0.055** | **+0.043** |
| hybrid without entanglement | 287 (24) | **+0.086** | +0.054 | +0.041 |
| MLP control | 285 (22) | +0.056 | +0.053 | +0.041 |
| bottleneck control | 263 (0) | +0.066 | +0.045 | +0.033 |
| ref `gru_uni` | 38,019 | +0.058 | +0.066 | +0.054 |
| ref PatchTST | 75,215 | +0.112 | +0.072 | +0.059 |
| ref LightGBM | – | +0.131 | +0.085 | +0.065 |

  Validation ranks of the hybrid among 24 models: 10 / 9 / 10.
- DM (seed 0, Holm): h1: no-ent better than hybrid (p 8e-22), bottleneck better than hybrid (p 1e-15), hybrid better than MLP (p 1e-22); hybrid vs gru_uni n.s. (p 0.62); PatchTST better. h5: hybrid ≈ no-ent ≈ MLP (p = 1.0), hybrid > bottleneck, PatchTST > hybrid (p 4e-4), gru_uni n.s. h15: hybrid ≈ no-ent ≈ MLP, hybrid > bottleneck, vs PatchTST n.s. (p 0.07), vs gru_uni n.s. All beat persistence.
- Ramps: ramp skill 0.07–0.14, below PatchTST/gru_uni (0.13–0.18); bottleneck ramp F1 ≈ 0 (predicts tiny changes).
- **Provisional reading:** with 24 circuit parameters the hybrid QNN is comparable to univariate networks with ~130–260× more parameters at h5/h15 (statistically indistinguishable from gru_uni), but the entangling circuit gives **no measurable benefit over the same model without entanglement or with a matched classical MLP** (h5/h15 identical; h1 the no-entanglement variant is better). Consistent with Bowles et al. 2024. No quantum advantage is claimed. Single validation month; test pending.

## 5. Decisions this session (all provisional; assistant as data expert)

1. Backend `default.qubit` (exact, 113× faster) instead of `lightning.qubit`; documented as a speed choice.
2. Hybrid input univariate (same as PatchTST and gru_uni); 4 qubits main; trainable input scaling init π/2; exact simulation in training.
3. Classical controls: matched MLP (same encoder/decoder, k chosen to match 3nL) and bottleneck (no middle); entanglement ablation = same circuit without CNOTs.
4. Same training protocol as all networks (Adam 1e-3, batch 512, patience 3, cap 30, grad clip 1); rule in nb 18: if the hybrid had hit the cap, all models of nb 18 would get cap 60 in group `week5_qnn_e60` (not triggered → group `week5_qnn`, cap 30).
5. Overfit check = memorisation of 64 samples (ratio ≤ 0.3); the 1% check of nb 17 kept as reported.
6. Main hybrid = `hqnn_noreup_q4_l2` (validation screen).
7. QRC: per-minute input = tanh of the last n 1-minute changes / s; brickwork with CZ; 5n observables; exponential kernels; **ridge readout** (Elastic Net did not converge on the collinear aggregates); n = 8, depth 4; input scale chosen on validation from {π/8, π/4, π/2, π}, same grid for qrc/qrc_noent/rf; flag if at grid edge (synthetic data chose the smallest scale = near-linear regime: if real data does too, that is itself a finding).
8. Data fraction = most recent contiguous block of train; fixed optimiser steps per epoch; LightGBM rounds by early stopping on validation at every fraction (rerun at 100%); gru_uni replaces the plan's LSTM as the univariate recurrent reference.
9. Qubit sweep reuses the QRC input scale chosen at 8 qubits for all n.
10. Noise: inference only; p1 depolarizing after each rotation, p2 on both qubits after each two-qubit gate, readout flip analytic, 1,024 shots; hardware-like points from arXiv 2607.11637 Table 4 (ibm_marrakesh Heron r2: CZ 3e-3, readout 1.25e-2; ibm_torino Heron r1: CZ 4e-3, readout 2.41e-2), p1 = p2/10 assumed (not reported there). 8-qubit QRC: no density-matrix noise (≈0.4 s per minute on a laptop → hours per setting); noise on the 4-qubit QRC instead.
11. Synthetic-test finding (to verify on real data): QRC trained on exact states collapses under shot noise (tiny feature variations in the near-linear regime are destroyed); refitting the readout on noisy states (noise-aware readout, nb 22 cell 6) recovers it; hybrid QNN robust to Heron-like noise. Hypothesis.
12. Test protocol for quantum models (nb 23): frozen set = hybrid main, no-ent, MLP, bottleneck (3 seeds), QRC, QRC no-ent, RF (3 reservoirs), raw, and the hybrid with Heron r2-like noise (secondary). Fraction/qubit-sweep models stay validation-only.

## 6. Remaining work (in order)

1. Run nb 19 (QRC), read outputs; check scale-at-edge flag.
2. Run nb 20; Week 5 checklist; **announce Week 5 complete** when all main items are done: hybrid QNN ✓, matched control ✓, QRC (nb 19), epoch timing + sample size ✓, overfit sanity ✓ (memorisation), connected to evaluation module ✓, two main models end to end in one table (nb 20). Supplementary quantum kernel: not done (optional). Documents for Week 5: Weekly Report 5 (Claude Doc, English + Persian), Methods notes for the hybrid/QRC design.
3. Week 6: nb 21 (overnight), nb 22, then nb 23 (ONE test run). Then: three main figures (A, B, C) + ablation/noise table, one-sentence main finding, Methods draft (paper Doc `7a9ac5a7-d74c-445f-9964-fa5a7b3258de`), Weekly Report 6, Meeting 3 packet (questions: is the finding enough for a paper; venue; scope of Sprint 4).
4. Still open from before: Meeting 2 itself and minutes (+ Meeting 1 minutes); Outcome columns of the meeting packets.

## 7. Documents

No Claude Doc was edited in this session. Existing: proposal `1c7f11e9-9c9b-40f6-ad87-6c447798161e`, Weekly Report 3 `31fce096-d7b7-4a08-9a3a-c60cfa6ea41c`, Weekly Report 4 `2e8f16b5-9c76-454e-858f-e4c304d9a0ab`, Meeting 2 packet `2d9ab7c8-e3aa-451f-b9dc-809f8fb15e39`, paper draft `7a9ac5a7-d74c-445f-9964-fa5a7b3258de`.

## 8. Opening prompt for the next chat

```text
I am continuing my bachelor project (forecasting AI data-center GPU power on MIT Supercloud; classical, foundation and hybrid quantum models; conformal intervals; battery MPC). We are in Week 5 (Sprint 3: hybrid quantum-classical models). Read the project files first, in this order: Project_Context_FA.txt, HANDOFF_week1_complete.md (Week 1 quantum decisions 4-7 in Section 12.2), HANDOFF_week2_complete_data.md, HANDOFF_week2_session3.md, HANDOFF_week2_session4.md, HANDOFF_week2_session5.md, claude/HANDOFF_week3_complete.md, claude/HANDOFF_week4_session1.md and claude/HANDOFF_week5_session1.md (most important: code, notebooks 17-23, results of notebooks 17-18, decisions and remaining work in Section 6). Later files override earlier ones.

Rules:
- Always answer me in Persian. Anything in notebooks, code comments, scripts or repo docs is in English.
- Everything runs on my Mac (M1, 8 GB, conda env dcpower, kernel "Python (dcpower)", JupyterLab from notebooks/). Colab only if a single experiment would need more than a night.
- If this chat is linked to my Mac, request access to ~/Desktop/Uni Project/datacenter-power-forecasting, write new notebooks and modules directly into the repo and read the *_outputs.md files yourself; I run the notebooks and delete files. If not linked, give me one zip with only NEW files in project layout and one unzip command.
- Whole notebooks; test new code on synthetic data (in your own workspace on a fake copy of the repo) before I run it; outputs come back via src/tools/nb_outputs_to_md.py (I save first).
- Terminal: one command per block, no # comments; project path has a space: ~/"Desktop/Uni Project/datacenter-power-forecasting"; use /opt/miniconda3/envs/dcpower/bin/python. VPN on for pip and Hugging Face; check every install with pip --dry-run first.
- No Git instructions. Answer data-science design questions yourself as a data expert, state the decision and mark it provisional. Cautious claims; mark hypotheses; never claim quantum advantage; our circuits are classically simulable.
- The test split is closed for all quantum models until notebook 23 (one run, frozen protocol).
- Documents (proposal, weekly reports, meeting packets, paper draft) are Claude Docs; read both tabs before editing.
- Tell me explicitly when Week 5 is complete (and later Week 6), after checking the plan's full list.

Current state: notebooks 17 and 18 are run (main hybrid hqnn_noreup_q4_l2: validation skill +0.074/+0.055/+0.043, statistically like gru_uni at h5/h15 with 287 parameters, but no benefit over the same circuit without entanglement or a matched classical MLP). Notebooks 19-23 are in the repo, tested on synthetic data, not yet run. Start with Section 6 step 1 of claude/HANDOFF_week5_session1.md: I will run notebook 19 (QRC) or have run it; read the outputs and interpret them. Give me only the next action.
```

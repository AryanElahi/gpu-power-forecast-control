# Paper draft — Section IV. Methods (draft of 12 October 2026)

Copy of Section IV of the paper draft (Claude Doc: https://claude.ai/code/artifact/7a9ac5a7-d74c-445f-9964-fa5a7b3258de). Reference numbers are those of the draft's reference list; [28]–[41] were added with this section and are listed at the end.

## IV. Methods

Every model forecasts the same origins with the same 60-minute input and is scored in kW against persistence. All design choices (architectures, hyperparameters, input scales, penalties) were made on the validation split. The test split was then evaluated once per model group under a frozen protocol (Section IV-I). Quantum circuits were simulated exactly on a laptop; they are classically simulable, and we make no claim of computational advantage.

### A. Forecasting task

At each usable origin t (Section III-E) a model forecasts P(t + h) for h = 1, 5 and 15 minutes from the 60 minutes P(t − 59), …, P(t). Networks and quantum models are direct multi-horizon models with one output per horizon and a loss masked to the horizons usable at that origin. They see the window with local scaling and predict the scaled change:

```latex
x_k(t) = \frac{P(t-k) - P(t)}{s}, \quad k = 0,\dots,59, \qquad \hat{y}_h(t) = \frac{\hat{P}(t+h) - P(t)}{s}
```

Here s is the standard deviation of one-minute changes on the training split. Local scaling replaced Min–Max scaling after a validation comparison (mean MAE 0.4125 vs 0.4412 kW for the LSTM), because Min–Max is fragile under level shifts between splits. Tree and linear models are fitted on the change P(t + h) − P(t) with tabular features. The main cross-family comparison is univariate; models with the 12 covariates (Section III-C) form a separate arm, and calendar inputs are hour sin/cos and minute-of-hour mod 5.

### B. Classical and foundation benchmark

The benchmark has 20 models in six families. Every trained model ran with three seeds.

| Family | Models | Fitting |
| --- | --- | --- |
| Naive | persistence; persistence + rolling 7-day mod-5 profile; daily seasonal naive | none |
| Linear | ridge AR on the 59 lagged changes, + calendar, + covariates | penalty chosen on validation |
| Statistical | AutoARIMA (1,1,5), AutoETS (ANN) [statsforecast] | fitted on 12.7 days (17–29 July), then run as fixed-parameter Kalman or ETS filters |
| Gradient boosting | LightGBM [33], one model per horizon on the change, univariate and with covariates | Optuna [34] (15 trials or 300 s per study), early stopping on validation |
| Networks | LSTM and GRU (2 × 64, dropout 0.1, with calendar channels); univariate GRU; N-HiTS [24], PatchTST [25], DLinear and NLinear [26] | shared protocol below |
| Foundation (zero-shot) | Chronos-Bolt small (48 M) [10], TimesFM 2.5 200M [11] | none; context 512 chosen on validation from {60, 512, 2048} and {64, 512} |

All networks share one training protocol, so only the architecture differs: MSE on the masked horizons, Adam (learning rate 10⁻³), batch 512, gradient clipping at 1, early stopping on mean validation MAE (patience 3, cap 20–30 epochs; no run reached the cap), no per-model tuning. The neuralforecast architectures run as plain PyTorch modules inside this loop rather than through the library's own fit. PatchTST was sized by us (hidden 64, 4 heads, 2 layers, patch 16, stride 8, RevIN; 75,215 parameters); N-HiTS keeps library defaults (676,301). Foundation models give the median as the point forecast. Their results are secondary because their pre-training may overlap our period [13].

### C. Hybrid quantum neural network

The hybrid QNN is a classical encoder, an n-qubit variational circuit and a classical decoder, trained end to end:

```latex
\theta = a \tanh(W_e x + b_e) \in \mathbb{R}^n, \qquad q_i = \langle 0 | U^\dagger(\theta, \phi)\, Z_i\, U(\theta, \phi) | 0 \rangle, \qquad \hat{y} = W_d\, q + b_d \in \mathbb{R}^3
```

The encoder is a linear map from the 60-step window to n angles. The scale a is trainable and starts at π/2. U encodes θ with one RX rotation per qubit (angle embedding), then applies L layers. Each layer has a general rotation Rot(φ₁, φ₂, φ₃) on every qubit and a ring of CNOTs with range (l mod (n − 1)) + 1; this equals the strongly entangling layers of [30]. The readout is the n single-qubit expectations ⟨Zᵢ⟩, and the decoder maps them to the three horizons.

The main model has n = 4 qubits, L = 2 layers and no data re-uploading [29]: 287 trainable parameters, 24 of them in the circuit. It was chosen on validation (seed 0) from four designs (L ∈ {2, 4}, with or without re-uploading); the margin, 0.0024 kW mean MAE, is about the size of seed noise. Three controls keep the encoder and decoder and change only the middle block:

| Variant | Middle block | Parameters (middle) |
| --- | --- | --- |
| Hybrid QNN | 4-qubit circuit, 2 layers | 287 (24) |
| No entanglement | same circuit without CNOTs (a product state) | 287 (24) |
| Matched MLP | n → k → n with tanh, k chosen to match 3nL | 285 (22) |
| Bottleneck | cos θᵢ, no trainable middle | 263 (0) |

All variants use the network protocol of Section IV-B (cap 30 epochs; no run reached it). Training simulates the exact state vector with PennyLane `default.qubit` [40], with back-propagation and parameter broadcasting over the batch. This was 113 times faster than `lightning.qubit` with adjoint differentiation, and the two agree to 10⁻⁷. One epoch on the full training split (196,700 origins) took about 5 s for 4 qubits and 2.3 min for 8 qubits and 4 layers, so no subsampling was needed.

Two checks preceded the main runs. In a memorisation test on 64 origins (500 epochs) the training loss fell to 0.21 of its start for the hybrid and 0.12 for the MLP (pass threshold 0.3). The variance of the loss gradient over 20 random initialisations was measured for 2–8 qubits as a diagnostic of barren-plateau-type concentration [35]. Every variant tried is logged.

### D. Quantum reservoir computing

The quantum reservoir [31], [17] is a fixed random circuit; only a linear readout is trained. Each minute is processed once, so the reservoir cost does not grow with the number of training passes.

1. **Input.** For each minute, u(t) = tanh([ΔP(t), …, ΔP(t − 7)] / s): the last eight one-minute changes, squashed to (−1, 1).
2. **Reservoir.** Eight qubits receive RY(a·uᵢ). Four brickwork layers follow, each with fixed random Rot gates and CZ gates on alternating neighbour pairs. Parameters are drawn once per reservoir seed.
3. **Features.** The exact state gives ⟨Xᵢ⟩, ⟨Yᵢ⟩, ⟨Zᵢ⟩, ⟨ZᵢZᵢ₊₁⟩ and ⟨XᵢXᵢ₊₁⟩: 40 features per minute, computed once for all 303,840 minutes (about 85 s per reservoir).
4. **Memory.** Exponential kernels with decay λ ∈ {0, 0.5, 0.8, 0.95} aggregate the features over the 60-minute window, giving 160 features per origin.
5. **Readout.** A ridge regression per horizon predicts the scaled change, in closed form from one eigendecomposition for all penalties. The penalty comes from a grid of 17 values (10⁻⁷ to 10, times n_train), chosen per horizon on validation. The readout has 483 coefficients.

The input scale a was chosen on validation from {π/8, π/4, π/2, π}, with the same grid for the reservoir and its controls. The smallest value won, so the grid was extended once to π/32 and π/16, under a rule fixed before the rerun: no further extension. The final choices are π/32 for the reservoir and the random-feature control and π/16 for the version without entanglement; π/32 and π/16 differ by about 10⁻⁴ kW. The readout therefore prefers the nearly linear regime of the reservoir (Section V).

Three controls use the same input, kernels and readout: the reservoir without CZ gates (a product state); size-matched classical random features cos(aWu + b) with 40 features [32]; and the raw inputs u. Each reservoir and control ran with three random seeds. An Elastic Net readout was planned but did not converge on the collinear aggregates, so ridge is used throughout.

### E. Quantum kernel ridge regression (supplementary)

A fidelity kernel tests a third route to quantum features. The window (k = 1, …, 59) is reduced by PCA, fitted on the training subsample, to n whitened components z. Two layers of an IQP-type feature map [28] prepare |ψ(z)⟩: Hadamard on every qubit, RZ(2a zᵢ), then IsingZZ(2a² zᵢ zⱼ) on ring neighbours. The kernel and its two classical controls are:

```latex
k_{\mathrm{Q}}(z, z') = |\langle \psi(z) | \psi(z') \rangle|^2, \qquad k_{\mathrm{RBF}}(z, z') = \exp\!\left(-\tfrac{a^2}{2} \lVert z - z' \rVert^2\right)
```

The second control is the same map without IsingZZ gates, a product state with a closed form. Kernel ridge regression per horizon uses one eigendecomposition of the Gram matrix for all 17 penalties; the bandwidth a ∈ {0.01, 0.03, 0.1, 0.3, 1} and the penalty are chosen on validation. The kernels were fitted on two training blocks: the most recent 1% of origins (1,953 usable for all horizons) and the most recent 5%, thinned to every second origin (4,911). They were run for 4 and 8 qubits, giving 60 fits. The off-diagonal mean and variance of the Gram matrix on 500 points measure kernel concentration [36]. The kernels are deterministic and were evaluated on validation only; they are not part of the test run.

### F. Noise model

Noise enters at inference only: models trained on exact states are evaluated with noisy circuits, which mimics deployment on hardware without retraining. The model has three parts. Depolarizing noise p₁ follows each one-qubit rotation and p₂ acts on both qubits after each two-qubit gate. Symmetric readout flips have probability p_ro, applied analytically to ⟨Z⟩. Finite sampling uses 1,024 shots. Two hardware-like points take two-qubit and readout errors from the IBM Heron devices in [41]:

| Setting | p₂ | p_ro | p₁ |
| --- | --- | --- | --- |
| Heron r2-like (ibm_marrakesh) | 3 × 10⁻³ | 1.25 × 10⁻² | p₂/10 (assumed) |
| Heron r1-like (ibm_torino) | 4 × 10⁻³ | 2.41 × 10⁻² | p₂/10 (assumed) |

We also sweep depolarizing rates p₁ / p₂ = 10⁻⁴ / 10⁻³, 10⁻³ / 10⁻², 3·10⁻³ / 3·10⁻² and 10⁻² / 10⁻¹. Noisy circuits are simulated with density matrices in PennyLane (`default.mixed`) rather than a vendor noise model. The 8-qubit reservoir received shot and readout noise only, because density-matrix simulation of all minutes would take hours per setting; gate noise was applied to a 4-qubit reservoir instead. As a recovery attempt, the reservoir readout was also refitted on noisy features (a noise-aware readout). Choosing the input scale under noise was not tried.

### G. Data-fraction and qubit-count experiments

Training fractions of 1, 5, 20 and 100% are the most recent contiguous block of training origins: 1,967, 9,835, 39,340 and 196,700 origins (1.4, 6.8, 29.4 and 147.7 days). A recent block keeps the gap to validation fixed, which a random subsample would not. Networks keep a fixed number of optimiser steps per epoch, ⌈196,700 / 512⌉, so smaller fractions are not trained for fewer steps. LightGBM uses the tuned hyperparameters with boosting rounds set by early stopping at every fraction, and min_data_in_leaf capped at n/20. The fraction study covers the hybrid QNN, the MLP control, the reservoir, the random-feature control, LightGBM, PatchTST and the univariate GRU.

The qubit sweep uses n ∈ {2, 4, 6, 8} on 100% of the data, for the hybrid, its version without entanglement and the MLP control, and for the reservoir, its version without entanglement and the random-feature and raw controls. The reservoir input scale chosen at 8 qubits is reused for every n. Both studies use three seeds and validation only. The trainable-parameter count of every model is logged for the accuracy-versus-parameters comparison.

### H. Metrics and statistical tests

The primary metric is MAE in kW and its skill against persistence on the same origins, skill = 1 − MAE / MAE_persistence. We also report RMSE, sMAPE, R² on the level and on the change, and MAE with the five-minute-dip targets removed at h = 1. Ramp metrics use the ramp label of Section III-G: MAE on ramp and on normal steps, and the precision, recall and F1 of predicted ramps (|P̂(t + h) − P(t)| / L24(t) above the same threshold). Skill per calendar week shows how stable each model is over time.

Pairs of models are compared with the Diebold–Mariano test [37] on absolute errors. The variance uses Newey–West [38] with lag 60, because overlapping windows make errors at nearby origins dependent. The origins are treated as one sequence despite the gaps at invalid minutes, which is an approximation. Each table of pairwise tests is corrected with the Holm procedure [39]. For seeded models the tests use seed 0, and tables report the mean and standard deviation over three seeds.

### I. Frozen test protocol and reproducibility

The test split was opened twice in total, once per model group, each time after all choices for that group were final:

1. **Freeze.** The model set, hyperparameters, penalties, input scales and foundation-model contexts are fixed. Models are trained on the training split only, with no refit on train plus validation.
2. **Reproduce.** Frozen settings re-create every stored validation forecast before the test is touched. The largest difference was 3·10⁻⁶ kW for the classical benchmark (networks trained on MPS, re-predicted on CPU) and 0.0 kW for all eight quantum model families.
3. **Record.** A freeze record stores the protocol, the git commit and the SHA-256 hash of every input file: the dataset, the code and each checkpoint or readout (25 files for the quantum group).
4. **Test once.** Test forecasts are written with their own SHA-256 hash. Any later fix is reported as such.

The classical benchmark (20 models) was evaluated on 9 October 2026. Its protocol was recorded twice with identical content, because the first run was interrupted before any test forecast was written. The quantum group was evaluated on 12 October 2026: the hybrid QNN and its three controls, the reservoir and its three controls, and, as a secondary result, the hybrid with Heron r2-like noise. The data-fraction, qubit-sweep and kernel models stayed on validation only. The quantum freeze record names the Week 4 commit, because the Week 5–6 code had not been committed when it was written; the file hashes are therefore the binding record.

### J. Software and compute

All experiments ran on one Apple M1 laptop (8 GB RAM) with Python 3.11, PyTorch 2.14.1, PennyLane 0.45.1, LightGBM 4.7.0, statsforecast 2.1.1, neuralforecast 3.3.0 and polars 1.44.2; the full environment is pinned in the repository's lock file. Only the per-file reduction of the raw telemetry used Google Colab (Section III-C). Runs are logged as JSON records and in Weights & Biases.

| Model | Training time | Notes |
| --- | --- | --- |
| Hybrid QNN, 4 qubits | about 70 s (4.2 s per epoch) | exact simulation, full training split |
| Matched MLP control | about 5 s | same protocol |
| Hybrid QNN, 8 qubits | 7–15 min | qubit sweep |
| Quantum reservoir, 8 qubits | about 85 s (features) + under 1 s (ridge) | per reservoir, all minutes |
| Random-feature control | under 1 s | same readout |
| Quantum kernel, 8 qubits, 4,911 points | 21 min for all 60 fits | 4 and 8 qubits, two blocks, three kernels |

The code is public; the processed data are not, because of the dataset licence (Section III-A).

## References added with Section IV

28. V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," Nature 567, 209–212, 2019.
29. A. Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier," Quantum 4, 226, 2020.
30. M. Schuld, A. Bocharov, K. M. Svore, N. Wiebe, "Circuit-centric quantum classifiers," Physical Review A 101, 032308, 2020.
31. K. Fujii, K. Nakajima, "Harnessing disordered-ensemble quantum dynamics for machine learning," Physical Review Applied 8, 024030, 2017.
32. A. Rahimi, B. Recht, "Random features for large-scale kernel machines," NeurIPS, 2007.
33. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," NeurIPS, 2017.
34. T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," KDD, 2019.
35. J. R. McClean et al., "Barren plateaus in quantum neural network training landscapes," Nature Communications 9, 4812, 2018.
36. S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes, "Exponential concentration in quantum kernel methods," Nature Communications 15, 5200, 2024.
37. F. X. Diebold, R. S. Mariano, "Comparing predictive accuracy," Journal of Business & Economic Statistics 13(3), 253–263, 1995.
38. W. K. Newey, K. D. West, "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix," Econometrica 55(3), 703–708, 1987.
39. S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian Journal of Statistics 6(2), 65–70, 1979.
40. V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations," arXiv:1811.04968, 2018.
41. IBM Heron device error rates, Table 4 of arXiv:2607.11637, 2026 (complete: title, authors).

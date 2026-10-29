# Code map

How the source code, notebooks, reports and freeze records fit together (state of 14 October 2026, end of Week 7 and the project delivery). Run every command from the repository root in the conda environment `dcpower` (Python 3.11).

## 1. Pipeline at a glance

```
MIT Supercloud S3 (2.2 TiB raw, CC BY-NC-ND)
  │  notebook 02 + src/data/build_series.py           (per-file reduction on Google Colab, ~17 h)
  ▼
per-GPU minute streams (Google Drive)
  │  src/data/build_series_lowmem.py, build_log_features.py   (notebooks 05 / 05 v2)
  ▼
data/processed/series_full*.parquet  (1-minute power + 12 covariates, 4 Mar – 30 Sep 2021)
  │  src/data/build_model_dataset.py  (notebook 06: validity rules, split, windows, ramp labels)
  ▼
data/processed/model_dataset.parquet + model_dataset_meta.json
  │
  ├─ benchmark   notebooks 07–15 (validation) → 16 (frozen test)        reports/baselines, classical, week3, week4
  ├─ quantum     notebooks 17–22b (validation) → 23 (frozen test)       reports/week5, week6
  └─ week 7      24 conformal (val → test) · 25 SHAP (val) · 26 MPC (val → test) · 27 band MPC (val → test)
                                                                         reports/week7
```

## 2. Scripts

| Script | Purpose |
|---|---|
| `scripts/reproduce.py` | Runs notebooks 06–27 in order with nbclient, saves executed copies to `notebooks/_reproduced/`, exports outputs and logs to `reports/reproduce_log.csv`. Without `--run` it only prints the plan. Test-split notebooks (16, 23, 24, 26, 27) are skipped unless `--include-test`. |
| `scripts/verify_freeze.py` | Recomputes the SHA-256 of every file listed in the freeze records under `reports/` and reports MATCH / CHANGED / MISSING. On 14 October 2026: 6 records, 47 inputs, all MATCH. |
| `src/tools/nb_outputs_to_md.py` | Exports a saved notebook's outputs to `<name>_outputs.md` (+ figures). |
| `src/run_experiment.py`, `src/tracking.py` | YAML-driven training runs with JSON records in `experiments/runs/` and Weights & Biases. |

Typical use:

```
python scripts/reproduce.py
python scripts/reproduce.py --run --stage week7
python scripts/verify_freeze.py --quiet
```

## 3. Modules and the notebooks that import them

Imports were read from the notebooks on 14 October 2026.

| Module | What it contains | Imported by notebooks |
|---|---|---|
| `src/data/build_series.py` | Per-file reduction and time-weighted aggregation of GPU streams into minute power (CLI) | 02 (called as a script) |
| `src/data/build_series_lowmem.py` | Same aggregation in batches of streams (out-of-memory fix) | 05, 05 v2 (script) |
| `src/data/build_log_features.py` | Slurm-log covariates: allocated GPUs, GPU jobs, all jobs, queue | 02, 05, 05 v2 (script) |
| `src/data/build_model_dataset.py` | Validity rules, 70/15/15 split, usable origins per horizon, ramp labels, covariate changes | 06 (script) |
| `src/evaluation/metrics.py` | MAE, RMSE, sMAPE, R², skill, ramp metrics, Diebold–Mariano | 07, 08, 09, 11, 12, 14, 15, 16, 20, 22, 23 |
| `src/evaluation/rolling.py` | Rolling-origin helpers | 09 |
| `src/evaluation/compare.py` | DM with Newey–West lag 60 and Holm; ramp breakdown; weekly skill | 12, 16, 18, 19, 22b, 23 |
| `src/evaluation/frozen.py` | Frozen re-implementation of every benchmark model for the test run; reproduction check | 16, 25 |
| `src/evaluation/conformal.py` | Conformal scores, static / rolling / ACI (delayed feedback) / Mondrian / mondrian_pc, selection | 24, 27 |
| `src/models/tabular_features.py` | Lag, calendar and covariate features for trees and linear models | 08, 09 (and `frozen.py`) |
| `src/models/statespace.py` | AutoARIMA / AutoETS run as fixed filters | 08 |
| `src/models/rnn.py` | Shared data loader and training loop; LSTM, GRU | 09–11, 13–15, 17–23, 22b |
| `src/models/nf_models.py` | N-HiTS, PatchTST, DLinear, NLinear inside the shared loop | 10, 11 |
| `src/models/quantum.py` | Hybrid QNN, no-entanglement, MLP and bottleneck controls; noise | 17, 18, 21, 22, 23 |
| `src/models/qrc.py` | Quantum reservoir, no-entanglement, random-feature and raw controls | 19, 21, 22, 23 |
| `src/models/week6.py` | Data-fraction and qubit-sweep runners | 21, 22b |
| `src/models/qkernel.py` | Quantum kernel ridge, product-state and RBF controls, concentration | 22b |
| `src/control/battery_mpc.py` | Battery model, ramp-limiting MPC (cvxpy, Clarabel), closed-loop simulation, ramp metrics | 26, 27 |
| `src/control/scenario_mpc.py` | Uncertainty-aware MPC on three band trajectories | 27 |

`src/control/battery_mpc.py`, `scenario_mpc.py`, `src/evaluation/conformal.py`, `src/models/quantum.py` and `qrc.py` are hashed in freeze records: do not edit them; put changes in new modules.

## 4. Notebooks

| Notebook | Stage | Test split | Main outputs |
|---|---|---|---|
| 00 setup check, 01 PennyLane warm-up | Week 1 | – | environment checks |
| 02 data download and exploration | Week 2 (historical; downloads raw data, old project path) | – | streams, development series |
| 03, 04, 05, 05 v2 EDA | Week 2 | – | `reports/figures/eda*`, `reports/eda_full_summary_v2.json` |
| 06 modelling dataset | data | – | `data/processed/model_dataset*` |
| 07 baselines, 08 classical models, 09 Week 3 completion | benchmark | – | `reports/baselines`, `classical`, `week3` |
| 10–15 deep, foundation and validation analysis | benchmark | – | `reports/week4/*val*`, `experiments/week4_*` |
| 16 test evaluation | benchmark | **opened once, 9 Oct** | `reports/week4/test_*` |
| 17–20 hybrid QNN, QRC | quantum | – | `reports/week5` |
| 21, 22, 22b fraction, qubits, noise, kernel | quantum | – | `reports/week6/*val*` |
| 23 quantum test | quantum | **opened once, 12 Oct** | `reports/week6/quantum_test_*` |
| 24 conformal | week 7 | **opened once, 13 Oct** | `reports/week7/conformal_*`, figures 01–04 |
| 25 SHAP | week 7 | – (validation only) | `reports/week7/shap_*`, figures 05–06 |
| 26 battery MPC | week 7 | **opened once, 14 Oct** | `reports/week7/mpc_*`, figures 07–09 |
| 27 uncertainty-aware MPC | week 7 | **opened once, 14 Oct** | `reports/week7/mpc_band_*`, figure 10 |

Delivery figures used in the reports and slides: `reports/figures/delivery/` (test skill of the 12 main models; MPC ramp event at 2 kWh).

## 5. Freeze records

| Group | Record | Inputs hashed |
|---|---|---|
| Classical and foundation benchmark | `reports/week4/test_freeze.json` (+ identical `test_freeze_20261009_191127.json`) | dataset, baseline and classical summaries |
| Quantum models | `reports/week6/quantum_test_freeze.json` | dataset, `quantum.py`, `qrc.py`, 12 checkpoints, 10 readouts |
| Conformal intervals | `reports/week7/conformal_test_freeze.json` | dataset, stored forecasts, `conformal.py` |
| Battery MPC | `reports/week7/mpc_test_freeze.json` | dataset, test forecasts, `battery_mpc.py` |
| Uncertainty-aware MPC | `reports/week7/mpc_band_test_freeze.json` | test forecasts, intervals, `conformal_val.json`, `battery_mpc.py`, `scenario_mpc.py`, `conformal.py` |

The records name git commit `96597605…` (Week 4): the Week 5–7 code was not committed when they were written, so the SHA-256 hashes are the binding record.

## 6. Housekeeping notes

- `src/eval/` is an empty legacy folder and `src/data/build_series.py.bak_unweighted` an old copy; both can be deleted.
- `requirements.lock` predates shap 0.51.0 and cvxpy 1.9.3; refresh it with `pip freeze`.
- Outputs of notebooks 00–05 were never exported to markdown.
- Raw and processed MIT Supercloud data are never committed (licence CC BY-NC-ND 4.0).

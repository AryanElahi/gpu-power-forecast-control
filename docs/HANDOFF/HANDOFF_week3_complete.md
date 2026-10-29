# HANDOFF — Week 3 (Thursday 8 – Friday 9 October 2026)

Bachelor project: forecasting AI data center GPU power (MIT Supercloud), classical / foundation / hybrid quantum models, conformal intervals, battery MPC.
This file continues `HANDOFF_week2_session5.md` (read it first). Sections of earlier handoffs not repeated or replaced here remain valid. Later files override earlier ones.

**WEEK 3 IS COMPLETE (all main items of the project plan, checked by notebook 09 cell 11).** Done: Meeting 1 outcome recorded (decisions taken by the assistant as data expert at the student's request; minutes skipped for now), modelling dataset builder (notebook 06), shared evaluation module, reference baselines (notebook 07), AutoARIMA / AutoETS / LightGBM (notebook 08), LSTM / GRU from YAML configs with three seeds, W&B tracking, rolling-origin module, one Week 3 benchmark table (notebook 09). Weekly Report 2 (both tabs) finalised and due 9 October. Weekly Report 3 written and the proposal updated in both tabs (Section 8). **Still open:** Meeting 1 minutes and the packet's Outcome column (skipped by the student for now).

---

## 0. Working setup (changes)

- **Project root moved:** `~/Desktop/Uni Project/datacenter-power-forecasting/` (space in the name: quote the path in Terminal, e.g. `~/"Desktop/Uni Project/datacenter-power-forecasting"`). The old `~/Desktop/Other/...` is wrong.
- Rules unchanged: Persian in chat, English in code and docs, whole notebooks as `.ipynb` (tested on synthetic data first), outputs returned as Markdown from `src/tools/nb_outputs_to_md.py` (save first), one Terminal command per block, no Git instructions, cautious claims.
- New delivery pattern that worked well: a **zip in project-folder layout**, installed with one command: `unzip -o ~/Downloads/<name>.zip -d ~/"Desktop/Uni Project/datacenter-power-forecasting"`.
- The student runs JupyterLab with `/opt/miniconda3/envs/dcpower/bin/jupyter lab` from `notebooks/`. Terminal `python` is the base env; use the full dcpower path for scripts.
- Long runs: `caffeinate -dis` in a separate Terminal; lid open. The student's internet is unstable (affects only W&B, which falls back to offline).
- The student does not want to answer data-science design questions: "answer them yourself as a data expert" (the supervisor is not a data person). Make sensible decisions, state them, mark them provisional.

## 1. Environment fixes (important for reinstalls)

- Installed in dcpower this week: statsforecast 2.1.1, statsmodels 0.15.0, lightgbm 4.7.0, optuna 5.0.0, scikit-learn 1.9.1 (pip). torch 2.14.1, wandb 0.30.0, pyyaml were already there.
- **lightgbm needs `libomp.dylib`** on macOS. conda installs failed silently because the Anaconda **Terms of Service are not accepted** (`CondaToSNonInteractiveError`; same cause as in Week 1). Working fix: copy the libomp bundled with scikit-learn: `.../site-packages/sklearn/.dylibs/libomp.dylib` → `/opt/miniconda3/envs/dcpower/lib/libomp.dylib` (done by a one-off notebook cell). Repeat after recreating the env. Alternative: `conda tos accept ...` then `conda install -c conda-forge llvm-openmp` (not tried).
- Packages via `sys.executable -m pip` inside a notebook cell show `[*]` without output for minutes; installing from Terminal with `/opt/miniconda3/envs/dcpower/bin/python -m pip install ...` is clearer.
- W&B: credentials in `~/.netrc` work; runs log **online** (project `datacenter-power-forecasting`, entity `aryan-faghirollahy-uok`).
- `requirements.lock` not refreshed this week (do it).

## 2. Meeting 1 outcome (student: meeting went well; supervisor left data choices to the student, who asked the assistant to decide)

| # | Decision | Outcome |
|---|---|---|
| 1 | Window 4 Mar – 30 Sep 2021 | Approved |
| 2 | Relative ramp \|dP\|/L24, train p95 | Approved; undefined L24 (< 720 valid min) → no ramp label |
| 3 | Maintenance event window | Approved |
| 4 | Unscheduled outage 29 Mar (uwin) | Approved |
| 5 | New York wall clock, DST hour excluded | Approved |
| 6 | Five-minute dip kept, mod-5 feature | Approved |
| 7 | Covariates | **Changed:** main cross-family comparison univariate; covariates a separate arm, used only after the gain is confirmed on validation (now confirmed, Section 6) |
| 8 | Hour sin/cos yes, weekday no | Approved |
| 9 | Scaling | **Changed (narrowed):** Min-Max on train vs window-local scaling, chosen on validation, metrics in kW. **Decided in Week 3: local scaling for networks** (Section 7) |
| 10 | PM100 second dataset | Approved with condition: schema check in Weeks 3–4 (whole-node power, exclusive jobs, 20 s, 2020; rank stability only); deferred if no sensible 1-min series |
| — | 10-second experiment | Dropped (would need the raw data again) |
| — | sMAPE | Kept (zero minutes absent from valid data) |

The Meeting 1 packet's Outcome column was **not** filled in the Claude Doc yet; minutes in `docs/meetings/` not written (student skipped).

## 3. Modelling dataset: `src/data/build_model_dataset.py` (v1.1) + `notebooks/06_model_dataset.ipynb`

- Input `data/processed/series_full_features.parquet` → `data/processed/model_dataset.parquet` (303,840 × 68, 45.6 MiB) + `model_dataset_meta.json` (rules, counts, split, thresholds, persistence, train stats, sha256 of input, reference checks).
- Flags: `outage`, `mwin`, `uwin`, `dst_gap` (computed from America/New_York, not hard-coded), `susp_zero`, `no_gpu` (util/temp filled 0), `valid`.
- `uwin` rule: for each outage run outside mwin, from min(first susp_zero of the day, run start) to 23:59. On real data: two runs on 29 Mar (12:22–15:03 and 15:12; 8 minutes of telemetry in between), both give uwin 11:32–23:59 (748 min; 536 valid minutes removed). v1.1 counts **days** (check `uwin_days`), v1.0 counted runs.
- Features (causal): `P_kW`, `mod5`, `hour_sin/cos`, `L24_kW` (mean of valid P over 1,440 min ending at t, null if < 720 valid), `n_valid_in60`, `d1_*`/`d5_*` of 12 covariates (null unless both ends valid).
- Per horizon h ∈ {1, 5, 15}: `ok_h` (all 60 input minutes and target valid), `split_h` (pair rule of nb 05), `ramp_h` (relative, train p95), `ramp99_h`.
- **Real-data results:** valid 282,452 (92.96%); valid_before_uwin 282,988 (= nb 05, exact); usable samples train / val / test: h1 196,700 / 42,575 / 42,575; h5 196,668 / 42,567 / 42,567; h15 196,588 / 42,547 / 42,547 (60-min rule loses ~0.2%). **Final relative ramp thresholds (p95): 0.0295 / 0.0399 / 0.0568**; p99 0.0563 / 0.0800 / 0.1095; absolute p95 0.821 / 1.075 / 1.522 kW. Ramp rates p95 val / test: h1 0.045 / 0.040, h5 0.058 / 0.071, h15 0.065 / 0.089. Persistence test MAE 0.2407 / 0.4000 / 0.6221 (unchanged, test untouched by uwin). L24 null on 22,039 minutes (mostly after maintenance).
- Independent checks (nb 06 cell 8): brute-force L24 and n_valid_in60, covariate changes, ok/split/ramp labels, **leakage test** (rebuild on the first 66% → no causal column changes): all passed. All 16 reference checks ok with v1.1.

## 4. Shared evaluation and helper modules (all in repo)

| File | Content |
|---|---|
| `src/evaluation/metrics.py` | MAE, RMSE, sMAPE, R², R²_delta (on the change), skill vs persistence (MAE, RMSE), MAE ramp/normal, ramp precision/recall/F1 (predicted ramp: \|ŷ−P(t)\|/L24 > thr), MAE without mod-0 targets (h=1), Diebold–Mariano on absolute errors with Newey–West lag 60 (samples treated as one sequence despite gaps: approximation) |
| `src/evaluation/rolling.py` | `per_block` (skill per day/week of a fixed model), `rolling_refit` (re-fit before each block on all samples with target before the block; callbacks) |
| `src/models/tabular_features.py` | feature matrices: lags (59), cal (+mod5 one-hot, hour), cal_cov (+24 covariate changes + alloc_gap), gbm / gbm_cov (+P/L24−1, L24) |
| `src/models/statespace.py` | AutoARIMA → SARIMAX Kalman filter with fixed params; AutoETS (ANN/AAN/AAdN) recursion; rolling h-step forecasts with NaN skipping; verified vs statsforecast fitted (≤ 1e-8) and statsmodels forecasts (0 diff) |
| `src/models/rnn.py` | LSTM/GRU direct 3-horizon network, masked loss, scaling minmax/local, calendar and optional covariate channels, early stopping on mean val MAE |
| `src/run_experiment.py` | YAML-driven runner (`--config`, `--set k=v`, `--tracking-mode`); writes `experiments/<group>/<name>/` (config, history, predictions_val, metrics, model.pt) and logs |
| `src/tracking.py` | JSON record in `experiments/runs/<group>/<name>.json` always + W&B run (mode auto: online, else offline) |
| `configs/tracking.yaml`, `configs/rnn_lstm.yaml`, `configs/rnn_gru.yaml` | 2 layers × 64 hidden, dropout 0.1, MSE, Adam 1e-3, batch 512, ≤ 20 epochs, patience 3, grad clip 1. Sizes are **our choice**; the Alberta layer sizes were not found in the handoffs (check the PDF / literature.md entry 2) |

## 5. Notebooks 07 and 08 (validation only; test closed)

- **07 baselines:** persistence (reproduces nb 06 exactly), persistence + mod-5 (static train mean: almost useless; rolling causal 7-day mean: +5.6% at h=1, none at h=5/15), daily seasonal naive (skill −16 / −10 / −7), ridge AR on changes (lags; + calendar; + covariates), α chosen on val (some at grid edge 10,000; differences in the 4th decimal).
- **H3 out of sample confirmed:** covariates lower AR MAE by 1.5 / 1.8 / 1.4% (DM p ≈ 1e-6); largest coefficients d5_mem_used_sum, d1_gpus_alloc.
- **08:** AutoARIMA (1,1,5) and AutoETS ANN (α 0.70) fitted on 12.7 days (17–29 Jul) and run as fixed-parameter filters: skill only +0.01 to +0.03 (ARIMA at h=1 not significant, p 0.17). LightGBM direct per h on the change, Optuna 15 trials or 300 s per study (selection and early stopping on validation → slightly optimistic), refit with 3 seeds (sd ≤ 0.0008 kW). LightGBM covariates −1.0 / −2.1 / −1.7% MAE (DM p ≤ 0.005). Top gain feature d5_util_gpu_mean (17%).

## 6. Week 3 benchmark table (validation; `reports/week3/benchmark_val.csv`)

| Model | MAE h1 | MAE h5 | MAE h15 | skill h1 / h5 / h15 |
|---|---|---|---|---|
| persistence | 0.2866 | 0.4299 | 0.6048 | 0 |
| pers_mod5_rolling | 0.2707 | 0.4302 | 0.6059 | +0.056 / −0.001 / −0.002 |
| auto_arima (1,1,5) | 0.2838 | 0.4180 | 0.5885 | +0.010 / +0.028 / +0.027 |
| auto_ets ANN | 0.2778 | 0.4178 | 0.5930 | +0.031 / +0.028 / +0.020 |
| ar_lags_cal | 0.2677 | 0.4163 | 0.5869 | +0.066 / +0.032 / +0.030 |
| ar_lags_cal_cov | 0.2637 | 0.4090 | 0.5786 | +0.080 / +0.049 / +0.043 |
| LSTM (3 seeds, local) | 0.2631 ± 0.0056 | 0.3998 ± 0.0010 | 0.5724 ± 0.0043 | +0.082 / +0.070 / +0.054 |
| GRU (3 seeds, local) | 0.2583 ± 0.0034 | 0.4002 ± 0.0025 | 0.5704 ± 0.0036 | +0.099 / +0.069 / +0.057 |
| LightGBM (3 seeds) | 0.2491 | 0.3934 | 0.5659 | +0.131 / +0.085 / +0.065 |
| **LightGBM + cov (3 seeds)** | **0.2464** | **0.3850** | **0.5559** | **+0.140 / +0.104 / +0.082** |

Ranking stable across horizons: ML > deep > linear > statistical > naive. Gains come mostly from ramp steps for linear models; LightGBM also improves normal steps (0.226 → 0.199 at h=1). Ramp recall stays ≤ ~0.18. LSTM 52,419 params, GRU 39,363; ~50 s/epoch on the Mac, best epoch 4–9, 6–11 min per run.

## 7. Week 3 findings (provisional)

1. **Scaling decision:** LSTM seed 0, minmax vs local: mean val MAE 0.4412 vs 0.4125; minmax at h=1 ≈ persistence (0.2852). → **local scaling** for networks (likely because Min-Max is fragile under level shifts between splits; hypothesis).
2. **Weekly skill varies strongly:** h=1, week of 26 Jul skill ≈ +0.21 to +0.24; **week of 16 Aug all models worse than persistence** (AR −0.11, GRU −0.05, LightGBM −0.04). Hypothesis (not tested): models learned the large July five-minute dip and over-correct when the dip nearly vanishes after the 10 Aug maintenance (session 4). Test: per-week skill with and without mod-0 targets; rolling mod-5 features. Report weekly skill on test.
3. Weekly expanding-window re-fit of the AR changes nothing (0.2637 → 0.2632): parameter drift is not the issue at this scale.
4. Multi-output networks stop early (4–9 epochs); h=1 still improving when h=15 degrades → per-horizon networks may help (Week 4 option).
5. Alberta comparison: not comparable (1 s vs 1 min, 300-step vs 60-step window, 90 s vs 1/5/15 min, single run, no naive baseline). At 1-min resolution persistence alone has R² > 0.98 → a high R² without a naive baseline proves little (point for the paper).

## 8. Documents this week

- **Weekly Report 2** (`https://claude.ai/code/artifact/cacd28a3-fb6f-425c-b39e-02af4438c68f`): English tab already updated (rev 18, Meeting 1 outcome, modelling dataset, links line "Code: repository … · W&B: no training runs yet"); **Persian tab synced** this session (rev 13). Student to send to supervisor (due 9 Oct).
- **Weekly Report 3** (new, due Fri 16 Oct; as-of chip 2026-10-16): `https://claude.ai/code/artifact/31fce096-d7b7-4a08-9a3a-c60cfa6ea41c`; tabs English (body `9bf44d64-3915`) and فارسی (body `e81e2bbc-f48b`); tabs listed on the left. Sections: work done (table + code/W&B line), key results (benchmark table, five bullets), blockers and decisions (libomp, scaling, weekly skill; four decisions), plan for Week 4 (checklist). Student may add direct W&B links.
- **Proposal** (`1c7f11e9-…`): English body rev 33, Persian body rev 22. Changes in both tabs: as-of 9 Oct (17 Mehr), valid 282,452 (93.0%), 29 Mar wording, PM100 conditional, EDA table "confirmed at Meeting 1", covariates = separate arm (row + Methods paragraph), resolution row (10-s dropped), split row (local scaling for networks; trees/linear on changes), ramp row (0.0295/0.0399/0.0568; no label with < 12 valid hours), invalid-minutes row (60-min input rule), metrics row (weekly skill), new subsection "Preliminary benchmark (validation)" after the protocol table, Section 9 sentence + six data questions ticked. Figures unchanged (Figure 5 still absolute thresholds).

## 9. Next steps

1. **Documents (done this session):** Weekly Report 3 and proposal (Section 8). Remaining: Meeting 1 packet Outcome column and minutes, if the student wants; update Weekly Report 3 at the end of Week 4 only if asked.
2. **Week 4** (Sprint 2 end): PatchTST and N-HiTS (neuralforecast), Chronos and TimesFM zero-shot (Colab GPU; leakage caveat, secondary results), final table with seeds, ramp-error analysis, DM between best models, ramp-event figure, parameter counts and train/inference times, draft Related Work and Data sections. Then **freeze the benchmark** and open test once (rolling-origin, per-week skill). Decide before test: fixed hyperparameters and rounds; refit on train+val or train only.
3. Smaller: refresh `requirements.lock`; PM100 schema check; per-week skill without mod-0 targets (finding 2); optional per-horizon networks; LightGBM rolling-window features (plan item "rolling mean/std" not added; expected small).

## 10. Files created this week (all should be in the repo)

`src/data/build_model_dataset.py`, `src/evaluation/{metrics,rolling}.py`, `src/models/{tabular_features,statespace,rnn}.py`, `src/run_experiment.py`, `src/tracking.py`, `configs/{tracking,rnn_lstm,rnn_gru}.yaml`, `notebooks/06_model_dataset.ipynb`, `07_baselines.ipynb`, `08_classical_models.ipynb`, `09_week3_completion.ipynb` (+ `_outputs.md` and figures), `data/processed/model_dataset.parquet` + meta (data, not in Git), `reports/{baselines,classical,week3}/`, `reports/figures/{model_dataset,baselines,classical,week3}/`, `reports/model_dataset_build.log`, `experiments/` (runs, wandb; keep `experiments/wandb/` out of Git).

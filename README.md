# Forecasting AI Data Center Power Demand

**A benchmark of classical, foundation and hybrid quantum–classical models on MIT Supercloud GPU power, with calibrated uncertainty and battery model-predictive control.** Bachelor project (Electrical Engineering, Control), delivered October 2026.

Status (14 October 2026): all technical work is complete (Weeks 1–7 of the plan, plus an uncertainty-aware MPC extension). Delivery documents: results report (English and Persian), final research report and presentation slides; links in `docs/DOCUMENTS.md`. Code map: `docs/CODE_MAP.md`.

## Main results (test month, opened once per model group)

The target is the aggregate GPU power of the MIT Supercloud cluster (448 V100 GPUs) at 1-minute resolution, 4 March – 30 September 2021 (303,840 minutes, 93.0% valid), forecast 1, 5 and 15 minutes ahead from a 60-minute window. Skill = 1 − MAE / MAE of persistence on the same origins. Every choice was made on the validation month (29 July – 30 August); the test month (30 August – 30 September) was opened once per model group under a frozen protocol with SHA-256 freeze records.

1. **Forecasting (RQ1).** LightGBM with 12 job-scheduler and telemetry covariates is the best of 29 models at every horizon: −11.4 / −9.0 / −6.2% MAE against persistence (Diebold–Mariano with Holm, p < 1e-7). No model predicts ramp onsets.
2. **Quantum models (RQ2).** A 287-parameter hybrid QNN (4 qubits) is statistically equal to LightGBM, PatchTST and GRU at 15 minutes, but no quantum model beats its matched classical control consistently; no quantum advantage is claimed.
3. **Uncertainty (RQ3).** Adaptive conformal inference gives 0.900 coverage of 90% intervals in all 12 model × horizon cells; ramp minutes are covered only 18–49%.
4. **Battery control (RQ4).** A 2 kWh battery under MPC cuts minutes above a 1 kW/min grid ramp limit from 2.2% to 0.03%; a perfect forecast would halve the battery; planning on the conformal band lowers the worst grid ramp by 21% against persistence.

| Model (test split) | Trainable params | Skill h1 | Skill h5 | Skill h15 |
|---|---|---|---|---|
| LightGBM + covariates | trees | +0.114 | +0.090 | +0.062 |
| LightGBM (power only) | trees | +0.064 | +0.046 | +0.033 |
| TimesFM 2.5, zero-shot (secondary) | 231 M | +0.056 | +0.041 | +0.019 |
| PatchTST | 75,215 | +0.048 | +0.039 | +0.029 |
| GRU (univariate) | 38,019 | +0.036 | +0.041 | +0.031 |
| Hybrid QNN (4 qubits, 24 circuit params) | 287 | +0.020 | +0.034 | +0.029 |
| Hybrid QNN without entanglement | 287 | +0.020 | +0.028 | +0.025 |
| Matched classical MLP control | 285 | +0.009 | +0.029 | +0.026 |
| Quantum reservoir (8 qubits) | 483 | +0.016 | +0.022 | +0.015 |
| Classical random-feature control | 483 | +0.015 | +0.019 | +0.012 |

| Battery MPC, test month, R = 1 kW/min | Excess ramp at 2 kWh (kW) | Max grid ramp at 2 kWh (kW/min) | Needed battery |
|---|---|---|---|
| No battery | 567 | 7.4 | – |
| Perfect forecast (oracle) | 0.9 | 1.9 | 1 kWh |
| Uncertainty-aware MPC, LightGBM band | 14.1 | 3.5 | 2 kWh |
| LightGBM point forecast | 15.3 | 4.4 | 2 kWh |
| Persistence | 15.5 | 4.4 | 2 kWh |
| No forecast (myopic) | 32.3 | 5.5 | 2 kWh |

All circuits are small and simulated exactly (PennyLane `default.qubit`). Results come from one cluster, one period and one test month; see the limitations in the final report.

## Repository map

| Path | Content |
|---|---|
| `src/data/` | `build_series.py`, `build_series_lowmem.py`, `build_log_features.py`, `build_model_dataset.py` (data pipeline and modelling dataset v1.1) |
| `src/evaluation/` | `metrics.py`, `rolling.py`, `compare.py` (DM with Holm), `frozen.py` (frozen re-implementation for the test run), `conformal.py` (conformal intervals, ACI) |
| `src/models/` | `tabular_features.py`, `statespace.py`, `rnn.py`, `nf_models.py`, `quantum.py` (hybrid QNN and controls), `qrc.py` (quantum reservoir and controls), `week6.py`, `qkernel.py` |
| `src/control/` | `battery_mpc.py` (battery model, MPC, simulation), `scenario_mpc.py` (uncertainty-aware MPC) |
| `scripts/` | `reproduce.py` (run notebooks 06–27 in order), `verify_freeze.py` (check the freeze-record hashes) |
| `src/run_experiment.py`, `src/tracking.py`, `configs/` | YAML-driven runs; JSON records in `experiments/runs/` plus Weights & Biases |
| `src/tools/nb_outputs_to_md.py` | Export a saved notebook's outputs to Markdown |
| `notebooks/` | 00–05 data and EDA · 06 modelling dataset · 07–16 benchmark (16 = frozen test) · 17–23 and 22b quantum models (23 = frozen test) · 24 conformal · 25 SHAP · 26 battery MPC · 27 uncertainty-aware MPC |
| `reports/` | Tables, JSON summaries and freeze records per week; `reports/figures/` (incl. `delivery/`) |
| `experiments/` | One folder per run: config, history, predictions, metrics, checkpoint |
| `docs/` | `CODE_MAP.md`, `data_quality.md`, `literature.md`, `DOCUMENTS.md` (index of the Claude Docs), `HANDOFF/`, `weekly/`, `meetings/`, `paper/` |

## Week by week

| Week | Work | Key outputs |
|---|---|---|
| 1 | Literature (26 papers, 10 summaries), repository, environment, PennyLane test, proposal | `docs/literature.md`, `docs/papers/` |
| 2 | 2.2 TiB of raw telemetry reduced on Colab (17 h, 0 failures), time-axis and double-count fixes, EDA, data-quality report | `data/processed/series_full*.parquet`, notebooks 01–05 |
| 3 | Modelling dataset, evaluation module, baselines, AutoARIMA/AutoETS, LightGBM (Optuna), LSTM/GRU (3 seeds), W&B | notebooks 06–09, `reports/week3/benchmark_val.csv` |
| 4 | N-HiTS, PatchTST, DLinear, NLinear, univariate GRU, Chronos-Bolt and TimesFM zero-shot; ramp analysis, DM with Holm; frozen test run (9 Oct) | notebooks 10–16, `reports/week4/` |
| 5 | Hybrid QNN + matched MLP, bottleneck and no-entanglement controls; quantum reservoir + random-feature, raw and no-entanglement controls | notebooks 17–20, `reports/week5/` |
| 6 | Data fraction 1/5/20/100%, qubits 2/4/6/8, inference noise (1,024 shots, IBM Heron-like), quantum kernel, frozen quantum test (12 Oct); Methods draft | notebooks 21–23, 22b, `reports/week6/` |
| 7 | Conformal intervals (split, rolling, ACI, Mondrian) with a frozen test (13 Oct); SHAP; battery model and MPC in four scenarios (14 Oct); uncertainty-aware MPC (14 Oct) | notebooks 24–27, `reports/week7/` |
| 8 | Delivery: results report (EN/FA), final research report, slides, code map, reproduction scripts, licence | `docs/`, `scripts/` |

No supervisor meeting was held during the project; the delivery meeting is the first.

## Reproducing

Environment: conda env `dcpower` (Python 3.11), packages pinned in `requirements.lock`. Run from the repository root:

```
python scripts/reproduce.py
python scripts/reproduce.py --run
python scripts/verify_freeze.py
```

The first command prints the plan, the second runs the validation-only notebooks 06–25 in order (executed copies and their `_outputs.md` go to `notebooks/_reproduced/`), the third checks that every file hashed in the six freeze records is unchanged. Notebooks that open the test split (16, 23, 24, 26, 27) run only with `--include-test`, which writes a second freeze record. Notebooks can also be run by hand from `notebooks/` with the kernel "Python (dcpower)"; export outputs with `python src/tools/nb_outputs_to_md.py notebooks/<name>.ipynb`.

The MIT Supercloud data are CC BY-NC-ND 4.0: raw and processed data are never committed; only code is published (MIT licence, `LICENSE`).

---

# Data pipeline (Weeks 1–2)

## Data source

MIT Supercloud Datacenter Challenge dataset (CC BY-NC-ND 4.0), public AWS S3 bucket, no account needed.

- Prefix: `s3://mit-supercloud-dataset/datacenter-challenge/202201/`
- List: `aws s3 ls --no-sign-request s3://mit-supercloud-dataset/datacenter-challenge/202201/`
- Used: `gpu/` (one CSV per job, 100 ms), `slurm-log.csv` (~103 MB), `labelled_jobids.csv`
- Not used: `cpu/`, `node-data.csv` (~2.27 GB)
- Raw and processed data are never committed (license: NoDerivatives); only scripts are published.

## Building the GPU power series

One command builds the 1-minute aggregate series from the raw MIT Supercloud GPU files.
Files are downloaded, aggregated per minute and deleted immediately, so disk use stays small.
The run is resumable: re-running the same command skips finished files and retries failed ones.
A VPN connection is required for downloads (see `docs/data_quality.md`).

```bash
caffeinate -i /opt/miniconda3/envs/dcpower/bin/python -u src/data/build_series.py \
  --files data/interim/dev_window_files.parquet \
  --stream-dir data/interim/streams_dev \
  --tmp-dir data/interim/_tmp_raw \
  --out data/processed/series_dev.parquet \
  --win-start 2021-09-10 --win-end 2021-09-17 --workers 8
```

For multi-hour runs, start it from Terminal with `caffeinate -i nohup ... > log 2>&1 &`.

### Development window (2021-09-10 to 2021-09-17, file clock = America/New_York)

| Item | Value |
|---|---|
| Input | 7,435 GPU files, 134.54 GiB raw CSV (downloaded, then deleted) |
| Intermediate | 7,435 per-file stream Parquets, 111.2 MiB (`data/interim/streams_dev/`) |
| Output | `data/processed/series_dev.parquet`, 10,080 rows, 559.8 KiB |
| First pass | 368.4 min at 5.70 MiB/s with 8 workers; 310 files failed with transient SSL errors |
| Retry | 37.2 min at 5.24 MiB/s; 0 failures |
| Series rebuild from existing streams | 1.8 s (measured; output identical to the saved file) |

### Output columns

| Column | Meaning |
|---|---|
| `minute_epoch`, `minute_local` | Minute start in the file clock (New York local time) |
| `power_W` | Time-weighted sum of GPU power over all job streams (energy / 60 s) |
| `util_gpu_mean`, `util_mem_mean`, `temp_gpu_mean`, `temp_mem_mean` | Time-weighted means over streams |
| `mem_used_sum`, `mem_free_sum` | Time-weighted sums over streams (MiB) |
| `n_gpu_equiv` | Sum of stream weights (full-GPU equivalents) |
| `n_gpu_streams` | Raw count of distinct (file, gpu_index) streams in the minute |
| `n_edge_streams` | Streams in their first or last minute (partial coverage) |

Minutes with no job stream (722 in the development window) have `n_gpu_streams = 0` and null metrics.
The aggregation rule and all data issues are documented in `docs/data_quality.md`.

## Full dataset on Google Colab

The raw GPU files total 2,269.5 GiB, far above a local download budget. The per-file stage of
`build_series.py` (download, per-minute aggregation, delete raw CSV) therefore runs on Google Colab:
raw data flows over Google's network, and only the stream Parquets (about 2 GB for the whole
dataset, estimated from a compression ratio of about 1,239x) are copied back. Building the series
(`build_series` over the streams), the log features and all analysis run locally.

**Setup.** Google Drive folder `MyDrive/dcpower/` with `build_series.py` and
`all_files_desc.parquet` (all 109,946 files with job start and end in the file clock, sorted by job
end, newest first). Raw CSVs go to the Colab local disk (`--tmp-dir /content/tmp_raw`); streams go
to Drive (`--stream-dir .../streams_full`). The window is the whole period
(`--win-start 2021-01-01 --win-end 2021-10-02`), so no stream is clipped.

**Monthly chunks.** Files are processed in chunks by the month in which the job ended, newest first.
After chunk k, the series is complete from the start of that month onward, so every finished chunk
yields a usable, contiguous series and an interrupted session wastes nothing (the run is resumable).

| Chunk | Jobs ending in | Raw size |
|---|---|---|
| 1 | September and 1 October | 437.5 GiB |
| 2 | August | 385.0 GiB |
| 3 | July | 369.7 GiB |
| 4 | June | 367.6 GiB |
| 5 | May | 295.1 GiB |
| 6 | April | 218.9 GiB |
| 7 | March and February | 195.7 GiB |

**Throughput (free Colab, 2 vCPUs, 12 GB RAM).** Small tests: 13.9 MiB/s with 8 workers,
20.4 MiB/s with 16, 16.7 MiB/s with 32 (network-bound, CPU about 45%). Chunk 1 ran at about
43 MiB/s with 16 workers and was CPU-bound (100% on both cores), so the whole dataset needs roughly
15 hours, i.e. two to three Colab sessions. Free Colab sessions last at most 12 hours and usage
limits are not published.

**Risks.** Google Drive has unpublished per-user operation and bandwidth quotas; the full run writes
about 110,000 small files. If `Input/output error` appears, write streams to the Colab disk and copy
them to Drive in batches. MIT Supercloud data is CC BY-NC-ND: keep the Drive folder private.

## Full series (Week 2 result)

The whole MIT Supercloud GPU dataset is processed. Raw files were never stored on the Mac: Google Colab downloads each file from S3, reduces it to one-minute rows per GPU ("stream") on Google Drive, and deletes the raw file.

| Chunk | Jobs ending | Files | Raw size | Failed |
|---|---|---|---|---|
| 1 | 2021-09-01 .. 10-02 | 26,479 | 437.5 GiB | 0 |
| 2 | 2021-08-01 .. 09-01 | 22,710 | 385.0 GiB | 0 |
| 3 | 2021-07-01 .. 08-01 | 16,525 | 369.7 GiB | 0 |
| 4 | 2021-06-01 .. 07-01 | 11,881 | 367.6 GiB | 0 |
| 5 | 2021-05-01 .. 06-01 | 14,862 | 295.1 GiB | 0 |
| 6 | 2021-04-01 .. 05-01 | 7,582 | 218.9 GiB | 0 |
| 7 | 2021-01-01 .. 04-01 | 9,907 | 195.7 GiB | 0 |
| **Total** | | **109,946** | **2,269.5 GiB** | **0** |

Throughput 42–45 MiB/s (16 workers); about 17 h of Colab processing in total.

### Low-memory aggregation

The one-shot aggregation in `build_series.py` loads all streams into RAM and was killed (out of memory, 13.6 GB) from about 77k streams on. `src/data/build_series_lowmem.py` applies the same rules in batches of 2,000 streams, writes per-minute partial sums, and adds them up at the end. Validated on chunks 1–3 against the original series: max difference 2.9e-11 W for power, counts identical.

```
python src/data/build_series_lowmem.py --stream-dir <streams_full> --work-dir <local tmp dir> --out series_full_2021.parquet --win-start 2021-01-01 --win-end 2021-10-02 --batch 2000
```

Optional `--files chunk_k.parquet ...` restricts the run to the streams of given file lists. Batches are resumable (finished partials are skipped).

### Outputs

| File | Content |
|---|---|
| `data/processed/series_full_2021.parquet` | Full series, 2021-01-01 .. 2021-10-02 (file clock), 394,560 rows, 18.5 MiB |
| `data/processed/series_full.parquet` | Analysis window 2021-03-04 .. 2021-09-30, 303,840 rows (notebook 05 v2, cell 4) |
| `data/processed/series_full_features.parquet` | Analysis window with Slurm-log features and flags, 20 columns (notebook 05 v2, cell 5) |

EDA on the full series: `notebooks/05_eda_full_v2.ipynb` (figures in `reports/figures/eda_full_v2/`, summary in `reports/eda_full_summary_v2.json`). Validity rules and data issues: `docs/data_quality.md`, Sections 15–22.

### Tools

`src/tools/nb_outputs_to_md.py` exports the outputs of a saved notebook to Markdown (`python src/tools/nb_outputs_to_md.py notebooks/<name>.ipynb`).

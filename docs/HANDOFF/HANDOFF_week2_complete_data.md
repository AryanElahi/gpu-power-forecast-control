# Handoff: Week 2 (data engineering and EDA), state at the end of 2 October 2026

**Project:** Forecasting AI Data Center Power Demand: A Benchmark of Classical, Foundation, and Hybrid Quantum Models with Uncertainty Quantification and Grid-Aware Control
**Version:** 4.0. Supersedes `HANDOFF_week2_progress.md` (v3.0) and continues `HANDOFF_week1_complete.md` (v2.0, still valid for Week 1 rules, protocol and decisions).
**Date:** Friday 2 October 2026 (10 Mehr 1405)
**Purpose:** a self-contained record of everything done in Week 2, every verified number, every decision, every correction, the running Colab job, and the exact remaining steps, so that a new chat can continue without the earlier conversations.

**Read together with:** `Project_Context_FA.txt` (master plan, v1.0) and `HANDOFF_week1_complete.md` (Week 1). This file replaces `HANDOFF_week2_progress.md`; you do not need to upload that one any more.

---

## 0. How to use this file

| If you need | Go to |
|---|---|
| Status in one minute | 1 |
| Working rules (updated) | 2 |
| Environment (Mac, Colab, Drive) | 3 |
| Verified facts about the data | 4 |
| Time frame (critical) | 5 |
| Pipeline code: `build_series.py` v3 and `build_log_features.py` | 6 |
| Development-window run and validation | 7 |
| Outages and maintenance days | 8 |
| EDA results (all numbers) | 9 |
| Data-volume analysis and the dataset decision | 10 |
| **Colab full-dataset run (in progress)** | 11 |
| Files on the Mac and on Google Drive | 12 |
| All Week 2 decisions | 13 |
| Unverified items and watch-outs | 14 |
| Correction history | 15 |
| **Remaining work, step by step** | 16 |
| Meeting 1 agenda (draft) | 17 |
| Week 3 preview | 18 |
| Step logs (notebook 02, notebook 03, Colab) | 19 |
| **Opening prompt for the new chat** | 20 |

---

## 1. Snapshot

- **Week 2 is NOT complete.** Both completion criteria of the plan are met (one command builds the series from raw data; the ramp threshold is a concrete number, provisional), but the longer series, the final proposal, Weekly Report 2 and the Meeting 1 packet are still open.
- **Done in Week 2:** Slurm log understood; GPU jobs identified; `gpu/` indexed; file format understood; time-frame rule found and validated; pipeline written, run on the development week (7,435 files, 134.5 GiB), failures retried, **aggregation bug found and fixed (time weighting)**; outage and maintenance flags; Slurm-log covariates; `docs/data_quality.md`; README sections; EDA notebook with seven analyses and a written summary; provisional ramp thresholds; data-volume analysis; PM100 size checked; **Colab route designed, tested and launched for the whole dataset**.
- **Running right now:** chunk 1 of the whole-dataset run on Google Colab (jobs ending in September and on 1 October: 26,479 files, 437.5 GiB). Last observed: 134.9 of 437.5 GiB after 52 min 57 s, 8,450 of 26,479 files, 0 failures, 43.54 MiB/s, CPU 100%, 91 GB free on the Colab disk, about 2 h remaining. See Section 11.
- **Next:** finish chunk 1, bring its streams to the Mac, build and validate a September series; then the proposal, the Meeting 1 agenda, Weekly Report 2 (due Friday 9 October 2026); continue the other Colab chunks in later sessions.

| # | Week 2 task (plan) | Status |
|---|---|---|
| 1 | Check the full run, validate the series | Done |
| 2 | Quick look at the series; run time and sizes in README | Done |
| 3 | Decide empty minutes and edge minutes | Done |
| 4 | Covariates from the log; labelled files | Done |
| 5 | `docs/data_quality.md` | Done (Sections 1 to 14) |
| 6 | EDA notebook and provisional ramp threshold | Done |
| 7 | Decide the longer series | In progress: whole dataset via Colab, chunk 1 running |
| 8 | Finalize the proposal and fixed-decisions table; five-minute EDA summary | Open |
| 9 | Weekly Report 2 (bilingual, four-part template), due Friday 9 October 2026 | Open |
| 10 | Meeting 1 preparation | Agenda drafted (Section 17); packet open |
| 11 | Housekeeping | Mostly done; leftovers in Section 16 |

---

## 2. Working rules (updated; Week 1 rules still apply)

- **Chat in Persian, always.** (The assistant slipped into English twice in this session; do not.) Anything written into notebooks, code comments, scripts and repo docs is in **English**.
- **One step at a time:** one cell or one action per message, then wait for the pasted output. The student sometimes pastes code instead of output; ask for what is printed below the cell.
- Answer in chat; create files only when asked or when clearly a deliverable. Week 1 deliverables (proposal, weekly report) are **Claude Docs with an English and a Persian tab**.
- **Git: the student handles it completely. Do not give Git instructions or Git checks.** (He explicitly asked this.) At most a one-line reminder that data must never be committed.
- **Terminal (zsh) lessons:** never put `#` comments in commands to paste (zsh does not treat `#` as a comment by default; an apostrophe in a comment, e.g. `run's`, opened a quote and produced the `quote>` prompt). Give Terminal commands **one per code block, without comments**. A new Terminal window starts in `~`, so start with `cd ~/Desktop/Other/datacenter-power-forecasting`. `rm` on more than 100 files triggers a zsh confirmation (`y`).
- **Notebook lessons:** the notebook working directory is `notebooks/`, so use an absolute root: `ROOT = Path.home() / "Desktop/Other/datacenter-power-forecasting"` (or `PROJECT_DIR = Path.cwd().parent`). When a fix is needed, give the **whole corrected cell**, not a line edit (two line edits went wrong: merged lines). Avoid reusing variable names: a loop `for old, new in edits` overwrote DataFrames called `old` and `new`.
- **Edit scripts from a notebook cell** (read, replace, `py_compile`, print a check) rather than asking the student to edit in an editor: one manual edit was never saved and an old version ran silently. Always verify with a content check (`"new_column" in src`) and the file's modification time.
- Cautious claims: mark unverified points; never claim quantum advantage; validate any time conversion against GPU file timestamps before using it for selection.
- **Downloads:** the student's internet budget is **about 100 GB in total, and large downloads only at night.** Heavy raw-data work runs on **Google Colab** (Section 11); only small outputs come back.
- Long local runs go to Terminal with `caffeinate -i nohup ... &`; the Colab run needs the browser tab open and the Mac awake (`caffeinate -d` in a Terminal window).
- **Announce explicitly** when Week 2 is complete.

---

## 3. Environment

### 3.1 Mac

- Project root: `~/Desktop/Other/datacenter-power-forecasting/` (absolute `/Users/aryaneuse/Desktop/Other/datacenter-power-forecasting/`).
- Conda env `dcpower`, kernel `Python (dcpower)`; Terminal Python `/opt/miniconda3/envs/dcpower/bin/python`; Python 3.11.
- Versions seen: polars 1.44.2, matplotlib 3.11.2, numpy 2.4.6, pandas 3.0.6, statsmodels 0.15.0, scipy 1.17.1.
- VPN needed for S3 from the Mac; it truncates any single HTTP response at 8 MiB, so downloads use 4 MiB Range chunks. Measured Mac throughput: 5.45 to 5.70 MiB/s with 8 threads.
- Notebooks: `notebooks/02_data_download_explore.ipynb` (data work; the file was accidentally named `02_data_download_explore.ipynb.ipynb`; the student was asked to rename it in JupyterLab, not confirmed), `notebooks/03_eda.ipynb` (EDA).

### 3.2 Google Colab (free tier)

- Runtime type CPU. Measured: Python 3.13, polars 1.44.2 (installed with `pip -q install polars==1.44.2`), **2 vCPUs, 12 GiB RAM, about 87 to 88 GB free on `/content`**.
- S3 is reached directly from Colab without VPN (the script uses `https://mit-supercloud-dataset.s3.amazonaws.com/datacenter-challenge/202201/gpu`). The VPN is needed only to open Colab in the browser.
- Colab FAQ (checked 2 October 2026): resources are not guaranteed or unlimited, limits fluctuate and are not published; free notebooks run at most 12 hours; Google Drive enforces unpublished per-user operation and bandwidth quotas (exceeding them gives `Input/output error`).

### 3.3 Google Drive

- About 9 GB free (15 GB shared with Gmail and Photos).
- Folder `MyDrive/dcpower/` contains: `build_series.py` (v3, time-weighted), `dev_window_files.parquet`, `all_files_desc.parquet`, `chunk_1.parquet` (written by Colab cell 5), `streams_test/` (280 test outputs clipped to the development week; **obsolete, delete**), `streams_full/` (the whole-dataset streams, filling up).
- Keep the folder private (MIT Supercloud licence CC BY-NC-ND).

---

## 4. Verified facts about the data

### 4.1 `slurm-log.csv` (`data/raw/`, 103,165,427 bytes)

- 395,914 jobs, 29 columns (`id_job, id_array_job, id_array_task, id_user, kill_requid, nodes_alloc, nodelist, cpus_req, derived_ec, exit_code, gres_used, array_max_tasks, array_task_pending, constraints, flags, mem_req, partition, priority, state, timelimit, time_submit, time_eligible, time_start, time_end, time_suspended, track_steps, tres_alloc, tres_req, job_type`). All 395,914 have `time_end > time_start`.
- Times are integer Unix seconds in **true UTC**. `gres_used` is null in all rows.
- **GPU jobs:** regex `(?:^|,)100[12]=(\d+)` on `tres_alloc` (codes 1001 gpu:tesla, 1002 gpu:volta from `tres-mapping.txt`); the captured number is the GPU count. **99,344 GPU jobs**, February to 1 October 2021; per-job GPUs: 1: 78,111; 2: 18,860; 4: 1,593; up to 128.
- **`nodelist` holds node names such as `r6760045-n386398`**, the same strings as in GPU file names.
- Load grows over time (about 1.5k GPU jobs in February, about 23k in September).

### 4.2 `gpu/` folder

- `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/`, sub-folders `0000` to `0099`; **109,946 files, 2,269.5 GiB**; index in `data/interim/gpu_file_index.parquet` (`sub, name, size, id_job`).
- File name `<id_job>-r<digits>-n<digits>.csv`. **Node id = the whole `r...-n...` string** (matches `nodelist`); the `r...` part alone is not unique. Meaning of `n...` unverified.
- One file = one node of one job, all GPUs of that node interleaved. `gpu_index` is the index within the node (0, 1).
- 98,176 distinct jobs have files; 1,168 GPU jobs (about 1.2%) have none; every file's job is in the log; every file joins to log start and end.
- File clock range of all files: 2021-02-05 16:35:15 to 2021-10-01 11:01:58.

### 4.3 File content

- Columns: `timestamp` (epoch float), `gpu_index`, `utilization_gpu_pct`, `utilization_memory_pct`, `memory_free_MiB`, `memory_used_MiB`, `temperature_gpu`, `temperature_memory`, `power_draw_W`, `pcie_link_width_current`.
- Sampling about every 0.103 to 0.107 s, **about 560 samples per minute**. Per-stream median of full-minute sample counts: 539 (1st percentile), 551 (5th), 562 (median), 581 (95th and 99th). Interior minutes occasionally have fewer samples: minimum 287, 0.1% quantile 461, 1% 484, 5% 543, median 570, maximum 582.
- **No non-numeric values**: all 3,210,563 stream-minute rows of the development week have zero nulls in every metric.
- Idle GPUs inside jobs draw about 27 to 47 W; power per GPU on normal days 58 to 98 W (minute means), far below the V100 maximum (about 300 W).

### 4.4 Labelled jobs

- `labelled_jobids.csv`: 3,430 jobs, columns `id_job, model`; `labelled_job_stats.csv`: 26 model types with counts (inception3/4, vgg11/16/19, bert-base-uncased, distilbert-base-uncased, U-Nets U3/U4/U5 with 32/64/128, graph networks dimenet, schnet, pna, conv, and others).
- Only **6 of 5,379** GPU jobs active in the development week are labelled (about 0.1%): **not used as a covariate**.

---

## 5. Time frame (critical)

**Rule (empirical):** Slurm log times are UTC; GPU file timestamps are America/New_York wall-clock times stored as epoch seconds. Conversion used everywhere:

```python
def log_to_file_clock(col):
    return (pl.from_epoch(col, time_unit="s").dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York").dt.replace_time_zone(None).dt.epoch("s"))
```

The series lives in the file clock (`minute_local`, New York local time). Window dates are converted with `calendar.timegm(datetime(...).timetuple())` (naive epoch, independent of the Mac's zone).

**Evidence (three independent lines):**
1. On 25 files (February to October, both sides of the DST change) residuals between converted log times and file times: at most 1.9 s at the start and 9.3 s at the end (means 0.7 s and 1.2 s).
2. All long cluster-wide outages start within 15 minutes after **local** midnight in the file clock (Section 8).
3. GPUs allocated according to the log versus GPU equivalents measured in the files on normal days: correlation 0.9975, mean difference 0.8 GPUs (0.2%).

**Exception found this session:** when jobs are killed at the start of a maintenance outage, GPU files keep recording for **up to about 3 minutes** after the log's job end (2021-09-16: log ends 00:00:24 to 00:00:27, telemetry until about 00:02:55 at idle power). The 9.3 s bound holds for normal job ends only. Mechanism unverified.

**Consequences:** the local hour 02:00 to 02:59 on 14 March 2021 does not exist; the ambiguous November hour is outside the data.

---

## 6. Pipeline code

### 6.1 `src/data/build_series.py` (v3, 243 lines; also uploaded to Drive)

**Per file, in parallel (ThreadPoolExecutor), resumable:**
1. Download with HTTP Range chunks of 4 MiB, up to 5 retries per chunk with exponential backoff; if a partial file exists in `--tmp-dir`, the download resumes from its size (line 52); the final size is verified.
2. `pl.scan_csv` (lazy, streaming), cast `timestamp` and metrics to Float64 with `strict=False`, **clip to the window** `[t0, t1)` (file clock).
3. Mean of each metric per `(gpu_index, minute)` plus `n_samples`. Output columns: `gpu_index, minute_epoch, power_W, util_gpu, util_mem, temp_gpu, temp_mem, mem_used, mem_free, n_samples, file`.
4. Write `<stem>.parquet.tmp`, rename to `<stem>.parquet`, delete the raw CSV.
5. At start, files whose `<stem>.parquet` exists are skipped (line 163); failures go to `<stream-dir>/_failures.csv` (written **only when failures exist**, so an old file can stay behind and be stale).
6. At the end, `build_series()` aggregates all stream Parquets into the series, writes `--out`, and prints `series rows; minutes with no stream; power_W min/max`.

**Arguments:** `--files` (Parquet with at least `sub, name, size`), `--stream-dir`, `--tmp-dir`, `--out`, `--win-start`, `--win-end` (YYYY-MM-DD, file clock, end exclusive), `--workers` (default 8), `--limit`, `--max-size-mib`.

**`build_series()` (rewritten this session; time-weighted):**
- A stream is `(file, gpu_index)`. For each stream, `_m0`/`_m1` = first and last minute; a minute is **interior** if strictly between them.
- `_full` = median `n_samples` over the stream's interior minutes; fallback = global median of interior minutes.
- Weight `_w` = 1 for interior minutes; `min(n_samples / _full, 1)` for the first and last minute.
- Per minute: `power_W` = Σ(power × w) (energy / 60 s); `util_gpu_mean, util_mem_mean, temp_gpu_mean, temp_mem_mean` = weighted means (nulls excluded from numerator and denominator); `mem_used_sum, mem_free_sum` = weighted sums; **`n_gpu_equiv`** = Σw; `n_gpu_streams` = raw count of distinct streams; **`n_edge_streams`** = count of non-interior stream-minutes.
- Joined to a complete minute grid `[t0, t1)`; empty minutes: `n_gpu_streams = 0`, `n_edge_streams = 0`, `n_gpu_equiv = 0.0`, metrics null.
- Series columns: `minute_epoch, power_W, util_gpu_mean, util_mem_mean, temp_gpu_mean, temp_mem_mean, mem_used_sum, mem_free_sum, n_gpu_equiv, n_gpu_streams, n_edge_streams, minute_local`.
- Note: because the weights use each stream's first and last minute, streams must be **unclipped** (or clipped consistently) for the weights to mean "job start/end". Development-week streams are clipped at the window edges; the effect is only at the window boundaries.

**Backup of v2:** `src/data/build_series.py.bak_unweighted` = the old unweighted version. `src/` has **never been committed**, so this `.bak` is the only copy of v2 (needed to reproduce `series_dev_v1_unweighted.parquet`). Keep it; keep it out of Git.

### 6.2 `src/data/build_log_features.py` (new, 148 lines)

```bash
/opt/miniconda3/envs/dcpower/bin/python -u src/data/build_log_features.py --log data/raw/slurm-log.csv --series data/processed/series_dev.parquet --out data/processed/series_dev_features.parquet
```

- Loads the log (`id_job, tres_alloc, time_submit, time_start, time_end`), extracts `n_gpu`, converts times with `log_to_file_clock`.
- **Cluster-wide gaps** over the whole log: sort jobs by start, running maximum of end; a gap is where the next start exceeds the running maximum end. Found **42 gaps**, **9 long (≥ 6 h)**.
- **Maintenance days** = local date of (gap start + 2 h) for long gaps: 2021-01-21, 2021-03-02, 2021-03-18, 2021-04-15, 2021-05-20, 2021-06-17, 2021-07-15, 2021-08-10, 2021-09-16.
- Per minute (state sampled at **minute end, t + 60 s**, so no information from after the minute): `active_jobs`, `active_gpu_jobs`, `gpus_alloc` (Σ n_gpu of running GPU jobs), `queued_jobs`, `queued_gpu_jobs` (submit ≤ t < start). Counting by sorted `searchsorted` (fast on the full log).
- Flags: `cluster_gap` (minute entirely inside a log gap), **`outage` = cluster_gap AND `n_gpu_streams == 0`** (telemetry takes precedence over the log), `maint_day` (whole local day).
- Empty-minute rule: outage → metrics stay null; cluster up but no GPU stream → `power_W`, `mem_used_sum`, `mem_free_sum`, `n_gpu_equiv` set to 0 (utilization and temperature stay null).
- Output `data/processed/series_dev_features.parquet`: 10,080 rows, **20 columns**: the 12 series columns + `cluster_gap, active_jobs, active_gpu_jobs, gpus_alloc, queued_jobs, queued_gpu_jobs, outage, maint_day`.
- Printed checks of the final run: outage 674; cluster up, no GPU job (set to 0) 48; maint_day 1,440; null `power_W` 674; cluster-gap minutes with telemetry (kept) 2; `gpus_alloc` max 426.
- `--outage-min-hours` (default 6) is our choice and is documented.

---

## 7. Development-window run and validation

**Window:** 2021-09-10 00:00 to 2021-09-17 00:00 (file clock), 10,080 minutes; files selected by job-time overlap: `data/interim/dev_window_files.parquet` (7,435 files, 134.54 GiB, columns `sub, name, size, id_job, start_epoch, end_epoch, duration_h`).

### 7.1 Runs

| Run | Result |
|---|---|
| First pass (Terminal, 8 workers) | 368.4 min, 5.70 MiB/s; done 7,095, skipped 30 (the earlier test), **failed 310** |
| Failures | all 310 `URLError(SSLEOFError(8, '[SSL: ...`, i.e. transient SSL drops (VPN or network); 0 had outputs; 310 partial files (4,572 MiB) left in `_tmp_raw/` |
| Retry | partial files deleted (`rm -f data/interim/_tmp_raw/*`), run-1 list kept as `_failures_run1.csv`; same command, log `build_dev_retry.log`: 37.2 min, 5.24 MiB/s, done 310, skipped 7,125, **failed 0** |
| Stale file | `streams_dev/_failures.csv` from run 1 was not overwritten (script writes it only when failures exist); removed |
| Rebuild with v3 | log `build_dev_rebuild.log`; all 7,435 skipped; series rebuilt |
| Rebuild timing | calling `build_series()` from the notebook: **1.8 s**, output identical to the saved file |

**Sizes:** 134.54 GiB raw CSV → 7,435 stream Parquets, **111.2 MiB** (ratio about 1,239x) → `series_dev.parquet` 559.8 KiB (v1 unweighted: 526.2 KiB). Stream-minute rows: 3,210,563 (interior 3,189,457); 10,553 streams, all with interior minutes.

### 7.2 Validation and the aggregation bug

- v1 (unweighted sum of stream means) gave power max 64.5 kW and up to **540 streams in one minute** (cluster has 448 GPUs).
- Peak minute 2021-09-16 15:31: 410 streams, 0 partial, 157.4 W per stream, mean utilization 79.3%. **Log cross-check:** exactly 4 GPU jobs active cluster-wide, **410 GPUs allocated** on 205 nodes (205 files × 2 GPUs): jobs 24269420549306 (16 GPUs, 8 nodes), 28000495821100 (8, 4), **48436739809721 (384 GPUs, 192 nodes)**, 30715379517168 (2, 1); one user (5042570016904), one array job (16618712154521); starts 15:04:25 to 15:24:55, first end 15:32:05. **The peak is real.**
- Minute 15:32: 538 streams, 512 partial (most about 47 samples, about 5 s). v1 49.37 kW; time-weighted **8.22 kW** (`n_gpu_equiv` 129.4). **The unweighted method double-counted GPUs in job-turnover minutes.** Fixed with time weighting (Section 6.1).
- Physical GPUs per minute with the correct node key `(r...-n..., gpu_index)`: max **418** (any coverage and full coverage). With the wrong key (`r...` only) one got a meaningless 92.
- v3 versus v1: 3,764 minutes changed; mean change -0.074 kW; weekly energy 4,672.5 kWh (v1) versus 4,660.9 kWh (v3), **-0.25%**; `n_gpu_equiv` max **426**; `n_edge_streams` max 512; minimum power 7.88 W (v1 52.7 W). Quantiles v3: p1 3.33, p50 30.79, p95 36.94, p99 41.38, max 64.53 kW (v1: p1 3.39, p25 27.16, p50 30.83, p75 33.91, p95 36.98, p99 41.40).
- Empty minutes: 722 (v1 run 1: 732, before the retry). Null counts: 722 in every metric; none in time columns or `n_gpu_streams`.

### 7.3 Log covariates (development week)

| Covariate | mean | max |
|---|---|---|
| `active_jobs` | 546.7 | 1,213 |
| `active_gpu_jobs` | 229.0 | 324 |
| `gpus_alloc` | 318.3 | 426 |
| `queued_jobs` | 261.6 | 1,182 |
| `queued_gpu_jobs` | 136.5 | 369 |

- `gpus_alloc` versus `n_gpu_equiv` on normal days: correlation **0.9975**, mean difference 0.81 (1% to 99%: -2.47 to 9.62), mean relative difference 0.22%. Consistent with the 1.2% of GPU jobs without files (unverified).
- Same-minute correlation with power on normal days: `active_gpu_jobs` 0.891, `n_gpu_equiv` 0.884, `gpus_alloc` 0.878, `queued_gpu_jobs` 0.280, `queued_jobs` 0.260, `active_jobs` 0.074, `util_gpu_mean` 0.023.

---

## 8. Outages and maintenance days

### 8.1 Whole log: cluster-wide gaps of at least 10 minutes (19 gaps, 126.9 h)

| Gap start (local) | Gap end (local) | Minutes |
|---|---|---|
| 2021-01-21 00:02:20 | 2021-01-21 14:45:30 | 883.2 |
| 2021-01-21 15:59:16 | 2021-01-21 16:22:03 | 22.8 |
| 2021-03-02 00:01:00 | 2021-03-02 18:24:48 | 1,103.8 |
| 2021-03-02 19:09:53 | 2021-03-02 19:47:07 | 37.2 |
| 2021-03-18 00:12:05 | 2021-03-18 14:03:02 | 831.0 |
| 2021-03-29 12:21:13 | 2021-03-29 15:04:49 | 163.6 (midday, looks unscheduled) |
| 2021-04-15 00:03:59 | 2021-04-15 12:03:02 | 719.0 |
| 2021-04-15 12:08:24 | 2021-04-15 12:40:10 | 31.8 |
| 2021-04-15 13:09:53 | 2021-04-15 16:15:41 | 185.8 |
| 2021-05-20 00:00:17 | 2021-05-20 09:42:37 | 582.3 |
| 2021-06-17 00:15:22 | 2021-06-17 12:05:38 | 710.3 |
| 2021-06-17 13:28:46 | 2021-06-17 13:49:52 | 21.1 |
| 2021-07-14 23:58:03 | 2021-07-15 16:24:25 | 986.4 |
| 2021-08-09 23:59:46 | 2021-08-10 10:49:13 | 649.4 |
| 2021-08-10 22:25:36 | 2021-08-10 22:39:29 | 13.9 |
| 2021-09-16 00:00:28 | 2021-09-16 09:55:20 | 594.9 |
| 2021-09-16 11:23:42 | 2021-09-16 11:49:55 | 26.2 |
| 2021-09-16 11:56:17 | 2021-09-16 12:17:09 | 20.9 |
| 2021-09-16 14:20:24 | 2021-09-16 14:51:04 | 30.7 |

Long outages start around local midnight, mostly on the **third Thursday** of the month (21 Jan, 18 Mar, 15 Apr, 20 May, 17 Jun, 15 Jul, 16 Sep); exceptions 2 Mar and 10 Aug (Tuesdays); none in February. That they are scheduled maintenance is **unverified** (no official schedule seen). After an outage the cluster runs a few very large jobs of one user for several hours (likely post-maintenance tests, unverified), then normal load returns.

### 8.2 Development week (16 September)

- Hourly log activity: zero jobs of any kind active and zero submitted from 01:00 to 08:59; 09:00 to 17:00 only 1 to 13 GPU jobs per hour with very large allocations; normal operation from about 18:00 (139 to 218 GPU jobs per hour by evening). `active_gpu_jobs` equals `active_gpu_jobs_with_file` almost everywhere, so empty minutes are not missing files.
- Empty-minute runs (8, all on 16 Sep): 00:03 to 09:54 (592), 14:21 to 14:58 (38), 11:15 to 11:48 (34), 17:00 to 17:25 (26), 11:57 to 12:16 (20), 10:30 to 10:34 (5), 10:51 to 10:55 (5), 10:16 to 10:17 (2).
- Classification of the 722 empty minutes: **674 inside a cluster-wide gap (outage)**, **48 with the cluster up but no GPU job** (10:30 to 10:34, 11:15 to 11:23, 14:51 to 14:58, 17:00 to 17:25). A minute only partly inside a gap counts as "cluster up" (conservative).
- Two minutes (00:01 and 00:02) are inside the log gap but have telemetry (35 and 2 streams, 134 W and 50 W, idle level): the telemetry tail after forced job kills (Section 5).

### 8.3 Decision (implemented in `build_log_features.py`)

| Case | Minutes (dev week) | `power_W` |
|---|---|---|
| Outage: in a cluster-wide gap and no telemetry | 674 | null (masked in modelling) |
| Cluster up, no GPU job | 48 | 0 |

Maintenance days are excluded from the ramp threshold and from EDA statistics and are reported separately in evaluation. If the schedule was announced in advance, `maint_day` is a legitimate calendar covariate (ask the supervisor).

---

## 9. EDA results (`notebooks/03_eda.ipynb`, development week)

All statistics use the **8,640 contiguous normal-day minutes** (2021-09-10 to 2021-09-15); the maintenance day 2021-09-16 is the last day of the window and is excluded (`valid = ~outage & ~maint_day`). ΔP only between consecutive valid minutes. One week is short: **every number here is provisional**. Figures in `reports/figures/eda/`.

| # | Analysis (figure) | Result |
|---|---|---|
| 1 | Setup | 10,080 rows, 20 columns; valid 8,640; maint_day 1,440; outage 674; zero-power minutes 48 (all on 16 Sep); normal days min 21.63, median 31.07, **max 42.20 kW** (below the paper's about 45 kW) |
| 2 | Week, one day (14 Sep), one hour (14 Sep 10:00) (`01_series_week_day_hour.png`) | Mean 31.2 kW, std 4.0 kW. Slow downward trend over six days (about 40 to 23 kW). Load moves in **steps of several kW** (e.g. +5 kW at about 09:40, back at about 11:30, a drop to about 22 kW at about 18:30) with small minute noise (about ±0.5 kW); a 2-minute dip of about 3 kW at 10:20 that reverts. Maintenance day: zeros, test spikes up to about 64 kW, recovery to about 25 kW |
| 3 | Daily profile (`02_daily_profile.png`) | Daily means 36.0, 33.7, 30.2, 30.4, 30.8, 26.0 kW (Fri 10 to Wed 15). Hour of day explains **0.20** of within-day hourly variance, close to the about 0.16 expected from fitting 24 means to 144 noise points (rough estimate): **no reliable daily cycle**. No visible weekend effect (one weekend only) |
| 4 | ACF/PACF (`03_acf_pacf.png`) | ACF(P): 0.995 (1 min), 0.985 (5), 0.961 (15), 0.869 (60), 0.509 (360), 0.182 (1440); first below 0.5 at **379 min**; no daily bump. PACF(P): 0.995, 0.166, 0.076, then within noise (AR(2)-like). ACF(ΔP): **-0.193** at lag 1, -0.054, -0.028, +0.078 at lag 5 (95% band ±0.021) |
| 5 | ΔP distribution and ramp threshold (`04_dP_distribution_ramp_threshold.png`) | See table below |
| 6 | Lagged covariate correlations (train only) | See table below |
| 7 | Power per GPU vs utilization (`05_power_per_gpu_vs_util.png`) | corr(util, total power) 0.023; **corr(util, power per GPU) 0.576**; corr(n_gpu, total power) 0.884. Composition test: corr(Δutil, Δn_gpu_equiv) -0.217, corr(Δutil, Δgpus_alloc) -0.136; mean Δutil -0.082 pp when GPUs are added (825 minutes), +0.068 when removed (1,467), -0.005 otherwise (6,347): direction supports a composition effect, magnitude small. Power per GPU 58 to 98 W at utilization 30 to 55%, in parallel bands (job mixes, unverified) |

**Split (normal days, chronological 70/15/15):** train 6,048 minutes (ends 2021-09-14 04:47), validation 1,296 (ends 2021-09-15 02:23), test 1,296.

**Ramp statistics on the training part** (|P(t+h) − P(t)| with both ends in train):

| h | std (kW) | median \|Δ\| | p90 | **p95 = threshold** | p99 | max | excess kurtosis | share up | ramps in val | ramps in test |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 min | 0.357 | 0.166 | 0.470 | **0.618** | 1.133 | 7.105 | 50.4 | 0.502 | 55 | 48 |
| 5 min | 0.556 | 0.227 | 0.727 | **1.005** | 2.183 | 7.335 | 23.0 | 0.531 | 86 | 72 |
| 15 min | 0.827 | 0.354 | 1.158 | **1.684** | 3.281 | 7.520 | 8.5 | 0.563 | 136 | 103 |

**Provisional ramp threshold (h = 1): |ΔP| > 0.618 kW/min**, 1.9% of the training median power. Because it is close to normal minute noise, a **second, severe tier at p99** (1.13, 2.18, 3.28 kW) is proposed for the supervisor.

**Lagged correlations with the future change P(t+h) − P(t)** (train; level = value at t; change = change over the last minute; 95% band ±0.025):

| Covariate | level h=1 | level h=5 | level h=15 | change h=1 | change h=5 | change h=15 |
|---|---|---|---|---|---|---|
| active_gpu_jobs | 0.001 | -0.033 | -0.084 | 0.238 | 0.214 | 0.106 |
| gpus_alloc | 0.002 | -0.047 | -0.113 | **0.252** | **0.215** | 0.089 |
| queued_gpu_jobs | 0.014 | 0.037 | 0.060 | 0.054 | 0.037 | 0.058 |
| queued_jobs | 0.009 | 0.027 | 0.055 | 0.016 | 0.004 | 0.014 |
| n_gpu_equiv | -0.009 | -0.058 | -0.122 | 0.166 | 0.152 | 0.033 |
| util_gpu_mean | -0.102 | -0.064 | -0.020 | **-0.352** | -0.243 | -0.159 |
| own last ΔP | | | | -0.288 | -0.182 | -0.161 |

Interpretation: **changes, not levels, carry predictive information**; Δ allocated GPUs is partly mechanical (allocation visible at minute end, power fully only in the next minute); Δ utilization is negative (partly composition, partly mean reversion); queues are weak. Univariate only; joint value to be tested by ablation.

**Modelling implications:** persistence will be a very hard baseline (report skill relative to it); a 60-minute input window covers the short-term linear dynamics; seasonal-naive with a daily season is not meaningful on this evidence; heavy tails argue against Gaussian intervals; a two-part model (number of active GPUs × power per GPU) is a candidate physics-informed design.

A Markdown summary cell with these nine findings is at the end of `03_eda.ipynb`.

---

## 10. Data-volume analysis and the dataset decision

**Constraint:** the student's internet budget is about 100 GB in total, large downloads only at night.

**MIT windows on the Mac (index + log only, no download):**

| Window | New download |
|---|---|
| 2 weeks from 2021-02-01 | 8.0 GiB (unrepresentative low load) |
| 4 weeks from 2021-02-01 | 18.7 GiB (unrepresentative) |
| 6 weeks from 2021-02-01 | 106.8 GiB |
| 2 weeks from 2021-09-06 (includes the dev week) | 95.6 GiB nominal, but see below |

**Reuse rule (important):** development-week streams are clipped to 10 to 17 September. Only files whose job lies entirely inside that week (6,930 of 7,435) are reusable; the other 505 (long jobs) must be downloaded again for any longer window. With the correct rule, even **2 extra days (8 to 17 Sep) cost 111.5 GB**; 8 to 21 Sep 172.6 GB; 3 Sep to 1 Oct 379.0 GB; 10 Sep to 1 Oct 281.5 GB. **No MIT extension fits the budget on the Mac.**

**PM100 (Marconi100, Zenodo record 8129258):** about **101 MB**, 231,238 jobs, May to October 2020, node-level power (all node components) every 20 s, only jobs running exclusively on their nodes; a newer version of the record exists. Not downloaded yet (the Zenodo-API metadata cell 54 was written but not run).

**Outcome:** run the per-file stage on **Google Colab**: raw data flows over Google's network; only stream Parquets (about 2 GB for everything, estimated from the 1,239x ratio) come back. Proposed for Meeting 1: **MIT Supercloud, whole period, as main data; PM100 as second dataset** for rank stability. (The plan's fallback "replace with PM100" is no longer needed if the Colab run completes.)

**What is lost by aggregation:** everything below one minute. The protocol's supplementary **10-second** experiment would need the raw data again; if the supervisor wants it, the Colab run should produce 10-second streams in the same pass (ask early; not implemented).

---

## 11. Colab whole-dataset run (in progress)

### 11.1 Throughput tests (all on `dev_window_files.parquet`, outputs in `MyDrive/dcpower/streams_test/`)

| Test | Files | Workers | Speed | CPU |
|---|---|---|---|---|
| Colab cell 2 | 30 (0.77 GiB) | 8 | 13.88 MiB/s, 1.0 min | not measured |
| Colab cell 3 | 100 new (3.22 GiB) | 16 | 20.41 MiB/s, 2.7 min | mean 48%, median 41% |
| Colab cell 4 | 150 new (2.83 GiB) | 32 | 16.69 MiB/s, 3.0 min | mean 43%, median 33% |
| Chunk 1 (first 53 min) | 8,450 of 26,479 | 16 | **43.54 MiB/s** | **100%** |

Small tests were network/overhead-bound (small files); real chunks are CPU-bound on 2 vCPUs. **16 workers** is the setting. Estimated whole dataset: about 15 hours (two to three sessions).

### 11.2 Design

- `MyDrive/dcpower/all_files_desc.parquet`: all 109,946 files (`sub, name, size, id_job, start_fc, end_fc, cum_GiB`), sorted by job end in the file clock, **newest first** (built on the Mac, cell 55; 3.2 MiB).
- **Window = whole period** (`--win-start 2021-01-01 --win-end 2021-10-02`), so streams are not clipped and nothing is ever re-downloaded.
- **Monthly chunks by job end.** After chunk k the series is complete from the start of that month (every job active after that date ends after it).

| Chunk | Jobs ending in | Raw size | Cumulative | Series complete from |
|---|---|---|---|---|
| 1 | Sep 2021 and 1 Oct | 437.5 GiB | 437.5 | 2021-09-01 |
| 2 | Aug | 385.0 | 822.5 | 2021-08-01 |
| 3 | Jul | 369.7 | 1,192.2 | 2021-07-01 |
| 4 | Jun | 367.6 | 1,559.8 | 2021-06-01 |
| 5 | May | 295.1 | 1,854.9 | 2021-05-01 |
| 6 | Apr | 218.9 | 2,073.8 | 2021-04-01 |
| 7 | Mar and Feb | 195.7 | 2,269.5 | whole period |

- Raw CSVs go to the Colab disk (`/content/tmp_raw`, worst case about 21 GiB with 16 workers × 1.3 GiB); streams go to Drive (`streams_full/`).
- The student's internet use for everything: Colab UI plus the final copy of streams, about 2 to 3 GB.

### 11.3 Colab cells (notebook `dcpower_stage1.ipynb` in Colab)

**Cell 1 (run after every reconnect):**

```python
from google.colab import drive
drive.mount("/content/drive")
import os, platform
BASE = "/content/drive/MyDrive/dcpower"
print("files in", BASE, ":", os.listdir(BASE))
!pip -q install polars==1.44.2
import polars as pl
print("python", platform.python_version(), "| polars", pl.__version__)
```

**Cell 5 (one chunk; set `CHUNK`; resumable; live text progress bar):**

```python
import os, re, sys, shutil, threading, subprocess, statistics, psutil, time, calendar
from datetime import datetime
from tqdm import tqdm
import polars as pl

BASE = "/content/drive/MyDrive/dcpower"
CHUNK = 1          # 1 = Sep/Oct, 2 = Aug, 3 = Jul, 4 = Jun, 5 = May, 6 = Apr, 7 = Mar+Feb

bounds = ["2021-10-02", "2021-09-01", "2021-08-01", "2021-07-01", "2021-06-01",
          "2021-05-01", "2021-04-01", "2021-01-01"]
ep = lambda d: calendar.timegm(datetime.fromisoformat(d).timetuple())
hi, lo = ep(bounds[CHUNK - 1]), ep(bounds[CHUNK])

full = pl.read_parquet(f"{BASE}/all_files_desc.parquet")
part = full.filter((pl.col("end_fc") >= lo) & (pl.col("end_fc") < hi))
part_path = f"{BASE}/chunk_{CHUNK}.parquet"
part.select("sub", "name", "size").write_parquet(part_path)
print(f"chunk {CHUNK}: jobs ending {bounds[CHUNK]} .. {bounds[CHUNK - 1]} | "
      f"{part.height:,} files | {part['size'].sum() / 2**30:,.1f} GiB")

os.makedirs(f"{BASE}/streams_full", exist_ok=True)
os.makedirs("/content/tmp_raw", exist_ok=True)
cmd = ["python", "-u", f"{BASE}/build_series.py",
       "--files", part_path,
       "--stream-dir", f"{BASE}/streams_full",
       "--tmp-dir", "/content/tmp_raw",
       "--out", f"/content/series_after_chunk_{CHUNK}.parquet",
       "--win-start", "2021-01-01", "--win-end", "2021-10-02",
       "--workers", "16"]

prog = re.compile(r"files ([\d,]+)/([\d,]+)\s+done=([\d,]+) skipped=([\d,]+) failed=([\d,]+)\s+"
                  r"([\d.]+)/([\d.]+) GiB\s+([\d.]+) MiB/s")
state = {"files": "0/?", "failed": "0", "rate": "0"}
cpu, running = [], True
bar = tqdm(total=1.0, unit="GiB", file=sys.stdout, mininterval=1, dynamic_ncols=True,
           bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} GiB [{elapsed}<{remaining}] {postfix}")

def heartbeat():
    while running:
        c = psutil.cpu_percent(interval=2); cpu.append(c)
        free_gb = shutil.disk_usage("/content").free / 1e9
        bar.set_postfix_str(f"files {state['files']} | failed {state['failed']} | "
                            f"{state['rate']} MiB/s | CPU {c:.0f}% | disk free {free_gb:.0f} GB", refresh=True)
threading.Thread(target=heartbeat, daemon=True).start()

t = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
other = []
for line in proc.stdout:
    m = prog.search(line)
    if m:
        f_i, f_n, done, skip, fail, g_i, g_n, rate = m.groups()
        bar.total = float(g_n); bar.n = float(g_i)
        state.update(files=f"{f_i}/{f_n}", failed=fail, rate=rate)
        bar.refresh()
    else:
        other.append(line.rstrip())
proc.wait(); running = False; time.sleep(2.5); bar.close()

print("\n".join(other))
print(f"\nchunk {CHUNK} | exit code: {proc.returncode} | wall time: {(time.time() - t) / 3600:.2f} h")
```

### 11.4 Operating rules

- Keep the Colab tab open; run `caffeinate -d` in a Mac Terminal window so the Mac does not sleep.
- **On disconnect:** run cell 1, then cell 5 with the same `CHUNK`; finished files are skipped.
- **If `failed` > 0** at the end of a chunk: run cell 5 again with the same `CHUNK` (only failures and missing files are retried; partial downloads in `/content/tmp_raw` resume).
- If `disk free` falls below about 20 GB, stop and investigate (should not happen).
- **If `Input/output error` or a sudden rise in failures appears (Drive quota):** stop; switch to writing streams to `/content/streams_local` and copying them to Drive in batches (tar every N minutes). Not implemented yet.
- With about 43 MiB/s, two chunks fit in one session (e.g. chunks 2 and 3, about 5 h).
- The series printed at the end of each chunk on Colab is irrelevant (the series is built on the Mac).

### 11.5 After a chunk: bring the streams to the Mac and build the series (NOT done yet)

1. On Colab, pack the streams of the finished chunk(s) into one archive on the Colab disk and copy it to Drive (one file is cheaper to download than tens of thousands): e.g. `!cd /content/drive/MyDrive/dcpower && zip -q -r -0 /content/streams_full.zip streams_full && cp /content/streams_full.zip /content/drive/MyDrive/dcpower/` (Parquet is already compressed; `-0` stores). Check the size first; Drive has about 9 GB free.
2. Download the zip from Drive in the browser (at night if large; after chunk 1 it should be a few hundred MB, estimate), unzip to `data/interim/streams_full/` on the Mac.
3. Build the series on the Mac with **only the files of the finished chunks as `--files`** (download `chunk_1.parquet` from Drive, or rebuild it from `all_files_desc.parquet`). **Never pass `all_files_desc.parquet` on the Mac**: every missing file would be downloaded over the student's internet.
4. Series window for chunk 1: `--win-start 2021-09-01 --win-end 2021-10-01` (September only, 43,200 minutes). Do not end at 2021-10-02: the minutes after the last job end on 1 October would be classed as "cluster up, no GPU" and set to 0.
5. Then `build_log_features.py` with `--series` pointing to the new series and a new `--out`; validate as in Section 7 (row count, empty minutes, null counts, `n_gpu_equiv` ≤ 448, power range, `gpus_alloc` vs `n_gpu_equiv`, outage = null power, maint_day count = 1 day for September).
6. Re-run the EDA on September (four weeks, one maintenance day) and recompute the ramp thresholds; compare with the development-week numbers.
7. Delete `MyDrive/dcpower/streams_test/` (obsolete, clipped to the development week).

---

## 12. Files

### 12.1 Mac (`data/` is local only and must never be committed)

| Path | Content |
|---|---|
| `data/raw/slurm-log.csv`, `tres-mapping.txt`, `README.md`, `labelled_jobids.csv`, `labelled_job_stats.csv` | Dataset files |
| `data/raw/tsfm_lineage.csv` | Week 1 lineage check |
| `data/raw/gpu_samples/` | About 26 test files (about 150 MiB); can be deleted |
| `data/interim/gpu_file_index.parquet` | 109,946 files (`sub, name, size, id_job`) |
| `data/interim/dev_window_files.parquet` | Development-week selection (7,435 files) |
| `data/interim/all_files_desc.parquet` | All files with file-clock start/end, newest end first (for Colab) |
| `data/interim/streams_dev/` | 7,435 stream Parquets (clipped to the dev week) + `_failures_run1.csv` |
| `data/interim/_tmp_raw/` | Empty |
| `data/interim/build_dev.log`, `build_dev_retry.log`, `build_dev_rebuild.log`, `build_features_dev.log` | Run logs |
| `data/interim/minute_streams_test/`, `data/processed/series_test.parquet`, `data/processed/series_dev_test.parquet` | Obsolete test outputs; can be deleted |
| `data/processed/series_dev.parquet` | **Development series, v3 time-weighted** (12 columns) |
| `data/processed/series_dev_v1_unweighted.parquet` | Unweighted v1, kept for comparison |
| `data/processed/series_dev_features.parquet` | **Series + log features + flags** (20 columns): EDA input |
| `src/data/build_series.py` | Pipeline v3 (243 lines) |
| `src/data/build_series.py.bak_unweighted` | Only copy of v2 |
| `src/data/build_log_features.py` | Flags and covariates (148 lines) |
| `docs/data_quality.md` | Data-quality report, Sections 1 to 14 (164 + 12 lines) |
| `README.md` | Sections "Building the GPU power series" (measured numbers) and "Full dataset on Google Colab" |
| `notebooks/02_data_download_explore.ipynb` | Data work, cells 1 to 56 (rename from `.ipynb.ipynb` if not done) |
| `notebooks/03_eda.ipynb` | EDA cells 1 to 7 + summary |
| `reports/figures/eda/01_series_week_day_hour.png` ... `05_power_per_gpu_vs_util.png` | Five EDA figures |

**Git status at the last check (student handles it):** `.gitignore` did not ignore `data/`, `.ipynb_checkpoints/` or the `.bak`; `src/`, `docs/data_quality.md`, `reports/` and both notebooks were untracked. Suggested ignore rules (only if asked): `data/`, `*.parquet`, `*.csv`, `.ipynb_checkpoints/`, `wandb/`, `*.bak*`, `*.log`.

### 12.2 Google Drive (`MyDrive/dcpower/`)

`build_series.py` (v3), `dev_window_files.parquet`, `all_files_desc.parquet`, `chunk_1.parquet`, `streams_test/` (delete), `streams_full/` (filling).

### 12.3 Claude Docs from Week 1

- Research proposal (English and Persian tabs): https://claude.ai/code/artifact/1c7f11e9-9c9b-40f6-ad87-6c447798161e
- Weekly Report Week 1 (English and Persian tabs): https://claude.ai/code/artifact/43bd3fb4-44f7-41f6-bbb2-ad521fef6980

---

## 13. All Week 2 decisions (numbered; superseded ones marked)

1. GPU jobs via `tres_alloc` codes 1001/1002 (not `gres_used`).
2. Time frame: log UTC, files New York wall clock; series in the file clock; `log_to_file_clock` for every log time.
3. Stream key `(file, gpu_index)`; never sum raw samples.
4. *(Superseded by 10)* Edge minutes keep the mean of available samples.
5. *(Superseded by 12)* Empty minutes null pending a decision.
6. Development subset: 2021-09-10 to 2021-09-17 (file clock), selected by job-time overlap.
7. Download-and-aggregate streaming design (raw files deleted at once).
8. Long local runs in Terminal with `caffeinate -i nohup`.
9. Notebook numbering: `02_data_download_explore.ipynb`, `03_eda.ipynb`.
10. **Time-weighted aggregation** (interior weight 1; edge weight `min(n_samples / stream median, 1)`), adding `n_gpu_equiv` and `n_edge_streams`. Replaces the planned edge-minute sensitivity check.
11. **Node id = `r...-n...`**; physical GPU = `(node, gpu_index)`.
12. **Empty minutes:** outage → null and masked; cluster up without GPU jobs → 0.
13. **Flags:** `cluster_gap` (log), `outage` (gap and no telemetry; telemetry wins), `maint_day` (local day of a gap ≥ 6 h, dated by gap start + 2 h). Maintenance days excluded from the ramp threshold and EDA statistics; reported separately in evaluation.
14. **Log covariates** sampled at minute end, built by a separate script `build_log_features.py`; `build_series.py` stays log-free.
15. Labelled job types not used (0.1% coverage).
16. **Provisional ramp thresholds** = p95 of |P(t+h) − P(t)| on the first 70% of normal days, per horizon (0.618 / 1.005 / 1.684 kW); a p99 severe tier proposed. Final values after the longer series.
17. EDA on normal days only; ΔP only between consecutive valid minutes.
18. **Whole dataset via Google Colab**, newest-first monthly chunks, whole-period window (no clipping), 16 workers; only streams come back to the Mac.
19. Proposed: MIT whole period as main data, PM100 as second dataset (to be decided in Meeting 1).
20. Development-week streams are not reused for longer series (clipped).

---

## 14. Unverified items and watch-outs

- Mechanism behind the time offset; telemetry tail after forced job ends (up to about 3 min).
- Meaning of the `n...` part of node names.
- Cause of the 1,168 GPU jobs without files.
- Idle baseline variation (27 vs 47 W).
- Whether the long outages are scheduled maintenance announced in advance; the 2021-03-29 midday gap.
- Post-maintenance "test jobs" interpretation.
- Our peak 64.5 kW (post-maintenance) vs the paper's about 45 kW; normal-day max 42.2 kW in the dev week.
- 79% single-GPU jobs (log) vs the paper's 85%.
- Positive ACF(ΔP) at lag 5 (+0.078): cause unknown.
- Composition explanation of the negative Δutilization correlation (supported in sign only).
- All EDA numbers come from one week with a strong downward trend; daily and weekly seasonality cannot be assessed.
- Colab: unpublished limits; Drive operation quotas with about 110,000 small files; whole-run speed extrapolated from chunk 1.
- Size of the whole-dataset streams (about 2 GB) extrapolated from the dev-week ratio.
- Python 3.13 on Colab vs 3.11 on the Mac (script ran correctly on both so far).
- Whether the notebook was renamed from `.ipynb.ipynb`.

---

## 15. Correction history (be transparent about it)

- Cells 9 and 10 (earlier in Week 2): time relationship first read the wrong way round; cells 14, 16, 17, 20 used a conversion 8 hours off. Caught in cell 22, fixed in cell 24. Daily tables of cells 14, 16, 17 must not be quoted; window selection redone (6,957 files, 113 GiB superseded by 7,435 files, 134.5 GiB).
- Cells 28 to 37: v2 aggregation double-counted GPUs in job-turnover minutes (up to 540 streams, 49.4 vs 8.2 kW at 15:32); replaced by time weighting. A first hypothesis (edge minutes explain the 15:31 peak) was rejected: 15:31 had no partial streams and the log confirmed 410 GPUs.
- Cell 32: node key built from `r...` only (wrong, gave "92 physical GPUs"); correct key `r...-n...` from `nodelist`.
- Cell 37 failed with `ColumnNotFoundError`: the script edit had not been saved; fixed by patching the file from the notebook (cell 39).
- Cell 47: two minutes in a log gap had telemetry; the outage rule now requires no telemetry (cell 48).
- Cell 51: `TypeError` because cell 48's loop variables `old, new` overwrote DataFrames; fixed by reading files directly.
- Cell 52 assumed all dev-week files reusable; cell 53 corrected this (clipping), raising extension costs (2 extra days = 111.5 GB).
- An earlier remark that the old `build_series.py` "is in Git" was wrong: `src/` was never committed.
- EDA cell 2: a malformed line (mixing Polars and Pandas) and a merged line after a partial edit; fixed by replacing the whole cell.

---

## 16. Remaining work, step by step (one action per message)

**A. Colab chunk 1 (running).** When it ends, the student pastes the last lines (summary and `chunk 1 | exit code ... | wall time`). Check `failed`; if > 0 run cell 5 again with `CHUNK = 1`. Update the README throughput line if the final speed differs much from "about 43 MiB/s".

**B. September series on the Mac** (Section 11.5): zip on Colab, copy to Drive, download (night if large), unzip to `data/interim/streams_full/`, build with `--files chunk_1.parquet --win-start 2021-09-01 --win-end 2021-10-01 --out data/processed/series_sep.parquet` (new stream-dir, new tmp-dir; all files must be skipped, so check the first progress line says `still to download: 0.00 GiB` before anything downloads), then `build_log_features.py --out data/processed/series_sep_features.parquet`, validate, re-run EDA on September, recompute ramp thresholds. Before running on the Mac, confirm `still to download` is zero; abort otherwise.

**C. Further Colab chunks** (2 and 3 together, then 4 and 5, then 6 and 7) in later sessions; after each, repeat B with the union of chunk lists and an earlier `--win-start`.

**D. Proposal (task 8).** Read the Week 1 Claude Doc (link in 12.3) with the Claude Docs tools, update in both tabs: Section 3 Data (whole MIT dataset via Colab, target = job-attributed GPU power, time-frame rule, outages and flags, time-weighted aggregation, PM100 as second dataset), Section 5 protocol (ramp per horizon, p99 tier, persistence skill, outage masking), Section 9 open questions (from Section 17 here); add a **fixed-decisions table for approval**; add a five-minute EDA summary with four or five figures from `reports/figures/eda/` (or the September versions). Do not add dates that were not discussed (term start and Meeting 1 date unknown).

**E. Meeting 1 packet (task 10):** final proposal; four or five EDA figures; fixed-decisions table; `docs/data_quality.md`; agenda (Section 17). After the meeting: minutes in `docs/meetings/` the same day (template: date and attendees, presented, feedback, decisions, next steps with deadlines).

**F. Weekly Report 2 (task 9), Friday 9 October 2026:** bilingual Claude Doc like Week 1; four parts: (1) work done with commit/W&B links (placeholders for the student), (2) key number or figure (suggest: the time-weighting fix or the ramp threshold with the ΔP distribution), (3) blockers and decisions (data budget → Colab; aggregation bug; outages), (4) next week's plan (Colab chunks, September series, Week 3 framework).

**G. Housekeeping leftovers:** delete `MyDrive/dcpower/streams_test/`; optionally delete obsolete local test outputs (Section 12.1); confirm the notebook rename; README throughput update after chunk 1.

**H. Week 1 residuals:** fill commit and W&B links in the Week 1 report; obtain two pending papers (Colangelo 2026, QMI 2026); coordinate the Meeting 1 date.

**Optional, cheap:** download PM100 (about 101 MB) and check whether a 1-minute aggregate series can be built (node-level power, exclusive jobs only, 20 s sampling); run the Zenodo-API metadata cell first (latest version, file names, sizes).

**Week 2 is complete when:** the proposal is final, Weekly Report 2 is written, the Meeting 1 packet is ready, and at least the September series is built and validated (or the supervisor decides otherwise). Announce it explicitly.

---

## 17. Meeting 1 agenda (draft, discussed with the student in Persian)

**A. Decisions to take**
1. **Main dataset:** MIT Supercloud, whole period (2021-02-05 to 2021-10-01, about 238 days) via Colab; PM100 (about 0.1 GB) as second dataset for rank stability.
2. **Target definition:** job-attributed GPU power, not facility power (idle GPUs outside jobs are absent).
3. **Outages and maintenance days:** outage minutes null and masked; cluster up without GPU jobs = 0; maintenance days excluded from the ramp threshold and reported separately; may `maint_day` be a calendar covariate (only if announced in advance)?
4. **Ramp definition:** threshold per horizon on |P(t+h) − P(t)|; add a p99 severe tier because p95 at h = 1 (0.62 kW) is near normal noise.
5. **Mandatory persistence baseline:** ACF 0.995 at lag 1; report skill relative to persistence; seasonal-naive with a daily season is not meaningful on current evidence.
6. **10-second supplementary experiment:** keep or drop? If kept, it must be produced in the same Colab pass (raw data needed).

**B. Protocol confirmations with Week 2 evidence**
7. Input window 60 minutes (PACF cuts off after lag 2 to 3); longer look-back only as a side experiment (Week 1 item c).
8. Chronological 70/15/15 split: the dev week showed a 10 kW downward trend; re-check on the long series; is Min-Max on train enough under level shifts?
9. Features (Week 1 item g): lagged **changes** of covariates (Δ allocated GPUs +0.25, Δ utilization −0.35); queues weak; lags only.

**C. Information**
10. Time-frame rule and three lines of evidence.
11. Aggregation bug found and fixed (time weighting).
12. Peak 64.5 kW is real but post-maintenance; normal-day max 42.2 kW; the paper's about 45 kW not yet explained.
13. Nine monthly outages (about 127 h); 1,168 GPU jobs without files (1.2%); labelled model types unusable (0.1%).
14. Idea: two-part physics-informed model (number of active GPUs × power per GPU); utilization correlates 0.58 with power per GPU, 0.02 with total power.

**D. From the plan and Week 1 (unchanged)**
15. Approve the four RQs and scope; 1-minute resolution; horizons 1, 5, 15 minutes.
16. Co-authorship and target venue; thesis format.
17. Week 1 items (a) to (i): Alberta comparison with a naive baseline; DLinear and NLinear (PI-DLinear if code exists); look-back; two kinds of quantum comparison; QRC controls; equal hyperparameter budget and logging of all quantum variants; extra features; zero-shot as secondary because of leakage; ACI with delayed updates (empirical coverage only).

---

## 18. Week 3 preview (from the plan; starts after Meeting 1)

**Goal:** one evaluation framework for all models, plus statistical, ML and deep baselines.
- `src/eval/`: one function taking predictions and truth, returning all metrics including ramp metrics; rolling-origin evaluation on the test set; one config file (YAML or Hydra) per experiment.
- Naive and Seasonal Naive (daily season; Week 2 evidence says the daily season is weak, so persistence is the key reference); AutoETS and AutoARIMA (statsforecast).
- Features: lags 1 to 60, rolling mean and std, hour and day as sin/cos, lagged covariates (add lagged covariate changes per Week 2).
- LightGBM with limited Optuna tuning on validation.
- LSTM and GRU with the Alberta settings, compared with their table; document differences.
- Log all runs in W&B or MLflow.
- **Completion criterion:** all Week 3 models on three horizons with three seeds in one table.
- Week 2 inputs needed: the final series with flags (masking of outages), the split, the ramp thresholds per horizon, the persistence baseline.

---

## 19. Step logs

### 19.1 `notebooks/02_data_download_explore.ipynb`, cells 28 to 56 (this session; cells 1 to 27 are in `HANDOFF_week2_progress.md` v3, summarized: log download and structure, GPU job identification, `gpu/` index, file format, time-frame finding and correction, window selection, pipeline v2 tests)

| Cell | What | Result |
|---|---|---|
| 28 | Inspect `_failures.csv` | first `FileNotFoundError` (cwd = `notebooks/`); with absolute `ROOT`: 310 rows, all `SSLEOFError` |
| 29 | Pre-retry checks | 7,125 outputs; 310 partial files (4,572 MiB) in `_tmp_raw`; 0 failed files with outputs; resume and skip logic located in the script |
| 30 | Verify the retry | done 310, skipped 7,125, failed 0; 7,435 outputs; `_tmp_raw` empty; stale `_failures.csv` detected; series 10,080 × 10, 722 nulls per metric |
| 31 | Remove stale failures file; peak sanity | streams 2 to 540 (> 448); W per stream 26 to 191; p99 41.4 kW, max 64.5 kW |
| 32 | Stream rows at 15:31 and 15:32 | 948 rows (410 + 538); stream schema with `n_samples` |
| 33 | Double-counting test | 15:31: 0 partial; 15:32: 512 partial, 49.4 vs 8.0 kW (provisional FULL = 576); wrong node key gave 92 |
| 34 | Cross-check 15:31 with the log | 4 jobs, 410 GPUs, 205 nodes, one user and one array job; node names `r...-n...` in `nodelist` |
| 35 | Full-minute sample counts; physical GPUs with correct key | per-stream medians 539 to 581; max 418 physical GPUs |
| 36 | Show `build_series()` | lines 108 to 132 (unweighted sum) |
| 37 | Compare v3 vs v1 | first `ColumnNotFoundError` (edit not saved); after the patch: 3,764 minutes changed, energy −0.25%, 15:32 = 8.22 kW, max `n_gpu_equiv` 426 |
| 38 | Diagnose | script mtime unchanged; new code absent |
| 39 | Patch `build_series()` from the notebook | backup `.bak_unweighted`; 243 lines; compiled |
| 40 | First look; empty runs; week plot | 8 empty runs, all on 16 Sep |
| 41 | Hourly log activity 15 to 17 Sep | zero jobs 01:00 to 08:59 on 16 Sep; normal from about 18:00 |
| 42 | Cluster-wide gaps over the whole log | 19 gaps ≥ 10 min, 126.9 h (Section 8.1) |
| 43 | Sizes | 134.54 GiB raw, 111.2 MiB streams, 559.8 KiB series, ratio 1,239x |
| 44 | Measure and write README section | run 1 368.4 min, retry 37.2 min, rebuild 1.8 s identical; README +44 lines |
| 45 | Classify empty minutes | 674 in gaps, 48 cluster up |
| 46 | Write `build_log_features.py` | 144 lines, compiled |
| 47 | Inspect two conflicting minutes | telemetry tail after forced kills (Section 5) |
| 48 | Patch outage rule (`cluster_gap` + telemetry precedence) | 148 lines |
| 49 | Covariate sanity | no nulls; corr(gpus_alloc, n_gpu_equiv) 0.9975; same-minute correlations |
| 50 | Labelled files | 3,430 jobs, 26 models; 6 of 5,379 in window |
| 51 | Write `docs/data_quality.md` | 164 lines; stream nulls all 0 (after fixing the variable clash) |
| 52 | Cost of multi-week MIT windows | cheapest: Feb (8.0, 18.7, 106.8 GiB for 2, 4, 6 weeks) |
| 53 | Exact extension cost (correct reuse) | 6,930 reusable; +2 days = 111.5 GB; no extension fits 100 GB |
| 54 | Zenodo API metadata for PM100 | written, **not run** |
| 55 | `all_files_desc.parquet` | 109,946 files, 2,269.5 GiB, 3.2 MiB; cumulative table (Section 11.2) |
| 56 | Append to `data_quality.md` (Section 14) and README (Colab section) | +12 and +38 lines |

Variables defined in the notebook kernel (lost on restart): `ROOT, STREAMS, SCRIPT, SERIES, LOG, FAIL, LOGCSV, all_jobs, iv, gaps_all, fj, fj2, full, st, all_st, feat, mtime()`, `log_to_file_clock`, `t0, t1` (window epochs).

### 19.2 `notebooks/03_eda.ipynb`

| Cell | What | Output |
|---|---|---|
| 1 | Setup, libraries, valid minutes | Section 9 row 1 |
| 2 | Week, day, hour figure | `01_series_week_day_hour.png` (first attempt had a malformed line) |
| 3 | Daily profile | `02_daily_profile.png` |
| 4 | ACF/PACF | `03_acf_pacf.png` |
| 5 | ΔP distribution, ramp thresholds, split | `04_dP_distribution_ramp_threshold.png` |
| 6 | Lagged covariate correlations | table (Section 9) |
| 7 | Power per GPU vs utilization; composition test | `05_power_per_gpu_vs_util.png` |
| 8 | Markdown summary of nine findings | text cell |

The notebook uses `PROJECT_DIR = Path.cwd().parent`, input `data/processed/series_dev_features.parquet`, and variables `df, x, P, dP, n, n_tr, n_va`. To re-run on September, change `FEAT` to the new features file and the maintenance-day handling stays the same.

### 19.3 Colab (`dcpower_stage1.ipynb`)

| Cell | What | Result |
|---|---|---|
| 1 | Mount Drive, install polars, machine info | 2 vCPUs, 12 GiB RAM, 88 GB free; Python 3.13; polars 1.44.2 |
| 2 | 30-file test, 8 workers | 13.88 MiB/s; 30 outputs on Drive |
| 3 | 100-file test, 16 workers, CPU sampling | 20.41 MiB/s; CPU mean 48% |
| 4 | 150-file test, 32 workers, live progress bar | 16.69 MiB/s; CPU mean 43% |
| 5 | Chunk runner (Section 11.3) | chunk 1 running: 26,479 files, 437.5 GiB, 43.54 MiB/s, 0 failures at 31% |

### 19.4 Terminal commands used this session (reference)

```bash
cd ~/Desktop/Other/datacenter-power-forecasting
```

```bash
caffeinate -i /opt/miniconda3/envs/dcpower/bin/python -u src/data/build_series.py --files data/interim/dev_window_files.parquet --stream-dir data/interim/streams_dev --tmp-dir data/interim/_tmp_raw --out data/processed/series_dev.parquet --win-start 2021-09-10 --win-end 2021-09-17 --workers 8 2>&1 | tee data/interim/build_dev_rebuild.log
```

```bash
/opt/miniconda3/envs/dcpower/bin/python -u src/data/build_log_features.py --log data/raw/slurm-log.csv --series data/processed/series_dev.parquet --out data/processed/series_dev_features.parquet 2>&1 | tee data/interim/build_features_dev.log
```

```bash
caffeinate -d
```

---

## 20. Opening prompt for the new chat

**Upload:** `Project_Context_FA.txt`, `HANDOFF_week1_complete.md`, and this file (`HANDOFF_week2_complete_data.md`). Optional: `literature.md`. (`HANDOFF_week2_progress.md` is no longer needed.)

**Paste as the first message:**

```text
I am continuing my bachelor project at the end of Week 2 (data engineering and exploratory analysis); after Meeting 1 with my supervisor, Week 3 starts. Read the attached files first: Project_Context_FA.txt (master plan), HANDOFF_week1_complete.md (Week 1 rules, protocol and decisions) and HANDOFF_week2_complete_data.md (everything done in Week 2: pipeline, time-frame rule, time-weighted aggregation, outages, EDA numbers, the Colab whole-dataset run, decisions, and the remaining work in Section 16).

Rules:
- Always answer me in Persian. Anything written into notebooks, code comments, scripts or repo docs must be in English.
- Go ONE STEP AT A TIME: one cell or one action per message, then wait for my pasted output. When something must be fixed, give me the whole corrected cell, not a line edit.
- Edit scripts from a notebook cell (read, replace, py_compile, verify content and modification time), not by asking me to edit files by hand.
- Terminal commands: one per code block, no # comments (zsh), and remember a new Terminal starts in ~.
- Answer in chat; create a file or document only when I ask or when it is clearly a deliverable (the proposal and weekly reports are Claude Docs with English and Persian tabs).
- I run notebooks in local Jupyter on my Mac (conda env dcpower, kernel "Python (dcpower)"); the project root is ~/Desktop/Other/datacenter-power-forecasting/ and the notebook working directory is notebooks/. Heavy raw-data processing runs on Google Colab (Drive folder MyDrive/dcpower); my own internet budget is about 100 GB in total and large downloads only at night, so never make my Mac download raw GPU files.
- Do not give me any Git instructions; I handle Git myself.
- Keep claims cautious: mark unverified points, never claim quantum advantage, and tell me when a source could not be read.
- Tell me explicitly when Week 2 is complete.

Current state and important notes:
- Week 2 is NOT complete yet. Both plan criteria are met (one command builds the series; the ramp threshold is a concrete, provisional number), but the September series, the final proposal, Weekly Report 2 (due Friday 9 October 2026) and the Meeting 1 packet are still open (Section 16).
- Chunk 1 of the Colab whole-dataset run (jobs ending in September 2021, 26,479 files, 437.5 GiB) was last seen at 31% (134.9 of 437.5 GiB after 53 minutes, about 43 MiB/s, 0 failures, CPU 100%, 91 GB free on the Colab disk). I will paste its final lines; if failed > 0, I re-run the same cell with the same CHUNK.
- WARNING: on my Mac, never pass all_files_desc.parquet as --files to build_series.py; every missing file would be downloaded over my own internet. Use only the file lists of finished chunks (e.g. chunk_1.parquet), and check that the first progress line says "still to download: 0.00 GiB" before anything runs. For September use --win-start 2021-09-01 --win-end 2021-10-01 (not 2021-10-02).
- Unconfirmed: whether I renamed notebooks/02_data_download_explore.ipynb.ipynb to 02_data_download_explore.ipynb; the PM100 Zenodo metadata cell (cell 54) was written but never run, and PM100 is not downloaded.
- MyDrive/dcpower/streams_test/ is obsolete (clipped to the development week) and should be deleted after chunk 1.

Start with Section 16, step A: I will paste the final lines of the Colab chunk-1 cell; check them, then guide me through step B (bringing the streams to my Mac and building and validating the September series). Give me only the first action.
```

---

*End of file. Version 4.0, 2 October 2026.*

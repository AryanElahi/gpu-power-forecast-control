# Handoff: Week 2 in progress (data engineering)

**Project:** Forecasting AI Data Center Power Demand: A Benchmark of Classical, Foundation, and Hybrid Quantum Models with Uncertainty Quantification and Grid-Aware Control
**Version:** 3.0 (continues `HANDOFF_week1_complete.md` v2.0; read both)
**Date:** 2 October 2026 (10 Mehr 1405)
**Purpose:** a self-contained record of everything done in Week 2 so far, every verified fact, every decision, every correction and every open item, so a new chat can continue without reading the earlier conversation. Read together with `Project_Context_FA.txt` (master plan, v1.0) and `HANDOFF_week1_complete.md` (Week 1, rules, protocol, decisions).

---

## 0. How to use this file

| If you need | Go to |
|---|---|
| Status in one minute | 1 |
| Working rules (unchanged, short reminder) | 2 |
| Environment changes | 3 |
| Verified facts about the data | 4 |
| The time-zone finding (critical) | 5 |
| The pipeline (`build_series.py`) | 6 |
| Development window and the running job | 7 |
| Files on disk | 8 |
| Decisions made in Week 2 | 9 |
| Unverified items and watch-outs | 10 |
| Remaining Week 2 tasks, step by step | 11 |
| Meeting 1 agenda additions | 12 |
| Step log (cells 1 to 27) | 13 |
| Opening prompt for the new chat | 14 |

---

## 1. Snapshot

- **Week 2 is NOT complete.** Data access, data understanding and the series-building pipeline are done and tested. The full run on the development window was just launched (or is about to be) from Terminal.
- **Done:** `slurm-log.csv` downloaded and understood; GPU jobs identified; `gpu/` structure fully indexed (109,946 files, 2,269.5 GiB); file format understood (one file per node per job); time-zone relationship between log and GPU files established and validated; download throughput measured; `src/data/build_series.py` (v2) written and tested end to end on 30 files.
- **In progress:** full run of `build_series.py` on the development window 2021-09-10 to 2021-09-17 (7,435 files, 134.5 GiB, about 7 hours at 5.45 MiB/s).
- **Next:** check the run log, validate the series, then covariates, `docs/data_quality.md`, EDA notebook, ramp threshold, decision on a longer series, proposal, weekly report 2, Meeting 1 preparation (Section 11).

---

## 2. Working rules (unchanged; full list in `HANDOFF_week1_complete.md` Section 2)

- Chat in **Persian**; anything in notebooks, code comments, repo docs in **English**.
- **One step at a time:** one cell or one action per message, then wait for the pasted output.
- The student sometimes pastes the **code instead of its output**; ask for what is printed below the cell.
- Answer in chat; create files only when asked or when clearly a deliverable.
- Local Jupyter on the student's Mac, conda env `dcpower`, kernel `Python (dcpower)`.
- The student handles **Git**; only remind about `.gitignore` for data.
- Cautious claims; mark unverified points; never claim quantum advantage.
- **Announce explicitly** when Week 2 is complete.
- Cells: self-contained, `PROJECT_DIR = Path.cwd().parent`, print what was done, guard against re-running side effects.
- **New lesson from this session:** validate every time conversion against the GPU file timestamps before using it for selection (a sign error went unnoticed for several steps; see Section 5).
- Long runs (hours) go to **Terminal** with `caffeinate -i nohup ... &`, not inside a Jupyter cell.

---

## 3. Environment changes

- **Project root moved:** now `~/Desktop/Other/datacenter-power-forecasting/` (absolute: `/Users/aryaneuse/Desktop/Other/datacenter-power-forecasting/`). Notebooks still use `PROJECT_DIR = Path.cwd().parent`.
- **Python for Terminal runs:** `/opt/miniconda3/envs/dcpower/bin/python`.
- **Notebook for Week 2 data work:** `notebooks/02_data_download_explore.ipynb` (cells 1 to 27, Section 13). The EDA notebook will be `notebooks/03_eda.ipynb` (renumbered from the plan's `02_eda.ipynb` to avoid a clash).
- **Disk:** the Mac had 13 GiB free (94% full); the student freed about 20 GiB, so about 33 GiB should be free (unverified; check with `df -h ~`).
- **Network:** unchanged (VPN required; 8 MiB truncation per HTTP response; Range chunks of 4 MiB work). Measured throughput: **1.17 MiB/s single thread, 5.45 MiB/s with 8 threads** (small sample, approximate).
- **Unknown:** whether the student's internet or VPN has a data cap (the full window downloads about 134 GiB). Asked twice, not yet answered.
- No new packages were installed in Week 2 (polars 1.44.2 handles everything so far).

---

## 4. Verified facts about the data

### 4.1 `slurm-log.csv` (data/raw/, 103,165,427 bytes)

- **395,914 rows (jobs), 29 columns:** `id_job, id_array_job, id_array_task, id_user, kill_requid, nodes_alloc, nodelist, cpus_req, derived_ec, exit_code, gres_used, array_max_tasks, array_task_pending, constraints, flags, mem_req, partition, priority, state, timelimit, time_submit, time_eligible, time_start, time_end, time_suspended, track_steps, tres_alloc, tres_req, job_type`.
- `time_*` are integer Unix seconds and are **true UTC** (Section 5). Raw range: submit 2021-01-01 to 2021-10-01; start 2021-01-05 to 2021-10-01. No row has start or end <= 0 or end < start.
- `partition`: normal 249,548; xeon-p8 146,176; gaia 82; test 80; db 28.
- `job_type`: OTHER 266,622; LLSUB:BATCH 96,347; LLMAPREDUCE:MAP 25,844; LLSUB:INTERACTIVE 7,101.
- `constraints`: xeon-g6 192,481; `\N` 188,499 (MySQL-style null, not a category); xeon-g6&6248 14,845; xeon-e5 60; opteron 26; xeon-p8 2; opteron&6274 1.
- **`gres_used` is null in all rows**, so it cannot identify GPU jobs.

### 4.2 Reference files (data/raw/)

- `tres-mapping.txt` (111 B): 1 cpu, 2 mem, 3 energy, 4 node, 5 billing, 6 fs, 7 vmem, 8 pages, **1001 gpu:tesla, 1002 gpu:volta**.
- `README.md` (7,616 B), `labelled_jobids.csv` (82,332 B), `labelled_job_stats.csv` (339 B): downloaded, **not yet inspected**.

### 4.3 GPU jobs in the log

- Identified by the regex `(?:^|,)100[12]=(\d+)` on `tres_alloc` (the captured number is the GPU count). Example: `1=20,2=170000,4=1,5=20,1002=1`.
- **99,344 GPU jobs.** GPUs per job: 1: 78,111 (about 79%); 2: 18,860; 3: 23; 4: 1,593; 5: 6; ...; 14: 71; 16: 254; 32: 15; 64: 18; 128: 17. (The dataset paper says about 85% single-GPU; our count differs, likely a different population; unverified.)
- GPU jobs start in **February 2021** and end on 1 October 2021; job counts grow over time (about 1.5k in February, about 23k in September), so cluster load is not stationary.

### 4.4 `gpu/` folder

- Prefix `s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/`, 100 sub-folders `0000/` to `0099/`.
- **109,946 files, 2,269.5 GiB** in total (index saved in `data/interim/gpu_file_index.parquet`, columns `sub, name, size, id_job`). Sub-folder `0000/`: 1,047 files, 22.4 GiB.
- File name: `<id_job>-r<digits>-n<digits>.csv`. The first part is **`id_job`** (verified: every file's id_job exists in the log). Within one job the `r…` part differs between files (likely the node id) while the `n…` part is shared (meaning unverified).
- **98,176 distinct id_job** have files; **1,168 GPU jobs in the log (about 1.2%) have no file** (cause unverified); **0** files belong to jobs absent from the log.
- File sizes in the development window: median about 1.05 MiB, mean 16.6 MiB, max **1,324 MiB** (a 793-hour job). Many long jobs exist (527 jobs longer than 24 h overlap the development window).

### 4.5 GPU file format

- 10 columns: `timestamp` (float epoch with milliseconds), `gpu_index`, `utilization_gpu_pct`, `utilization_memory_pct`, `memory_free_MiB`, `memory_used_MiB`, `temperature_gpu`, `temperature_memory`, `power_draw_W`, `pcie_link_width_current`.
- Sampling about every 0.103 to 0.107 s, **about 560 samples per minute** (not 600). Not perfectly regular.
- **One file = one node of one job**, holding all GPUs of that node (verified on a 2-GPU single-node job and a 3-node, 6-GPU job: three files with two GPUs each). Rows of different GPUs are interleaved and have equal counts.
- Power in watts. Idle GPUs inside jobs draw about **27 to 47 W** (baseline differs between files; cause unverified); loaded GPUs up to about 287 W in the samples seen.
- Recording covers the job's lifetime: file duration equals log duration (Section 5).

---

## 5. Time-zone finding (critical)

**Rule (empirical):** the Slurm log times are **true UTC**; the GPU file `timestamp` values are **America/New_York wall-clock times stored as epoch seconds**. So `file_time = log_time − 5 h` in winter (EST, before 14 March 2021) and `− 4 h` in summer (EDT).

**Validation:** on 25 local files (February, both sides of the 14 March DST change, summer, September, October) the residual between converted log times and file times is at most 1.9 s at the start and 9.3 s at the end (means 0.7 s and 1.2 s). The mechanism is unverified; report it as an empirical consistency, not as a documented fact.

**Conversion used everywhere (Polars):**
```python
def log_to_file_clock(col):
    return (pl.from_epoch(col, time_unit="s").dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York").dt.replace_time_zone(None).dt.epoch("s"))
```

**Consequences:**
- The aggregate series lives in the **file clock** (New York local time). Column name `minute_local`, not `minute_utc`.
- In local time the hour 02:00 to 02:59 on **14 March 2021 does not exist** (a one-hour gap in a full-length series). The ambiguous November hour is outside the data range.
- Local time is natural for daily and weekly profiles (MIT is in Massachusetts).

**Correction history (be transparent about it):** in cells 9 and 10 the relationship was first interpreted the wrong way round (log assumed local, files UTC), and cells 14, 16, 17 and 20 used a conversion with the wrong direction (8 hours off in the file frame). This was caught in cell 22 (a "window" file lay entirely outside the window) and fixed in cell 24. Consequences: the monthly and daily tables of cells 14, 16 and 17 have day boundaries shifted by several hours (totals per month approximately right; do not quote their daily numbers), the earlier "one job starts in the non-existent DST hour" was an artifact of the wrong conversion, and the first window selection (6,957 files, 113 GiB) was superseded by the corrected one (7,435 files, 134.5 GiB).

---

## 6. Pipeline: `src/data/build_series.py` (v2, 192 lines)

**Per file, in parallel (ThreadPoolExecutor, default 8 workers), resumable:**
1. Download with HTTP Range chunks of 4 MiB, up to 5 retries per chunk with exponential backoff, verify the final size.
2. `pl.scan_csv` (lazy, streaming engine), cast `timestamp` and all metrics to Float64 with `strict=False` (robust to non-numeric values; none seen yet), **clip to the window** `[t0, t1)` in the file clock.
3. Mean of each metric per `(gpu_index, minute)`; `n_samples` per minute. Output columns: `gpu_index, minute_epoch, power_W, util_gpu, util_mem, temp_gpu, temp_mem, mem_used, mem_free, n_samples, file`.
4. Write `<stem>.parquet.tmp`, rename to `<stem>.parquet` (the final name marks completion), delete the raw CSV.
5. Existing outputs are skipped (resume); failures are recorded in `<stream-dir>/_failures.csv` and the run continues.

**Series:** over all stream Parquets, per minute: `power_W` = sum of per-stream minute means; `util_gpu_mean`, `util_mem_mean`, `temp_gpu_mean`, `temp_mem_mean` = means over streams; `mem_used_sum`, `mem_free_sum` = sums; `n_gpu_streams` = number of distinct `(file, gpu_index)`. Joined to a **complete minute grid** of the window; minutes with no stream get `n_gpu_streams = 0` and null power (decision pending, Section 11).

**Arguments:** `--files` (Parquet with `sub, name, size`), `--stream-dir`, `--tmp-dir`, `--out`, `--win-start`, `--win-end` (YYYY-MM-DD, file clock; window end exclusive), `--workers`, `--limit`, `--max-size-mib`. Window days are converted with `calendar.timegm` (independent of the Mac's time zone).

**Tests passed:** eager vs lazy identical on a 35.7 MiB file (0.3 s vs 0.1 s); clipping on a job crossing the window start removed exactly 540 rows (270 minutes × 2 GPUs) and the clipped output starts at 2021-09-10 00:00:00; end-to-end test on 30 files under 20 MiB: 30 outputs, no `.tmp` left, raw folder empty, all minutes inside the window, series of 10,080 rows (7,095 minutes empty because only 30 files), power 25.4 to 478.8 W.

**Not yet tested:** memory use on the largest files (1.3 GiB) under the streaming engine; behaviour of a full 7,435-file run.

**Design notes:** raw 100 ms samples are never summed; per-minute means are summed across streams. Edge minutes of each job are partial (fewer samples) and keep the mean of available samples; a sensitivity check is pending. The series is **job-attributed GPU power, not facility power** (idle GPUs outside jobs, CPUs, network, storage, cooling are excluded).

---

## 7. Development window and the running job

- **Why not 1 to 2 months (as planned):** monthly volumes are 200 to 385 GiB (February about 19 GiB but low-load); the full dataset is about 2.27 TiB (about 5 days at 5.45 MiB/s). A continuous 7-day window is the development subset.
- **Window:** 2021-09-10 00:00 to 2021-09-17 00:00, **file clock**, i.e. 10,080 minutes. September is the busiest month. Selection by **time overlap** of the job (`start < window end` and `end >= window start`), which includes long jobs that started earlier.
- **Selection:** `data/interim/dev_window_files.parquet`: **7,435 files, 134.5 GiB, 5,338 jobs, 527 jobs longer than 24 h**. Columns: `sub, name, size, id_job, start_epoch, end_epoch, duration_h` (epochs in the file clock).
- **Disk needs:** raw files are deleted after processing; worst case about 7.4 GiB of temporary raw files (8 largest files at once); stream outputs small (tens to hundreds of MiB, estimate).
- **Full-run command (Terminal, one line):**
```bash
cd ~/Desktop/Other/datacenter-power-forecasting && caffeinate -i nohup /opt/miniconda3/envs/dcpower/bin/python src/data/build_series.py --files data/interim/dev_window_files.parquet --stream-dir data/interim/streams_dev --tmp-dir data/interim/_tmp_raw --out data/processed/series_dev.parquet --win-start 2021-09-10 --win-end 2021-09-17 --workers 8 > data/interim/build_dev.log 2>&1 &
```
- **Monitor:** `tail -n 3 ~/Desktop/Other/datacenter-power-forecasting/data/interim/build_dev.log; df -h ~ | tail -1`
- **If interrupted:** run the same command again (finished files are skipped). Preconditions: VPN on, charger connected, lid open, iCloud Desktop sync off.
- **Fallback if disk gets tight:** two passes, first with `--max-size-mib 200 --workers 8`, then without the size limit and `--workers 2`.
- **Fallback if there is a data cap:** shrink the window (e.g. 3 days) by rebuilding `dev_window_files.parquet` with new dates.

---

## 8. Files on disk (all under `data/` are local only and gitignored)

| Path | Content |
|---|---|
| `data/raw/slurm-log.csv` | Slurm log (verified size) |
| `data/raw/tres-mapping.txt`, `README.md`, `labelled_jobids.csv`, `labelled_job_stats.csv` | Reference files (the last three not yet inspected) |
| `data/raw/tsfm_lineage.csv` | From Week 1 (foundation-model lineage check) |
| `data/raw/gpu_samples/` | About 26 sample GPU files (about 150 MiB) used for tests; can be deleted after Week 2 |
| `data/interim/gpu_file_index.parquet` | Full listing of `gpu/` (109,946 rows) |
| `data/interim/dev_window_files.parquet` | Corrected development-window selection |
| `data/interim/streams_dev/` | Per-file minute tables of the window (30 from the test; filling up during the full run) |
| `data/interim/_tmp_raw/` | Temporary raw downloads (should be empty when idle) |
| `data/interim/build_dev.log` | Log of the full run |
| `data/interim/minute_streams_test/`, `data/processed/series_test.parquet` | Obsolete outputs of the v1 test (cell 19); can be deleted |
| `data/processed/series_dev_test.parquet` | Series from the 30-file test (partial) |
| `data/processed/series_dev.parquet` | Will hold the development series after the full run |
| `src/data/build_series.py` | Pipeline v2 (code, may be committed) |

`.gitignore` must keep `data/`, `*.csv`, `*.parquet`, `*.log` (suggest adding), `docs/papers/`, `wandb/`.

---

## 9. Decisions made in Week 2

1. GPU jobs are identified via `tres_alloc` codes 1001/1002, not `gres_used` (all null).
2. Time frame: log = UTC; GPU files = New York wall clock as epoch; series in the file clock (`minute_local`); log times converted with `log_to_file_clock` (Section 5).
3. Stream key `(file, gpu_index)`; per-minute mean per stream, then sum over streams; never sum raw samples.
4. Edge minutes keep the mean of the available samples (sensitivity check pending).
5. Empty minutes: `n_gpu_streams = 0`, power null, until a reasoned decision (zero vs missing) is recorded.
6. Development subset: continuous 7-day window 2021-09-10 to 2021-09-17 (file clock), selected by job-time overlap, instead of 1 to 2 months (volume). For pipeline development only.
7. Download-and-aggregate streaming design (raw files deleted immediately), so the full dataset can be processed later with little disk.
8. Long runs from Terminal with `caffeinate -i nohup`.
9. Notebook numbering: `02_data_download_explore.ipynb` (data work), `03_eda.ipynb` (EDA).

---

## 10. Unverified items and watch-outs

- Mechanism behind the time-frame rule (validated on 25 files only); end residual up to 9 s.
- Meaning of the `n…` part of file names; the `r…` part as the node id.
- Idle GPUs outside jobs are absent (series = job-attributed power).
- Cause of the 1,168 GPU jobs without files.
- Variation of idle baseline power (27 vs 47 W).
- Whether `nvidia-smi` wrote non-numeric values anywhere (handled by `strict=False`, but count nulls after the run).
- Bias from partial edge minutes.
- Memory behaviour of the streaming engine on 1.3 GiB files.
- Whether 79% single-GPU jobs vs the paper's 85% reflects a different population.
- Internet/VPN data cap (unknown).
- Daily/monthly tables of cells 14, 16, 17 were computed with the wrong conversion; do not quote their daily numbers.
- **A one-week series is too short** for weekly seasonality, a stable ACF and a defensible ramp threshold; the threshold from this window is provisional only.

---

## 11. Remaining Week 2 tasks (one step at a time)

1. **Check the full run:** read `build_dev.log` (throughput, failures), confirm `_failures.csv` is empty or retry, `_tmp_raw/` empty, number of stream Parquets equals 7,435, `series_dev.parquet` has 10,080 rows; count empty minutes, null metrics, `n_gpu_streams` range, power range (sanity: at most 448 GPUs × about 300 W ≈ 134 kW; the paper reports an aggregate peak of about 45 kW for the whole period).
2. **Quick look at the series:** plot the week, zoom to one day and one hour; record the run time and input/output sizes in `README.md` (completion criterion: one command builds the series).
3. **Decide empty minutes** (zero vs missing) with a written reason; decide edge-minute handling (sensitivity check: drop edge minutes vs keep).
4. **Covariates from the log:** active and queued job counts per minute (convert log times with `log_to_file_clock`; queued = `time_submit <= t < time_start`, active = `time_start <= t < time_end`, GPU jobs only and possibly all jobs); inspect `labelled_jobids.csv` and `labelled_job_stats.csv` (job to model type).
5. **`docs/data_quality.md`:** time-frame rule and residuals; DST gap on 14 March; 1,168 jobs without files; irregular sampling (about 560 samples/min); partial edge minutes; idle baseline variation; one file per node; non-numeric values (if any); empty minutes decision; job-attributed vs facility power; the correction history in Section 5.
6. **EDA notebook `notebooks/03_eda.ipynb`** (English text): full series and zooms; daily profile (weekly profile is limited with one week); ACF and PACF; distribution of ΔP and ramp statistics in kW/min; **provisional** ramp threshold = 95th percentile of |ΔP| on the training portion; correlation of covariates with power (simultaneous and lagged). Supplementary: brief comparison with PM100.
7. **Decide the longer series:** options (a) extend the window to several weeks (e.g. August to September, about 750 GiB, about 1.6 days of download), (b) full dataset in the background (about 2.27 TiB, about 5 days; exceeds the plan's 3-day risk signal), (c) keep one week for development and ask the supervisor. The ramp threshold and the 70/15/15 split need the final series. Recommend raising this in Meeting 1.
8. **Finalize the proposal** (Claude Doc from Week 1) and the fixed-decisions table for approval; five-minute EDA summary with four to five key plots.
9. **Weekly report 2** (Friday 9 October 2026), four-part template, bilingual as in Week 1.
10. **Meeting 1 preparation** and, afterwards, minutes in `docs/meetings/`.
11. Housekeeping: delete obsolete test outputs (Section 8); remind `.gitignore`; update `README.md` with the pipeline command.

**Week 2 completion criterion (from the plan):** one command builds the whole series from the raw data, and the ramp threshold is a concrete number. Deliverables: series script, EDA notebook, `docs/data_quality.md`, final proposal.

**Residual Week 1 items (still open):** fill commit and W&B links in the Week 1 report; obtain the two pending papers (Colangelo 2026, QMI 2026); coordinate the Meeting 1 date with the supervisor.

---

## 12. Meeting 1 agenda additions (from Week 2)

- Data volume: 2.27 TiB in total; development on one week; choice of the final series length (Section 11, item 7) and its effect on the split and the ramp threshold.
- Time-frame finding (log UTC vs files New York local) and its empirical validation.
- The target is job-attributed GPU power, not facility power; idle GPUs outside jobs are absent.
- Empty-minute and edge-minute decisions.
- 1,168 GPU jobs without files (about 1.2%).
- All Week 1 agenda items (`HANDOFF_week1_complete.md` Section 12.3) still apply.

---

## 13. Step log (`notebooks/02_data_download_explore.ipynb`)

| Cell | What | Result |
|---|---|---|
| 1 | Download `slurm-log.csv` in 4 MiB Range chunks | 103,165,427 bytes, 94 s, SIZE OK |
| 2 | Structure of the log | 395,914 rows, 29 columns |
| 3 | Time range, categoricals, sanity | Unix seconds; `gres_used` all null |
| 4 | Small reference files, `tres-mapping.txt` | GPU codes 1001/1002 |
| 5 | GPU jobs via `tres_alloc` | 99,344 GPU jobs, Feb to Oct 2021 |
| 6 | List `gpu/` | 100 sub-folders; `0000/` 1,047 files, 22.4 GiB |
| 7 | One small GPU file | 10 columns, about 0.1 s sampling, watts |
| 8 | Link file to log | id_job matches; duration equal; offset −4 h |
| 9 (fixed) | Offsets across months (18 files) | −5 h in Feb, −4 h from mid-March |
| 10 (fixed) | DST boundary check | consistent with US DST (interpretation later reversed) |
| 11 | 2-GPU single-node file | both GPUs in one file, interleaved |
| 12 | Multi-node file | one node per file |
| 13 | All 3 files of a 3-node job | 6 GPUs in total, overlapping times |
| 14 (fixed) | Index all of `gpu/`, coverage, monthly volume | 109,946 files, 2,269.5 GiB; 1,168 jobs without files |
| 15 | Throughput test | 1.17 MiB/s (1 thread), 5.45 MiB/s (8 threads) |
| 16, 17 | Daily volumes, choose window | September window chosen (tables used the wrong conversion) |
| 18 | Per-file aggregation prototype | about 560 samples/min, partial edge minutes |
| 19 | `build_series.py` v1 on local files | structure works (obsolete outputs) |
| 20 | Window selection (wrong conversion) | superseded |
| 21 | Lazy vs eager | identical; lazy faster |
| 22 | Clipping test | exposed the 8 h error |
| 23 | Diagnosis on 25 files | constant −4/−5 h; durations equal |
| 24 (rewritten) | Corrected rule, validation, new selection | residual ≤ 9.3 s; 7,435 files, 134.5 GiB |
| 25 | Clipping test on a boundary-crossing job | exactly 540 rows removed; starts at window start |
| 26 | `build_series.py` v2 + 30-file test | 10,080-row grid; works |
| 27 | Sanity of output folders | 30 outputs, nothing left over |

---

## 14. Opening prompt for the new chat

**Upload:** `Project_Context_FA.txt`, `HANDOFF_week1_complete.md`, `HANDOFF_week2_progress.md` (this file). Optional: `literature.md`.

**Paste as the first message:**

```text
I am continuing my bachelor project in the middle of Week 2 (data engineering and exploratory analysis). Read the attached files first: Project_Context_FA.txt (master plan), HANDOFF_week1_complete.md (Week 1, rules, protocol, decisions) and HANDOFF_week2_progress.md (everything done in Week 2 so far, the time-zone finding, the pipeline, and the remaining tasks in Section 11).

Rules:
- Answer me in Persian. Anything written into notebooks, code comments or repo docs must be in English.
- Go ONE STEP AT A TIME: one cell or one action per message, then wait for my pasted output. No multi-cell batches unless I ask.
- Answer in chat; create a file or document only when I ask or when it is clearly a deliverable.
- I run everything in local Jupyter on my Mac (conda env dcpower, kernel "Python (dcpower)"); long runs go to Terminal with caffeinate and nohup. The project root is ~/Desktop/Other/datacenter-power-forecasting/. The VPN must be on for downloads and truncates any single download at 8 MiB.
- I handle Git myself; do not give Git instructions. Just remind me about .gitignore for data.
- Keep claims cautious: mark unverified points, never claim quantum advantage, and tell me when a source could not be read. Validate any time conversion against the GPU file timestamps before using it.
- Tell me explicitly when Week 2 is complete.

The full run of src/data/build_series.py on the development window (2021-09-10 to 2021-09-17, 7,435 files, 134.5 GiB) was launched from Terminal with the log at data/interim/build_dev.log. Start with step 1 of Section 11: help me check the run (I will paste the tail of the log), then validate the resulting series. Give me only the first action.
```

---

*End of file. Version 3.0, 2 October 2026.*

# Data quality report: MIT Supercloud GPU power series

Scope: development window 2021-09-10 to 2021-09-17 (file clock), built with
`src/data/build_series.py` and `src/data/build_log_features.py`. Numbers below refer to this
window unless stated otherwise. Statements marked *unverified* are empirical observations whose
mechanism has not been confirmed from documentation.

## 1. Sources

- `slurm-log.csv`: 395,914 jobs, 29 columns; times are Unix seconds in true UTC. GPU jobs are
  identified from `tres_alloc` codes 1001/1002 (`gres_used` is null in all rows): 99,344 GPU jobs.
- `gpu/`: 109,946 files, 2,269.5 GiB. One file holds one node of one job, with all GPUs of that
  node interleaved; `nvidia-smi` sampling about every 0.103 to 0.107 s.
- Development window: 7,435 files, 134.54 GiB raw CSV, aggregated to 7,435 per-file stream
  Parquets (111.2 MiB) and one 1-minute series of 10,080 rows.

## 2. Time frame

**Rule (empirical):** Slurm log times are UTC; GPU file timestamps are America/New_York wall-clock
times stored as epoch seconds. Log times are converted with `log_to_file_clock()`
(UTC to New York local, then dropped time zone). The series is in the file clock (`minute_local`).

**Evidence:**
- On 25 files across February to October and both sides of the DST change, the residual between
  converted log times and file times is at most 1.9 s at job start and 9.3 s at job end
  (means 0.7 s and 1.2 s).
- All long cluster-wide outages in the log start within 15 minutes after local midnight in the
  file clock (Section 7), which is consistent with scheduled maintenance at local midnight.
- GPUs allocated according to the log and GPU equivalents observed in the files agree closely on
  normal days (correlation 0.997, mean difference 0.8 GPUs, 0.2%).
- The mechanism behind the offset is *unverified*.

**Consequences:** the local hour 02:00 to 02:59 on 14 March 2021 does not exist (one-hour gap in a
full-length series); the ambiguous November hour is outside the data range.

**Exception (new):** when jobs are killed at the start of a maintenance outage, GPU files can keep
recording for up to about 3 minutes after the job end in the log (observed on 2021-09-16:
log ends 00:00:24 to 00:00:27, telemetry until about 00:02:55, at idle power). The 9.3 s bound
above holds for normal job ends only.

## 3. Identifiers and streams

- File name: `<id_job>-r<digits>-n<digits>.csv`. `id_job` matches the log for every file.
- **Node id is the combined `r...-n...` string**, matching entries of the log's `nodelist`
  (e.g. `r6760045-n386398`). The `r...` part alone is not unique across nodes. The meaning of the
  `n...` part is *unverified*.
- A stream is one `(file, gpu_index)` pair. A physical GPU is `(node, gpu_index)`; with this key
  the maximum number of physical GPUs active in one minute is 418 (cluster: 448 GPUs).

## 4. Sampling

- About 560 samples per fully covered minute, not 600; per-stream medians range from 539 to 581
  (1st to 99th percentile), so a single global constant is not accurate.
- Interior minutes (job running the whole minute) occasionally have fewer samples
  (minimum 287; 0.1% quantile 461): irregular sampling, not missing activity.
- Non-numeric values are read with `strict=False`. Null per-minute means in the stream Parquets
  (3,210,563 stream-minute rows): `power_W` 0, `util_gpu` 0, `util_mem` 0, `temp_gpu` 0, `temp_mem` 0, `mem_used` 0, `mem_free` 0.

## 5. Aggregation rule (time-weighted)

**Problem found:** the first version summed per-stream minute means. When one GPU switches from
one job to the next within a minute, both streams counted as a full minute, so power was counted
up to twice. Raw stream counts reached 540 in one minute, above the 448 GPUs of the cluster.
Example: 2021-09-16 15:32, when a 410-GPU group of jobs ended about 5 s into the minute:
unweighted 49.4 kW, time-weighted 8.2 kW.

**Rule now used:** each stream-minute has weight 1 for interior minutes and
`min(n_samples / full, 1)` for a stream's first and last minute, where `full` is the median
sample count over that stream's interior minutes (global median as fallback).
`power_W` = weighted sum (energy / 60 s); utilization and temperature are weighted means;
memory is a weighted sum. `n_gpu_equiv` (sum of weights) peaks at 426.

**Effect:** 3,764 minutes changed, but the mean change is small (energy over the week
4,672.5 kWh unweighted vs 4,660.9 kWh weighted, -0.25%).
The error was concentrated in job-turnover minutes, exactly where ramps matter. This rule
replaces the earlier plan of a sensitivity check that drops or keeps edge minutes.
The unweighted series is kept as `data/processed/series_dev_v1_unweighted.parquet` for comparison.

## 6. Download failures and coverage

- First pass: 310 of 7,435 files failed, all with `SSLEOFError` (transient network or VPN drops).
  Partial downloads were deleted and the files retried: 0 failures. Run-1 failures are kept in
  `data/interim/streams_dev/_failures_run1.csv`.
- 1,168 GPU jobs in the log (about 1.2%) have no GPU file (cause *unverified*). Consistent with
  this, GPUs allocated in the log exceed GPU equivalents in the files by 0.8 on average.
- Rebuilding the series from existing streams reproduces the saved file exactly (1.8 s).

## 7. Cluster outages and maintenance days

From the log, the cluster had 42 cluster-wide gaps (no job of any kind running); 9 last at least
6 hours and start around local midnight, mostly on the third Thursday of a month:
2021-01-21, 2021-03-02, 2021-03-18, 2021-04-15, 2021-05-20, 2021-06-17, 2021-07-15, 2021-08-10,
2021-09-16 (about 127 hours in gaps of 10 minutes or more). An unscheduled-looking midday gap of
2.7 h occurred on 2021-03-29. That the long outages are scheduled maintenance is *unverified*
(no official schedule seen).

After an outage, the cluster runs a few very large jobs from one user for several hours
(likely post-maintenance tests, *unverified*) before normal load returns. On 2021-09-16 normal
operation resumed around 18:00.

**Decision on empty minutes (minutes with no GPU stream):**

| Case | Minutes | `power_W` | Reason |
|---|---|---|---|
| Outage: inside a cluster-wide gap and no GPU telemetry | 674 | null | Cluster down; zero would create artificial ramps of about 30 kW. Masked in modelling. |
| Cluster up, no GPU job | 48 | 0 | True job-attributed power, not missing data. |

Flags: `cluster_gap` (log only), `outage` (cluster gap and no telemetry; 2 minutes are in a
cluster gap but still have telemetry and keep their values), `maint_day` (the whole local day of
a gap of at least 6 h, dated by gap start + 2 h). Maintenance days are excluded from the ramp
threshold and reported separately in evaluation. If the schedule is known in advance,
`maint_day` is a legitimate calendar covariate.

## 8. Peak power

The maximum is 64.5 kW at 2021-09-16 15:31, from 410 GPUs at about 157 W each. The log confirms
exactly 410 allocated GPUs in 4 jobs of one user and one array job (one job with 384 GPUs on
192 nodes). The peak is real but occurs in the post-maintenance period, not in normal operation.
The 99th percentile is about 41 kW. The dataset paper reports an aggregate peak of about 45 kW for
the whole period; the reason for the difference is *unverified* (possibly a different
aggregation or treatment of this period).

## 9. Target definition

The series is **job-attributed GPU power**: GPUs are recorded only while allocated to a job.
Idle GPUs outside jobs are absent, so this is not facility or node power. Idle GPUs inside jobs
draw about 27 to 47 W (baseline differs between files; cause *unverified*).

## 10. Log covariates

`active_jobs`, `active_gpu_jobs`, `gpus_alloc`, `queued_jobs`, `queued_gpu_jobs` are sampled at
the end of each minute (the moment its power value is complete), so they contain no information
from after the minute. Same-minute correlation with power on normal days: active GPU jobs 0.89,
allocated GPUs 0.88, queued GPU jobs 0.28, all active jobs 0.07. Only lagged values may be used
for forecasting.

## 11. Labelled jobs

`labelled_jobids.csv` labels 3,430 jobs with 26 model types (CNNs, BERT variants, U-Nets, graph
networks). Only 6 of 5,379 GPU jobs in the development window are labelled, so model type is not
used as a covariate for the aggregate series.

## 12. Correction history

- Week 2, cells 9 and 10: the time relationship was first interpreted the wrong way round, and
  cells 14, 16, 17 and 20 used a conversion 8 hours off. Caught in cell 22, fixed in cell 24.
  The daily tables of cells 14, 16 and 17 must not be quoted; the window selection was redone
  (6,957 files, 113 GiB superseded by 7,435 files, 134.5 GiB).
- Week 2, cells 32 to 37: unweighted aggregation double-counted GPUs in job-turnover minutes;
  replaced by the time-weighted rule (Section 5). A node key using only the `r...` part was
  wrong; the correct key is `r...-n...` (Section 3).
- Week 2, cell 47: two minutes inside a cluster gap had GPU telemetry; the outage flag now
  requires both a log gap and no telemetry (Section 7).

## 13. Open items

- Mechanism of the time offset; telemetry tail after forced job ends.
- Cause of the 1,168 GPU jobs without files.
- Idle baseline variation (27 vs 47 W).
- Difference between our peak (64.5 kW) and the paper's (about 45 kW).
- Whether maintenance days are scheduled and announced in advance.
- Whether 79% single-GPU jobs (log) vs the paper's 85% reflects a different population.
- One week is too short for weekly seasonality and a stable ramp threshold (series length to be
  decided, see the proposal).

## 14. Additions after the EDA

- **Normal-day maximum.** On the six normal days of the development window (2021-09-10 to
  2021-09-15) power ranges from 21.6 to 42.2 kW. The maximum is below the aggregate peak of about
  45 kW reported in the dataset paper, consistent with the 64.5 kW peak (Section 8) belonging to the
  post-maintenance period rather than normal operation (*unverified*).
- **Clipping of development-window streams.** Stream Parquets in `data/interim/streams_dev/` are
  clipped to 2021-09-10 to 2021-09-17. For 505 of the 7,435 files the job crosses a window boundary,
  so these streams are incomplete outside the window and cannot be reused for a longer series.
  The full-dataset run (README, "Full dataset on Google Colab") uses the whole period as its window,
  so its streams are not clipped.

## 15. Full-series coverage and the February ramp-up period

The full series (`series_full_2021.parquet`, 2021-01-01 .. 2021-10-02, 394,560 minutes) has no GPU telemetry in January. Stream coverage starts on 5 February (0.6% of minutes), 8 February 29%, complete from 9 February. The last complete day is 30 September; the file clock ends on 1 October 11:01.

February is a ramp-up period: mean power 1.9 kW with about 25 GPU-equivalents, against 16.8 kW in March. Telemetry also covers fewer allocated GPUs early on: median `n_gpu_equiv / gpus_alloc` on valid minutes is 0.88 (February), 0.93 (March), 0.96 (April), 0.99 (May) and 1.00 (June to September). Early power is therefore slightly under-counted.

## 16. Analysis window

Decision (provisional, Meeting 1): **2021-03-04 .. 2021-09-30**, 211 days, 303,840 minutes. 4 March is the first day after the maintenance event window of 2 March. Valid minutes: 282,988 (93.1%). Split 70/15/15 by index: training to 2021-07-29 16:47, validation to 2021-08-30 08:23, test to 2021-09-30 23:59.

## 17. Maintenance event window

Rule: the last hour (23:00–23:59) of the day before a maintenance day, the maintenance day and the whole day after are excluded from statistics, thresholds and scoring, and reported separately. In the window: 7 maintenance days (18 Mar, 15 Apr, 20 May, 17 Jun, 15 Jul, 10 Aug, 16 Sep), 20,580 minutes. Reason: restarts after maintenance produce jumps up to ±31 kW (e.g. 11 Aug 2021) and the 64.5 kW raw maximum (16 Sep). This replaces the earlier `post_maint_zero` rule. Implemented in notebook 05 (flag `mwin`); to be moved into the modelling-dataset builder.

## 18. Unscheduled outage (29 March 2021)

A cluster-wide gap outside any maintenance window: outage 12:22–15:12 (163 minutes). Telemetry stopped earlier, at 11:32: 49 minutes with zero power while the log shows GPUs allocated, plus one zero minute with `gpus_alloc = 0`. Rule (decided): exclude from the start of the telemetry loss to the end of that day (11:32–23:59), like a maintenance event (flag `uwin`, to be implemented). 28 March has the lowest daily mean of the window (7.7 kW).

## 19. Time axis and the daylight-saving gap

The series is on the New York wall clock of the GPU files, stored as naive epoch seconds (Section 5 of the time-frame rule); Slurm log times (UTC) are converted to this clock. Consequence: on 2021-03-14 the wall-clock hour 02:00–02:59 does not exist. The telemetry is empty there while the log shows jobs running across it (up to 128 GPUs), so the series has 60 artificial zero minutes. They are flagged `dst_gap` and excluded. Log and telemetry changes align at lag 0–1 minute (correlation 0.53 and 0.66), which confirms that both are on the same clock. The DST end (7 November) is outside the data.

## 20. Zero-power minutes, suspicious zeros and the covariate fill rule

In the window: 585 minutes with the cluster up but no GPU stream (power set to 0): 475 inside maintenance windows, 60 in the DST gap, 50 on 29 March. Suspicious zeros (power 0 outside `mwin` while `gpus_alloc > 0`, i.e. telemetry gaps) are excluded (49 minutes, all on 29 March). On minutes without any GPU stream the weighted means `util_gpu_mean`, `util_mem_mean`, `temp_gpu_mean`, `temp_mem_mean` are undefined (null); they are filled with 0 and flagged `no_gpu`. All other covariates are never null.

## 21. Stream overlaps and the raw maximum

`n_gpu_streams` reaches 540 although the cluster has 448 GPUs: in handover minutes a job ending and a job starting on the same GPU are both counted as streams. Time weighting keeps `n_gpu_equiv` at most 429, equal to the log maximum of `gpus_alloc`. The raw maximum of 64.5 kW lies inside the 16 September maintenance window; the maximum on valid minutes is 49.6 kW.

## 22. Full series versus the earlier July–September series

The July–September slice of the full series and `series_julsep.parquet` (built from chunks 1–3 only) have identical stream and edge counts but small differences in weighted columns (power up to 2.5 W, `n_gpu_equiv` up to 0.08). Likely cause (not verified): the global fallback median of `n_samples`, used for streams without interior minutes, now comes from all streams instead of chunks 1–3. On the same set of streams the low-memory and the original aggregation agree to 3e-11 W.

## 23. Modelling dataset v1.1 (Week 3)

`src/data/build_model_dataset.py` (notebook 06) writes `data/processed/model_dataset.parquet` (303,840 × 68) and `model_dataset_meta.json`. Flags: `outage`, `mwin`, `uwin`, `dst_gap`, `susp_zero`, `no_gpu`, `valid`. Valid minutes: 282,452 (92.96%). A forecast origin t is usable for horizon h when all 60 input minutes and the target minute t + h are valid and both lie in the same split. Usable origins train / validation / test: h1 196,700 / 42,575 / 42,575; h5 196,668 / 42,567 / 42,567; h15 196,588 / 42,547 / 42,547. Relative ramp thresholds (training p95 of |ΔP| / L24): 0.0295 / 0.0399 / 0.0568; no ramp label when L24 has fewer than 720 valid minutes (22,039 minutes, mostly after maintenance). The `uwin` rule (unscheduled outage of 29 March) runs from min(first suspicious zero of the day, outage start) to 23:59 and counts days, not runs (v1.0 counted runs). A leakage test (rebuild on the first 66% of the data) changed no causal column.

## 24. Known bug: ARIMA parameter names in the classical summary (harmless)

In `reports/classical/classical_summary.json` (notebook 08) the ARIMA parameter **names** are wrong: they read `lag1 … lag7` because a feature-name variable overwrote them. The **values** are correct and in SARIMAX order for ARIMA(1,1,5): ar.L1 −0.3969; ma.L1 … ma.L5 0.1615, −0.1832, −0.2339, −0.2454, 0.1547; sigma2 0.2281. `src/evaluation/frozen.py` reads the values by position, and the frozen reproduction of all validation forecasts was exact (Week 4, notebook 16). No result is affected; fix the naming only if notebook 08 is rerun.

## 25. Test-protocol records (Weeks 4 and 6)

- Classical benchmark: two freeze records with identical content (`reports/week4/test_freeze.json`, `test_freeze_20261009_191127.json`); the first run stopped before any test forecast was written, so the test split was evaluated once (9 October 2026).
- Quantum models: `reports/week6/quantum_test_freeze.json` (12 October 2026, 13:48). Its `git_head` is the Week 4 commit (`96597605…`) because the Week 5–6 code was not committed when it was written; the SHA-256 hashes of the 25 input files (dataset, code, checkpoints, readouts) are the binding record. Validation reproduction before the test: max difference 0.0 kW for all 8 model families.

## 26. Five-minute dip and the week of 16 August

The Week 3 hypothesis (models over-correct the five-minute dip after it nearly vanished on 10 August) was rejected in Week 4: weekly skill stays near zero or negative in that week also without the minute-mod-5 targets. Alternative, untested hypothesis: a change of job mix after the 10 August maintenance (power per GPU fell from about 103 W to 67–77 W on 8–9 August).

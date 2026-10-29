# HANDOFF — Week 2, session 3 (Friday 2 October 2026)

Bachelor project: forecasting AI data center GPU power (MIT Supercloud), classical / foundation / hybrid quantum models, conformal intervals, battery MPC.
This file continues `HANDOFF_week2_complete_data.md` (v of 2 Oct, morning). Read that file for everything before this session (pipeline, time-frame rule, time-weighted aggregation, outages, dev-week EDA, Colab design, Section 16 remaining work). This file records only what happened in session 3 and the updated plan.

**WEEK 2 IS NOT COMPLETE.** Both plan criteria were already met (one command builds the series; ramp threshold is a concrete provisional number). Still open: whole-dataset chunks 4–7, the remaining EDA on Jul–Sep (and later on the full series), the 5-minute periodicity investigation, the proposal update, Weekly Report 2 (due Friday 9 October 2026) and the Meeting 1 packet.

---

## 0. New working setup (from the next session on)

- **Chat (claude.ai Project)** = thinking, reasoning, explaining, interpreting outputs, deciding, writing the proposal / weekly report / meeting packet (Claude Docs with English and Persian tabs). The chat writes precise task specifications for Claude Code and short Colab cells.
- **Claude Code (on the Mac, project root)** = executes only: writes and runs code, edits scripts with verification, saves figures, and returns raw outputs in a fixed report format. No interpretation, no decisions.
- **Colab** = heavy raw-data processing; cells are pasted by the student.
- Prompts for both are in `PROMPT_chat.md` and `PROMPT_claude_code.md`.

---

## 1. Environment and paths (unchanged unless stated)

| Item | Value |
|---|---|
| Mac project root | `~/Desktop/Other/datacenter-power-forecasting/` |
| Python | `/opt/miniconda3/envs/dcpower/bin/python` (conda env `dcpower`, Python 3.11, kernel "Python (dcpower)") |
| Notebooks | `notebooks/` (notebook working directory; paths inside notebooks are relative, e.g. `../data/processed`) |
| Colab Drive folder | `MyDrive/dcpower/` (mounted at `/content/drive/MyDrive/dcpower`), Colab Python 3.13 |
| Internet budget | about 100 GB total for the student; large downloads only at night; **never download raw GPU files on the Mac** |
| Git | handled by the student only; never give Git instructions |

---

## 2. Colab whole-dataset run: status

### 2.1 Finished chunks (all verified from the cell output)

| Chunk | Jobs ending | Files | Raw size | failed | exit | Wall time | Rate | Minutes with no stream after chunk (whole window 2021-01-01..10-02, 394,560 rows) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2021-09-01 .. 2021-10-02 | 26,479 | 437.5 GiB | 0 | 0 | 2.91 h | 43.34 MiB/s | 322,191 |
| 2 | 2021-08-01 .. 2021-09-01 | 22,710 | 385.0 GiB | 0 | 0 | 2.70 h (16:00–18:41) | 41.89 MiB/s | 284,292 |
| 3 | 2021-07-01 .. 2021-08-01 | 16,525 | 369.7 GiB | 0 | 0 | 3.13 h (18:41–21:49, incl. about 30 min aggregation) | 44.56 MiB/s | 244,654 |

- Power min/max of the whole-window series after chunk 3: 4.76 W / 64,531.7 W (the 64.5 kW post-maintenance peak of 16 Sep).
- Cumulative: 1,192.2 of 2,269.5 GiB (about 53%). The series is complete from **2021-07-01** onward.
- README speed "about 43 MiB/s" is still correct; no README change needed.

### 2.2 Remaining chunks

| Chunk | Jobs ending | Raw size | Estimate (processing only) |
|---|---|---|---|
| 4 | June | 367.6 GiB | about 2.4–2.5 h |
| 5 | May | 295.1 GiB | about 1.9–2.0 h |
| 6 | April | 218.9 GiB | about 1.4 h |
| 7 | March + February | 195.7 GiB | about 1.3 h |

Plus the aggregation step after each chunk (see 2.4). Plan: chunks 4 and 5 in one Colab session, 6 and 7 in the next.

### 2.3 What happened to the Colab session tonight (important for the next start)

1. The student turned Colab cell 5 into a loop cell ("Colab cell 6") that runs `CHUNKS = [2, 3]` back to back. It ran chunks 2 and 3 and printed `ALL QUEUED CHUNKS FINISHED`.
2. During the chunk-3 aggregation the browser showed "Connecting" and "Not connected to runtime"; internet was fine; a page reload did not help at first; the runtime was alive and finished (CPU value in the bar kept updating). Cause most likely a busy runtime (aggregation of about 65,700 streams from Drive), **not verified**.
3. `series_after_chunk_3.parquet` (9.31 MiB) was copied from `/content` to `MyDrive/dcpower/` (sizes equal; shape 394,560 × 12; minute_epoch 1609459200 .. 1633132740) and downloaded to the Mac (`data/processed/series_after_chunk_3.parquet`).
4. Claude wrote cell 6 v2/v3 with a pre-check that lists `streams_full/` to confirm that chunks 1–3 are complete. On a fresh Drive mount the listing ran at about **5,000 entries per 47 s** (about 10–11 min for about 65,700 entries); the first attempt was interrupted, the second appeared stuck and the student stopped it.
5. Claude wrote **cell 6 v4 without any pre-check** (Appendix A). After a session restart, the student ran the cell and got `FileNotFoundError: /content/drive/MyDrive/dcpower/all_files_desc.parquet` because **Drive was not mounted** (cell 1 not run after the restart). The traceback also showed a v3 line, so the old cell content may still be in the notebook.
6. **Chunk 4 has NOT started.**

**Next Colab steps, in order:** run cell 1 (mount, confirm `Mounted at /content/drive`) → make sure the loop cell contains v4 (first line `# Colab cell 6 (v4)`) → run it → check the first lines: `=== chunk 4: ... | 367.6 GiB` and `still to download` about 367.6 GiB (if much larger, stop at once: earlier chunks would be re-downloaded) → watch `failed` and RAM.

### 2.4 Known issue: aggregation after every chunk

`build_series.py` always aggregates all stream Parquets in `--stream-dir` into a series at the end. With a growing folder this took about 30 min after chunk 3 and will grow; RAM risk is unknown (v4 shows RAM in the bar). Options for a decision: (a) keep it (its output is useful: the Mac series came from it); (b) add a `--no-series` flag to the script (script edit via verified procedure, then upload to Drive). Not decided.

### 2.5 Unverified

- Completeness of `streams_full/` for chunks 1–3 (expected 26,479 + 22,710 + 16,525 = 65,714 stream Parquets; never counted because the pre-check did not finish). Risk judged small (the runtime finished chunk 3 normally). Check once later, preferably after all chunks, with plenty of time (listing about 10+ min).
- Whether leftover `.tmp` files exist in `streams_full/`.
- Stream file naming = `Path(name).stem + ".parquet"` (from the handoff description of `<stem>.parquet`; not checked against real names).

### 2.6 Change of plan versus Section 16 step B of the previous handoff

Streams are **not** zipped or downloaded to the Mac any more. The series is built on Colab and only the series Parquet (about 10 MB) is downloaded. This saves the student's internet and removes the risk of accidental raw downloads on the Mac. For the final series: after chunk 7, use `series_after_chunk_7.parquet` (whole window, unclipped) and slice on the Mac, or build once on Colab with the exact window. The Mac warning stays: never run `build_series.py` on the Mac with `all_files_desc.parquet`.

---

## 3. Housekeeping done this session

- Renamed `notebooks/02_data_download_explore.ipynb.ipynb` → `02_data_download_explore.ipynb` (tab closed first, `mv` in Terminal).
- Found and fixed the same double extension for `notebooks/01_pennylane_warmup.ipynb.ipynb` → `01_pennylane_warmup.ipynb` (was not recorded before).
- `ls -la notebooks/` after the renames: `00_setup_check.ipynb`, `01_pennylane_warmup.ipynb`, `02_data_download_explore.ipynb`, `03_eda.ipynb`, `04_eda_jul_sep.ipynb` (new), `wandb/`, `.ipynb_checkpoints/`.
- `03_eda.ipynb` showed "modified now" in JupyterLab (opened). **Do not re-run it**: it holds the dev-week figures and numbers.
- Still open: delete `MyDrive/dcpower/streams_test/` (obsolete); `.ipynb_checkpoints` names follow the notebooks and are harmless.

---

## 4. PM100 metadata (notebook 02, new last cell, labelled "Cell 57"; metadata only, no data download)

| | Record 8129258 | Record 10127767 (latest) |
|---|---|---|
| Published | 2023-07-21, version 1.0 | 2023-11-24 (version field empty) |
| File | `job_table.parquet` | `job_table.parquet` |
| Size | 101.0 MiB | **273.9 MiB** |
| License | CC BY 4.0 | CC BY 4.0 |
| Concept record | 8129257 | 8129257 |

- Other work (ORNL HPC digital-twin code; a 2025 carbon study) uses record **10127767**, so that is the version to use. The "about 101 MB" in the earlier handoff refers to the old version and must be corrected.
- Per the paper's description, each job row contains lists of power values per timestamp (CPU, memory, node power); building a 1-minute aggregate series looks possible but the schema is **not verified**.
- Job count: earlier handoff says 231,238; one other source gives 231,116 jobs; probably a version difference, **not verified**.
- **Source that could not be read:** the Zenodo record page (robots disallow automated access), so the changelog of the new version is unknown; why the file grew about 2.7× is unknown.
- Plan: inspect the schema on Colab later (no Mac download needed); the decision on PM100 belongs to Meeting 1.

---

## 5. Jul–Sep series on the Mac

### 5.1 Files

| File | Content |
|---|---|
| `data/processed/series_after_chunk_3.parquet` | Colab output, whole window 2021-01-01..10-02, unclipped, 394,560 × 12; complete only from 2021-07-01 |
| `data/processed/series_julsep.parquet` | slice 2021-07-01 ≤ minute_local < 2021-10-01 (file clock), 132,480 rows, 7.30 MiB (notebook 04 cell 1) |
| `data/processed/series_julsep_features.parquet` | output of `build_log_features.py`, 132,480 × 20, 7.69 MiB |

Assumption, **not verified**: slicing the unclipped whole-window series equals building with `--win-start 2021-07-01 --win-end 2021-10-01` (streams are unclipped, aggregation per minute should be identical).

### 5.2 Slice checks (cell 1)

132,480 rows = 92 days × 1,440; all steps 60 s; 2,480 null-power minutes, only on four days: 2021-07-15 (982), 2021-08-10 (684), 2021-08-11 (92), 2021-09-16 (722); no zero minutes before the features step; power 4.8 W .. 64,531.7 W; `n_gpu_equiv` max 429.00 (≤ 448).

### 5.3 `build_log_features.py` output

Command (Terminal): `/opt/miniconda3/envs/dcpower/bin/python -u src/data/build_log_features.py --log data/raw/slurm-log.csv --series data/processed/series_julsep.parquet --out data/processed/series_julsep_features.parquet` (absolute paths were used).

```
log rows: 395,914; with end > start: 395,914
cluster-wide gaps: 42; long (>= 6.0 h): 9
maintenance days in the whole log: 2021-01-21, 03-02, 03-18, 04-15, 05-20, 06-17, 07-15, 08-10, 09-16
minutes: 132,480; outage: 2,316; cluster up, no GPU job (set to 0): 164; maint_day: 4,320
null power_W after rules: 2,316; cluster-gap minutes with GPU telemetry (kept, not outage): 7
gpus_alloc max: 429 (cluster has 448 GPUs)
```

2,316 + 164 = 2,480 (every empty minute has a label); maint_day = 3 days (15 Jul, 10 Aug, 16 Sep).

Columns (20): `minute_epoch, power_W, util_gpu_mean, util_mem_mean, temp_gpu_mean, temp_mem_mean, mem_used_sum, mem_free_sum, n_gpu_equiv, n_gpu_streams, n_edge_streams, minute_local, cluster_gap, active_jobs, active_gpu_jobs, gpus_alloc, queued_jobs, queued_gpu_jobs, outage, maint_day`. Nulls: power_W, mem_used_sum, mem_free_sum 2,316; util_gpu_mean, util_mem_mean, temp_gpu_mean, temp_mem_mean 2,480 (also null on the 164 zero minutes: no GPU, no utilization). **Modelling note:** these covariates need a fill rule or a flag on zero minutes.

---

## 6. EDA on Jul–Sep (`notebooks/04_eda_jul_sep.ipynb`, cells 1–6 run; figures in `reports/figures/eda_julsep/`)

All numbers are **provisional (Jul–Sep only)**; they will be recomputed on the full series by changing the input path and re-running the notebook.

### 6.1 Monthly level (cell 2)

| Month | Null minutes | Mean kW | Median kW | p99 kW | Max kW |
|---|---|---|---|---|---|
| 2021-07 | 982 | 34.28 | 34.19 | 47.16 | 49.57 |
| 2021-08 | 660 | 27.81 | 28.69 | 44.90 | 48.31 |
| 2021-09 | 674 | 27.27 | 26.94 | 40.83 | 64.53 |

- Level shift of about 7 kW from July to August/September.
- Hypothesis (not verified): the paper's "about 45 kW" matches the level of normal peaks (monthly p99 41–47 kW, Jul/Aug max 48–50 kW); 64.5 kW is an exceptional post-maintenance event.

### 6.2 Whole series, split and daily means (cell 3, `01_series_jul_sep.png`)

- Chronological 70/15/15 by minute index: train < 2021-09-03 09:36 ≤ val < 2021-09-17 04:48 ≤ test. The maintenance day 16 Sep is inside validation.
- Per split, normal days only: train mean 30.98 kW (std 7.97, min 0.0, max 49.57; 89,856 valid minutes); val 30.40 (5.13, 17.73, 42.20; 18,432); test **25.56** (5.78, 10.96, 41.22; 19,872). Test is about 5.4 kW below train.
- Daily means (normal days): min 13.2, max 46.0 kW; first 7 days 34.2, last 7 days 23.1 kW (downward trend of about 11 kW over three months).
- **New finding: post-maintenance tail.** 2021-08-11 has 92 zero-power minutes: the 10 Aug outage continues past midnight; the log shows the cluster up but no GPU job yet, so the rule sets 0. `maint_day` is a calendar-day flag, so these minutes count as a normal day (they caused train min 0.0) and create an artificial large ramp (0 → normal level).
- **EDA-only rule (provisional, for Meeting 1):** zero-power minutes on the day after a maintenance day are flagged `post_maint_zero` and excluded from statistics (`valid = ~outage & ~maint_day & ~post_maint_zero`). Not in `build_log_features.py` yet.

### 6.3 Daily and weekly profiles (cell 4, `02_daily_weekly_profile.png`)

Valid minutes 128,068 of 132,480 (92 post-maintenance zero minutes excluded).
- **Daily:** hour of day explains 0.072 of within-day variance (hourly means minus day mean); permutation null (hours shuffled within days, 1,000 runs) mean 0.011, 95th pct 0.017, p = 0.000. Shape: decline from about 03:00 to a minimum of about −1.7 kW near 09:00, about +0.9 kW from afternoon to night; peak-to-trough about 2.6 kW vs hourly sd about ±4 kW. **Caveat:** the permutation breaks the within-day autocorrelation of hourly deviations, so the null is probably too low and p too optimistic. Cautious reading: a weak daily cycle exists (the dev week was too short to see it).
- **Weekly:** weekday explains 0.049 of the variance of detrended daily means (centred 7-day rolling mean removed; 83 days); null mean 0.073, 95th pct 0.146, p = 0.693. **No weekday effect.** Weekday means (kW, detrended): Mon −0.93, Tue +0.53, Wed +0.29, Thu +1.40, Fri +0.81, Sat +0.18, Sun −1.48 (sd 3.4–5.2, n 10–13).

### 6.4 ACF, PACF, persistence (cell 5, `03_acf_pacf.png`)

ACF on pairs where both minutes are valid; PACF on the longest valid segment (51,082 min, 2021-08-11 12:38 .. 2021-09-15 23:59).
- ACF(P): 0.997 (1), 0.992 (5), 0.985 (15), 0.959 (60), 0.858 (360), 0.760 (720), 0.642 (1440), 0.401 (2880); first below 0.5 at **2,096 min** (dev week: 379); no daily bump (ACF at 1440 equals the minimum over 600–1440). Slow decay is driven by trend / level shifts (non-stationarity).
- ACF(ΔP): −0.170 (1), −0.096 (2), −0.032 (3), −0.086 (4), **+0.109 (5)**, +0.115 (10), +0.113 (15); 95% band ±0.005.
- PACF(P): +0.998, +0.138, +0.103, +0.040, +0.053, +0.002 (lags 1–6); band ±0.009.
- **Persistence baseline** P(t+h) = P(t), valid pairs only (kW):

| h | split | pairs | MAE | RMSE | p95 abs ΔP |
|---|---|---|---|---|---|
| 1 | train | 89,751 | 0.343 | 0.665 | 1.075 |
| 1 | val | 18,431 | 0.252 | 0.402 | 0.730 |
| 1 | test | 19,872 | 0.252 | 0.408 | 0.736 |
| 5 | train | 89,699 | 0.463 | 1.009 | 1.422 |
| 5 | val | 18,427 | 0.374 | 0.638 | 1.106 |
| 5 | test | 19,872 | 0.464 | 0.816 | 1.465 |
| 15 | train | 89,627 | 0.655 | 1.394 | 1.993 |
| 15 | val | 18,417 | 0.564 | 0.903 | 1.763 |
| 15 | test | 19,872 | 0.751 | 1.262 | 2.474 |

Train p95 of |P(t+h) − P(t)| (1.08 / 1.42 / 1.99 kW) is larger than the dev-week thresholds (0.62 / 1.01 / 1.68 kW): the provisional ramp thresholds will change. Note: these p95 values are over all valid train pairs, not yet the formal "normal days of the first 70%" threshold computation.

### 6.5 Five-minute periodicity (cell 6, `03b_dP_5min_check.png`) — OPEN INVESTIGATION

- ACF(ΔP) lags 1–30 shows positive spikes at 5, 10, 15, 20, 25, 30 (about +0.09 to +0.115) and negative values between.
- By minute of hour mod 5 (valid consecutive minutes): mean ΔP **−0.208 kW** at mod 0, **+0.205 kW** at mod 1, about 0 at mod 2–4; mean |ΔP| 0.376 / 0.369 vs about 0.277 at mod 2–4. Mean edge streams 1.08–1.21 (not higher at mod 0); mean GPU streams about 333.7 everywhere.
- The minutes with the largest mean |ΔP| are all at :x0/:x5 or the minute after (40, 30, 16, 25, 5, 36, 55, 31).
- **Reading:** every 5 minutes, on the wall-clock boundary (:00, :05, ...), power drops about 0.2 kW for one minute and recovers the next minute. This is comparable to the h = 1 persistence MAE (0.25 kW); a model with minute-of-hour features could learn it and "beat" persistence for a possibly artificial reason.
- **Hypotheses:** (1) real: a cluster-wide periodic process (monitoring, health check) briefly reduces GPU power; (2) artefact: the GPU logger misses some seconds every 5 minutes (e.g. log rotation), and the time-weighted aggregation then sees less coverage in that minute, lowering the summed power. Job start/end is unlikely (edge streams not higher). In the dev week ACF(ΔP) at lag 5 was already +0.078 (noted, not investigated).
- **Next test (cell 7, Appendix B, written but not run):** by minute mod 5, compare `n_gpu_equiv` and its change, power per GPU-equivalent and its change, and the coverage ratio `n_gpu_equiv / n_gpu_streams`. If `n_gpu_equiv` dips at mod 0 while W per GPU-equivalent is flat → coverage artefact (fix in the aggregation, e.g. per-stream mean over covered seconds). If `n_gpu_equiv` is flat and W per GPU dips → real dip (document; decide on features and fairness).
- If inconclusive: inspect raw sample timestamps of a few stream files around :00/:05 (requires raw files → on Colab, never on the Mac).

---

## 7. Proposal review (Claude Doc, read this session; NOT edited)

- Artifact link: `https://claude.ai/code/artifact/1c7f11e9-9c9b-40f6-ad87-6c447798161e`; Doc id `4X6RRgndxG69SsM5wyeFvV`; tabs English (`27ba9808-32b2`, content node `1031885e-f609`, rev 12) and فارسی (`74a5bf79-9ada`, content node `df5361d7-61d0`). **The Persian tab was not read**; read it before editing.
- The student said: wait with edits until the data numbers are ready; just keep the list.

**Change list:**
1. **Section 3 Data:** "aggregate peak about 45 kW" → 64.5 kW post-maintenance peak, normal levels / p99 41–47 kW, the paper's 45 kW not yet explained; "summed over all jobs" → time-weighted aggregation; "about 330,000 steps (about 238 days)" → 238 days × 1,440 ≈ 342,700 minutes minus about 127 h of outages (final number from the full series); add time-frame rule (log UTC, GPU files New York clock), outage / maint_day flags, empty-minute rule, Colab pipeline; PM100: not "backup" but second dataset, record 10127767 (273.9 MiB), license CC BY 4.0.
2. **Section 4 Methods:** persistence is the mandatory baseline (skill relative to persistence); seasonal naive kept with the note that the daily cycle is weak; two-part model (active GPUs × power per GPU) only as an idea.
3. **Section 5 Protocol:** ramp threshold per horizon on |P(t+h) − P(t)| (p95, plus p99 severe tier) from normal days of the first 70%; masking of outage minutes; maintenance days (and post-maintenance tail) reported separately; skill vs persistence; sMAPE is undefined on zero-power minutes (0/0) → exclude them or use MASE (question); 10-second experiment as an open question.
4. **Section 9 Open questions:** add agenda items A–C of the previous handoff Section 17 plus the new items in Section 8 below.
5. **New sections:** fixed-decisions table for approval; five-minute EDA summary with 4–5 figures (from `reports/figures/eda_julsep/`, later full series).
6. Sections 1, 2, 6, 7, 8 unchanged. Do not add dates that were not discussed.

---

## 8. New items for the Meeting 1 agenda (add to Section 17 of the previous handoff)

1. **Post-maintenance tail:** extend the maintenance event beyond the calendar day (zero minutes after an outage that crosses midnight)? Put the rule into `build_log_features.py`?
2. **Five-minute periodicity:** result of the cell-7 test; if real, may minute-of-hour features be used? If artefact, fix in aggregation.
3. **Level shift and split:** train mean 31.0 kW vs test 25.6 kW (Jul–Sep); is Min-Max on train enough, or should models work on differences / locally normalised levels? Re-check on the full series.
4. **Covariates on zero minutes:** fill rule or flag for util/temp nulls.
5. **sMAPE on zero minutes** (or MASE).
6. **Daily cycle weak but detectable (share 0.072), no weekly effect:** hour sin/cos features yes, weekday features no (provisional).
7. **PM100 version:** 10127767; schema check on Colab.

---

## 9. Remaining work for Week 2 (updated list; replaces Section 16 of the previous handoff)

| Step | Task | Status |
|---|---|---|
| A | Check chunk-1 output | **done** (plus chunks 2 and 3) |
| B | Series on the Mac | **done for Jul–Sep** via Colab series + slice (not September only); validated |
| C | Colab chunks 4–7 | **open**: chunk 4 not started (mount first, cell 6 v4) |
| C2 | Count streams in `streams_full/` (65,714 expected for chunks 1–3, then all) and leftover `.tmp` | open, low priority |
| H1 | EDA cell 7: 5-minute periodicity test | **next on the Mac** |
| H2 | Ramp thresholds per horizon (p95 and p99, normal days of the first 70%, excluding post-maintenance tail), compare with dev week, ΔP distribution figure | open |
| H3 | Covariates: correlations of levels and lagged changes with ΔP, power per GPU vs utilization (repeat dev-week cell 7 on Jul–Sep) | open |
| H4 | Re-run notebook 04 on the full series after chunk 7 (change input path) | open |
| D | Proposal update (both tabs) | open (after the numbers) |
| E | Weekly Report 2 (Claude Doc, English + Persian tabs), due **Friday 9 October 2026** | open |
| F | Meeting 1 packet (agenda Section 17 + Section 8 here, five-minute EDA summary, decisions table) | open |
| G | Housekeeping: delete `MyDrive/dcpower/streams_test/`; correct PM100 size in docs; record the notebook renames | partly done (renames) |
| I | PM100 schema check on Colab | open, optional before Meeting 1 |

**Week 2 is complete when:** chunks 4–7 are done (or a documented decision to stop at Jul–Sep for Meeting 1), the EDA (H1–H3) is finished with provisional numbers, the proposal is updated, Weekly Report 2 and the Meeting 1 packet exist. Tell the student explicitly when this is reached.

---

## 10. Decisions taken in this session

1. Download only the Colab-built series to the Mac; no stream zip.
2. EDA continues on Jul–Sep now (deadline and early error detection); all numbers marked provisional; re-run on the full series later by changing the input path.
3. EDA figures go to `reports/figures/eda_julsep/`; dev-week figures in `reports/figures/eda/` and `03_eda.ipynb` stay untouched.
4. EDA-only `post_maint_zero` exclusion (provisional).
5. Proposal edits wait until numbers are ready.
6. Colab: chunks 4 and 5 next, no pre-check.

---

## Appendix A — Colab cell 6 v4 (run after cell 1 mount)

```python
# Colab cell 6 (v4): run chunks back to back without any pre-check of finished chunks;
# copy each chunk's series to Drive; RAM shown in the progress bar.
import os, re, sys, shutil, threading, subprocess, psutil, time, calendar
from datetime import datetime
from tqdm import tqdm
import polars as pl

BASE = "/content/drive/MyDrive/dcpower"
CHUNKS = [4, 5]          # 4 = Jun, 5 = May

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

bounds = ["2021-10-02", "2021-09-01", "2021-08-01", "2021-07-01", "2021-06-01",
          "2021-05-01", "2021-04-01", "2021-01-01"]
ep = lambda d: calendar.timegm(datetime.fromisoformat(d).timetuple())
prog = re.compile(r"files ([\d,]+)/([\d,]+)\s+done=([\d,]+) skipped=([\d,]+) failed=([\d,]+)\s+"
                  r"([\d.]+)/([\d.]+) GiB\s+([\d.]+) MiB/s")

log("reading all_files_desc.parquet ...")
full = pl.read_parquet(f"{BASE}/all_files_desc.parquet")
log(f"all_files_desc: {full.height:,} files | queue: {CHUNKS}")

def chunk_part(k):
    hi, lo = ep(bounds[k - 1]), ep(bounds[k])
    return full.filter((pl.col("end_fc") >= lo) & (pl.col("end_fc") < hi))

def run_chunk(k):
    part = chunk_part(k)
    part_path = f"{BASE}/chunk_{k}.parquet"
    part.select("sub", "name", "size").write_parquet(part_path)
    print(f"\n=== chunk {k}: jobs ending {bounds[k]} .. {bounds[k - 1]} | "
          f"{part.height:,} files | {part['size'].sum() / 2**30:,.1f} GiB | start {time.strftime('%H:%M')}", flush=True)
    os.makedirs(f"{BASE}/streams_full", exist_ok=True)
    os.makedirs("/content/tmp_raw", exist_ok=True)
    out_local = f"/content/series_after_chunk_{k}.parquet"
    cmd = ["python", "-u", f"{BASE}/build_series.py", "--files", part_path,
           "--stream-dir", f"{BASE}/streams_full", "--tmp-dir", "/content/tmp_raw",
           "--out", out_local,
           "--win-start", "2021-01-01", "--win-end", "2021-10-02", "--workers", "16"]
    log(f"chunk {k}: starting build_series.py (its own skip check can take a few minutes)")
    state = {"files": "0/?", "failed": "0", "rate": "0", "running": True, "agg_note": False}
    bar = tqdm(total=1.0, unit="GiB", file=sys.stdout, mininterval=1, dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} GiB [{elapsed}<{remaining}] {postfix}")
    def heartbeat():
        while state["running"]:
            c = psutil.cpu_percent(interval=2)
            ram = psutil.virtual_memory()
            free_gb = shutil.disk_usage("/content").free / 1e9
            bar.set_postfix_str(f"chunk {k} | files {state['files']} | failed {state['failed']} | "
                                f"{state['rate']} MiB/s | CPU {c:.0f}% | RAM {ram.used / 1e9:.1f}/{ram.total / 1e9:.1f} GB | "
                                f"disk free {free_gb:.0f} GB", refresh=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    t = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        m = prog.search(line)
        if m:
            f_i, f_n, done, skip, fail, g_i, g_n, rate = m.groups()
            bar.total = float(g_n); bar.n = float(g_i)
            state.update(files=f"{f_i}/{f_n}", failed=fail, rate=rate)
            bar.refresh()
            if f_i == f_n and not state["agg_note"]:
                state["agg_note"] = True
                bar.write(f"[{time.strftime('%H:%M:%S')}] chunk {k}: all files handled; aggregating the series (no progress output, can take 30+ min)")
        else:
            bar.write(f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}")
    proc.wait(); state["running"] = False; time.sleep(2.5); bar.close()
    if os.path.exists(out_local):
        dst = f"{BASE}/series_after_chunk_{k}.parquet"
        shutil.copy2(out_local, dst)
        log(f"series copied to Drive: {dst} | sizes equal: {os.path.getsize(out_local) == os.path.getsize(dst)}")
    print(f"chunk {k} | exit code: {proc.returncode} | failed: {state['failed']} | "
          f"wall time: {(time.time() - t) / 3600:.2f} h | end {time.strftime('%H:%M')}", flush=True)

for k in CHUNKS:
    run_chunk(k)
log("ALL QUEUED CHUNKS FINISHED")
```

Next sessions: `CHUNKS = [6, 7]`. If a chunk is interrupted, re-run it with the same number (finished streams are skipped).

## Appendix B — Notebook 04, cell 7 (written, not run; depends on `df`, `ok_dp` from cells 1–6)

```python
# Cell 7: Coverage artifact or real power dip? Compare, by minute of hour mod 5, the time-weighted GPU
# count (n_gpu_equiv), its change, power per GPU-equivalent and its change, and stream counts.
d2 = (df.with_columns(pl.col("minute_local").dt.minute().alias("mm"),
                      (pl.col("power_W") / pl.col("n_gpu_equiv")).alias("W_per_gpu"))
        .with_columns(pl.col("n_gpu_equiv").diff().alias("d_ngpu"),
                      pl.col("W_per_gpu").diff().alias("d_Wpg"),
                      pl.col("n_gpu_streams").diff().alias("d_streams"))
        .with_columns(pl.Series("ok_dp", ok_dp))
        .filter(pl.col("ok_dp") & (pl.col("n_gpu_equiv") > 0)))

chk = (d2.group_by((pl.col("mm") % 5).alias("mm_mod5"))
         .agg(pl.len().alias("n"),
              pl.col("n_gpu_equiv").mean().round(3).alias("mean_ngpu_eq"),
              pl.col("d_ngpu").mean().round(4).alias("mean_d_ngpu_eq"),
              pl.col("n_gpu_streams").mean().round(2).alias("mean_streams"),
              pl.col("d_streams").mean().round(4).alias("mean_d_streams"),
              pl.col("W_per_gpu").mean().round(3).alias("mean_W_per_gpu"),
              pl.col("d_Wpg").mean().round(4).alias("mean_d_W_per_gpu"))
         .sort("mm_mod5"))
print("by minute of hour mod 5 (valid consecutive minutes, n_gpu_equiv > 0):")
print(chk)

d2 = d2.with_columns((pl.col("n_gpu_equiv") / pl.col("n_gpu_streams")).alias("cover"))
cov = d2.group_by((pl.col("mm") % 5).alias("mm_mod5")).agg(
    pl.col("cover").mean().round(5).alias("mean_cover"),
    pl.col("cover").quantile(0.05).round(5).alias("p05_cover")).sort("mm_mod5")
print("\ncoverage ratio n_gpu_equiv / n_gpu_streams by minute mod 5:")
print(cov)

for col, lab in [("mean_d_ngpu_eq", "n_gpu_equiv"), ("mean_d_W_per_gpu", "W per GPU-equivalent")]:
    v0 = chk.filter(pl.col("mm_mod5") == 0)[col][0]; v1 = chk.filter(pl.col("mm_mod5") == 1)[col][0]
    print(f"{lab}: mean change at mod5 = 0: {v0:+.4f} | at mod5 = 1: {v1:+.4f}")
```

Note for Claude Code: notebook 04 cells 1–6 define `df` (with columns `post_maint_zero`, `valid`), `dp`, `ok_dp`, `n`, `i_tr`, `i_va`, `FIG`, `acf_valid`, `mu`, `mu_y`. If cell 7 is run in a fresh kernel, cells 1–6 must run first (cell 4 permutation test takes about a minute).

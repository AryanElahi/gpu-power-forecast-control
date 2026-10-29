# HANDOFF — Week 2, session 5 (Saturday 3 October 2026)

Bachelor project: forecasting AI data center GPU power (MIT Supercloud), classical / foundation / hybrid quantum models, conformal intervals, battery MPC.
This file continues `HANDOFF_week2_session4.md` (read it first; it continues session 3, which continues `HANDOFF_week2_complete_data.md`). This file records only what happened in session 5 and the updated plan. Sections of earlier handoffs that are not repeated or replaced here are still valid.

**WEEK 2 IS COMPLETE except the confirmation of housekeeping (G).** Done in this session: Colab chunks 4–7 (all 109,946 GPU files processed), a new low-memory aggregation (validated), the full series, the full-series EDA (notebook 05, v1 and v2; H1–H4 done), the proposal update in both tabs with five figures and the processing method (D), Weekly Report 2 (E), the Meeting 1 packet (F), a plain-language explainer document, and the housekeeping instructions (G; the student has to run them and confirm, Section 16). The unscheduled-outage rule (Section 10.1) is decided but not yet implemented in code. Links and ids of all documents: Section 15.

---

## 0. Working setup (unchanged from session 4, plus one tool)

- No Claude Code. Chat writes ready-to-run notebook cells, whole notebooks (`.ipynb` files to download) and Terminal commands; the student runs them in JupyterLab (kernel "Python (dcpower)") or Colab and pastes the output. Rules of `PROMPT_chat.md` still apply except the Claude Code parts (answer in Persian; code, notebooks and repo docs in English; no Git instructions; never download raw GPU files on the Mac; cautious claims, provisional labels).
- **New in this session:** the student prefers to receive **whole notebooks as files** and to run them cell by cell; outputs are passed back as one Markdown file made by `src/tools/nb_outputs_to_md.py` (Section 3.4). The notebook must be **saved (Cmd+S) before the export**, because the script reads the `.ipynb` on disk (first export was empty for this reason).
- Colab cells can be "all in one" (a cell that writes a script to Drive and runs it), so nothing has to be uploaded to Drive by hand (Section 2.5).

---

## 1. Environment (unchanged)

| Item | Value |
|---|---|
| Mac project root | `~/Desktop/Other/datacenter-power-forecasting/` |
| Handoff files in the repo | `docs/HANDOFF/` |
| Python on the Mac | `/opt/miniconda3/envs/dcpower/bin/python`; polars 1.44.2, numpy 2.4.6, matplotlib 3.11.2 (versions printed by notebook 05 cell 1) |
| Notebooks | `notebooks/` (working directory; relative paths `../data/...`) |
| Colab Drive folder | `MyDrive/dcpower/` |

The overnight `caffeinate -dis` setup of session 4 (Section 2.2 there) is no longer needed (Colab work is finished).

---

## 2. Colab whole-dataset run: FINISHED

### 2.1 Chunks 4 and 5 (overnight, loop cell v3)

| Chunk | Jobs ending | Files | Raw size | failed | Download/processing | Rate | Exit code | End (Colab clock) |
|---|---|---|---|---|---|---|---|---|
| 4 | 2021-06-01 .. 07-01 | 11,881 | 367.6 GiB | 0 | 3:53:51 (all files handled 00:51) | 45.03 MiB/s | **−9** | 02:25 (wall 3.90 h) |
| 5 | 2021-05-01 .. 06-01 | 14,862 | 295.1 GiB | 0 | 2:00:16 (all files handled 04:23) | 42.92 MiB/s | **−9** | 04:26 (wall 2.00 h) |

- Exit code −9 = SIGKILL. Both processes were killed **during the series aggregation** (after "all files handled"): chunk 4 aggregated about 1.5 h (00:51 → 02:25) before the kill, chunk 5 only about 3 min. Most likely cause: out of memory (13.6 GB RAM) when aggregating about 77k–92k streams at once; the risk was predicted in session 3, Section 2.4. **Not verified** from a kernel log.
- The per-file stage (stream Parquets) had finished before the kill, so the streams were complete (confirmed by the pre-check of the next run: chunks 4 and 5 `missing 0`). `series_after_chunk_4/5.parquet` were **not** created (no "series copied" line).
- The loop cell still printed "ALL QUEUED CHUNKS FINISHED" (the loop does not stop on a non-zero exit code).

### 2.2 Loop cell v5 (replaces v3/v4; code in Appendix B)

Changes against v3:
- **Pre-check of all earlier chunks** (1 .. min(CHUNKS) − 1): lists `streams_full/` (progress every 5,000 entries), counts `.tmp` leftovers, notes a stale `_failures.csv`, and compares expected stems per chunk. Stem rule: `Path(name).stem` of `all_files_desc.name` = stream file name without `.parquet`. **Verified** (all chunks `missing 0`; a wrong rule would show missing = expected).
- **`SKIP_AGG = True`:** when the progress line reaches `files N/N`, the cell waits 10 s and terminates `build_series.py` (SIGTERM → exit code **−15**, expected), so no time is wasted on the aggregation that is OOM-killed anyway.
- **Post-check after each chunk** (fresh listing; `expected | missing` for that chunk).

### 2.3 Chunks 6 and 7 (loop cell v5, Saturday morning, Colab clock)

Pre-check at 05:30: `92,457 entries | 92,457 stream parquets | leftover .tmp: 0`; chunks 1–5 `missing 0`; listing took 17 s.

| Chunk | Jobs ending | Files | Raw size | failed | Processing | Rate | Exit | Post-check |
|---|---|---|---|---|---|---|---|---|
| 6 | 2021-04-01 .. 05-01 | 7,582 | 218.9 GiB | 0 | 1:22:57 (05:30 → 06:53) | 45.20 MiB/s | −15 (stopped before aggregation) | 100,039 parquets, 0 `.tmp`, missing 0 |
| 7 | 2021-01-01 .. 04-01 | 9,907 | 195.7 GiB | 0 | 1:15:56 (06:53 → 08:09) | 44.16 MiB/s | −15 (stopped before aggregation) | **109,946** parquets, 0 `.tmp`, missing 0 |

### 2.4 All chunks (C and C2 done)

| Chunk | Files | Raw size |
|---|---|---|
| 1 | 26,479 | 437.5 GiB |
| 2 | 22,710 | 385.0 GiB |
| 3 | 16,525 | 369.7 GiB |
| 4 | 11,881 | 367.6 GiB |
| 5 | 14,862 | 295.1 GiB |
| 6 | 7,582 | 218.9 GiB |
| 7 | 9,907 | 195.7 GiB |
| **Total** | **109,946** (= all rows of `all_files_desc.parquet`) | **2,269.5 GiB** |

Zero failed files in all chunks. Throughput over chunks 4–7: 42.9–45.2 MiB/s (README value "about 43 MiB/s" still correct).

### 2.5 Low-memory aggregation: `build_series_lowmem.py` (new; code in Appendix A)

**Why:** the one-shot aggregation in `build_series.py` loads all streams into RAM and is OOM-killed on Colab from about 77k streams on.

**What it does** (same rules as `build_series()` v3, written from the handoff description of those rules, not from the original code; validated below):
- Reads the stream Parquets in **batches** (default 2,000 files, 16 reader threads; Drive reading is latency-bound).
- Per batch: per stream (`file`, `gpu_index`) first/last minute, interior = strictly between; `full` = median `n_samples` of the stream's interior minutes; weight w = 1 (interior) or min(n_samples / full, 1) (first/last minute); per-minute **partial sums**: Σ power·w, Σ mem_used·w, Σ mem_free·w, Σ w, stream count, edge count, and numerator/denominator of the four weighted means (util_gpu, util_mem, temp_gpu, temp_mem; nulls excluded). Written to `--work-dir` as `part_XXXX.parquet`, plus `hist_XXXX` (histogram of interior `n_samples`) and `noint_XXXX` (rows of streams without interior minutes).
- After all batches: the **global fallback median** of `n_samples` is computed exactly from the merged histogram (mean of the two middle values for an even count, as polars `median`); streams without interior minutes are then weighted with it; partials are summed **incrementally** (one batch at a time, low RAM); complete minute grid [t0, t1); empty minutes: counts 0, `n_gpu_equiv` 0.0, metrics null.
- `minute_local` = `pl.from_epoch(minute_epoch)` (naive; same convention as the original series, see Section 5).
- **Resumable:** a batch whose three files exist is skipped (the `part` file is written last). Partials live on the Colab disk (`/content/agg_*`) and are lost when the runtime is deleted.
- Arguments: `--stream-dir`, `--out`, `--work-dir`, `--win-start`, `--win-end` (file clock, end exclusive), optional `--files` (one or more Parquets with column `name`, e.g. `chunk_1.parquet`; restricts the streams; aborts if any expected stream is missing), `--batch` (2000), `--readers` (16).
- Tested in the chat on synthetic data (250 files, nulls, gaps, streams without interior minutes) against a one-shot reference: max difference 4.5e-13.

**Colab cell 7 "all in one"** (`colab_cell7_lowmem.py`, 227 lines; the script text is embedded and written to `MyDrive/dcpower/build_series_lowmem.py`): (1) validation run on chunks 1–3 (`--files chunk_1..3.parquet`, work dir `/content/agg_c123`) and comparison with `series_after_chunk_3.parquet` (built by the original code); (2) automatic pass rule: same rows, no minute only in one series, `power_W`, `n_gpu_equiv`, `n_gpu_streams`, `n_edge_streams` max diff ≤ 1e-3 and no null mismatch; (3) if passed and `RUN_FULL = True`: full run on all streams (work dir `/content/agg_all`) and copy to Drive.

**Validation result (passed):**

| Column | max abs diff | null mismatch | NaN old / new |
|---|---|---|---|
| power_W | 2.9e-11 | 0 | 0 / 0 |
| util_gpu_mean | 4.3e-14 | 0 | 0 / 0 |
| util_mem_mean | 1.8e-14 | 0 | 0 / 0 |
| temp_gpu_mean | 3.6e-14 | 0 | 0 / 0 |
| temp_mem_mean | 4.3e-14 | 0 | 0 / 0 |
| mem_used_sum | 3.7e-09 | 0 | 0 / 0 |
| mem_free_sum | 5.6e-09 | 0 | 0 / 0 |
| n_gpu_equiv | 1.7e-13 | 0 | 0 / 0 |
| n_gpu_streams | 0 | 0 | 0 / 0 |
| n_edge_streams | 0 | 0 | 0 / 0 |

Rows 394,560 / 394,560; minutes only in one series: 0. → Batched partial sums reproduce the one-shot aggregation up to float summation order.

**Full run:** exit code 0; `series_full_2021.parquet` copied to Drive, sizes equal, **18.5 MiB**. Window 2021-01-01 .. 2021-10-02 (file clock), 394,560 rows, built from all 109,946 stream Parquets.

**Not recorded:** run times of the validation and the full run, number of batches, the global fallback median value (the subprocess output, including the `batch ...` progress lines, is **not shown** in the Colab cell output; progress can only be seen by counting `part_*.parquet` files in the Colab file browser). If the script is run again, capture the output (e.g. `capture_output=True` and print, or `tee` to a log file).

### 2.6 Files on Drive after this session (`MyDrive/dcpower/`)

| File / folder | Content |
|---|---|
| `streams_full/` | 109,946 stream Parquets (complete) |
| `series_full_2021.parquet` | **final full series**, 18.5 MiB (Section 2.5) |
| `build_series_lowmem.py` | written by Colab cell 7 |
| `series_after_chunk_1/2/3.parquet` | old intermediate series (chunk 3 used as validation reference) |
| `chunk_1.parquet` .. `chunk_7.parquet` | file lists per chunk (`sub, name, size`) |
| `all_files_desc.parquet`, `build_series.py` | unchanged |

`series_after_chunk_4/5/6/7.parquet` do not exist. The Colab runtime may be deleted (Drive files stay). `streams_test/` still to be deleted (G).

---

## 3. Notebook 05 (`notebooks/05_eda_full.ipynb` v1 and `05_eda_full_v2.ipynb` v2)

### 3.1 Purpose and design

Replaces "H4 = re-run notebook 04 on the full series": a new notebook with the notebook-04 analyses plus analyses that need the long series. Written in the chat, tested on synthetic data with the same columns and size before delivery, delivered as `.ipynb` files. Cell 1 is needed by every cell; each cell states which other cells it needs. Cell 5 runs `build_log_features.py` through `subprocess` with `sys.executable` (replaces the broken Terminal-command cell of notebook 04, session 4 Section 3.2). All key numbers are collected in a dict `R` and saved as JSON by cell 15.

| Cell | Content |
|---|---|
| 1 | Setup, paths, input checks, versions |
| 2 | Raw full series: rows, steps, nulls, min/max, per-month and per-day coverage, automatic first/last day with stream coverage ≥ 0.9 (`COV_EDGE`) |
| 3 | Jul–Sep slice of the full series vs `series_julsep.parquet` |
| 4 | Analysis window (`WIN_START`, `WIN_END`; None = automatic) and slice → `data/processed/series_full.parquet` |
| 5 | `build_log_features.py` → `data/processed/series_full_features.parquet` |
| 6 | Validity rules, maintenance event window, DST gap, fill rule, suspicious zeros, split 70/15/15, `pairs(h)` helper |
| 7 | Monthly level, telemetry/log coverage (`n_gpu_equiv / gpus_alloc`), split statistics, daily-mean figure (01) |
| 8 | Weekly table; decomposition P = GPU-equivalents × W per GPU; largest 7-day level changes (02) |
| 9 | Daily profile with two permutation nulls (A: hours shuffled within day; B: circular shift per day, keeps autocorrelation); profile per month; weekday effect on detrended daily means (03) |
| 10 | Masked ACF via FFT (global mean, pairs where both valid), ACF of dP, PACF (Durbin–Levinson on the longest valid segment), persistence MAE/RMSE per split and horizon (04) |
| 11 | H1 five-minute dip per month |
| 12 | H2 ramp thresholds (absolute; v2 also relative), monthly ramp rates, stability table (05) |
| 13 | Largest one-minute jumps in train with context |
| 14 | H3 covariates: nulls, level correlation, correlation of past 5-min changes with future dP, (v2) partial correlation and incremental R², log/telemetry lead–lag |
| 15 | Summary JSON |

Pair-split rule (as in notebook 04): train = both ends < i_tr; val = start ≥ i_tr and end < i_va; test = start ≥ i_va; i_tr = int(0.7·n), i_va = int(0.85·n).

### 3.2 v1 (window 2021-02-09 .. 2021-09-30, automatic)

Run fully. Outputs: figures `reports/figures/eda_full/` (01–05), `reports/eda_full_summary.json`. Its window files (`series_full.parquet`, `series_full_features.parquet`) were **overwritten by v2**. Main v1 findings are kept in Sections 4, 6 and 8 where they differ.

### 3.3 v2 (window 2021-03-04 .. 2021-09-30; decisions of this session)

Changes: `WIN_START = date(2021, 3, 4)`; `dst_gap` flag; `Wpg` null when `n_gpu_equiv = 0` (v1 had NaN in March from one 0/0 minute); util/temp fill rule with `no_gpu` flag; unscheduled-outage listing; `is_in(full_days.to_list())` (v1 raised a polars DeprecationWarning); h = 1 persistence without mod-0 targets; relative ramp; partial correlation and incremental R². Outputs: figures `reports/figures/eda_full_v2/`, `reports/eda_full_summary_v2.json`, data files in `data/processed/` (window v2).

### 3.4 Output export tool `src/tools/nb_outputs_to_md.py` (new)

Reads a **saved** notebook and writes `<name>_outputs.md` (text outputs in code blocks, ANSI codes removed, errors marked, cells without output marked) plus `<name>_outputs_figures/*.png` (linked). Usage: `%run ../src/tools/nb_outputs_to_md.py 05_eda_full_v2.ipynb` in a notebook cell or `python src/tools/nb_outputs_to_md.py notebooks/05_eda_full_v2.ipynb` in Terminal; optional second argument = output path. Result of this session: `notebooks/05_eda_full_v2_outputs.md` (62 KB) and `notebooks/05_eda_full_v2_outputs_figures/`.

---

## 4. Facts about the raw full series (notebook 05 cell 2; `series_full_2021.parquet`)

- 394,560 rows, step 60 s, 12 columns, 2021-01-01 00:00 .. 2021-10-01 23:59 (file clock). Nulls: 63,780 in each metric column (minutes without any stream), none in counts.
- **January has no GPU data at all**; the first stream minutes are on 5 Feb (coverage 0.0056), 8 Feb 0.29, full coverage from **9 Feb**. Last full day 30 Sep; 1 Oct 0.46 (file clock ends 2021-10-01 11:01:58, see complete_data handoff).
- Power: min 0.4 W, **max 64.53 kW** (inside the 16 Sep maintenance window; the maximum over valid minutes is 49.59 kW). `n_gpu_equiv` max 429.0; `n_gpu_streams` max **540 > 448 GPUs** (stream overlaps at handover minutes: one job ending and another starting on the same GPU in the same minute; weights keep `n_gpu_equiv` ≤ 429). `gpus_alloc` max 429 (log), equal to the telemetry maximum.
- Days inside 9 Feb – 30 Sep with coverage < 0.9: the 8 maintenance days (2 Mar 0.21, 18 Mar 0.41, 15 Apr 0.35, 20 May 0.40, 17 Jun 0.50, 15 Jul 0.32, 10 Aug 0.53, 16 Sep 0.50) and **29 Mar (0.85, unscheduled outage)**.
- Per month (all minutes): Feb mean 1.89 kW (gpu_eq 18), Mar 16.07, Apr 19.78, May 26.10, Jun 33.26, Jul 34.28, Aug 27.88, Sep 27.30 kW.
- Slurm log (`build_log_features.py` output): 395,914 rows (all with end > start); 42 cluster-wide gaps, 9 long (≥ 6 h); maintenance days in the whole log: 2021-01-21, 03-02, 03-18, 04-15, 05-20, 06-17, 07-15, 08-10, 09-16.
- **Jul–Sep slice vs `series_julsep.parquet` (cell 3):** 132,480 rows each, no minute only in one; `n_gpu_streams` and `n_edge_streams` identical; small differences in weighted columns: power_W max 2.54 W, n_gpu_equiv 0.081, util_gpu 0.027, mem_used_sum 22.7, mem_free_sum 2,594.6; no null mismatch, no NaN. Likely explanation (**not verified**): only the weights of streams without interior minutes differ, because the global fallback median of `n_samples` is now computed over all streams of Feb–Sep instead of chunks 1–3. Negligible (2.5 W against about 30 kW). The validation in Section 2.5 (same stream set, differences ≈ 0) supports this explanation.

---

## 5. Time axis: confirmed New York wall clock; DST gap found

- Check cell (run after notebook 05 v1): `minute_local` of the full series equals the UTC rendering of `minute_epoch` (2021-01-01 00:00 ↔ epoch 1609459200), and equals `minute_local` of `series_julsep.parquet` for the same epochs. → `build_series_lowmem.py` uses the **same convention as the original pipeline**; no bug.
- This is the documented rule of `HANDOFF_week2_complete_data.md`, Section 5: GPU file timestamps are **America/New_York wall-clock times stored as epoch seconds**; the series lives in this file clock; Slurm log times are true UTC and are converted with `log_to_file_clock`. "minute_epoch" is therefore a naive wall-clock epoch, not true UTC.
- **New consequence found:** on 2021-03-14 (US DST start) the wall-clock hour **02:00–02:59 does not exist**. The telemetry has no samples there, so the series has 60 empty minutes (power set to 0 by the empty-minute rule because the cluster is up), while the log (converted to the file clock) shows jobs running across this hour (`gpus_alloc` up to 128). These 60 minutes are not real time; they are flagged `dst_gap` and excluded (v2). Persistence pairs across the gap (01:59 → 03:00) are treated as broken; negligible. The DST end (7 Nov) is outside the window.
- Supporting evidence that log and telemetry are on the same clock: changes of `gpus_alloc` and `n_gpu_equiv` correlate at lag 0 (0.53) and +1 min (0.66), nothing at |lag| ≥ 2 (a 4–5 h offset would show no lag-0 peak). Maintenance outages start 00:01–00:19 in the file clock (consistent with complete_data Section 5).
- Hours in notebook 05 (daily profile) are therefore New York local hours (label in the figure is correct).

---

## 6. Analysis window decision: 2021-03-04 .. 2021-09-30 (provisional, for Meeting 1)

**Decided by the student in this session.** Reasons (all from notebook 05 v1):
- **February is a ramp-up regime:** mean 1.91 kW with about 25 GPU-equivalents; after the 2 Mar maintenance the level jumps to about 15–17 kW (largest 7-day change of the whole series, +15.5 kW at 7 Mar).
- **Telemetry is incomplete early:** median `n_gpu_equiv / gpus_alloc` (valid minutes) Feb 0.88, Mar 0.93, Apr 0.96, May 0.99, Jun–Sep 1.00 → early power is under-counted (some allocated GPUs have no telemetry). March and April are kept although still below 1.0 (note for the Data section).
- With February, the train split was dominated by the low regime (v1: train mean 22.5 kW, std 12.2; val 30.6 / 8.9; test 27.7 / 5.8) and the absolute ramp thresholds produced monthly ramp rates near 0 in Feb–Apr and 0.06–0.14 in May–Sep.
- 4 Mar = the first day after the 2 Mar maintenance event window (1 Mar 23:00 – 3 Mar 23:59).

Result: **211 days, 303,840 minutes**. Differs from the proposal's "about 238 days" → Meeting 1 item. February can be described as the commissioning / ramp-up period in the Data section.

---

## 7. Validity rules and data preparation (notebook 05 v2, cell 6; provisional)

| Flag | Definition | Minutes (v2 window) |
|---|---|---|
| `outage` | from `build_log_features.py` (cluster-wide log gap without telemetry; power null) | 5,548 |
| `mwin` | maintenance event window: 23:00–23:59 of the day before + maintenance day + whole day after (session 4) | 20,580 |
| `dst_gap` | 2021-03-14 02:00–02:59 (non-existent wall-clock hour) | 60 |
| `susp_zero` | power = 0, outside `mwin` and `dst_gap`, `gpus_alloc` > 0 (telemetry gap while the log shows GPUs allocated); `EXCLUDE_SUSP_ZERO = True` | 49 (all on 29 Mar, 11:32–12:20) |
| `no_gpu` | not outage and `n_gpu_streams` = 0 (cluster up, no GPU stream; power set to 0 by the empty-minute rule) | 585 |
| `valid` | not outage, not `mwin`, not `dst_gap`, power not null, not `susp_zero` | **282,988 (93.14%)** |

- `build_log_features.py` on the v2 window: 303,840 minutes; outage 5,548; cluster up / no GPU job (set to 0) 585; `maint_day` 10,080 (7 days); 13 cluster-gap minutes with GPU telemetry kept (not outage); `gpus_alloc` max 429; run time 1 s.
- **Fill rule (decision 4):** `util_gpu_mean`, `util_mem_mean`, `temp_gpu_mean`, `temp_mem_mean` are null exactly on the 585 `no_gpu` minutes → filled with 0, flag `no_gpu` kept. After the fill: no util/temp nulls outside outages; all other covariates never null.
- Zero-power minutes: 585 (475 inside `mwin`, 110 outside: 60 = DST gap, 50 on 29 Mar of which 49 `susp_zero`). **One** zero minute stays valid (29 Mar, `gpus_alloc` = 0 in the log, right before the unscheduled outage) → train minimum 0.0 kW. Removed by the unscheduled-outage rule (decision 10, Section 10.1).
- **Unscheduled outage:** 2021-03-29 12:22–15:12 (163 min), not a maintenance day; telemetry was lost about 50 min earlier (11:32). 28 Mar has the lowest daily mean of the window (7.70 kW).
- `W per GPU` (`Wpg`) = power / `n_gpu_equiv`, null when `n_gpu_equiv` = 0.
- `mod5` = (minute_epoch // 60) mod 5.
- **Split (v2):** train 2021-03-04 00:00 .. < 2021-07-29 16:48; val .. < 2021-08-30 08:24; test .. 2021-09-30 23:59. Maintenance days: train 18 Mar, 15 Apr, 20 May, 17 Jun, 15 Jul; val 10 Aug; test 16 Sep.

---

## 8. Full-series EDA results (notebook 05 v2 unless stated; provisional)

### 8.1 Level and splits (cell 7, valid minutes)

| Month | valid min | outage min | zero min | mean kW | median | p01 | p99 | max | gpu_eq | alloc | eq/alloc (median) | W per GPU | util_gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021-03 (from 4 Mar) | 37,108 | 993 | 134 | 16.79 | 16.25 | 3.21 | 25.75 | 27.59 | 179.3 | 195.4 | 0.931 | 93.7 | 50.7 |
| 2021-04 | 40,260 | 933 | 0 | 19.85 | 19.74 | 11.62 | 29.29 | 31.20 | 220.5 | 234.5 | 0.959 | 90.6 | 45.3 |
| 2021-05 | 41,700 | 580 | 287 | 26.31 | 29.82 | 8.86 | 44.45 | 46.95 | 269.0 | 276.5 | 0.988 | 99.6 | 49.9 |
| 2021-06 | 40,260 | 726 | 0 | 33.83 | 32.53 | 24.05 | 47.05 | 49.59 | 352.3 | 352.7 | 1.000 | 96.1 | 49.9 |
| 2021-07 | 41,700 | 982 | 0 | 34.83 | 34.47 | 24.47 | 47.19 | 49.57 | 336.9 | 338.6 | 1.000 | 103.1 | 52.3 |
| 2021-08 | 41,700 | 660 | 116 | 28.52 | 28.94 | 13.14 | 44.88 | 48.31 | 350.9 | 351.2 | 1.000 | 81.3 | 42.3 |
| 2021-09 | 40,260 | 674 | 48 | 27.51 | 27.02 | 14.73 | 40.84 | 42.20 | 320.2 | 321.1 | 1.000 | 85.9 | 48.3 |

| Split | valid min | mean kW | std | min | median | max |
|---|---|---|---|---|---|---|
| train | 197,716 | 26.28 | 9.67 | 0.00 | 26.85 | 49.59 |
| val | 42,636 | 29.51 | 8.73 | 11.04 | 29.63 | 48.31 |
| test | 42,636 | 27.61 | 5.86 | 10.96 | 27.42 | 42.20 |

Daily means (197 days with ≥ 1,200 valid minutes): min 7.70 kW (28 Mar), max 46.05 kW (29 Jul); first 30 days 17.46, last 30 days 27.55 kW. Figure `01_series_full.png`.

### 8.2 Level shifts (cell 8)

- Variance of log daily mean power 0.137 (v1 with Feb: 0.791). Shares: **GPU count 0.956**, W per GPU 0.160, 2·cov −0.116 → the level is set by the number of active GPUs; more GPUs go with slightly lower power per GPU (negative covariance). v1 (with Feb): 0.877 / 0.031 / 0.092.
- Largest changes of the centred 7-day mean: **10–18 May** rise (+13.6 to +17.7 kW; weekly GPU-equivalents 115 → 289 → 344 in weeks W18–W20; started before the 20 May maintenance) and **8–9 Aug** drop (−13.0 / −13.2 kW; GPU count about constant, W per GPU about 103 → 67–77, util down; job-mix change). v1 also had 7 Mar +15.5 kW (end of the February regime).
- Weekly table (W06–W39) in the v1 output; queued GPU jobs vary strongly (7 to 340 per week).

### 8.3 Daily and weekly profiles (cell 9)

- 188 days with 24 full hours. Hour of day explains **0.0626** of within-day variance; null A mean 0.0053 (95th 0.0082), null B (circular shift) mean 0.0054 (95th 0.0128); p < 0.001 for both. The circular-shift null fixes the p-value issue of the earlier shuffle-only test.
- Profile (kW vs day mean, New York hours): minimum −1.18 at 09:00 (−0.89 to −1.18 from 06:00 to 09:00), maximum +0.74 at 17:00; peak-to-trough about 1.9 kW.
- Per month peak-to-trough: Mar 0.97, Apr 1.01, May 2.58, Jun 1.52, Jul 3.25, Aug 4.14, Sep 2.39 kW; minimum hour 6–9 in all months → the amplitude grows with the level (multiplicative effect, provisional).
- Weekly: 195 detrended days; weekday share 0.0491, null mean 0.0316, 95th 0.0639, **p = 0.152 (not significant)**; weekday means Mon −0.80, Tue +0.49, Wed +0.41, Thu +0.63, Fri +0.89, Sat −0.10, Sun −1.10 kW.
- v1 (with Feb): daily share 0.0581 (p < 0.001), weekly p = 0.109.

### 8.4 ACF, PACF, persistence (cell 10)

- ACF(P): lag 1 0.999, 5 0.997, 15 0.995, 60 0.985, 360 0.938, 720 0.894, 1440 0.847, 2880 0.754, 10080 0.511; **first lag below 0.5: 10,444 min (about 7.3 days)**. No peak at 1 day (ACF 0.847 = minimum over 600–1440) or 1 week → the ACF is driven by slow level changes, not by cycles. (v1 with Feb: never below 0.5 within 14 days.)
- ACF(dP): lag 1 **−0.247**, 2 −0.064, 3 −0.020, 4 −0.091, **5 +0.099**, 6 −0.052, **10 +0.100, 15 +0.110, 20 +0.094, 30 +0.112** (band ±0.004) → partial reversal of each change in the next minute, plus the five-minute periodicity (H1) in the correlation structure.
- PACF(P) on the longest valid segment (50,340 min, 12 Aug 00:00 – 15 Sep 22:59): 1 +0.998, 2 +0.125, 3 +0.089, 4 +0.058, 5 +0.068, 6 +0.002 (band ±0.009) → meaningful up to lag 5.

Persistence P(t+h) = P(t), valid pairs (kW):

| h | split | pairs | MAE | RMSE | p95 |dP| | mean P |
|---|---|---|---|---|---|---|
| 1 | train | 197,706 | 0.2650 | 0.4339 | 0.821 | 26.28 |
| 1 | val | 42,634 | 0.2864 | 0.5108 | 0.963 | 29.51 |
| 1 | test | 42,634 | 0.2407 | 0.3838 | 0.704 | 27.61 |
| 5 | train | 197,674 | 0.3526 | 0.6190 | 1.075 | 26.28 |
| 5 | val | 42,626 | 0.4300 | 0.8969 | 1.383 | 29.51 |
| 5 | test | 42,626 | 0.4000 | 0.6779 | 1.233 | 27.61 |
| 15 | train | 197,595 | 0.4845 | 0.8228 | 1.525 | 26.28 |
| 15 | val | 42,606 | 0.6051 | 1.1784 | 1.980 | 29.50 |
| 15 | test | 42,606 | 0.6221 | 0.9988 | 1.976 | 27.61 |
| 1 | train, no mod-0 target | 158,169 | 0.2568 | 0.4228 | 0.790 | 26.30 |
| 1 | val, no mod-0 target | 34,108 | 0.2710 | 0.4812 | 0.899 | 29.54 |
| 1 | test, no mod-0 target | 34,108 | 0.2351 | 0.3709 | 0.692 | 27.63 |

Persistence MAE is about 1–2% of the level; removing mod-0 targets lowers the h = 1 MAE by about 2–5%. Figure `04_acf.png`.

### 8.5 H1 — five-minute dip (cell 11): confirmed on the full series, CLOSED

282,976 consecutive valid minutes. Whole series: mod-0 mean dP **−0.138 kW** (dW per GPU −0.434 W), mod-1 **+0.133 kW** (+0.419 W), mod-2–4 +0.002 kW; mean |dP| 0.300 / 0.297 / 0.242 kW (n 56,589 / 56,598 / 169,789).

| Month | dP mod-0 (kW) | dP mod-1 | dP mod-2–4 | dW/GPU mod-0 (W) | dW/GPU mod-1 |
|---|---|---|---|---|---|
| Mar | −0.021 | +0.017 | +0.001 | −0.126 | +0.121 |
| Apr | −0.021 | +0.018 | +0.001 | −0.115 | +0.095 |
| May | −0.080 | +0.070 | +0.004 | −0.336 | +0.299 |
| Jun | −0.187 | +0.185 | +0.000 | −0.544 | +0.538 |
| Jul | −0.366 | +0.364 | +0.000 | −1.083 | +1.078 |
| Aug | −0.177 | +0.167 | +0.004 | −0.482 | +0.458 |
| Sep | −0.098 | +0.091 | +0.001 | −0.305 | +0.300 |

The dip exists in every month of the window (not in Feb, v1), is fully recovered in the next minute, lives in power per GPU (not in GPU count) and varies with the job mix (consistent with session 4: job-driven). Data not corrected (decision 3).

### 8.6 H2 — ramp thresholds (cell 12)

**Absolute definition** |P(t+h) − P(t)| (kW), train valid pairs; rates with train thresholds (expected 0.05 / 0.01):

| h | n_train | std | median | p90 | p95 | p99 | max | exkurt | share up | rate95 val | rate99 val | rate95 test | rate99 test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 197,706 | 0.434 | 0.164 | 0.603 | 0.821 | 1.499 | 10.83 | 41.0 | 0.492 | 0.071 | 0.017 | 0.036 | 0.008 |
| 5 | 197,674 | 0.619 | 0.220 | 0.767 | 1.075 | 2.157 | 17.82 | 74.4 | 0.522 | 0.077 | 0.024 | 0.066 | 0.015 |
| 15 | 197,595 | 0.823 | 0.297 | 1.076 | 1.525 | 2.930 | 17.55 | 38.0 | 0.557 | 0.077 | 0.024 | 0.084 | 0.021 |

h = 1 without pairs touching a mod-0 minute (n 118,625): p95 0.735, p99 1.391.

**Relative definition (new, decision 2):** |P(t+h) − P(t)| / L24(t), L24(t) = mean of valid P over the past 1,440 minutes ending at t (causal; ≥ 720 valid minutes required).

| h | n_train | p95 rel | p99 rel | p95 at 30 kW | p99 at 30 kW | rate95 val | rate99 val | rate95 test | rate99 test |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 193,392 | **0.0299** | 0.0577 | 0.90 kW | 1.73 kW | 0.044 | 0.010 | 0.039 | 0.010 |
| 5 | 193,360 | **0.0409** | 0.0834 | 1.23 kW | 2.50 kW | 0.056 | 0.018 | 0.068 | 0.017 |
| 15 | 193,281 | **0.0580** | 0.1155 | 1.74 kW | 3.46 kW | 0.064 | 0.020 | 0.085 | 0.022 |

Monthly ramp rate (p95, all splits, train thresholds):

| Month | abs h1 | abs h5 | abs h15 | rel h1 | rel h5 | rel h15 |
|---|---|---|---|---|---|---|
| Mar | 0.006 | 0.016 | 0.024 | 0.030 | 0.052 | 0.068 |
| Apr | 0.007 | 0.016 | 0.021 | 0.032 | 0.036 | 0.042 |
| May | 0.057 | 0.072 | 0.063 | 0.085 | 0.082 | 0.068 |
| Jun | 0.064 | 0.068 | 0.069 | 0.032 | 0.036 | 0.036 |
| Jul | 0.115 | 0.075 | 0.066 | 0.064 | 0.040 | 0.035 |
| Aug | 0.068 | 0.077 | 0.078 | 0.045 | 0.058 | 0.067 |
| Sep | 0.037 | 0.068 | 0.087 | 0.040 | 0.070 | 0.089 |

Stability across months (coefficient of variation; max/min):

| h | CV abs p95 | CV rel p95 | CV abs p99 | CV rel p99 | max/min abs p95 | max/min rel p95 |
|---|---|---|---|---|---|---|
| 1 | 0.750 | **0.440** | 0.751 | 0.778 | 19.5 | **2.8** |
| 5 | 0.493 | **0.334** | 0.598 | 0.521 | 5.0 | **2.3** |
| 15 | 0.447 | **0.353** | 0.598 | 0.561 | 4.3 | **2.5** |

Conclusions (provisional):
- The relative definition removes most of the level effect at p95 (max/min 19.5 → 2.8 at h = 1; val/test rates at h = 1 close to 0.05). **Proposed main definition: relative, p95.** Absolute kW and p99 kept as sensitivity analyses.
- At p99 the relative definition hardly helps, and the monthly relative p95 rate still varies 0.03–0.09 → **the frequency of ramps itself changes over time** (May and Sep high), independent of the level. A finding for the report; for RQ3, coverage must be reported separately for ramp and normal steps (as planned), and ramp counts reported with rates.
- Figure `05_ramp_thresholds.png` (absolute histograms).
- v1 (with Feb) absolute p95 0.751 / 1.006 / 1.430, p99 1.387 / 2.040 / 2.803 kW. Session 4 (Jul–Sep, boundaries excluded): p95 1.04 / 1.32 / 1.83 kW.

### 8.7 Largest one-minute jumps in train (cell 13)

Train h = 1 pairs: |dP| > 3 kW: 324; > 5 kW: 80; > 10 kW: 3. Of the 15 largest (|dP| 8.0–10.8 kW): **none has a change in GPU count** (`gpus_alloc` and `n_gpu_equiv` constant within ±1, except 22 May 07:16 where `n_gpu_equiv` rose 332.5 → 376 at constant `gpus_alloc` 376, i.e. telemetry of already-allocated GPUs appearing); all are changes of power per GPU (e.g. 93 → 67 W). They are clustered and back-and-forth: 6 on **18 May 18:51–21:16** (±8.5–10 kW), 12 Jun 15:13 (and 15:14 in v1), 15 Jun 03:22/03:26, 17 Jul 11:27/11:31, plus 20 Jul 22:28, 22 Jul 23:44, 29 Jul 00:19. Interpretation (**hypothesis**): a large multi-GPU job switching between phases. The largest ramps are transient excursions of power per GPU, not level changes; the Slurm log cannot announce them. (The turbulent post-maintenance jumps of session 4, e.g. 11 Aug, are now inside `mwin`.)

### 8.8 H3 — covariates (cell 14): DONE, promising (provisional)

Nulls after the fill rule: none in valid minutes for any covariate.

Train, valid minutes: level correlation with P and correlation of the past 5-min change of the covariate with the future change P(t+h) − P(t):

| Covariate | corr level P | h1 | h5 | h15 |
|---|---|---|---|---|
| util_gpu_mean | 0.254 | −0.262 | −0.291 | −0.245 |
| util_mem_mean | 0.482 | −0.270 | −0.289 | −0.243 |
| temp_gpu_mean | 0.230 | −0.167 | −0.200 | −0.180 |
| temp_mem_mean | 0.265 | −0.180 | −0.212 | −0.191 |
| mem_used_sum | 0.895 | +0.023 | +0.011 | +0.005 |
| mem_free_sum | 0.849 | +0.036 | +0.064 | +0.059 |
| n_gpu_equiv | 0.953 | +0.045 | +0.061 | +0.053 |
| gpus_alloc | 0.942 | +0.080 | +0.093 | +0.076 |
| active_gpu_jobs | 0.922 | +0.070 | +0.096 | +0.083 |
| active_jobs | 0.483 | +0.007 | +0.013 | +0.012 |
| queued_gpu_jobs | 0.299 | +0.020 | +0.050 | +0.049 |
| queued_jobs | −0.188 | +0.000 | −0.001 | +0.002 |
| P (own past change) | 1 | −0.252 | −0.289 | −0.242 |

**Partial correlation** of each covariate's past 5-min change with the future change, after removing the past of P (OLS on dP over 1 and 5 min, level P, mod-5 dummies of t and t + h), and **incremental R²** of all covariate changes together (train, in-sample):

| h | n | R² base | util_gpu | util_mem | temp_gpu | temp_mem | mem_used | mem_free | n_gpu_equiv | gpus_alloc | active_gpu_jobs | active_jobs | queued_gpu | queued | ΔR² all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 197,664 | 0.142 | −0.104 | −0.112 | −0.067 | −0.076 | +0.111 | +0.087 | +0.150 | **+0.181** | +0.154 | +0.018 | +0.036 | +0.002 | **+0.033** |
| 5 | 197,633 | 0.109 | −0.105 | −0.096 | −0.072 | −0.078 | +0.115 | +0.122 | +0.183 | **+0.205** | +0.191 | +0.026 | +0.070 | +0.001 | **+0.043** |
| 15 | 197,553 | 0.073 | −0.090 | −0.081 | −0.074 | −0.080 | +0.091 | +0.105 | +0.152 | **+0.167** | +0.160 | +0.023 | +0.065 | +0.004 | **+0.032** |

Lead–lag of 1-min changes, `gpus_alloc` (log) at t vs `n_gpu_equiv` (telemetry) at t + lag: lags −5..−1 ≈ 0 (−0.011 to +0.002), **lag 0 0.528, lag +1 0.657**, lags +2..+5 ≈ 0 (0.014 to −0.007). Median `n_gpu_equiv / gpus_alloc` (valid, alloc > 0) 0.997.

Conclusions (provisional; corrects the first reading in the chat):
- Covariates carry information **beyond the past of P**: ΔR² +0.033 / +0.043 / +0.032 on top of base R² 0.142 / 0.109 / 0.073 (about +23% / +39% / +44% relative). Strongest: changes of GPU counts (`gpus_alloc`, `active_gpu_jobs`, `n_gpu_equiv`, partial r 0.15–0.21), then memory (0.09–0.12), then util/temp (negative, −0.07 to −0.11). `active_jobs`, `queued_jobs` ≈ 0; `queued_gpu_jobs` small (0.04–0.07).
- Interpretation (**hypothesis**): newly allocated GPUs ramp up over the next minutes (warm-up), so a rise in allocation predicts a later rise in power; negative util/temp = mean reversion.
- Caveats: linear, in-sample on train (not yet checked on val); base R² is small (most of dP is unpredictable); the log leads telemetry by at most about 1 minute, so the value of the log is the **allocation change**, not an early warning.
- Consequence: exogenous covariates (GPU counts, memory, util) are worth testing in the models; to be confirmed on val.

---

## 9. Decisions in this session

1. **SKIP_AGG** in the Colab loop (v5); aggregation replaced by `build_series_lowmem.py` after all chunks.
2. `build_series_lowmem.py` accepted as the aggregation for the full series (validated to ≈ 1e-11 against the original on chunks 1–3).
3. Time axis: New York wall clock confirmed; the DST-gap hour 2021-03-14 02:00–02:59 is excluded (`dst_gap`).
4. **Analysis window 2021-03-04 .. 2021-09-30** (211 days, 303,840 min); February = ramp-up regime, excluded from modelling (provisional, Meeting 1).
5. **Ramp definition: relative (|dP| / L24), p95 as the main tier**; absolute kW and p99 as sensitivity (provisional, Meeting 1).
6. **Five-minute dip:** target unchanged; mod-5 feature given to models; h = 1 metrics also reported without mod-0 targets (confirms session 4 decision 2).
7. **util/temp on minutes without GPU streams:** fill 0 + flag `no_gpu`.
8. Suspicious zeros (telemetry gaps while GPUs are allocated) stay excluded.
9. EDA on the full series is complete (H1–H4); notebook 05 v2 is the reference EDA.
10. **Unscheduled outage window** (29 Mar 11:32–23:59) excluded like a maintenance event (Section 10.1; to be implemented in the modelling-dataset builder).

---

## 10. Pending decisions and open points

### 10.1 Unscheduled outage window (DECIDED by the student at the end of the session; not yet implemented)

Rule: an unscheduled cluster outage (log gap outside a maintenance event window) is excluded like a maintenance event, from the start of the telemetry loss before the gap to the end of that day. For the window this concerns only **29 Mar 2021: excluded from 11:32 to 23:59** (first suspicious zero 11:32, outage 12:22–15:12). This removes the one remaining valid zero-power minute (train min 0.0 kW) and about 10 h of valid minutes; effect on the Section 8 statistics expected to be negligible (numbers in Section 8 are **before** this rule). Generic form for the code: flag `uwin` = from the first `susp_zero` minute (or the outage start if there is none) on the day of an outage outside `mwin` until 23:59 of that day; `valid` excludes `uwin`. Implement in the modelling-dataset builder (Section 10.2), not in notebook 05.

### 10.2 Other open points

- Where the rules live: maintenance window, `dst_gap`, `susp_zero`, fill rule, unscheduled-outage window are implemented **only in notebook 05**. Session 4 planned to move the maintenance window into `build_log_features.py` after Meeting 1. Proposal: write one modelling-dataset builder (`src/data/build_model_dataset.py`) that applies all rules and adds features (lags, mod-5, hour sin/cos, covariate changes, L24, valid mask, split).
- The Jul–Sep difference explanation (global fallback median, Section 4) is not verified.
- Global fallback median value and run time of the lowmem full run not recorded.
- Telemetry coverage below 1.0 in March–April (0.93–0.96): early power slightly under-counted; mention in Data section.
- H3 results are in-sample; confirm on val in the modelling phase.
- Optional J (sub-minute shape of the dip, raw data on Colab) and I (PM100 schema check) unchanged.

---

## 11. Proposal change list (additions to session 3 Section 7 and session 4 Section 7; final numbers now available)

**Applied in this session** (Section 15.1), plus the processing-method subsection and five figures. Rule for later edits: edit the proposal only when the student asks; read the Persian tab first.
- **Section 3 Data:**
  - Whole MIT Supercloud GPU telemetry processed on Colab: 109,946 job files, 2,269.5 GiB raw, 0 failed; time-weighted aggregation to 1 minute; batched low-memory aggregation validated against the one-shot version.
  - Data period: GPU telemetry from 5 Feb 2021 (full coverage from 9 Feb) to 1 Oct 2021 11:01 (file clock); **analysis window 4 Mar – 30 Sep 2021 = 211 days, 303,840 minutes** (replace "about 330,000 steps, about 238 days"); February described as a ramp-up period (low load, incomplete telemetry) and excluded.
  - Valid minutes 282,988 (93.1%); outages 5,548 min; maintenance event windows 20,580 min (7 maintenance days in the window); DST gap 60 min; one unscheduled outage (29 Mar, 163 min).
  - Levels: monthly means 16.8 (Mar) to 34.8 kW (Jul); valid maximum 49.6 kW; the 64.5 kW raw peak lies inside the 16 Sep maintenance window (excluded). Up to 429 GPU-equivalents of 448 GPUs.
  - Time axis: New York wall clock (file clock); log converted from UTC; DST gap.
  - Telemetry vs log coverage: `n_gpu_equiv / gpus_alloc` 0.93 (Mar) → 1.00 (Jun–Sep).
- **Section 5 Protocol:**
  - Splits 70/15/15 by time on the new window (boundaries in Section 7); each split contains maintenance days.
  - **Ramp: relative threshold** |dP| / L24 at p95 (0.030 / 0.041 / 0.058 for h = 1 / 5 / 15) from train; absolute kW and p99 as sensitivity; ramp frequency varies over time → report counts and rates, coverage per ramp/normal step.
  - Persistence MAE as reference (h = 1 / 5 / 15 test: 0.241 / 0.400 / 0.622 kW); skill relative to persistence.
  - Five-minute dip: mod-5 feature allowed; h = 1 metrics also without mod-0 targets.
  - Zero-power minutes are now almost absent from valid data (one minute, removed by the rule of 10.1) → the sMAPE problem on zero minutes mostly disappears (note).
- **Section 4 Methods:** exogenous covariates (GPU counts from the log, memory, util) justified by H3 (ΔR² +0.03 to +0.04 beyond the past of P); seasonal naive weak (daily share 0.06, no weekly effect).
- **EDA summary section:** use figures from `reports/figures/eda_full_v2/` (01 series with splits and maintenance, 02 level decomposition, 03 daily/weekly profile, 04 ACF, 05 ramp thresholds).

---

## 12. Meeting 1 agenda (updates; items of sessions 3 and 4 still valid unless replaced)

1. **(new) Analysis window 4 Mar – 30 Sep 2021 (211 days)** instead of about 238 days; February ramp-up regime and incomplete telemetry; describe February in the Data section only?
2. **(replaces session 4 item 3) Ramp definition:** relative |dP| / L24, p95 main tier (stable across months at p95, val/test rates near 5% at h = 1); absolute and p99 as sensitivity; ramp frequency non-stationary (May, Sep high). Is L24 (causal 24 h mean) acceptable as the scale?
3. **(new) Covariates (H3):** allocation changes and memory/util add information beyond the past of P (in-sample ΔR² +0.03 to +0.04). Use them as exogenous inputs in all model families (classical, foundation with covariates if supported, hybrid)?
4. **(new) Time axis and DST gap:** series in New York wall clock; the non-existent hour of 14 Mar excluded. OK?
5. **(new) Unscheduled outage (29 Mar):** excluded from the telemetry loss (11:32) to the end of the day, like a maintenance event (decision 10, Section 10.1). OK?
6. **Maintenance event window** (session 4 item 1): unchanged proposal; now applied on the full series (20,580 min, 6.8%).
7. **Five-minute dip** (session 4 item 2): confirmed in every month Mar–Sep; proposal unchanged (mod-5 feature; report h = 1 with and without mod-0 targets).
8. **Level shift and split** (session 3 item 3, updated): with the new window split means 26.3 / 29.5 / 27.6 kW (std 9.7 / 8.7 / 5.9); level driven by GPU count (96% of log daily variance); normalisation choice still open (Min-Max on train vs differences / local scaling, e.g. by L24).
9. **Daily cycle / weekly** (session 3 item 6, updated): daily share 0.063 (p < 0.001 with the conservative null), amplitude grows with level; weekly p = 0.15 → hour features yes, weekday features no (provisional).
10. Session 3 items 4 (covariates on zero minutes → now fill 0 + flag), 5 (sMAPE on zero minutes → mostly moot, see Section 11), 7 (PM100 version) as before.

---

## 13. Remaining work for Week 2 (replaces session 4 Section 9)

| Step | Task | Status |
|---|---|---|
| A, B | Chunk outputs checked; Jul–Sep series on the Mac | done |
| C | Colab chunks 4–7 | **done** (Section 2) |
| C2 | Count streams / leftover `.tmp` | **done**: 109,946, 0 `.tmp`, all chunks missing 0 |
| — | Full series (low-memory aggregation, validated) | **done** (Section 2.5) |
| H1 | Five-minute dip | **done, confirmed on full series** (8.5) |
| H2 | Ramp thresholds | **done** (8.6; relative definition proposed) |
| H3 | Covariates | **done** (8.8) |
| H4 | EDA on the full series | **done** (notebook 05 v2) |
| — | Decision 10.1 (unscheduled outage window) | **decided**; implement in the modelling-dataset builder (Week 3) |
| D | Proposal update (both tabs; Section 11) | **done** (Section 15.1) |
| E | Weekly Report 2 (Claude Doc, English + Persian), due **Friday 9 October 2026** | **done** (Section 15.3); the student adds commit and W&B links |
| F | Meeting 1 packet (Section 12 + sessions 3/4 items) | **done** (Section 15.2) |
| G | Housekeeping (Drive clean-up, local obsolete files, notebook-04 cell fix, PM100 size, README and data_quality additions, scripts into the repo) | **instructions given** (Section 16); waiting for the student's confirmation |
| I | PM100 schema check on Colab | optional |
| J | Sub-minute shape of the 5-minute dip (raw data, Colab) | optional |

**Week 2 is complete when:** D, E and F exist (G recommended). **Reached in this session** for D, E, F; tell the student that Week 2 is complete once G is confirmed. Next phase (Week 3): modelling-dataset builder (Section 10.2), baselines (persistence, persistence + mod-5, linear/AR with and without covariates) on val.

---

## 14. Files created or changed in this session

| File | Where | Note |
|---|---|---|
| `build_series_lowmem.py` | repo `src/data/` (to add) and `MyDrive/dcpower/` | Appendix A |
| `colab_cell7_lowmem.py` | Colab cell (all-in-one) | validation + full run |
| Colab loop cell v5 | Colab notebook | Appendix B |
| `series_full_2021.parquet` | Drive and `data/processed/` | full series, 18.5 MiB |
| `notebooks/05_eda_full.ipynb` | repo | v1 (window from 9 Feb) |
| `notebooks/05_eda_full_v2.ipynb` | repo | v2, reference EDA |
| `data/processed/series_full.parquet` | repo data | v2 window slice, 303,840 rows |
| `data/processed/series_full_features.parquet` | repo data | v2 window with log features, 20 columns |
| `reports/figures/eda_full/`, `reports/eda_full_summary.json` | repo | v1 outputs |
| `reports/figures/eda_full_v2/`, `reports/eda_full_summary_v2.json` | repo | v2 outputs |
| `src/tools/nb_outputs_to_md.py` | repo | output export tool |
| `notebooks/05_eda_full_v2_outputs.md` + `_figures/` | repo | exported v2 outputs |
| `HANDOFF_week2_session5.md` | `docs/HANDOFF/` | this file |
| `README_full_series_section.md`, `data_quality_sections_15_22.md` | to append to `README.md` and `docs/data_quality.md` | Section 16 |
| Proposal, Meeting 1 packet, Weekly Report 2, explainer | Claude Docs | Section 15 |

Not verified in the chat: that `build_series_lowmem.py` and `05_eda_full.ipynb` (v1) were saved into the repo; ask the student.

---

---

## 15. Documents made in this session (Claude Docs)

All documents are Claude Docs owned by the student; the figures are uploaded into each document's own artifact (an image must be uploaded to every doc separately; a `$lid` of a blob does not resolve inside markdown, so blobs are created first and cited as `blob/<id>`).

### 15.1 Proposal (updated, both tabs)

Link `https://claude.ai/code/artifact/1c7f11e9-9c9b-40f6-ad87-6c447798161e` (doc id `4X6RRgndxG69SsM5wyeFvV`); English tab body `1031885e-f609` (rev 21 after this session), Persian tab body `df5361d7-61d0` (rev 10). Both tabs were read in full before editing; the Persian tab had no content beyond the English one.

Changes (same in both tabs):
- As-of date 2026-10-03 (Persian line: 11 Mehr 1405).
- Section 3 Data: main paragraph rewritten (109,946 files, about 2.2 TiB, time weighting, February ramp-up, window 4 Mar – 30 Sep 2021, 211 days, 303,840 minutes, covariates); new paragraph on validity (282,988 valid minutes = 93%, outages 5,548, maintenance windows 20,580, 29 Mar, DST hour, New York wall clock, monthly means 16.8–34.8 kW, max 49.6 kW, 64.5 kW raw peak inside a maintenance window); **Figure 1** (daily series); new subsection **"Processing 2.2 TiB of raw data without storing it"** (lead + 4 steps: index, per-file reduction on Colab, low-memory aggregation, local stage; "about 17 hours of Colab time" = sum of the seven chunk wall times, the aggregation runs are not included); PM100 now "second dataset", Zenodo 10127767, CC BY 4.0; new table of six EDA findings → consequences; **Figures 2–5** (level decomposition, daily/weekly profile, ACF, absolute ramp histograms; caption of Figure 5 says "absolute" thresholds because the main definition is now relative).
- Section 4 Methods: baselines row starts with persistence (reference for skill), persistence with a five-minute correction, daily seasonal naive; new sentence: models with exogenous inputs run with and without covariates.
- Section 5 Protocol: split with dates and the open scaling choice; relative ramp definition (0.030 / 0.041 / 0.058) with absolute and p99 as sensitivity; metrics add skill vs persistence and 1-minute metrics without dip minutes; uncertainty metrics add ramp counts; new rows "Invalid minutes" and "Five-minute dip".
- Section 9: six new open questions (window, relative ramp, maintenance/outage windows, mod-5 and hour features, covariates for all families, scaling).
- Sections 1, 2, 6, 7, 8 unchanged; no new dates.
- Numbers are before the 29-March rule (valid share drops by about 10 h when it is applied).

### 15.2 Meeting 1 packet (new)

Link `https://claude.ai/code/artifact/69a5bf15-6eb4-4cc0-8652-4d85f434fde3`; tabs English (body `6a0d064c-6e5b`) and فارسی (body `1128b277-4afc`). Content: lead (approval of proposal + ten decisions, link chip to the proposal); **Status of Sprint 1** table with a Status dropdown (enum `0d917354-0688`: Done / In progress / Open); **Decisions for approval** table with ten rows (window, ramp definition, maintenance window, unscheduled outage, time axis, five-minute dip, covariates, calendar features, scaling, second dataset), each with proposal, evidence, alternative and an empty **Outcome** dropdown (enum `e6d071e2-ab79`: Approved / Changed / Deferred) to fill in during the meeting; the seven open questions of the original proposal as a checklist; **Plan for Sprint 2** (six steps: modelling dataset, reference baselines, statistical/ML models, deep models, foundation models zero-shot, comparison with Diebold–Mariano).

### 15.3 Weekly Report 2 (new)

Link `https://claude.ai/code/artifact/cacd28a3-fb6f-425c-b39e-02af4438c68f`; tabs English (body `38f69c0d-4317`) and فارسی (body `9cb1b066-cda2`). Four-part template plus decisions: (1) Work done (table of eight items; "Commits: [add link] · W&B: [add link]" placeholders for the student); (2) Key results (window, Figure 1, table of seven numbers); (3) Blockers and how they were solved (five rows: data volume / VPN, wrong time conversion, double counting, Colab OOM, DST hour); (4) Decisions (seven, provisional); (5) Plan for Week 3 (checklist) and what is needed from the supervisor. Due Friday 9 October 2026.

### 15.4 Plain-language explainer (new, Persian only)

Link `https://claude.ai/code/artifact/104fade2-d69c-4183-b8f9-30c558a86958`, title «پروژه به زبان ساده» (body `88c42fda-8e2e`). For a reader without technical background: (1) the problem (data centers, power swings, battery); (2) basic concepts table (GPU, kW, time series, forecast and horizon, ramp, persistence, MAE, train/validation/test); (3) where the data came from (MIT Supercloud, telemetry, Slurm log, 2.2 TiB); (4) what we did, six steps with analogies (library catalogue, summaries in the cloud, counting votes per polling station, shift handover double counting, cleaning, log information); (5) six findings with Figures 1–3; (6) five problems and solutions; (7) next steps. One explanation was softened to "probably" (why there is no weekly cycle).

---

## 16. Housekeeping (G): instructions given, to be confirmed

Two files were delivered for appending: `README_full_series_section.md` (new README section "Full series (Week 2 result)": chunk table, low-memory aggregation with command line, outputs, tools) and `data_quality_sections_15_22.md` (new `docs/data_quality.md` Sections 15–22: February ramp-up and coverage, analysis window, maintenance event window, unscheduled outage, time axis and DST gap, zero minutes and fill rule, stream overlaps and raw maximum, full series vs Jul–Sep). The student appends them with `cat >>` in Terminal.

Steps given to the student (Persian, in the chat):
1. **Google Drive:** delete `MyDrive/dcpower/streams_test/`; delete `series_after_chunk_1.parquet` and `series_after_chunk_2.parquet`; keep `series_after_chunk_3.parquet` (validation reference), `streams_full/`, `series_full_2021.parquet`, `chunk_*.parquet`, `all_files_desc.parquet`, the two scripts. Delete the Colab runtime.
2. **Scripts into the repo:** `build_series_lowmem.py` → `src/data/`; `nb_outputs_to_md.py` → `src/tools/`; check `notebooks/05_eda_full.ipynb` and `05_eda_full_v2.ipynb` are there.
3. **Obsolete local files to the macOS Trash** (moved, not deleted, so recoverable): `data/raw/gpu_samples/`, `data/interim/minute_streams_test/`, `data/processed/series_test.parquet`, `data/processed/series_dev_test.parquet`, Claude Code backups (`data/interim/04_eda_jul_sep_backup_before_cell7.ipynb` and, if they exist, `..._before_cell8.ipynb`, `series_julsep_backup_before_cell8.parquet`, `reports/figures/eda_julsep_backup_before_cell8/`). Kept on purpose: `series_dev_v1_unweighted.parquet` (comparison) and `src/data/build_series.py.bak_unweighted` (only copy of v2).
4. **Notebook 04:** change the code cell at index 1 (Terminal command) to Raw.
5. **PM100 size:** find mentions with `grep` and correct "about 101 MB" (record 8129258) to 273.9 MiB, record 10127767.
6. **README and data_quality:** append the two files.
7. Git and `.gitignore`: left to the student (no Git instructions).

Results not yet reported. Next session: ask for the outputs of the `ls`/`grep` checks.

---

## 17. Next session (start of Week 3)

1. Confirm G (Section 16).
2. After Meeting 1: write the minutes in `docs/meetings/` the same day (date and attendees, presented, feedback, decisions, next steps with deadlines); record the Outcome column of the packet; update proposal and decisions accordingly.
3. Modelling-dataset builder `src/data/build_model_dataset.py`: apply `outage`, `mwin`, `uwin` (29 Mar 11:32–23:59), `dst_gap`, `susp_zero`; fill rule with `no_gpu`; features (lags, mod-5, hour sin/cos, covariate changes, L24); split; valid mask.
4. Shared evaluation module and reference baselines on validation (persistence, persistence + mod-5, daily seasonal naive, linear AR with and without covariates); confirm the H3 gain out of sample.


---

## Appendix A — `build_series_lowmem.py` (as run on Colab)

```python
#!/usr/bin/env python
"""build_series_lowmem.py: low-memory, resumable version of build_series() (time-weighted, v3 rules).

Reads the stream Parquets in batches, computes per-minute partial sums per batch, writes them to
--work-dir, and adds them up at the end. Only per-stream quantities are needed for the weights, so
the result equals the one-shot aggregation (up to float summation order).

Rules (same as build_series.py v3):
  stream = (file, gpu_index); m0/m1 = first/last minute; interior = strictly between.
  full = median n_samples over the stream's interior minutes; fallback = global median of interior minutes.
  w = 1 (interior), min(n_samples / full, 1) (first/last minute).
  power_W = sum(power*w); util/temp = weighted means (nulls excluded); mem_* = weighted sums;
  n_gpu_equiv = sum(w); n_gpu_streams = distinct streams; n_edge_streams = non-interior stream-minutes.
  complete minute grid [t0, t1); empty minutes: counts 0, n_gpu_equiv 0.0, metrics null.

Streams without interior minutes need the global median, so their rows are set aside per batch and
processed at the end (after the global median is known from the merged n_samples histogram).
"""
import argparse, calendar, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import polars as pl

COLS = ["file", "gpu_index", "minute_epoch", "power_W", "util_gpu", "util_mem", "temp_gpu", "temp_mem",
        "mem_used", "mem_free", "n_samples"]
MEANS = [("util_gpu", "util_gpu_mean"), ("util_mem", "util_mem_mean"),
         ("temp_gpu", "temp_gpu_mean"), ("temp_mem", "temp_mem_mean")]
KEY = ["file", "gpu_index"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ep(d):
    return calendar.timegm(datetime.fromisoformat(d).timetuple())


def read_one(p):
    df = pl.read_parquet(p, columns=COLS)
    return df.with_columns([pl.col(c).cast(pl.Float64) for c in COLS[3:]] +
                           [pl.col("gpu_index").cast(pl.Int64), pl.col("minute_epoch").cast(pl.Int64),
                            pl.col("file").cast(pl.Utf8)])


def add_interior(df):
    return (df.with_columns(pl.col("minute_epoch").min().over(KEY).alias("_m0"),
                            pl.col("minute_epoch").max().over(KEY).alias("_m1"))
              .with_columns(((pl.col("minute_epoch") > pl.col("_m0")) &
                             (pl.col("minute_epoch") < pl.col("_m1"))).alias("_int")))


def partial(df, t0, t1):
    """df has _int and _full (non-null). Returns per-minute partial sums inside [t0, t1)."""
    df = (df.with_columns(pl.when(pl.col("_int")).then(1.0)
                            .otherwise(pl.min_horizontal(pl.col("n_samples") / pl.col("_full"), pl.lit(1.0)))
                            .alias("_w"))
            .filter((pl.col("minute_epoch") >= t0) & (pl.col("minute_epoch") < t1)))
    aggs = [(pl.col("power_W") * pl.col("_w")).sum().alias("s_power"),
            (pl.col("mem_used") * pl.col("_w")).sum().alias("s_mem_used"),
            (pl.col("mem_free") * pl.col("_w")).sum().alias("s_mem_free"),
            pl.col("_w").sum().alias("s_w"),
            pl.len().cast(pl.Int64).alias("s_streams"),
            (~pl.col("_int")).cast(pl.Int64).sum().alias("s_edge"),
            pl.col("power_W").is_not_null().cast(pl.Int64).sum().alias("s_nn_power"),
            pl.col("mem_used").is_not_null().cast(pl.Int64).sum().alias("s_nn_mem_used"),
            pl.col("mem_free").is_not_null().cast(pl.Int64).sum().alias("s_nn_mem_free")]
    for c, _ in MEANS:
        aggs += [(pl.col(c) * pl.col("_w")).sum().alias(f"n_{c}"),
                 pl.when(pl.col(c).is_not_null()).then(pl.col("_w")).otherwise(0.0).sum().alias(f"d_{c}")]
    return df.group_by("minute_epoch").agg(aggs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", required=True, help="local folder for partials (resumable; one per file set)")
    ap.add_argument("--win-start", required=True)
    ap.add_argument("--win-end", required=True)
    ap.add_argument("--files", nargs="*", default=None,
                    help="optional Parquet(s) with column 'name' (e.g. chunk_1.parquet); restricts the streams")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--readers", type=int, default=16)
    a = ap.parse_args()

    t0, t1 = ep(a.win_start), ep(a.win_end)
    os.makedirs(a.work_dir, exist_ok=True)
    T = time.time()

    log(f"listing {a.stream_dir} ...")
    paths = sorted(e.path for e in os.scandir(a.stream_dir) if e.name.endswith(".parquet"))
    log(f"stream parquets in folder: {len(paths):,}")
    if a.files:
        want = set()
        for f in a.files:
            want |= {Path(n).stem for n in pl.read_parquet(f, columns=["name"])["name"].to_list()}
        have = {Path(p).name[:-len(".parquet")]: p for p in paths}
        miss = want - have.keys()
        paths = sorted(have[s] for s in want if s in have)
        log(f"restricted to --files: {len(want):,} expected | {len(paths):,} found | missing {len(miss):,}")
        if miss:
            log("ABORT: missing streams for the requested files"); sys.exit(2)

    batches = [paths[i:i + a.batch] for i in range(0, len(paths), a.batch)]
    log(f"batches: {len(batches)} x up to {a.batch:,} files | work dir {a.work_dir}")

    for b, bp in enumerate(batches):
        part_f = f"{a.work_dir}/part_{b:04d}.parquet"
        hist_f = f"{a.work_dir}/hist_{b:04d}.parquet"
        noint_f = f"{a.work_dir}/noint_{b:04d}.parquet"
        if os.path.exists(part_f) and os.path.exists(hist_f) and os.path.exists(noint_f):
            log(f"batch {b + 1}/{len(batches)}: done earlier, skipped")
            continue
        tb = time.time()
        with ThreadPoolExecutor(a.readers) as ex:
            df = pl.concat(list(ex.map(read_one, bp)), how="vertical_relaxed")
        rows = df.height
        df = add_interior(df)
        df = df.with_columns(pl.col("n_samples").filter(pl.col("_int")).median().over(KEY).alias("_full"))
        hist = (df.filter(pl.col("_int")).group_by("n_samples").agg(pl.len().cast(pl.Int64).alias("cnt")))
        noint = df.filter(pl.col("_full").is_null()).select(COLS + ["_int"])
        part = partial(df.filter(pl.col("_full").is_not_null()), t0, t1)
        del df
        noint.write_parquet(noint_f + ".tmp"); os.replace(noint_f + ".tmp", noint_f)
        hist.write_parquet(hist_f + ".tmp"); os.replace(hist_f + ".tmp", hist_f)
        part.write_parquet(part_f + ".tmp"); os.replace(part_f + ".tmp", part_f)   # part last = batch complete
        log(f"batch {b + 1}/{len(batches)}: {len(bp):,} files | {rows:,} stream-minutes | "
            f"{noint.height:,} rows without interior | {time.time() - tb:.0f} s | total {(time.time() - T) / 60:.1f} min")

    # global fallback median from the merged histogram (polars median: mean of the two middle values)
    hist = (pl.concat([pl.read_parquet(f"{a.work_dir}/hist_{b:04d}.parquet") for b in range(len(batches))])
              .group_by("n_samples").agg(pl.col("cnt").sum()).sort("n_samples"))
    n = int(hist["cnt"].sum())
    cum = hist["cnt"].cum_sum().to_list(); vals = hist["n_samples"].to_list()
    def kth(k):  # 0-based
        for v, c in zip(vals, cum):
            if k < c:
                return v
    gmed = kth((n - 1) // 2) if n % 2 else (kth(n // 2 - 1) + kth(n // 2)) / 2
    log(f"global median n_samples over {n:,} interior stream-minutes: {gmed}")

    noint = pl.concat([pl.read_parquet(f"{a.work_dir}/noint_{b:04d}.parquet") for b in range(len(batches))],
                      how="vertical_relaxed")
    n_noint_streams = noint.select(KEY).unique().height
    log(f"streams without interior minutes: {n_noint_streams:,} ({noint.height:,} rows) -> fallback median")
    log("adding up the partials (incrementally, to keep RAM low) ...")
    s = None
    def acc(s, p):
        return p if s is None else pl.concat([s, p], how="vertical_relaxed").group_by("minute_epoch").agg(pl.all().sum())
    for b in range(len(batches)):
        s = acc(s, pl.read_parquet(f"{a.work_dir}/part_{b:04d}.parquet"))
    if noint.height:
        s = acc(s, partial(noint.with_columns(pl.lit(float(gmed)).alias("_full")), t0, t1))
    ser = s.select(
        "minute_epoch",
        pl.col("s_power").alias("power_W"),
        *[pl.when(pl.col(f"d_{c}") > 0).then(pl.col(f"n_{c}") / pl.col(f"d_{c}")).otherwise(None).alias(o)
          for c, o in MEANS],
        pl.col("s_mem_used").alias("mem_used_sum"),
        pl.col("s_mem_free").alias("mem_free_sum"),
        pl.col("s_w").alias("n_gpu_equiv"),
        pl.col("s_streams").alias("n_gpu_streams"),
        pl.col("s_edge").alias("n_edge_streams"))
    grid = pl.DataFrame({"minute_epoch": pl.int_range(t0, t1, 60, eager=True).cast(pl.Int64)})
    ser = (grid.join(ser, on="minute_epoch", how="left")
               .with_columns(pl.col("n_gpu_equiv").fill_null(0.0),
                             pl.col("n_gpu_streams").fill_null(0),
                             pl.col("n_edge_streams").fill_null(0),
                             pl.from_epoch("minute_epoch", time_unit="s").alias("minute_local"))
               .sort("minute_epoch"))
    ser.write_parquet(a.out)
    empty = ser.filter(pl.col("n_gpu_streams") == 0).height
    log(f"series rows {ser.height:,}; minutes with no stream {empty:,}; "
        f"power_W min {ser['power_W'].min()} / max {ser['power_W'].max()}")
    log(f"written {a.out} | total {(time.time() - T) / 60:.1f} min")


if __name__ == "__main__":
    main()
```

## Appendix B — Colab loop cell v5 (as run for chunks 6 and 7)

```python
# Colab cell 6 (v5): pre-check of finished chunks, run queued chunks back to back,
# optional stop before the series aggregation (it is OOM-killed on Colab), post-check per chunk.
import os, re, sys, shutil, threading, subprocess, psutil, time, calendar
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import polars as pl

BASE = "/content/drive/MyDrive/dcpower"
SD = f"{BASE}/streams_full"
CHUNKS = [6, 7]          # 6 = Apr, 7 = Jan-Mar
SKIP_AGG = True          # True: stop build_series.py once all files are handled (no series; exit code -15 expected)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

bounds = ["2021-10-02", "2021-09-01", "2021-08-01", "2021-07-01", "2021-06-01",
          "2021-05-01", "2021-04-01", "2021-01-01"]
ep = lambda d: calendar.timegm(datetime.fromisoformat(d).timetuple())
prog = re.compile(r"files ([\d,]+)/([\d,]+)\s+done=([\d,]+) skipped=([\d,]+) failed=([\d,]+)\s+"
                  r"([\d.]+)/([\d.]+) GiB\s+([\d.]+) MiB/s")

log("reading all_files_desc.parquet ...")
full = pl.read_parquet(f"{BASE}/all_files_desc.parquet")
log(f"all_files_desc: {full.height:,} files")

def chunk_part(k):
    hi, lo = ep(bounds[k - 1]), ep(bounds[k])
    return full.filter((pl.col("end_fc") >= lo) & (pl.col("end_fc") < hi))

def list_streams():
    t = time.time(); stems, tmp, n = set(), 0, 0
    with os.scandir(SD) as it:
        for e in it:
            n += 1
            if n % 5000 == 0:
                log(f"  listed {n:,} entries ({time.time() - t:.0f} s)")
            if e.name.endswith(".parquet"):
                stems.add(e.name[:-len(".parquet")])
            elif e.name.endswith(".tmp"):
                tmp += 1
    log(f"listing done: {n:,} entries | {len(stems):,} stream parquets | leftover .tmp: {tmp} | {time.time() - t:.0f} s")
    if os.path.exists(f"{SD}/_failures.csv"):
        log("NOTE: _failures.csv exists (may be stale from an earlier run)")
    return stems

def check_chunk(k, stems):
    exp = {Path(n).stem for n in chunk_part(k)["name"].to_list()}
    miss = len(exp - stems)
    log(f"chunk {k}: expected {len(exp):,} | missing {miss:,}")
    if miss == len(exp) and miss > 0:
        log(f"  WARNING: nothing of chunk {k} found -> naming rule may differ; check by hand before trusting this")
    return miss

# ---- pre-check of all chunks before the queue ----
os.makedirs(SD, exist_ok=True)
log("listing streams_full ...")
stems = list_streams()
bad = [k for k in range(1, min(CHUNKS)) if check_chunk(k, stems) > 0]
log(f"queue: {CHUNKS}" + (f" | WARNING: incomplete earlier chunks {bad} -> stop and re-run them first" if bad else ""))

def run_chunk(k):
    part = chunk_part(k)
    part_path = f"{BASE}/chunk_{k}.parquet"
    part.select("sub", "name", "size").write_parquet(part_path)
    print(f"\n=== chunk {k}: jobs ending {bounds[k]} .. {bounds[k - 1]} | "
          f"{part.height:,} files | {part['size'].sum() / 2**30:,.1f} GiB | start {time.strftime('%H:%M')}", flush=True)
    os.makedirs("/content/tmp_raw", exist_ok=True)
    out_local = f"/content/series_after_chunk_{k}.parquet"
    cmd = ["python", "-u", f"{BASE}/build_series.py", "--files", part_path,
           "--stream-dir", SD, "--tmp-dir", "/content/tmp_raw", "--out", out_local,
           "--win-start", "2021-01-01", "--win-end", "2021-10-02", "--workers", "16"]
    log(f"chunk {k}: starting build_series.py (the skip check of existing streams can take a few minutes)")
    state = {"files": "0/?", "failed": "0", "rate": "0", "running": True, "agg_note": False, "stopped": False}
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
                if SKIP_AGG:
                    bar.write(f"[{time.strftime('%H:%M:%S')}] chunk {k}: all files handled; stopping before aggregation (SKIP_AGG)")
                    time.sleep(10)
                    state["stopped"] = True
                    proc.terminate()
                    break
                bar.write(f"[{time.strftime('%H:%M:%S')}] chunk {k}: all files handled; aggregating the series (no progress output, can take 30+ min)")
        else:
            bar.write(f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}")
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    state["running"] = False; time.sleep(2.5); bar.close()
    if os.path.exists(out_local) and not state["stopped"]:
        dst = f"{BASE}/series_after_chunk_{k}.parquet"
        shutil.copy2(out_local, dst)
        log(f"series copied to Drive: {dst} | sizes equal: {os.path.getsize(out_local) == os.path.getsize(dst)}")
    print(f"chunk {k} | exit code: {proc.returncode} | stopped before aggregation: {state['stopped']} | "
          f"failed: {state['failed']} | wall time: {(time.time() - t) / 3600:.2f} h | end {time.strftime('%H:%M')}", flush=True)
    log(f"post-check chunk {k}: listing streams_full ...")
    check_chunk(k, list_streams())

for k in CHUNKS:
    run_chunk(k)
log("ALL QUEUED CHUNKS FINISHED")
```

## Appendix C — Colab cell 7 "all in one" (script text shortened)

```python
# Colab cell 7 (all in one): write build_series_lowmem.py to Drive, validate on chunks 1-3,
# and if the check passes, build the full series and copy it to Drive.
import subprocess, os, shutil, polars as pl
BASE = "/content/drive/MyDrive/dcpower"
RUN_FULL = True   # False: only the validation

SCRIPT_TEXT = r'''<full text of build_series_lowmem.py, Appendix A>'''
with open(f"{BASE}/build_series_lowmem.py", "w") as fh:
    fh.write(SCRIPT_TEXT)
print("script written:", os.path.getsize(f"{BASE}/build_series_lowmem.py"), "bytes")
print("reference series exists:", os.path.exists(f"{BASE}/series_after_chunk_3.parquet"))

def run(args):
    cmd = ["python", "-u", f"{BASE}/build_series_lowmem.py", "--stream-dir", f"{BASE}/streams_full",
           "--win-start", "2021-01-01", "--win-end", "2021-10-02", "--batch", "2000"] + args
    return subprocess.run(cmd).returncode

# 1) validation on chunks 1-3
rc = run(["--files", f"{BASE}/chunk_1.parquet", f"{BASE}/chunk_2.parquet", f"{BASE}/chunk_3.parquet",
          "--work-dir", "/content/agg_c123", "--out", "/content/series_lowmem_c123.parquet"])
print("validation exit code:", rc)
old = pl.read_parquet(f"{BASE}/series_after_chunk_3.parquet")
new = pl.read_parquet("/content/series_lowmem_c123.parquet")
j = old.join(new, on="minute_epoch", how="full", suffix="_new", coalesce=True)
only_one = j.filter(pl.col("minute_local").is_null() | pl.col("minute_local_new").is_null()).height
print("rows old/new:", old.height, new.height, "| minutes only in one of them:", only_one)
passed = rc == 0 and old.height == new.height and only_one == 0
for c in old.columns:
    if c in ("minute_epoch", "minute_local"): continue
    a, b = j[c].cast(pl.Float64), j[c + "_new"].cast(pl.Float64)
    d, nm = (a - b).abs().max(), int((a.is_null() != b.is_null()).sum())
    print(f"{c:16s} max abs diff {d} | null mismatch {nm} | NaN old {a.is_nan().sum()} new {b.is_nan().sum()}")
    if c in ("power_W", "n_gpu_equiv", "n_gpu_streams", "n_edge_streams") and ((d or 0) > 1e-3 or nm > 0):
        passed = False
print("VALIDATION PASSED" if passed else "VALIDATION FAILED -> full run skipped; send the output")

# 2) full series (all 109,946 streams)
if passed and RUN_FULL:
    out = "/content/series_full_2021.parquet"
    rc = run(["--work-dir", "/content/agg_all", "--out", out])
    print("full run exit code:", rc)
    if rc == 0:
        dst = f"{BASE}/series_full_2021.parquet"
        shutil.copy2(out, dst)
        print("copied to Drive:", dst, "| sizes equal:", os.path.getsize(out) == os.path.getsize(dst),
              f"| {os.path.getsize(dst) / 2**20:.1f} MiB")
```

## Appendix D — notebook 05

The full code is in `notebooks/05_eda_full_v2.ipynb` (16 cells: a Markdown header with the change list and cells 1–15, Section 3.1). The time-zone check cell run after v1 (Section 5) is not part of the notebook; it compared `minute_local` with `datetime.fromtimestamp(epoch, tz=timezone.utc)`, joined `series_julsep.parquet` on `minute_epoch`, printed the first log rows, and listed first/last outage and suspicious-zero minutes per day.

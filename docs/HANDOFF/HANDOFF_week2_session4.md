# HANDOFF — Week 2, session 4 (Friday 2 October 2026, evening/night)

Bachelor project: forecasting AI data center GPU power (MIT Supercloud), classical / foundation / hybrid quantum models, conformal intervals, battery MPC.
This file continues `HANDOFF_week2_session3.md` (read it first; it continues `HANDOFF_week2_complete_data.md`). This file records only what happened in session 4 and the updated plan. Sections of session 3 that are not repeated here are still valid.

**WEEK 2 IS NOT COMPLETE.** Open: Colab chunks 4–7 (4 and 5 running overnight), H4 (re-run EDA on the full series), the proposal update, Weekly Report 2 (due Friday 9 October 2026) and the Meeting 1 packet. The Jul–Sep EDA is closed (H1, H2 done; H3 deliberately moved to the full series).

---

## 0. Working setup changed again: Claude Code is dropped

- **From now on there are NO Claude Code tasks.** Decided by the student in this session after Claude Code stopped a task (rule conflict about overwriting files in `data/`, Section 3.2).
- **Chat (claude.ai Project)** = thinking, interpreting, deciding, and writing **ready-to-run notebook cells and Terminal commands** for the student. The student runs them in JupyterLab (kernel "Python (dcpower)") or Terminal and pastes the output.
- **Colab** = heavy raw-data work, cells pasted by the student (unchanged).
- `PROMPT_claude_code.md` is obsolete. `PROMPT_chat.md` still applies except its parts about Claude Code (task format, "Claude Code only executes"). Rules that still hold: answer in Persian; code, notebooks and repo docs in English; one step per message (cheap read-only checks may be grouped); whole corrected cells, not line edits; Terminal commands one per block, no `#` comments; no Git instructions; never download raw GPU files on the Mac; cautious claims, provisional labels for Jul–Sep numbers.
- Cells for notebook 04 are written so that they either stand alone (read Parquet files) or say exactly which earlier cell's variables they need.

---

## 1. Environment (unchanged)

| Item | Value |
|---|---|
| Mac project root | `~/Desktop/Other/datacenter-power-forecasting/` (note: `HANDOFF_week1_complete.md` still says `~/Desktop/datacenter-power-forecasting/`; that is outdated) |
| Handoff files in the repo | `docs/HANDOFF/` (uppercase), including `Project_Context_FA.txt` |
| Python | `/opt/miniconda3/envs/dcpower/bin/python`; kernelspecs: `dcpower` (user-level, use this) and `python3` (inside the env) |
| Notebooks | `notebooks/` (working directory; relative paths `../data/...`) |
| Colab Drive folder | `MyDrive/dcpower/` |

---

## 2. Colab whole-dataset run

### 2.1 Chunk 4 started (running overnight, chunks 4 and 5 queued)

The loop cell that actually ran was still the **v3 version with the pre-check** (not v4). This time the listing of `streams_full/` took only **25 s** (earlier: about 10 min), so the pre-check finished:

```
listing done: 65,714 entries | 65,714 stream parquets | leftover .tmp: 0 | 25 s
chunk 1: expected 26,479 | missing 0
chunk 2: expected 22,710 | missing 0
chunk 3: expected 16,525 | missing 0
queue: [4, 5]
=== chunk 4: jobs ending 2021-06-01 .. 2021-07-01 | 11,881 files | 367.6 GiB | start 22:32
files to handle: 11,881 (367.61 GiB), workers: 16
still to download: 367.61 GiB
14% | 51.6/367.6 GiB [23:25 elapsed < 2:23:21] | files 1,980/11,881 | failed 0 | 37.58 MiB/s | CPU 100% | RAM 1.7/13.6 GB | disk free 91 GB
```

(Times are the Colab clock.)

- **C2 is done for chunks 1–3:** 65,714 stream Parquets = 26,479 + 22,710 + 16,525, no leftover `.tmp`, none missing.
- "still to download" = chunk size → nothing from earlier chunks is re-downloaded.
- Rate 37.6 MiB/s, a bit below the earlier about 43 MiB/s (README value "about 43 MiB/s" still fine as an average; check after the run).
- Estimate from start: chunk 4 about 2.7 h + about 40 min aggregation; chunk 5 about 2 h + aggregation (longer than before, more streams). Total about 6 h.
- Aggregation after every chunk (session 3, Section 2.4): kept as is for now (option a); watch RAM during aggregation.

### 2.2 Overnight setup on the Mac

Mac awake with `caffeinate -dis` in Terminal (stop with Ctrl + C in the morning), lid open, internal brightness at 0, external monitor may be switched off with its own power button. Do **not** use clamshell mode with the external monitor switched off (Mac may sleep; not tested). Optimized Battery Charging recommended if the Mac stays on power for days.

### 2.3 What to check in the morning

For each of chunks 4 and 5: `exit code`, `failed`, `wall time`, `series copied ... sizes equal: True`, and the build_series line `series rows; minutes with no stream; power_W min/max`.

Expected `minutes with no stream` (whole window 2021-01-01..10-02, 394,560 rows): after chunk 4 about **202k** (244,654 minus about 43,200 June minutes, plus the 17 June outage); after chunk 5 about **158k**. Series rows must stay 394,560.

If the run stopped: run Colab cell 1 (mount), then the loop cell with the interrupted chunk number; finished streams are skipped.

Next sessions: `CHUNKS = [6, 7]`. The cell in the notebook may still be v3; v3 works (pre-check now fast). Appendix A of session 3 (v4) is the alternative.

---

## 3. Notebook 04 (`notebooks/04_eda_jul_sep.ipynb`)

### 3.1 Content after this session

- Cells 1–7 as in session 3 (cell 7 = Appendix B of session 3; it had already been run by the student at the end of session 3, execution_count 9).
- New cells run by the student in JupyterLab: **Cell 8** (stream-level test, dev week), **Cell 9** (dip by month and ISO week) plus two short helper cells (`print(table("iso_week"))` and `pl.Config.set_tbl_rows(30); print(table("iso_week"))`), **Cell 10** (daily view around maintenance), **Cell 11** (concentration in jobs), **Cell 12** (H2 ramp thresholds, figure `04_ramp_thresholds.png`), **Cell 13** (largest one-minute jumps), **Cell 14** (maintenance-boundary rule). Exact cell indices were not checked. Code of cells 8–14 in Appendix A.
- The student should have saved the notebook (Cmd + S); not confirmed.

### 3.2 Known problems / state of files

- **The code cell at index 1 holds a Terminal command** (the `build_log_features.py` call), saved output `SyntaxError`. A full run (`Run All` or nbconvert) stops there. **Fix before H4:** change that cell to Raw (JupyterLab: select cell, press `R`, or cell type menu "Raw") or delete it; the command is recorded in session 3, Section 5.3.
- Cells 1 and 3–6 write files when re-run: cell 1 overwrites `data/processed/series_julsep.parquet` (deterministic slice, should be identical), cells 3–6 overwrite the four PNGs. Allowed only on purpose (H4).
- Claude Code created `data/interim/04_eda_jul_sep_backup_before_cell7.ipynb` (376,549 bytes, identical to the notebook before this session). It was given a second task (H1b) that would have created `data/interim/04_eda_jul_sep_backup_before_cell8.ipynb`, `data/interim/series_julsep_backup_before_cell8.parquet` and `reports/figures/eda_julsep_backup_before_cell8/`; whether any of these exist is **not verified** (Claude Code stopped). Check with `ls data/interim/ reports/figures/`; the backups can be deleted.
- `data/interim/streams_dev/` checked (read-only): 7,435 `.parquet`, 0 `.tmp`, only other entry `_failures_run1.csv`, 111.2 MiB; schema `gpu_index, minute_epoch, power_W, util_gpu, util_mem, temp_gpu, temp_mem, mem_used, mem_free, n_samples, file` (Int64/Float64/String); `minute_epoch` multiples of 60; median `n_samples` about 580 per minute (about 10 samples per second).
- New figure: `reports/figures/eda_julsep/04_ramp_thresholds.png` (histograms of |P(t+h) − P(t)|, h = 1, 5, 15, train pairs, log counts, p95/p99 lines).

---

## 4. H1 — the five-minute dip: RESULT (closed)

All Jul–Sep numbers are **provisional**.

### 4.1 Series level (cell 7, Jul–Sep, valid consecutive minutes, n_gpu_equiv > 0; n about 25,600 per class)

| mm mod 5 | mean n_gpu_equiv | mean change n_gpu_equiv | mean streams | mean change streams | mean W per GPU-eq | mean change W per GPU-eq |
|---|---|---|---|---|---|---|
| 0 | 333.133 | +0.0294 | 333.68 | −0.0344 | 89.843 | **−0.6073** |
| 1 | 333.143 | −0.0016 | 333.71 | +0.0266 | 90.448 | **+0.6039** |
| 2 | 333.165 | +0.0107 | 333.75 | +0.0223 | 90.434 | −0.012 |
| 3 | 333.132 | −0.0308 | 333.71 | −0.0331 | 90.441 | +0.0075 |
| 4 | 333.092 | −0.0202 | 333.69 | +0.005 | 90.456 | +0.0171 |

Coverage ratio `n_gpu_equiv / n_gpu_streams`: mean 0.9981–0.9983, p05 0.9920–0.9924, equal for all classes.
→ GPU count and coverage are flat; the whole dip is in power per GPU (−0.61 W × about 333 GPUs ≈ −0.20 kW = the series dip of −0.208 kW).

**Why this alone was not enough:** in `build_series.py` the per-stream minute value is the **mean** of the samples and interior minutes always have weight 1, so missed seconds in an interior minute would change neither `power_W` nor `n_gpu_equiv`. The "coverage artefact" hypothesis of session 3 could only act through edge minutes. Hence the stream-level test.

### 4.2 Stream level (cell 8, dev week 2021-09-10..17, interior stream-minutes, 16 Sep excluded)

Rows: 3,210,563 stream-minutes in total (as expected); kept 3,082,471 interior minutes, 6,024 streams, 3,076,447 within-stream consecutive steps.

| mm mod 5 | n | mean n_samples | mean rel. n_samples | mean P (W) | mean within-stream dP (W) | median dP | share dP < 0 |
|---|---|---|---|---|---|---|---|
| 0 | 616,208 | 567.48 | 0.99966 | 86.998 | **−0.2988** | −0.0056 | 0.5273 |
| 1 | 616,520 | 567.46 | 0.99963 | 87.293 | **+0.3076** | +0.0049 | 0.4768 |
| 2 | 616,680 | 567.45 | 0.99962 | 87.282 | +0.0242 | 0.0 | 0.4997 |
| 3 | 616,632 | 567.45 | 0.99962 | 87.301 | +0.0376 | +0.0005 | 0.4974 |
| 4 | 616,431 | 567.45 | 0.99962 | 87.323 | +0.0417 | +0.0005 | 0.4977 |

- Only minutes with full samples (rel ≥ 0.98): mean dP −0.2955 (mod 0), +0.3057 (mod 1), +0.024/+0.041/+0.042 (mod 2–4).
- Quantiles of within-stream dP (W), mod 0 / 1 / 2: p01 −27.6/−26.3/−26.7; p05 −9.07/−7.32/−7.72; p25 −0.80/−0.47/−0.59; p50 −0.006/+0.005/0.0; p75 0.50/0.83/0.60; p95 7.13/8.95/7.66; p99 27.2/28.5/27.9.
- Dev-week series mean dP (kW) vs sum of within-stream steps per minute (kW): mod 0 −0.1152 vs −0.1064; mod 1 +0.1003 vs +0.1095; mod 2–4 −0.0015..+0.0033 vs +0.009..+0.015.

### 4.3 Concentration in jobs (cell 11, dev week; one file = one job on one node)

- 5,037 files; total sum of dP at mod 0: −183,773 W; at mod 2 (placebo): +14,881 W.
- Most negative files by mod-0 sum vs the same number ranked on mod 2 (noise floor):

| top k files | mod-0 sum (W) | share of total | noise floor mod 2 (W) |
|---|---|---|---|
| 10 | −41,724 | 0.227 | −8,622 |
| 50 | −119,327 | 0.649 | −24,295 |
| 100 | −154,421 | 0.840 | −34,603 |
| 500 | −216,512 | 1.178 | −57,068 |
| 1,000 | −233,930 | 1.273 | −60,963 |

- Share of files with mean dP at mod 0 below the mean at mod 2–4: 0.576 (n = 5,013); files with ≥ 1,000 steps: 0.663 (n = 602) (expected 0.5 without an effect; sd about 0.02).
- The top 15 files: 1–2 GPUs each, mean GPU power 104–199 W (overall about 87 W), mean dP at mod 0 −3.4 to −10.8 W per step, about 0 at mod 2–4. Name parts (`r…`, `n…`) repeat among them; their meaning was not checked for this purpose.

### 4.4 Stability over time (cells 9, 10; Jul–Sep; simplified validity: no outage, no maintenance day, power > 0, consecutive minutes; 128,054 minutes)

By month, mean change at mod 0 (total kW / W per GPU-eq): Jul −0.360 / −1.065; Aug −0.164 / −0.443; Sep −0.096 / −0.304 (mod 1 mirror-image; mod 2–4 about +0.001 kW).

By ISO week, W per GPU-eq at mod 0: 26: −0.77; 27: −1.10; 28: −0.45; 29: −1.71; 30: −1.17; 31: −1.29; **32: −0.01**; 33: −0.13; 34: −0.18; 35: −0.21; 36: −0.45; 37: −0.23; 38: −0.26; 39: −0.34.

Daily around maintenance (W per GPU-eq at mod 0; maintenance days missing):
- 15 Jul: 8–14 Jul −0.52..−1.73; 16 Jul −0.55; **17–19 Jul −0.06, +0.06, −0.004**; 20 Jul −0.54; 21–22 Jul −1.97, −2.09 (abrupt return).
- 10 Aug: 3–9 Aug −0.63..−1.99; 11 Aug +0.75 (n = 267, restart day, unreliable); **12–15 Aug −0.03..+0.06**; 16–20 Aug −0.14, −0.19, −0.32, −0.11, −0.09 (slow return).
- 16 Sep: 9–15 Sep −0.18..−0.55; 17–23 Sep −0.21..−0.33 (no reset).

### 4.5 Conclusions (for the proposal and Meeting 1)

1. The dip is **not an artefact of our pipeline** (coverage, edge minutes, job composition and logging gaps all ruled out; `n_samples` is flat).
2. It is in the **measured power of the GPUs themselves** and comes **mostly from a small number of heavy jobs** (about 2% of files give about 65–84% of it, well above the noise floor) that lose a few watts per GPU on the wall-clock 5-minute boundary; a weaker, widespread tendency exists as well. Working hypothesis: periodic activity inside jobs (checkpoint, evaluation, logging, sync). **Mechanism not verified.** A system-wide cron job is less likely (the dip vanishes for days after two maintenance events and varies several-fold from day to day).
3. Its size **depends on the job mix**: large in July, small in late August and September, about 0 for some days after the 15 Jul and 10 Aug maintenance.
4. **Consequences:** do not "correct" the data. Persistence is only affected at h = 1 (for h = 5 and 15 origin and target share the same mod-5 class). A fixed minute-of-hour feature learned on train (large dip) would over-correct on test (small dip). Proposal: report models with and without a mod-5 feature, or add a "persistence + mod-5 correction" baseline, so no model "beats" persistence only through this pattern.
5. Optional later (Colab, raw data): sub-minute shape of the dip for one top file (e.g. `87628501384014-r8939293-n20853…`).

---

## 5. H2 — ramp thresholds (closed for Jul–Sep, provisional)

Definition (cell 12): |P(t+h) − P(t)| in kW; both ends valid (valid = not outage, not maintenance day, not post-maintenance zero minute); train = both ends before index 92,736 (2021-09-03 09:36); val = start ≥ 92,736 and end < 112,608 (2021-09-17 04:48); test = start ≥ 112,608. Checks matched cell 5 exactly (post-maintenance zero minutes 92; valid minutes 128,068; n_train 89,751 / 89,699 / 89,627; p95 1.075 / 1.422 / 1.993). Train median power 31.52 kW.

### 5.1 Current validity rule

| h | n_train | std | median abs | p90 | p95 | p99 | max | excess kurtosis | share up of ramps | p95 % of median | ramps val (n) | ramps test (n) | severe val | severe test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 89,751 | 0.665 | 0.192 | 0.783 | 1.075 | 2.209 | 31.19 | 184.3 | 0.490 | 3.41 | 361 (18,430) | 440 (19,871) | 61 | 65 |
| 5 | 89,699 | 1.009 | 0.249 | 0.950 | 1.422 | 4.036 | 33.23 | 100.0 | 0.521 | 4.51 | 562 (18,422) | 1,045 (19,867) | 56 | 121 |
| 15 | 89,627 | 1.394 | 0.369 | 1.336 | 1.993 | 5.552 | 40.81 | 108.1 | 0.543 | 6.32 | 699 (18,402) | 1,467 (19,857) | 52 | 175 |

Dev week for comparison: p95 0.618 / 1.005 / 1.684; p99 1.133 / 2.183 / 3.281; excess kurtosis 50 / 23 / 8.5.

- h = 1 without pairs touching a mod-0 minute (n = 53,851): p95 0.903, p99 2.123 → the 5-minute dip raises the h = 1 p95 by about 16%; p99 hardly.
- Ramp rates are not stationary: expected 5% (p95) and 1% (p99); observed h = 1 val 2.0% / test 2.2%; h = 5 3.1% / 5.3%; h = 15 3.8% / 7.4%; severe 0.28–0.88%.

### 5.2 Largest one-minute jumps in train (cell 13)

Train pairs h = 1: |dP| > 3 kW: 530; > 5 kW: 212; > 10 kW: 22. Of the 15 largest: **1 at 2021-07-14 23:57** (jobs killed before the 15 Jul maintenance; gpus_alloc 181 → 0, P 20.8 → 3.5 kW); **10 on 2021-08-11** (day after maintenance, 00:28–13:21; gpus_alloc jumps between 424, 256, 128 and 0; P jumps up to ±31 kW; looks like post-maintenance testing or restart, e.g. burn-in on all GPUs — **not verified**); 4 are genuine workload phase changes with a constant GPU count (16 Jul 19:14, 19:26, 19:42 and 3 Aug 06:43; W per GPU about 67–72 → 99–108 W).

### 5.3 Maintenance-boundary rule (cell 14)

Extra exclusion: whole day after a maintenance day (4,228 valid minutes = 3 × 1,440 − 92) and the last hour (23:00–23:59) of the day before it (180 minutes).

| h | rule | n_train | p95 | p99 | max | excess kurtosis | ramp share val | ramp share test |
|---|---|---|---|---|---|---|---|---|
| 1 | current | 89,751 | 1.075 | 2.209 | 31.19 | 184.3 | 0.020 | 0.022 |
| 1 | boundaries excluded | 86,853 | **1.043** | **1.855** | 12.65 | 30.2 | 0.022 | 0.023 |
| 5 | current | 89,699 | 1.422 | 4.036 | 33.23 | 100.0 | 0.031 | 0.053 |
| 5 | boundaries excluded | 86,841 | **1.322** | **3.039** | 17.82 | 59.3 | 0.035 | 0.060 |
| 15 | current | 89,627 | 1.993 | 5.552 | 40.81 | 108.1 | 0.038 | 0.074 |
| 15 | boundaries excluded | 86,811 | **1.833** | **4.031** | 17.55 | 37.7 | 0.046 | 0.081 |

- Maintenance boundaries mainly inflated p99 (−16 to −27% when excluded) and kurtosis; p95 changes 3–8%.
- Non-stationarity of ramp rates remains after the exclusion (property of the series, not of maintenance).
- **Provisional decision (for Meeting 1):** define a **maintenance event window** = last hour of the day before + maintenance day + whole day after; exclude it from threshold computation and EDA statistics, report it separately in evaluation. It replaces the EDA-only `post_maint_zero` rule of session 3. Implementation in `build_log_features.py` only after Meeting 1. Cost about 3% of minutes. (Calendar-hour rule is simple; a log-based rule, e.g. from the gap start, is an option.)
- **Provisional thresholds (Jul–Sep, boundaries excluded):** p95 1.04 / 1.32 / 1.83 kW; p99 (severe) 1.86 / 3.04 / 4.03 kW for h = 1 / 5 / 15.

---

## 6. Decision on the remaining EDA

The student asked whether the EDA helps without the full data. Answer recorded: the **rules and decisions** found on Jul–Sep (five-minute dip is real and job-driven, maintenance window, persistence behaviour by horizon, non-stationarity) hold for the full data and are the main value; the **numbers** are provisional and will change; the notebook is re-run on the full series by changing the input path (H4). Therefore:
- Jul–Sep EDA is closed after cell 14.
- **H3 (covariates) moved to the full series** (not done on Jul–Sep).
- Priority now: Colab chunks 4–7, then the documents (Weekly Report 2, Meeting 1 packet, proposal).

---

## 7. Proposal change list (additions to session 3, Section 7)

- Section 3 Data: five-minute periodic dip (real, job-driven, varying); maintenance event window (pre-hour, maintenance day, day after); post-maintenance testing day (11 Aug) as an example.
- Section 5 Protocol: thresholds per horizon with the maintenance window excluded; p99 severe tier (more meaningful at h = 1, where p95 is partly the 5-minute dip); ramp rates in val/test differ from 5%/1% (non-stationarity) → report counts and rates; models with and without a mod-5 feature, or a persistence + mod-5 baseline.
- Section 9 Open questions: the new Meeting 1 items below.
- Still: proposal edits only when the student asks; read the Persian tab first.

---

## 8. Meeting 1 agenda (updates to session 3, Section 8; items there still valid)

1. **(replaces item 1) Maintenance event window**: last hour before, the maintenance day, the whole day after (evidence: 14 Jul 23:57 kill, 11 Aug turbulent restart day, p99 −16 to −27%). Calendar rule or log-based rule? Into `build_log_features.py`?
2. **(replaces item 2) Five-minute dip**: real, in GPU power, mostly from a few heavy jobs, varying with the job mix. May minute-of-hour features be used? Proposed: report with and without, or a persistence + mod-5 baseline.
3. **Ramp tiers**: p95 at h = 1 is close to normal noise and partly the 5-minute dip; use p99 as the main ramp definition for h = 1? How to handle non-stationary ramp rates between train and test?
4. Items 3–7 of session 3 (level shift and split, covariates on zero minutes, sMAPE on zero minutes, weak daily cycle and no weekly effect, PM100 version) unchanged.

---

## 9. Remaining work for Week 2 (replaces session 3, Section 9)

| Step | Task | Status |
|---|---|---|
| A, B | Chunk outputs checked; Jul–Sep series on the Mac | done |
| C | Colab chunks 4–7 | **running**: 4 and 5 overnight; then 6 and 7 |
| C2 | Count streams / leftover `.tmp` | **done for chunks 1–3** (65,714, 0 `.tmp`); repeat once after chunk 7 (pre-check of the v3 cell does it) |
| H1 | Five-minute dip | **done** (Section 4) |
| H2 | Ramp thresholds | **done, provisional** (Section 5) |
| H3 | Covariates | **moved to the full series** |
| H4 | Re-run notebook 04 on the full series | open: first fix the Terminal-command cell (Section 3.2), slice/feature-build the full series, change input path, run all |
| D | Proposal update (both tabs) | open |
| E | Weekly Report 2 (Claude Doc, English + Persian), due **Friday 9 October 2026** | open |
| F | Meeting 1 packet | open |
| G | Housekeeping: delete `MyDrive/dcpower/streams_test/`; correct PM100 size in docs; delete Claude Code backups (Section 3.2) when no longer needed | open |
| I | PM100 schema check on Colab | optional |
| J | Sub-minute shape of the 5-minute dip (raw data, Colab) | optional |

**Week 2 is complete when:** chunks 4–7 are done (or a documented decision to stop earlier), H4 has run on the available full series (or Jul–Sep numbers are explicitly used for Meeting 1), the proposal is updated, Weekly Report 2 and the Meeting 1 packet exist. Tell the student explicitly when this is reached.

---

## 10. Decisions in this session

1. Claude Code dropped; chat gives cells and commands directly.
2. Five-minute dip: real, job-driven; data not corrected; fairness via with/without mod-5 feature (to be approved).
3. Maintenance event window (provisional, for Meeting 1); replaces the EDA-only `post_maint_zero` rule.
4. Jul–Sep EDA closed; H3 on the full series.
5. Aggregation after each Colab chunk kept for now.

---

## Appendix A — code of the new notebook-04 cells (as run)

### Cell 8 (standalone; dev-week streams)

```python
# Cell 8: Stream-level test of the 5-minute dip on the dev-week stream Parquets
# (2021-09-10..09-17 file clock; interior stream-minutes only; maintenance day 2021-09-16 excluded).
import polars as pl
SD = "../data/interim/streams_dev"
s = (pl.scan_parquet(f"{SD}/*.parquet")
       .select("file", "gpu_index", "minute_epoch", "power_W", "n_samples")
       .collect())
print(f"stream-minute rows (all): {s.height:,}  (expected 3,210,563)")
key = ["file", "gpu_index"]
s = (s.with_columns(pl.col("minute_epoch").min().over(key).alias("m0"),
                    pl.col("minute_epoch").max().over(key).alias("m1"))
      .filter((pl.col("minute_epoch") > pl.col("m0")) & (pl.col("minute_epoch") < pl.col("m1")))
      .with_columns(pl.from_epoch("minute_epoch", time_unit="s").alias("t"))
      .filter(pl.col("t").dt.date() != pl.date(2021, 9, 16))
      .sort(key + ["minute_epoch"])
      .with_columns(pl.col("n_samples").median().over(key).alias("ns_med"),
                    pl.col("power_W").diff().over(key).alias("dP_raw"),
                    pl.col("minute_epoch").diff().over(key).alias("dt"),
                    ((pl.col("minute_epoch") // 60) % 5).alias("mm_mod5"))
      .with_columns((pl.col("n_samples") / pl.col("ns_med")).alias("rel_ns"),
                    pl.when(pl.col("dt") == 60).then(pl.col("dP_raw")).otherwise(None).alias("dP_s")))
# ... tables A (by mod 5), B (full samples only), C (quantiles), D (series vs implied) as run in the session
```

(Full printed tables in Section 4.2. The complete cell is in the notebook.)

### Cell 9 (standalone; Jul–Sep, by month and ISO week)

Reads `series_julsep_features.parquet`; `Wpg = power_W / n_gpu_equiv`; `dP`, `dWpg` = first differences; keeps consecutive minutes where neither the minute nor the previous minute is outage, maintenance day or power ≤ 0; groups by month or ISO week and mod-5 class (m0, m1, m234); defines `f` and `table(by)`. Helper cells: `print(table("iso_week"))`, `pl.Config.set_tbl_rows(30)`.

### Cell 10 (needs `f` from cell 9)

Daily mean `dWpg` at m0, m1, m234 for 2021-07-08..22, 2021-08-03..20, 2021-09-09..23.

### Cell 11 (needs `s` from cell 8)

Per file: sum and mean of within-stream dP at mod 0 and mod 2; cumulative share of the most negative files vs the same ranking on mod 2; share of files with mean at mod 0 below mean at mod 2–4; top 15 files.

### Cell 12 (standalone; H2)

Rebuilds `g` (features + `date`), `maint`, `post`, `pmz`, `valid`, `P` (kW), `mm`, `i_tr = int(0.7 n)`, `i_va = int(0.85 n)`, function `pairs(h)`; prints the ramp table, the h = 1 mod-0 comparison, saves `04_ramp_thresholds.png`.

### Cell 13 (needs cell 12)

Counts train pairs with |dP| > 3/5/10 kW; prints the 15 largest jumps with P, n_gpu_equiv, gpus_alloc, W per GPU and days since the last maintenance.

### Cell 14 (needs cell 12)

```python
pre = [m - timedelta(days=1) for m in maint]
hour = g["minute_local"].dt.hour().to_numpy()
post_day = g["date"].is_in(post).to_numpy()
pre_hour = g["date"].is_in(pre).to_numpy() & (hour == 23)
valid2 = valid & ~post_day & ~pre_hour
# thresholds, max, kurtosis and val/test ramp shares for valid vs valid2 (Section 5.3)
```

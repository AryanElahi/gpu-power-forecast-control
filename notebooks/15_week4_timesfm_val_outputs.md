# Outputs of `15_week4_timesfm_val.ipynb`
Exported 2026-10-09 18:47 from the saved notebook file.

## Cell 1: setup
_execution count: 1_

```stderr
/opt/miniconda3/envs/dcpower/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

```
loaded google/timesfm-2.5-200m-pytorch | params 231,289,280 | device cpu
```

## Cell 2: SYNTHETIC check: output layout (mean + deciles), quantile order, NaN filling, sensible accuracy
_execution count: 2_

```
point shape (8, 15) | quantile shape (8, 15, 10)
deciles ordered: True | max |point - q50|: 0.0
{'ctx': 64, 'finite': True, 'skill_h1': 0.146, 'skill_h5': 0.016, 'skill_h15': 0.175, 'cover80': 0.5}
{'ctx': 512, 'finite': True, 'skill_h1': 0.273, 'skill_h5': 0.454, 'skill_h15': 0.804, 'cover80': 0.91}

SYNTHETIC CHECK PASSED
```

## Cell 3: REAL data: timing on 1,000 validation origins per context (decide before the full run)
_execution count: 3_

```
ctx    64: 3.8 ms/origin | est. full validation 3 min | skill on 1,000 origins +0.072 / +0.050 / +0.009
ctx   512: 31.7 ms/origin | est. full validation 23 min | skill on 1,000 origins +0.108 / +0.052 / +0.056
ctx  1024: 63.8 ms/origin | est. full validation 45 min | skill on 1,000 origins +0.125 / +0.047 / +0.049

estimated total for cell 4: 70 min (remove a context from CONTEXTS if too long)
```

## CONTEXTS = [64, 512]
_execution count: 4_

(no output)

## Cell 4: full validation run per context (resumable; long: see cell 3)
_execution count: 5_

```stderr
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /Users/aryaneuse/.netrc.
```

```
timesfm_2p5_200m_ctx64: 2.8 min | val MAE 0.2694 / 0.4133 / 0.5974 | skill +0.060 / +0.039 / +0.012 | 80% coverage 0.817 / 0.770 / 0.699
timesfm_2p5_200m_ctx512: 23.9 min | val MAE 0.2602 / 0.3976 / 0.5775 | skill +0.092 / +0.075 / +0.045 | 80% coverage 0.836 / 0.819 / 0.782
```

## Cell 5: choose the context on validation and add TimesFM to the benchmark table (next to Chronos)
_execution count: 6_

```
shape: (6, 10)
┌─────┬─────────┬────────┬───────────┬──────────┬────────────┬─────────┬───────────────┬───────────────┬─────────────────────┐
│ h   ┆ context ┆ MAE    ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 ┆ cover_q10_q90 ┆ width_q10_q90 ┆ infer_ms_per_sample │
│ --- ┆ ---     ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     ┆ ---           ┆ ---           ┆ ---                 │
│ i64 ┆ i64     ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     ┆ f64           ┆ f64           ┆ f64                 │
╞═════╪═════════╪════════╪═══════════╪══════════╪════════════╪═════════╪═══════════════╪═══════════════╪═════════════════════╡
│ 1   ┆ 64      ┆ 0.2694 ┆ 0.06      ┆ 1.401    ┆ 0.2178     ┆ 0.261   ┆ 0.817         ┆ 0.845         ┆ 3.96                │
│ 1   ┆ 512     ┆ 0.2602 ┆ 0.092     ┆ 1.388    ┆ 0.2089     ┆ 0.211   ┆ 0.836         ┆ 0.845         ┆ 33.69               │
│ 5   ┆ 64      ┆ 0.4133 ┆ 0.039     ┆ 2.254    ┆ 0.3017     ┆ 0.29    ┆ 0.77          ┆ 1.103         ┆ 3.96                │
│ 5   ┆ 512     ┆ 0.3976 ┆ 0.075     ┆ 2.205    ┆ 0.2883     ┆ 0.265   ┆ 0.819         ┆ 1.182         ┆ 33.69               │
│ 15  ┆ 64      ┆ 0.5974 ┆ 0.012     ┆ 2.893    ┆ 0.4385     ┆ 0.246   ┆ 0.699         ┆ 1.388         ┆ 3.96                │
│ 15  ┆ 512     ┆ 0.5775 ┆ 0.045     ┆ 2.809    ┆ 0.4231     ┆ 0.227   ┆ 0.782         ┆ 1.611         ┆ 33.69               │
└─────┴─────────┴────────┴───────────┴──────────┴────────────┴─────────┴───────────────┴───────────────┴─────────────────────┘

context chosen on validation (mean MAE over h): 512 min
h= 1: timesfm_2p5_200m rank 5 of 20 | chronos_bolt_small rank 16 | best lightgbm_cov
h= 5: timesfm_2p5_200m rank 3 of 20 | chronos_bolt_small rank 11 | best lightgbm_cov
h=15: timesfm_2p5_200m rank 8 of 20 | chronos_bolt_small rank 16 | best lightgbm_cov
```

## Cell 6: export outputs (save the notebook first with Cmd+S)
_execution count: 7_

```
written 15_week4_timesfm_val_outputs.md | code cells 8 | figures 0
```

## Cell 8
_execution count: None_

(no output)

# Outputs of `14_week4_chronos_val.ipynb`
Exported 2026-10-09 16:39 from the saved notebook file.

## Cell 1: setup
_execution count: 1_

```stderr
Loading weights: 100%|████████████████████| 143/143 [00:00<00:00, 29779.33it/s]
```

```
device mps | params 47,718,016 | val origins 42,575
```

## Cell 2: full validation run per context length (resumable)
_execution count: 2_

```stderr
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /Users/aryaneuse/.netrc.
```

```
chronos_bolt_small_ctx60: 0.4 min | val MAE 0.2770 / 0.4352 / 0.6509 | skill +0.033 / -0.012 / -0.076 | 80% interval coverage 0.809 / 0.803 / 0.743
chronos_bolt_small_ctx512: 1.2 min | val MAE 0.2808 / 0.4163 / 0.5955 | skill +0.020 / +0.032 / +0.015 | 80% interval coverage 0.811 / 0.818 / 0.802
chronos_bolt_small_ctx2048: 5.6 min | val MAE 0.2974 / 0.4291 / 0.5988 | skill -0.038 / +0.002 / +0.010 | 80% interval coverage 0.836 / 0.826 / 0.809
```

## Cell 3: choose the context on validation (mean MAE over horizons) and add Chronos to the benchmark table
_execution count: 3_

```
shape: (9, 10)
┌─────┬─────────┬────────┬───────────┬──────────┬────────────┬─────────┬───────────────┬───────────────┬─────────────────────┐
│ h   ┆ context ┆ MAE    ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 ┆ cover_q10_q90 ┆ width_q10_q90 ┆ infer_ms_per_sample │
│ --- ┆ ---     ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     ┆ ---           ┆ ---           ┆ ---                 │
│ i64 ┆ i64     ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     ┆ f64           ┆ f64           ┆ f64                 │
╞═════╪═════════╪════════╪═══════════╪══════════╪════════════╪═════════╪═══════════════╪═══════════════╪═════════════════════╡
│ 1   ┆ 60      ┆ 0.277  ┆ 0.033     ┆ 1.421    ┆ 0.2248     ┆ 0.165   ┆ 0.809         ┆ 0.883         ┆ 0.512               │
│ 1   ┆ 512     ┆ 0.2808 ┆ 0.02      ┆ 1.44     ┆ 0.2281     ┆ 0.137   ┆ 0.811         ┆ 0.868         ┆ 1.622               │
│ 1   ┆ 2048    ┆ 0.2974 ┆ -0.038    ┆ 1.467    ┆ 0.2444     ┆ 0.104   ┆ 0.836         ┆ 0.987         ┆ 7.841               │
│ 5   ┆ 60      ┆ 0.4352 ┆ -0.012    ┆ 2.364    ┆ 0.3179     ┆ 0.204   ┆ 0.803         ┆ 1.384         ┆ 0.512               │
│ 5   ┆ 512     ┆ 0.4163 ┆ 0.032     ┆ 2.289    ┆ 0.3028     ┆ 0.222   ┆ 0.818         ┆ 1.248         ┆ 1.622               │
│ 5   ┆ 2048    ┆ 0.4291 ┆ 0.002     ┆ 2.309    ┆ 0.3154     ┆ 0.21    ┆ 0.826         ┆ 1.296         ┆ 7.841               │
│ 15  ┆ 60      ┆ 0.6509 ┆ -0.076    ┆ 3.041    ┆ 0.4852     ┆ 0.148   ┆ 0.743         ┆ 1.94          ┆ 0.512               │
│ 15  ┆ 512     ┆ 0.5955 ┆ 0.015     ┆ 2.92     ┆ 0.4345     ┆ 0.18    ┆ 0.802         ┆ 1.8           ┆ 1.622               │
│ 15  ┆ 2048    ┆ 0.5988 ┆ 0.01      ┆ 2.958    ┆ 0.4355     ┆ 0.168   ┆ 0.809         ┆ 1.784         ┆ 7.841               │
└─────┴─────────┴────────┴───────────┴──────────┴────────────┴─────────┴───────────────┴───────────────┴─────────────────────┘

context chosen on validation (mean MAE over h): 512 min
h= 1: chronos_bolt_small rank 15 of 19 | MAE 0.2808 | best lightgbm_cov 0.2464
h= 5: chronos_bolt_small rank 10 of 19 | MAE 0.4163 | best lightgbm_cov 0.3850
h=15: chronos_bolt_small rank 15 of 19 | MAE 0.5955 | best lightgbm_cov 0.5559
written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/fm_val_summary.csv | /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/benchmark_val_week4_all.csv
```

## Cell 4: export outputs (save the notebook first with Cmd+S)
_execution count: 4_

```
written 14_week4_chronos_val_outputs.md | code cells 4 | figures 0
```

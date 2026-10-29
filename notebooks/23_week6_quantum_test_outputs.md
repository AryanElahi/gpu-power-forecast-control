# Outputs of `23_week6_quantum_test.ipynb`
Exported 2026-10-12 13:50 from the saved notebook file.

## Cell 1: setup and the frozen model set
_execution count: 1_

```
hybrid family: {'hqnn': 'hqnn_noreup_q4_l2', 'hqnn_noent': 'hqnn_noent_noreup_q4_l2', 'ctrl_mlp': 'ctrl_mlp_q4_l2', 'ctrl_bottleneck': 'ctrl_bottleneck_q4_l2'} 
QRC family: 8 qubits {'qrc': 'qrc_q8_d4', 'qrc_noent': 'qrc_noent_q8_d4', 'rf': 'ctrl_rf_q8_f40', 'raw': 'ctrl_raw_q8'} | scales {'qrc': 0.098, 'qrc_noent': 0.196, 'rf': 0.098, 'raw': 1.571}
usable origins: {1: {'val': 42575, 'test': 42575}, 5: {'val': 42567, 'test': 42567}, 15: {'val': 42547, 'test': 42547}}
freeze record exists: False (if True, the test split was already opened once)
```

## Cell 2: REPRODUCE every stored validation forecast from the frozen artifacts (test split untouched)
_execution count: 2_

```
  hybrid family: 3 s
  QRC family: 258 s
shape: (8, 4)
┌─────────────────────────┬────────┬─────────────────┬──────┐
│ model                   ┆ checks ┆ max_abs_diff_kW ┆ ok   │
│ ---                     ┆ ---    ┆ ---             ┆ ---  │
│ str                     ┆ u32    ┆ f64             ┆ bool │
╞═════════════════════════╪════════╪═════════════════╪══════╡
│ ctrl_bottleneck_q4_l2   ┆ 9      ┆ 0.0             ┆ true │
│ ctrl_mlp_q4_l2          ┆ 9      ┆ 0.0             ┆ true │
│ ctrl_raw_q8             ┆ 3      ┆ 0.0             ┆ true │
│ ctrl_rf_q8_f40          ┆ 9      ┆ 0.0             ┆ true │
│ hqnn_noent_noreup_q4_l2 ┆ 9      ┆ 0.0             ┆ true │
│ hqnn_noreup_q4_l2       ┆ 9      ┆ 0.0             ┆ true │
│ qrc_noent_q8_d4         ┆ 9      ┆ 0.0             ┆ true │
│ qrc_q8_d4               ┆ 9      ┆ 0.0             ┆ true │
└─────────────────────────┴────────┴─────────────────┴──────┘

REPRODUCTION PASSED: the frozen artifacts re-create the validation forecasts
```

## Cell 3: freeze record (written before the test split is opened)
_execution count: 3_

```
{
 "frozen_at": "2026-10-12 13:48:34",
 "git_head": "96597605437198fbfb9cb0a5da4aee2b50ccbbd4",
 "protocol": [
  "model set frozen (main configurations chosen on validation in notebooks 18-19)",
  "train only, no refit on train+val",
  "hybrid models from saved checkpoints",
  "QRC: fixed reservoirs (seeded), saved ridge readouts",
  "noise: inference only, Heron r2-like, 1024 shots",
  "one test run; later fixes reported"
 ],
 "main_hybrid": {
  "n_qubits": 4,
  "n_layers": 2,
  "reupload": false,
  "entangle": true
 },
 "qrc": {
  "n_qubits": 8,
  "depth": 4,
  "scales": {
   "qrc": 0.09817477042468103,
   "qrc_noent": 0.19634954084936207,
   "rf": 0.09817477042468103,
   "raw": 1.5707963267948966
  }
 },
 "noise": {
  "p1": 0.0003,
  "p2": 0.003,
  "p_ro": 0.0125,
  "shots": 1024
 },
 "models": [
  "ctrl_bottleneck_q4_l2",
  "ctrl_mlp_q4_l2",
  "ctrl_raw_q8",
  "ctrl_rf_q8_f40",
  "hqnn_noent_noreup_q4_l2",
  "hqnn_noreup_q4_l2",
  "hqnn_noreup_q4_l2_heron_r2",
  "qrc_noent_q8_d4",
  "qrc_q8_d4"
 ],
 "val_reproduction_max_diff_kW": 0.0
}
SHA-256 recorded for 25 files
```

## Cell 4: OPEN THE TEST SPLIT: forecasts of the frozen quantum set (one run; the noisy hybrid takes ~10 min)
_execution count: 4_

```
  hybrid family (+ noisy): 83 s
  QRC family: 15 s
test predictions: 3,192,225 rows | models 9 | sha256 334776b2b896728e...
```

## Cell 5: test metrics (mean +- sd over seeds), merged with the frozen Week 4 test benchmark; validation -> test
_execution count: 5_

```

=== TEST h = 1 min (29 models) ===
shape: (15, 9)
┌──────┬───────────────────┬────────────────────────────┬─────────┬───┬────────┬───────────┬──────────┬─────────┐
│ rank ┆ family            ┆ model                      ┆ n_seeds ┆ … ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ ramp_F1 │
│ ---  ┆ ---               ┆ ---                        ┆ ---     ┆   ┆ ---    ┆ ---       ┆ ---      ┆ ---     │
│ u32  ┆ str               ┆ str                        ┆ i64     ┆   ┆ f64    ┆ f64       ┆ f64      ┆ f64     │
╞══════╪═══════════════════╪════════════════════════════╪═════════╪═══╪════════╪═══════════╪══════════╪═════════╡
│ 1    ┆ ML                ┆ lightgbm_cov               ┆ 3       ┆ … ┆ 0.0001 ┆ 0.114     ┆ 1.005    ┆ 0.237   │
│ 2    ┆ ML                ┆ lightgbm                   ┆ 3       ┆ … ┆ 0.0001 ┆ 0.064     ┆ 1.118    ┆ 0.138   │
│ 3    ┆ foundation        ┆ timesfm_2p5_200m           ┆ 1       ┆ … ┆ null   ┆ 0.056     ┆ 1.125    ┆ 0.125   │
│ 6    ┆ deep              ┆ patchtst                   ┆ 3       ┆ … ┆ 0.0011 ┆ 0.048     ┆ 1.12     ┆ 0.102   │
│ 8    ┆ deep              ┆ gru_uni                    ┆ 3       ┆ … ┆ 0.0008 ┆ 0.036     ┆ 1.102    ┆ 0.175   │
│ 9    ┆ classical control ┆ ctrl_bottleneck_q4_l2      ┆ 3       ┆ … ┆ 0.001  ┆ 0.031     ┆ 1.172    ┆ 0.012   │
│ 12   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2    ┆ 3       ┆ … ┆ 0.002  ┆ 0.02      ┆ 1.132    ┆ 0.12    │
│ 13   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2          ┆ 3       ┆ … ┆ 0.0012 ┆ 0.02      ┆ 1.128    ┆ 0.128   │
│ 14   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2_heron_r2 ┆ 3       ┆ … ┆ 0.0024 ┆ 0.019     ┆ 1.134    ┆ 0.112   │
│ 15   ┆ quantum reservoir ┆ qrc_noent_q8_d4            ┆ 3       ┆ … ┆ 0.0001 ┆ 0.017     ┆ 1.152    ┆ 0.041   │
│ 17   ┆ quantum reservoir ┆ qrc_q8_d4                  ┆ 3       ┆ … ┆ 0.0005 ┆ 0.016     ┆ 1.144    ┆ 0.047   │
│ 18   ┆ classical control ┆ ctrl_rf_q8_f40             ┆ 3       ┆ … ┆ 0.0004 ┆ 0.015     ┆ 1.147    ┆ 0.051   │
│ 22   ┆ classical control ┆ ctrl_mlp_q4_l2             ┆ 3       ┆ … ┆ 0.0029 ┆ 0.009     ┆ 1.132    ┆ 0.128   │
│ 24   ┆ classical control ┆ ctrl_raw_q8                ┆ 1       ┆ … ┆ null   ┆ 0.004     ┆ 1.181    ┆ 0.002   │
│ 27   ┆ naive             ┆ persistence                ┆ 1       ┆ … ┆ null   ┆ 0.0       ┆ 1.252    ┆ 0.0     │
└──────┴───────────────────┴────────────────────────────┴─────────┴───┴────────┴───────────┴──────────┴─────────┘

=== TEST h = 5 min (29 models) ===
shape: (15, 9)
┌──────┬───────────────────┬────────────────────────────┬─────────┬───┬────────┬───────────┬──────────┬─────────┐
│ rank ┆ family            ┆ model                      ┆ n_seeds ┆ … ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ ramp_F1 │
│ ---  ┆ ---               ┆ ---                        ┆ ---     ┆   ┆ ---    ┆ ---       ┆ ---      ┆ ---     │
│ u32  ┆ str               ┆ str                        ┆ i64     ┆   ┆ f64    ┆ f64       ┆ f64      ┆ f64     │
╞══════╪═══════════════════╪════════════════════════════╪═════════╪═══╪════════╪═══════════╪══════════╪═════════╡
│ 1    ┆ ML                ┆ lightgbm_cov               ┆ 3       ┆ … ┆ 0.0001 ┆ 0.09      ┆ 1.589    ┆ 0.196   │
│ 2    ┆ ML                ┆ lightgbm                   ┆ 3       ┆ … ┆ 0.0002 ┆ 0.046     ┆ 1.681    ┆ 0.147   │
│ 4    ┆ foundation        ┆ timesfm_2p5_200m           ┆ 1       ┆ … ┆ null   ┆ 0.041     ┆ 1.674    ┆ 0.142   │
│ 5    ┆ deep              ┆ gru_uni                    ┆ 3       ┆ … ┆ 0.0006 ┆ 0.041     ┆ 1.678    ┆ 0.146   │
│ 6    ┆ deep              ┆ patchtst                   ┆ 3       ┆ … ┆ 0.0009 ┆ 0.039     ┆ 1.686    ┆ 0.126   │
│ 9    ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2          ┆ 3       ┆ … ┆ 0.0008 ┆ 0.034     ┆ 1.719    ┆ 0.118   │
│ 10   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2_heron_r2 ┆ 3       ┆ … ┆ 0.0008 ┆ 0.033     ┆ 1.724    ┆ 0.105   │
│ 11   ┆ classical control ┆ ctrl_bottleneck_q4_l2      ┆ 3       ┆ … ┆ 0.0011 ┆ 0.032     ┆ 1.761    ┆ 0.006   │
│ 12   ┆ quantum reservoir ┆ qrc_noent_q8_d4            ┆ 3       ┆ … ┆ 0.0001 ┆ 0.029     ┆ 1.745    ┆ 0.053   │
│ 13   ┆ classical control ┆ ctrl_mlp_q4_l2             ┆ 3       ┆ … ┆ 0.0017 ┆ 0.029     ┆ 1.711    ┆ 0.133   │
│ 14   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2    ┆ 3       ┆ … ┆ 0.0019 ┆ 0.028     ┆ 1.714    ┆ 0.127   │
│ 15   ┆ classical control ┆ ctrl_raw_q8                ┆ 1       ┆ … ┆ null   ┆ 0.024     ┆ 1.792    ┆ 0.0     │
│ 18   ┆ quantum reservoir ┆ qrc_q8_d4                  ┆ 3       ┆ … ┆ 0.0009 ┆ 0.022     ┆ 1.737    ┆ 0.088   │
│ 19   ┆ classical control ┆ ctrl_rf_q8_f40             ┆ 3       ┆ … ┆ 0.0003 ┆ 0.019     ┆ 1.735    ┆ 0.098   │
│ 28   ┆ naive             ┆ persistence                ┆ 1       ┆ … ┆ null   ┆ 0.0       ┆ 1.833    ┆ 0.0     │
└──────┴───────────────────┴────────────────────────────┴─────────┴───┴────────┴───────────┴──────────┴─────────┘

=== TEST h = 15 min (29 models) ===
shape: (15, 9)
┌──────┬───────────────────┬────────────────────────────┬─────────┬───┬────────┬───────────┬──────────┬─────────┐
│ rank ┆ family            ┆ model                      ┆ n_seeds ┆ … ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ ramp_F1 │
│ ---  ┆ ---               ┆ ---                        ┆ ---     ┆   ┆ ---    ┆ ---       ┆ ---      ┆ ---     │
│ u32  ┆ str               ┆ str                        ┆ i64     ┆   ┆ f64    ┆ f64       ┆ f64      ┆ f64     │
╞══════╪═══════════════════╪════════════════════════════╪═════════╪═══╪════════╪═══════════╪══════════╪═════════╡
│ 1    ┆ ML                ┆ lightgbm_cov               ┆ 3       ┆ … ┆ 0.0002 ┆ 0.062     ┆ 2.267    ┆ 0.134   │
│ 3    ┆ ML                ┆ lightgbm                   ┆ 3       ┆ … ┆ 0.0003 ┆ 0.033     ┆ 2.362    ┆ 0.112   │
│ 5    ┆ deep              ┆ gru_uni                    ┆ 3       ┆ … ┆ 0.0015 ┆ 0.031     ┆ 2.351    ┆ 0.103   │
│ 6    ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2          ┆ 3       ┆ … ┆ 0.0001 ┆ 0.029     ┆ 2.425    ┆ 0.039   │
│ 7    ┆ deep              ┆ patchtst                   ┆ 3       ┆ … ┆ 0.0004 ┆ 0.029     ┆ 2.371    ┆ 0.082   │
│ 9    ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2_heron_r2 ┆ 3       ┆ … ┆ 0.0003 ┆ 0.027     ┆ 2.43     ┆ 0.031   │
│ 10   ┆ classical control ┆ ctrl_mlp_q4_l2             ┆ 3       ┆ … ┆ 0.0014 ┆ 0.026     ┆ 2.413    ┆ 0.048   │
│ 12   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2    ┆ 3       ┆ … ┆ 0.0014 ┆ 0.025     ┆ 2.416    ┆ 0.039   │
│ 13   ┆ classical control ┆ ctrl_bottleneck_q4_l2      ┆ 3       ┆ … ┆ 0.0016 ┆ 0.024     ┆ 2.468    ┆ 0.0     │
│ 14   ┆ quantum reservoir ┆ qrc_noent_q8_d4            ┆ 3       ┆ … ┆ 0.0003 ┆ 0.021     ┆ 2.441    ┆ 0.018   │
│ 15   ┆ foundation        ┆ timesfm_2p5_200m           ┆ 1       ┆ … ┆ null   ┆ 0.019     ┆ 2.331    ┆ 0.108   │
│ 16   ┆ quantum reservoir ┆ qrc_q8_d4                  ┆ 3       ┆ … ┆ 0.0003 ┆ 0.015     ┆ 2.44     ┆ 0.04    │
│ 19   ┆ classical control ┆ ctrl_raw_q8                ┆ 1       ┆ … ┆ null   ┆ 0.014     ┆ 2.483    ┆ 0.0     │
│ 21   ┆ classical control ┆ ctrl_rf_q8_f40             ┆ 3       ┆ … ┆ 0.0005 ┆ 0.012     ┆ 2.441    ┆ 0.051   │
│ 27   ┆ naive             ┆ persistence                ┆ 1       ┆ … ┆ null   ┆ 0.0       ┆ 2.523    ┆ 0.0     │
└──────┴───────────────────┴────────────────────────────┴─────────┴───┴────────┴───────────┴──────────┴─────────┘

validation -> test skill of the Week 5 models:
shape: (27, 4)
┌─────┬────────────────────────────┬────────────┬───────────┐
│ h   ┆ model                      ┆ test_skill ┆ val_skill │
│ --- ┆ ---                        ┆ ---        ┆ ---       │
│ i64 ┆ str                        ┆ f64        ┆ f64       │
╞═════╪════════════════════════════╪════════════╪═══════════╡
│ 1   ┆ ctrl_bottleneck_q4_l2      ┆ 0.031      ┆ 0.066     │
│ 1   ┆ ctrl_mlp_q4_l2             ┆ 0.009      ┆ 0.056     │
│ 1   ┆ ctrl_raw_q8                ┆ 0.004      ┆ 0.058     │
│ 1   ┆ ctrl_rf_q8_f40             ┆ 0.015      ┆ 0.077     │
│ 1   ┆ hqnn_noent_noreup_q4_l2    ┆ 0.02       ┆ 0.086     │
│ 1   ┆ hqnn_noreup_q4_l2          ┆ 0.02       ┆ 0.074     │
│ 1   ┆ hqnn_noreup_q4_l2_heron_r2 ┆ 0.019      ┆ null      │
│ 1   ┆ qrc_noent_q8_d4            ┆ 0.017      ┆ 0.076     │
│ 1   ┆ qrc_q8_d4                  ┆ 0.016      ┆ 0.075     │
│ 5   ┆ ctrl_bottleneck_q4_l2      ┆ 0.032      ┆ 0.045     │
│ 5   ┆ ctrl_mlp_q4_l2             ┆ 0.029      ┆ 0.053     │
│ 5   ┆ ctrl_raw_q8                ┆ 0.024      ┆ 0.032     │
│ 5   ┆ ctrl_rf_q8_f40             ┆ 0.019      ┆ 0.042     │
│ 5   ┆ hqnn_noent_noreup_q4_l2    ┆ 0.028      ┆ 0.054     │
│ 5   ┆ hqnn_noreup_q4_l2          ┆ 0.034      ┆ 0.055     │
│ 5   ┆ hqnn_noreup_q4_l2_heron_r2 ┆ 0.033      ┆ null      │
│ 5   ┆ qrc_noent_q8_d4            ┆ 0.029      ┆ 0.041     │
│ 5   ┆ qrc_q8_d4                  ┆ 0.022      ┆ 0.041     │
│ 15  ┆ ctrl_bottleneck_q4_l2      ┆ 0.024      ┆ 0.033     │
│ 15  ┆ ctrl_mlp_q4_l2             ┆ 0.026      ┆ 0.041     │
│ 15  ┆ ctrl_raw_q8                ┆ 0.014      ┆ 0.021     │
│ 15  ┆ ctrl_rf_q8_f40             ┆ 0.012      ┆ 0.03      │
│ 15  ┆ hqnn_noent_noreup_q4_l2    ┆ 0.025      ┆ 0.041     │
│ 15  ┆ hqnn_noreup_q4_l2          ┆ 0.029      ┆ 0.043     │
│ 15  ┆ hqnn_noreup_q4_l2_heron_r2 ┆ 0.027      ┆ null      │
│ 15  ┆ qrc_noent_q8_d4            ┆ 0.021      ┆ 0.029     │
│ 15  ┆ qrc_q8_d4                  ┆ 0.015      ┆ 0.031     │
└─────┴────────────────────────────┴────────────┴───────────┘
```

## Cell 6: Diebold-Mariano with Holm (seed 0) on test: hybrid and QRC vs their controls and the classical references;
_execution count: 6_

```
shape: (57, 9)
┌─────┬─────────────────────────┬─────────────────────────┬────────┬───┬─────────┬────────────┬─────────────────────────┬──────────────────┐
│ h   ┆ model_a                 ┆ model_b                 ┆ MAE_a  ┆ … ┆ DM_stat ┆ DM_p_holm  ┆ better                  ┆ significant_5pct │
│ --- ┆ ---                     ┆ ---                     ┆ ---    ┆   ┆ ---     ┆ ---        ┆ ---                     ┆ ---              │
│ i64 ┆ str                     ┆ str                     ┆ f64    ┆   ┆ f64     ┆ f64        ┆ str                     ┆ bool             │
╞═════╪═════════════════════════╪═════════════════════════╪════════╪═══╪═════════╪════════════╪═════════════════════════╪══════════════════╡
│ 1   ┆ hqnn_noreup_q4_l2       ┆ ctrl_mlp_q4_l2          ┆ 0.2371 ┆ … ┆ -6.65   ┆ 9.6199e-10 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ hqnn_noent_noreup_q4_l2 ┆ 0.2371 ┆ … ┆ -3.77   ┆ 0.002783   ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ ctrl_bottleneck_q4_l2   ┆ 0.2371 ┆ … ┆ 3.9     ┆ 0.001709   ┆ ctrl_bottleneck_q4_l2   ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ qrc_q8_d4               ┆ 0.2371 ┆ … ┆ 1.02    ┆ 1.0        ┆ qrc_q8_d4               ┆ false            │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ ctrl_rf_q8_f40          ┆ 0.2371 ┆ … ┆ 0.37    ┆ 1.0        ┆ ctrl_rf_q8_f40          ┆ false            │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ ctrl_raw_q8             ┆ 0.2371 ┆ … ┆ -3.46   ┆ 0.008058   ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ lightgbm                ┆ 0.2371 ┆ … ┆ 15.35   ┆ 1.6692e-51 ┆ lightgbm                ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ patchtst                ┆ 0.2371 ┆ … ┆ 11.97   ┆ 2.0936e-31 ┆ patchtst                ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ gru_uni                 ┆ 0.2371 ┆ … ┆ 7.39    ┆ 4.6712e-12 ┆ gru_uni                 ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ persistence             ┆ 0.2371 ┆ … ┆ -2.42   ┆ 0.168454   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 1   ┆ ctrl_mlp_q4_l2          ┆ qrc_q8_d4               ┆ 0.2408 ┆ … ┆ 4.54    ┆ 0.000123   ┆ qrc_q8_d4               ┆ true             │
│ 1   ┆ hqnn_noent_noreup_q4_l2 ┆ qrc_q8_d4               ┆ 0.2382 ┆ … ┆ 2.74    ┆ 0.074793   ┆ qrc_q8_d4               ┆ false            │
│ 1   ┆ ctrl_bottleneck_q4_l2   ┆ qrc_q8_d4               ┆ 0.2343 ┆ … ┆ -3.58   ┆ 0.005506   ┆ ctrl_bottleneck_q4_l2   ┆ true             │
│ 1   ┆ qrc_q8_d4               ┆ ctrl_rf_q8_f40          ┆ 0.2364 ┆ … ┆ -2.17   ┆ 0.270848   ┆ qrc_q8_d4               ┆ false            │
│ 1   ┆ qrc_q8_d4               ┆ ctrl_raw_q8             ┆ 0.2364 ┆ … ┆ -6.59   ┆ 1.3526e-9  ┆ qrc_q8_d4               ┆ true             │
│ 1   ┆ qrc_q8_d4               ┆ lightgbm                ┆ 0.2364 ┆ … ┆ 17.35   ┆ 1.0513e-65 ┆ lightgbm                ┆ true             │
│ 1   ┆ qrc_q8_d4               ┆ patchtst                ┆ 0.2364 ┆ … ┆ 12.49   ┆ 3.6223e-34 ┆ patchtst                ┆ true             │
│ 1   ┆ qrc_q8_d4               ┆ gru_uni                 ┆ 0.2364 ┆ … ┆ 5.52    ┆ 7.7617e-7  ┆ gru_uni                 ┆ true             │
│ 1   ┆ qrc_q8_d4               ┆ persistence             ┆ 0.2364 ┆ … ┆ -3.33   ┆ 0.01234    ┆ qrc_q8_d4               ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ ctrl_mlp_q4_l2          ┆ 0.3871 ┆ … ┆ -1.57   ┆ 1.0        ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ hqnn_noent_noreup_q4_l2 ┆ 0.3871 ┆ … ┆ -2.45   ┆ 0.243878   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ ctrl_bottleneck_q4_l2   ┆ 0.3871 ┆ … ┆ -1.02   ┆ 1.0        ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ qrc_q8_d4               ┆ 0.3871 ┆ … ┆ -4.2    ┆ 0.000852   ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ ctrl_rf_q8_f40          ┆ 0.3871 ┆ … ┆ -4.56   ┆ 0.000192   ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ ctrl_raw_q8             ┆ 0.3871 ┆ … ┆ -3.27   ┆ 0.026954   ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ lightgbm                ┆ 0.3871 ┆ … ┆ 6.4     ┆ 6.9592e-9  ┆ lightgbm                ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ patchtst                ┆ 0.3871 ┆ … ┆ 1.44    ┆ 1.0        ┆ patchtst                ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ gru_uni                 ┆ 0.3871 ┆ … ┆ 3.33    ┆ 0.02265    ┆ gru_uni                 ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ persistence             ┆ 0.3871 ┆ … ┆ -7.51   ┆ 2.9349e-12 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 5   ┆ ctrl_mlp_q4_l2          ┆ qrc_q8_d4               ┆ 0.3883 ┆ … ┆ -2.76   ┆ 0.104445   ┆ ctrl_mlp_q4_l2          ┆ false            │
│ 5   ┆ hqnn_noent_noreup_q4_l2 ┆ qrc_q8_d4               ┆ 0.3882 ┆ … ┆ -3.22   ┆ 0.031262   ┆ hqnn_noent_noreup_q4_l2 ┆ true             │
│ 5   ┆ ctrl_bottleneck_q4_l2   ┆ qrc_q8_d4               ┆ 0.3879 ┆ … ┆ -2.95   ┆ 0.059985   ┆ ctrl_bottleneck_q4_l2   ┆ false            │
│ 5   ┆ qrc_q8_d4               ┆ ctrl_rf_q8_f40          ┆ 0.3915 ┆ … ┆ -2.01   ┆ 0.578845   ┆ qrc_q8_d4               ┆ false            │
│ 5   ┆ qrc_q8_d4               ┆ ctrl_raw_q8             ┆ 0.3915 ┆ … ┆ 1.12    ┆ 1.0        ┆ ctrl_raw_q8             ┆ false            │
│ 5   ┆ qrc_q8_d4               ┆ lightgbm                ┆ 0.3915 ┆ … ┆ 8.5     ┆ 1.0167e-15 ┆ lightgbm                ┆ true             │
│ 5   ┆ qrc_q8_d4               ┆ patchtst                ┆ 0.3915 ┆ … ┆ 3.97    ┆ 0.002215   ┆ patchtst                ┆ true             │
│ 5   ┆ qrc_q8_d4               ┆ gru_uni                 ┆ 0.3915 ┆ … ┆ 5.67    ┆ 5.7636e-7  ┆ gru_uni                 ┆ true             │
│ 5   ┆ qrc_q8_d4               ┆ persistence             ┆ 0.3915 ┆ … ┆ -4.23   ┆ 0.000769   ┆ qrc_q8_d4               ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ ctrl_mlp_q4_l2          ┆ 0.6042 ┆ … ┆ -2.18   ┆ 0.502773   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ hqnn_noent_noreup_q4_l2 ┆ 0.6042 ┆ … ┆ -2.83   ┆ 0.105836   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ ctrl_bottleneck_q4_l2   ┆ 0.6042 ┆ … ┆ -3.49   ┆ 0.013395   ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ qrc_q8_d4               ┆ 0.6042 ┆ … ┆ -5.71   ┆ 5.1784e-7  ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ ctrl_rf_q8_f40          ┆ 0.6042 ┆ … ┆ -6.01   ┆ 8.9577e-8  ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ ctrl_raw_q8             ┆ 0.6042 ┆ … ┆ -7.43   ┆ 5.9411e-12 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ lightgbm                ┆ 0.6042 ┆ … ┆ 1.93    ┆ 0.803369   ┆ lightgbm                ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ patchtst                ┆ 0.6042 ┆ … ┆ -0.17   ┆ 1.0        ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ gru_uni                 ┆ 0.6042 ┆ … ┆ 0.91    ┆ 1.0        ┆ gru_uni                 ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ persistence             ┆ 0.6042 ┆ … ┆ -8.67   ┆ 2.2367e-16 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ ctrl_mlp_q4_l2          ┆ qrc_q8_d4               ┆ 0.6061 ┆ … ┆ -4.48   ┆ 0.000258   ┆ ctrl_mlp_q4_l2          ┆ true             │
│ 15  ┆ hqnn_noent_noreup_q4_l2 ┆ qrc_q8_d4               ┆ 0.6058 ┆ … ┆ -4.76   ┆ 0.000075   ┆ hqnn_noent_noreup_q4_l2 ┆ true             │
│ 15  ┆ ctrl_bottleneck_q4_l2   ┆ qrc_q8_d4               ┆ 0.6084 ┆ … ┆ -2.42   ┆ 0.292707   ┆ ctrl_bottleneck_q4_l2   ┆ false            │
│ 15  ┆ qrc_q8_d4               ┆ ctrl_rf_q8_f40          ┆ 0.613  ┆ … ┆ -2.55   ┆ 0.213011   ┆ qrc_q8_d4               ┆ false            │
│ 15  ┆ qrc_q8_d4               ┆ ctrl_raw_q8             ┆ 0.613  ┆ … ┆ -0.37   ┆ 1.0        ┆ qrc_q8_d4               ┆ false            │
│ 15  ┆ qrc_q8_d4               ┆ lightgbm                ┆ 0.613  ┆ … ┆ 6.69    ┆ 1.0861e-9  ┆ lightgbm                ┆ true             │
│ 15  ┆ qrc_q8_d4               ┆ patchtst                ┆ 0.613  ┆ … ┆ 3.51    ┆ 0.013131   ┆ patchtst                ┆ true             │
│ 15  ┆ qrc_q8_d4               ┆ gru_uni                 ┆ 0.613  ┆ … ┆ 4.18    ┆ 0.000978   ┆ gru_uni                 ┆ true             │
│ 15  ┆ qrc_q8_d4               ┆ persistence             ┆ 0.613  ┆ … ┆ -3.48   ┆ 0.013529   ┆ qrc_q8_d4               ┆ true             │
└─────┴─────────────────────────┴─────────────────────────┴────────┴───┴─────────┴────────────┴─────────────────────────┴──────────────────┘
shape: (33, 8)
┌─────┬─────────────────────────┬────────┬────────────┬──────────┬──────────────┬────────────┬─────────┐
│ h   ┆ model                   ┆ n_ramp ┆ MAE_normal ┆ MAE_ramp ┆ skill_normal ┆ skill_ramp ┆ ramp_F1 │
│ --- ┆ ---                     ┆ ---    ┆ ---        ┆ ---      ┆ ---          ┆ ---        ┆ ---     │
│ i64 ┆ str                     ┆ i64    ┆ f64        ┆ f64      ┆ f64          ┆ f64        ┆ f64     │
╞═════╪═════════════════════════╪════════╪════════════╪══════════╪══════════════╪════════════╪═════════╡
│ 1   ┆ gru_uni                 ┆ 1659   ┆ 0.1972     ┆ 1.1      ┆ 0.007        ┆ 0.121      ┆ 0.169   │
│ 1   ┆ ctrl_mlp_q4_l2          ┆ 1659   ┆ 0.2047     ┆ 1.119    ┆ -0.03        ┆ 0.106      ┆ 0.176   │
│ 1   ┆ lightgbm                ┆ 1659   ┆ 0.1884     ┆ 1.12     ┆ 0.052        ┆ 0.106      ┆ 0.137   │
│ 1   ┆ patchtst                ┆ 1659   ┆ 0.1912     ┆ 1.12     ┆ 0.037        ┆ 0.106      ┆ 0.102   │
│ 1   ┆ hqnn_noent_noreup_q4_l2 ┆ 1659   ┆ 0.2017     ┆ 1.121    ┆ -0.016       ┆ 0.105      ┆ 0.156   │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ 1659   ┆ 0.2007     ┆ 1.121    ┆ -0.01        ┆ 0.104      ┆ 0.168   │
│ 1   ┆ qrc_q8_d4               ┆ 1659   ┆ 0.1992     ┆ 1.14     ┆ -0.003       ┆ 0.089      ┆ 0.049   │
│ 1   ┆ ctrl_rf_q8_f40          ┆ 1659   ┆ 0.1995     ┆ 1.143    ┆ -0.005       ┆ 0.087      ┆ 0.051   │
│ 1   ┆ ctrl_bottleneck_q4_l2   ┆ 1659   ┆ 0.1959     ┆ 1.166    ┆ 0.014        ┆ 0.069      ┆ 0.0     │
│ 1   ┆ ctrl_raw_q8             ┆ 1659   ┆ 0.2009     ┆ 1.181    ┆ -0.012       ┆ 0.057      ┆ 0.002   │
│ 1   ┆ persistence             ┆ 1659   ┆ 0.1986     ┆ 1.252    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 5   ┆ patchtst                ┆ 2969   ┆ 0.2889     ┆ 1.664    ┆ 0.009        ┆ 0.092      ┆ 0.154   │
│ 5   ┆ gru_uni                 ┆ 2969   ┆ 0.2856     ┆ 1.682    ┆ 0.02         ┆ 0.083      ┆ 0.127   │
│ 5   ┆ lightgbm                ┆ 2969   ┆ 0.2837     ┆ 1.682    ┆ 0.027        ┆ 0.082      ┆ 0.143   │
│ 5   ┆ ctrl_mlp_q4_l2          ┆ 2969   ┆ 0.2895     ┆ 1.697    ┆ 0.007        ┆ 0.074      ┆ 0.158   │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ 2969   ┆ 0.2873     ┆ 1.708    ┆ 0.014        ┆ 0.068      ┆ 0.153   │
│ 5   ┆ hqnn_noent_noreup_q4_l2 ┆ 2969   ┆ 0.2886     ┆ 1.708    ┆ 0.01         ┆ 0.068      ┆ 0.159   │
│ 5   ┆ ctrl_rf_q8_f40          ┆ 2969   ┆ 0.2912     ┆ 1.732    ┆ 0.001        ┆ 0.055      ┆ 0.106   │
│ 5   ┆ qrc_q8_d4               ┆ 2969   ┆ 0.29       ┆ 1.739    ┆ 0.005        ┆ 0.051      ┆ 0.092   │
│ 5   ┆ ctrl_bottleneck_q4_l2   ┆ 2969   ┆ 0.2834     ┆ 1.771    ┆ 0.028        ┆ 0.034      ┆ 0.0     │
│ 5   ┆ ctrl_raw_q8             ┆ 2969   ┆ 0.2844     ┆ 1.792    ┆ 0.024        ┆ 0.022      ┆ 0.0     │
│ 5   ┆ persistence             ┆ 2969   ┆ 0.2915     ┆ 1.833    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 15  ┆ gru_uni                 ┆ 3705   ┆ 0.4352     ┆ 2.351    ┆ 0.011        ┆ 0.068      ┆ 0.092   │
│ 15  ┆ lightgbm                ┆ 3705   ┆ 0.4326     ┆ 2.361    ┆ 0.017        ┆ 0.064      ┆ 0.109   │
│ 15  ┆ patchtst                ┆ 3705   ┆ 0.4362     ┆ 2.366    ┆ 0.009        ┆ 0.062      ┆ 0.097   │
│ 15  ┆ ctrl_mlp_q4_l2          ┆ 3705   ┆ 0.4347     ┆ 2.4      ┆ 0.012        ┆ 0.049      ┆ 0.074   │
│ 15  ┆ hqnn_noent_noreup_q4_l2 ┆ 3705   ┆ 0.4333     ┆ 2.41     ┆ 0.015        ┆ 0.045      ┆ 0.061   │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ 3705   ┆ 0.4315     ┆ 2.41     ┆ 0.02         ┆ 0.045      ┆ 0.066   │
│ 15  ┆ qrc_q8_d4               ┆ 3705   ┆ 0.4386     ┆ 2.439    ┆ 0.003        ┆ 0.033      ┆ 0.043   │
│ 15  ┆ ctrl_rf_q8_f40          ┆ 3705   ┆ 0.4399     ┆ 2.44     ┆ 0.001        ┆ 0.033      ┆ 0.054   │
│ 15  ┆ ctrl_bottleneck_q4_l2   ┆ 3705   ┆ 0.429      ┆ 2.482    ┆ 0.025        ┆ 0.016      ┆ 0.0     │
│ 15  ┆ ctrl_raw_q8             ┆ 3705   ┆ 0.435      ┆ 2.483    ┆ 0.012        ┆ 0.016      ┆ 0.0     │
│ 15  ┆ persistence             ┆ 3705   ┆ 0.4401     ┆ 2.523    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
└─────┴─────────────────────────┴────────┴────────────┴──────────┴──────────────┴────────────┴─────────┘
shape: (5, 13)
┌─────────────────────┬──────────────────────┬───────────────────┬──────────────┬───┬───────────────────────┬────────────────────┬───────────────┬─────────────┐
│ block_start         ┆ hqnn_noreup_q4_l2 h1 ┆ ctrl_mlp_q4_l2 h1 ┆ qrc_q8_d4 h1 ┆ … ┆ hqnn_noreup_q4_l2 h15 ┆ ctrl_mlp_q4_l2 h15 ┆ qrc_q8_d4 h15 ┆ gru_uni h15 │
│ ---                 ┆ ---                  ┆ ---               ┆ ---          ┆   ┆ ---                   ┆ ---                ┆ ---           ┆ ---         │
│ datetime[μs]        ┆ f64                  ┆ f64               ┆ f64          ┆   ┆ f64                   ┆ f64                ┆ f64           ┆ f64         │
╞═════════════════════╪══════════════════════╪═══════════════════╪══════════════╪═══╪═══════════════════════╪════════════════════╪═══════════════╪═════════════╡
│ 2021-08-30 00:00:00 ┆ 0.011                ┆ 0.008             ┆ 0.009        ┆ … ┆ 0.041                 ┆ 0.04               ┆ 0.026         ┆ 0.041       │
│ 2021-09-06 00:00:00 ┆ 0.099                ┆ 0.094             ┆ 0.09         ┆ … ┆ 0.059                 ┆ 0.062              ┆ 0.042         ┆ 0.073       │
│ 2021-09-13 00:00:00 ┆ -0.027               ┆ -0.059            ┆ -0.016       ┆ … ┆ 0.011                 ┆ -0.0               ┆ -0.011        ┆ 0.02        │
│ 2021-09-20 00:00:00 ┆ -0.046               ┆ -0.074            ┆ -0.031       ┆ … ┆ 0.012                 ┆ 0.006              ┆ 0.007         ┆ 0.004       │
│ 2021-09-27 00:00:00 ┆ -0.012               ┆ -0.028            ┆ -0.008       ┆ … ┆ 0.024                 ┆ 0.023              ┆ 0.008         ┆ 0.029       │
└─────────────────────┴──────────────────────┴───────────────────┴──────────────┴───┴───────────────────────┴────────────────────┴───────────────┴─────────────┘
```

## Cell 7: summary
_execution count: 7_

```
h= 1: hqnn_noreup_q4_l2 rank 13 skill +0.020 | qrc_q8_d4 rank 17 skill +0.016 | gru_uni rank 8 skill +0.036 | patchtst rank 6 skill +0.048 | lightgbm rank 2 skill +0.064
h= 5: hqnn_noreup_q4_l2 rank 9 skill +0.034 | qrc_q8_d4 rank 18 skill +0.022 | gru_uni rank 5 skill +0.041 | patchtst rank 6 skill +0.039 | lightgbm rank 2 skill +0.046
h=15: hqnn_noreup_q4_l2 rank 6 skill +0.029 | qrc_q8_d4 rank 16 skill +0.015 | gru_uni rank 5 skill +0.031 | patchtst rank 7 skill +0.029 | lightgbm rank 3 skill +0.033
```

## Cell 8: export outputs (save the notebook first with Cmd+S, then run this cell)
_execution count: 8_

```
written 23_week6_quantum_test_outputs.md | code cells 10 | figures 0
```

## Cell 9
_execution count: None_

(no output)

## Cell 10
_execution count: None_

(no output)

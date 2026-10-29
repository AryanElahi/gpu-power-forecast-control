# Outputs of `19_week5_qrc.ipynb`
Exported 2026-10-11 05:05 from the saved notebook file.

## Cell 1: setup
_execution count: 1_

```
pennylane 0.45.1 | features per minute (qrc): 40 | window features: 160
```

## Cell 2: unit checks (no real data)
_execution count: 2_

```
(a) broadcast reservoir vs one-sample circuit: max |diff| Z 1.6e-08, XX 3.6e-09
(b) no entanglement: change of single-qubit features of qubits 1..3 when only u_0 changes: 0.0e+00
(c) closed-form ridge vs sklearn Ridge: max |coef diff| 4.0e-14
(d) qrc        features per minute  40 | window features  160 | readout params (3 horizons) 483
(d) qrc_noent  features per minute  40 | window features  160 | readout params (3 horizons) 483
(d) rf         features per minute  40 | window features  160 | readout params (3 horizons) 483
(d) raw        features per minute   8 | window features   32 | readout params (3 horizons) 99
{'broadcast_equals_single_circuit': True, 'noent_is_product_state': True, 'ridge_closed_form_equals_sklearn': True} 
UNIT CHECKS PASSED
```

## Cell 3: DRY RUN on synthetic data (4 qubits, every reservoir type; deleted afterwards)
_execution count: 3_

```
qrc_q4_d4        files ok True | finite True | skill +0.029 / +0.047 / +0.079
qrc_noent_q4_d4  files ok True | finite True | skill +0.034 / +0.048 / +0.079
ctrl_rf_q4_f20   files ok True | finite True | skill +0.035 / +0.063 / +0.112
ctrl_raw_q4      files ok True | finite True | skill +0.034 / +0.059 / +0.097
DRY RUN PASSED
```

## Cell 4: real data, inputs and reservoir timing (10,000 minutes per configuration)
_execution count: 4_

```
minutes 303,840 | inputs per minute 8 | share of exact zeros 0.070 | |u| quantiles 50/95/99%: [0.359 0.957 0.998]
shape: (8, 4)
┌──────────┬──────────┬───────────┬─────────────────────┐
│ n_qubits ┆ entangle ┆ s_per_10k ┆ est_min_all_minutes │
│ ---      ┆ ---      ┆ ---       ┆ ---                 │
│ i64      ┆ bool     ┆ f64       ┆ f64                 │
╞══════════╪══════════╪═══════════╪═════════════════════╡
│ 2        ┆ true     ┆ 0.02      ┆ 0.01                │
│ 2        ┆ false    ┆ 0.02      ┆ 0.01                │
│ 4        ┆ true     ┆ 0.15      ┆ 0.07                │
│ 4        ┆ false    ┆ 0.14      ┆ 0.07                │
│ 6        ┆ true     ┆ 0.6       ┆ 0.3                 │
│ 6        ┆ false    ┆ 0.45      ┆ 0.23                │
│ 8        ┆ true     ┆ 3.11      ┆ 1.58                │
│ 8        ┆ false    ┆ 2.76      ┆ 1.4                 │
└──────────┴──────────┴───────────┴─────────────────────┘
```

## Cell 5: SCALE SCREEN, seed 0 (input scale a for qrc, qrc_noent and rf; raw has none). Resumable.
_execution count: 5_

```
[04:33:55] qrc_q8_d4_a0p393_s0: reservoir + readout
```

```stderr
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /Users/aryaneuse/.netrc.
```

```
[04:35:45] qrc_q8_d4_a0p393_s0: features 160 | readout params 483 | reservoir 96 s | val MAE 0.2659 / 0.4127 / 0.5864 kW | skill +0.072 / +0.040 / +0.030
[04:35:45] qrc_q8_d4_a0p785_s0: reservoir + readout
[04:37:36] qrc_q8_d4_a0p785_s0: features 160 | readout params 483 | reservoir 100 s | val MAE 0.2682 / 0.4141 / 0.5880 kW | skill +0.064 / +0.037 / +0.028
[04:37:36] qrc_q8_d4_a1p571_s0: reservoir + readout
[04:39:21] qrc_q8_d4_a1p571_s0: features 160 | readout params 483 | reservoir 93 s | val MAE 0.2722 / 0.4176 / 0.5919 kW | skill +0.050 / +0.029 / +0.021
[04:39:21] qrc_q8_d4_a3p142_s0: reservoir + readout
[04:41:04] qrc_q8_d4_a3p142_s0: features 160 | readout params 483 | reservoir 93 s | val MAE 0.2810 / 0.4267 / 0.6022 kW | skill +0.019 / +0.008 / +0.004
[04:41:04] qrc_noent_q8_d4_a0p393_s0: reservoir + readout
[04:42:43] qrc_noent_q8_d4_a0p393_s0: features 160 | readout params 483 | reservoir 88 s | val MAE 0.2649 / 0.4124 / 0.5870 kW | skill +0.076 / +0.041 / +0.030
[04:42:43] qrc_noent_q8_d4_a0p785_s0: reservoir + readout
[04:44:58] qrc_noent_q8_d4_a0p785_s0: features 160 | readout params 483 | reservoir 123 s | val MAE 0.2652 / 0.4130 / 0.5874 kW | skill +0.074 / +0.039 / +0.029
[04:44:58] qrc_noent_q8_d4_a1p571_s0: reservoir + readout
[04:47:01] qrc_noent_q8_d4_a1p571_s0: features 160 | readout params 483 | reservoir 112 s | val MAE 0.2669 / 0.4152 / 0.5893 kW | skill +0.069 / +0.034 / +0.026
[04:47:01] qrc_noent_q8_d4_a3p142_s0: reservoir + readout
[04:49:19] qrc_noent_q8_d4_a3p142_s0: features 160 | readout params 483 | reservoir 126 s | val MAE 0.2766 / 0.4238 / 0.5994 kW | skill +0.035 / +0.014 / +0.009
[04:49:19] ctrl_rf_q8_f40_a0p393_s0: reservoir + readout
[04:49:32] ctrl_rf_q8_f40_a0p393_s0: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2645 / 0.4117 / 0.5865 kW | skill +0.077 / +0.042 / +0.030
[04:49:32] ctrl_rf_q8_f40_a0p785_s0: reservoir + readout
[04:49:43] ctrl_rf_q8_f40_a0p785_s0: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2658 / 0.4125 / 0.5870 kW | skill +0.073 / +0.041 / +0.030
[04:49:43] ctrl_rf_q8_f40_a1p571_s0: reservoir + readout
[04:49:54] ctrl_rf_q8_f40_a1p571_s0: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2679 / 0.4155 / 0.5900 kW | skill +0.065 / +0.034 / +0.025
[04:49:54] ctrl_rf_q8_f40_a3p142_s0: reservoir + readout
[04:50:03] ctrl_rf_q8_f40_a3p142_s0: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2755 / 0.4243 / 0.5991 kW | skill +0.039 / +0.013 / +0.009
[04:50:03] ctrl_raw_q8_s0: reservoir + readout
[04:50:10] ctrl_raw_q8_s0: features 32 | readout params 99 | reservoir 0 s | val MAE 0.2700 / 0.4161 / 0.5920 kW | skill +0.058 / +0.032 / +0.021
shape: (13, 8)
┌───────────┬──────────────┬────────┬──────────┬──────────┬───────────┬──────────────┬─────────────┐
│ kind      ┆ variant      ┆ scale  ┆ skill_h1 ┆ skill_h5 ┆ skill_h15 ┆ val_MAE_mean ┆ ridge_at_ed │
│ ---       ┆ ---          ┆ ---    ┆ ---      ┆ ---      ┆ ---       ┆ ---          ┆ ge          │
│ str       ┆ str          ┆ f64    ┆ f64      ┆ f64      ┆ f64       ┆ f64          ┆ ---         │
│           ┆              ┆        ┆          ┆          ┆           ┆              ┆ bool        │
╞═══════════╪══════════════╪════════╪══════════╪══════════╪═══════════╪══════════════╪═════════════╡
│ qrc       ┆ qrc_q8_d4    ┆ 0.3927 ┆ 0.0721   ┆ 0.04     ┆ 0.0305    ┆ 0.42169      ┆ false       │
│ qrc       ┆ qrc_q8_d4    ┆ 0.7854 ┆ 0.0642   ┆ 0.0368   ┆ 0.0278    ┆ 0.42344      ┆ false       │
│ qrc       ┆ qrc_q8_d4    ┆ 1.5708 ┆ 0.05     ┆ 0.0287   ┆ 0.0214    ┆ 0.42725      ┆ false       │
│ qrc       ┆ qrc_q8_d4    ┆ 3.1416 ┆ 0.0194   ┆ 0.0075   ┆ 0.0044    ┆ 0.43663      ┆ true        │
│ qrc_noent ┆ qrc_noent_q8 ┆ 0.3927 ┆ 0.0757   ┆ 0.0407   ┆ 0.0295    ┆ 0.42144      ┆ true        │
│           ┆ _d4          ┆        ┆          ┆          ┆           ┆              ┆             │
│ …         ┆ …            ┆ …      ┆ …        ┆ …        ┆ …         ┆ …            ┆ …           │
│ raw       ┆ ctrl_raw_q8  ┆ 1.5708 ┆ 0.0578   ┆ 0.0321   ┆ 0.0213    ┆ 0.42604      ┆ true        │
│ rf        ┆ ctrl_rf_q8_f ┆ 0.3927 ┆ 0.0769   ┆ 0.0424   ┆ 0.0304    ┆ 0.4209       ┆ false       │
│           ┆ 40           ┆        ┆          ┆          ┆           ┆              ┆             │
│ rf        ┆ ctrl_rf_q8_f ┆ 0.7854 ┆ 0.0725   ┆ 0.0406   ┆ 0.0296    ┆ 0.42175      ┆ false       │
│           ┆ 40           ┆        ┆          ┆          ┆           ┆              ┆             │
│ rf        ┆ ctrl_rf_q8_f ┆ 1.5708 ┆ 0.0651   ┆ 0.0336   ┆ 0.0245    ┆ 0.42447      ┆ false       │
│           ┆ 40           ┆        ┆          ┆          ┆           ┆              ┆             │
│ rf        ┆ ctrl_rf_q8_f ┆ 3.1416 ┆ 0.0388   ┆ 0.013    ┆ 0.0095    ┆ 0.43296      ┆ false       │
│           ┆ 40           ┆        ┆          ┆          ┆           ┆              ┆             │
└───────────┴──────────────┴────────┴──────────┴──────────┴───────────┴──────────────┴─────────────┘
chosen scale at the edge of the grid (consider extending it): ['qrc', 'qrc_noent', 'rf']

chosen input scale per reservoir type (provisional, validation): {'qrc': 0.3927, 'qrc_noent': 0.3927, 'rf': 0.3927, 'raw': 1.5708} | 16.2 min
```

## Cell 6: THREE RESERVOIR SEEDS at the chosen scale (seed 0 reused from the screen). Resumable.
_execution count: 6_

```
[04:50:10] qrc_q8_d4_a0p393_s0: done earlier, skipped
[04:50:10] qrc_noent_q8_d4_a0p393_s0: done earlier, skipped
[04:50:10] ctrl_rf_q8_f40_a0p393_s0: done earlier, skipped
[04:50:10] ctrl_raw_q8_s0: done earlier, skipped
[04:50:10] qrc_q8_d4_a0p393_s1: reservoir + readout
[04:52:56] qrc_q8_d4_a0p393_s1: features 160 | readout params 483 | reservoir 154 s | val MAE 0.2656 / 0.4127 / 0.5864 kW | skill +0.073 / +0.040 / +0.031
[04:52:56] qrc_noent_q8_d4_a0p393_s1: reservoir + readout
[04:54:46] qrc_noent_q8_d4_a0p393_s1: features 160 | readout params 483 | reservoir 99 s | val MAE 0.2647 / 0.4122 / 0.5874 kW | skill +0.076 / +0.041 / +0.029
[04:54:46] ctrl_rf_q8_f40_a0p393_s1: reservoir + readout
[04:55:00] ctrl_rf_q8_f40_a0p393_s1: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2646 / 0.4115 / 0.5863 kW | skill +0.077 / +0.043 / +0.031
[04:55:00] qrc_q8_d4_a0p393_s2: reservoir + readout
[04:57:26] qrc_q8_d4_a0p393_s2: features 160 | readout params 483 | reservoir 131 s | val MAE 0.2658 / 0.4123 / 0.5871 kW | skill +0.072 / +0.041 / +0.029
[04:57:26] qrc_noent_q8_d4_a0p393_s2: reservoir + readout
[05:00:47] qrc_noent_q8_d4_a0p393_s2: features 160 | readout params 483 | reservoir 191 s | val MAE 0.2650 / 0.4123 / 0.5875 kW | skill +0.075 / +0.041 / +0.029
[05:00:47] ctrl_rf_q8_f40_a0p393_s2: reservoir + readout
[05:01:03] ctrl_rf_q8_f40_a0p393_s2: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2657 / 0.4128 / 0.5874 kW | skill +0.073 / +0.040 / +0.029

all runs handled in 10.9 min
```

## Cell 7: table (mean +- sd over reservoir seeds), merged with the Week 4 benchmark and the hybrid QNN (if available)
_execution count: 7_

```
shape: (12, 11)
┌─────┬─────────────────┬─────────┬──────────┬────────┬───────────┬──────────┬────────────┬─────────┬──────────┬─────────────┐
│ h   ┆ model           ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 ┆ n_params ┆ reservoir_s │
│ --- ┆ ---             ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     ┆ ---      ┆ ---         │
│ i64 ┆ str             ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     ┆ i64      ┆ f64         │
╞═════╪═════════════════╪═════════╪══════════╪════════╪═══════════╪══════════╪════════════╪═════════╪══════════╪═════════════╡
│ 1   ┆ qrc_noent_q8_d4 ┆ 3       ┆ 0.2649   ┆ 0.0001 ┆ 0.076     ┆ 1.434    ┆ 0.2115     ┆ 0.029   ┆ 483      ┆ 126.0       │
│ 1   ┆ ctrl_rf_q8_f40  ┆ 3       ┆ 0.265    ┆ 0.0007 ┆ 0.075     ┆ 1.415    ┆ 0.2125     ┆ 0.051   ┆ 483      ┆ 0.0         │
│ 1   ┆ qrc_q8_d4       ┆ 3       ┆ 0.2658   ┆ 0.0002 ┆ 0.073     ┆ 1.424    ┆ 0.213      ┆ 0.037   ┆ 483      ┆ 127.0       │
│ 1   ┆ ctrl_raw_q8     ┆ 1       ┆ 0.27     ┆ null   ┆ 0.058     ┆ 1.479    ┆ 0.2148     ┆ 0.012   ┆ 99       ┆ 0.0         │
│ 5   ┆ ctrl_rf_q8_f40  ┆ 3       ┆ 0.412    ┆ 0.0007 ┆ 0.042     ┆ 2.414    ┆ 0.2907     ┆ 0.127   ┆ 483      ┆ 0.0         │
│ 5   ┆ qrc_noent_q8_d4 ┆ 3       ┆ 0.4123   ┆ 0.0001 ┆ 0.041     ┆ 2.463    ┆ 0.288      ┆ 0.067   ┆ 483      ┆ 126.0       │
│ 5   ┆ qrc_q8_d4       ┆ 3       ┆ 0.4126   ┆ 0.0002 ┆ 0.04      ┆ 2.426    ┆ 0.2906     ┆ 0.119   ┆ 483      ┆ 127.0       │
│ 5   ┆ ctrl_raw_q8     ┆ 1       ┆ 0.4161   ┆ null   ┆ 0.032     ┆ 2.554    ┆ 0.2866     ┆ 0.002   ┆ 99       ┆ 0.0         │
│ 15  ┆ qrc_q8_d4       ┆ 3       ┆ 0.5866   ┆ 0.0004 ┆ 0.03      ┆ 3.097    ┆ 0.4127     ┆ 0.056   ┆ 483      ┆ 127.0       │
│ 15  ┆ ctrl_rf_q8_f40  ┆ 3       ┆ 0.5867   ┆ 0.0006 ┆ 0.03      ┆ 3.091    ┆ 0.4133     ┆ 0.071   ┆ 483      ┆ 0.0         │
│ 15  ┆ qrc_noent_q8_d4 ┆ 3       ┆ 0.5873   ┆ 0.0003 ┆ 0.029     ┆ 3.123    ┆ 0.4117     ┆ 0.022   ┆ 483      ┆ 126.0       │
│ 15  ┆ ctrl_raw_q8     ┆ 1       ┆ 0.592    ┆ null   ┆ 0.021     ┆ 3.217    ┆ 0.4103     ┆ 0.001   ┆ 99       ┆ 0.0         │
└─────┴─────────────────┴─────────┴──────────┴────────┴───────────┴──────────┴────────────┴─────────┴──────────┴─────────────┘

=== h = 1 min: validation ranking (28 models) ===
shape: (13, 6)
┌──────┬───────────────────┬─────────────────────────┬─────────┬──────────┬───────────┐
│ rank ┆ family            ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ skill_MAE │
│ ---  ┆ ---               ┆ ---                     ┆ ---     ┆ ---      ┆ ---       │
│ u32  ┆ str               ┆ str                     ┆ i64     ┆ f64      ┆ f64       │
╞══════╪═══════════════════╪═════════════════════════╪═════════╪══════════╪═══════════╡
│ 2    ┆ ML                ┆ lightgbm                ┆ 3       ┆ 0.2491   ┆ 0.131     │
│ 3    ┆ deep              ┆ patchtst                ┆ 3       ┆ 0.2543   ┆ 0.112     │
│ 7    ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.2618   ┆ 0.086     │
│ 10   ┆ quantum reservoir ┆ qrc_noent_q8_d4         ┆ 3       ┆ 0.2649   ┆ 0.076     │
│ 11   ┆ classical control ┆ ctrl_rf_q8_f40          ┆ 3       ┆ 0.265    ┆ 0.075     │
│ 12   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.2653   ┆ 0.074     │
│ 13   ┆ quantum reservoir ┆ qrc_q8_d4               ┆ 3       ┆ 0.2658   ┆ 0.073     │
│ 15   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.2673   ┆ 0.067     │
│ 16   ┆ classical control ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.2676   ┆ 0.066     │
│ 19   ┆ classical control ┆ ctrl_raw_q8             ┆ 1       ┆ 0.27     ┆ 0.058     │
│ 20   ┆ deep              ┆ gru_uni                 ┆ 3       ┆ 0.27     ┆ 0.058     │
│ 21   ┆ classical control ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.2706   ┆ 0.056     │
│ 27   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.2866   ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴───────────┘

=== h = 5 min: validation ranking (28 models) ===
shape: (13, 6)
┌──────┬───────────────────┬─────────────────────────┬─────────┬──────────┬───────────┐
│ rank ┆ family            ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ skill_MAE │
│ ---  ┆ ---               ┆ ---                     ┆ ---     ┆ ---      ┆ ---       │
│ u32  ┆ str               ┆ str                     ┆ i64     ┆ f64      ┆ f64       │
╞══════╪═══════════════════╪═════════════════════════╪═════════╪══════════╪═══════════╡
│ 2    ┆ ML                ┆ lightgbm                ┆ 3       ┆ 0.3934   ┆ 0.085     │
│ 4    ┆ deep              ┆ patchtst                ┆ 3       ┆ 0.3989   ┆ 0.072     │
│ 8    ┆ deep              ┆ gru_uni                 ┆ 3       ┆ 0.4016   ┆ 0.066     │
│ 9    ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.4064   ┆ 0.055     │
│ 10   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.4069   ┆ 0.054     │
│ 11   ┆ classical control ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.4073   ┆ 0.053     │
│ 13   ┆ classical control ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.4104   ┆ 0.045     │
│ 14   ┆ classical control ┆ ctrl_rf_q8_f40          ┆ 3       ┆ 0.412    ┆ 0.042     │
│ 15   ┆ quantum reservoir ┆ qrc_noent_q8_d4         ┆ 3       ┆ 0.4123   ┆ 0.041     │
│ 16   ┆ quantum reservoir ┆ qrc_q8_d4               ┆ 3       ┆ 0.4126   ┆ 0.04      │
│ 17   ┆ classical control ┆ ctrl_raw_q8             ┆ 1       ┆ 0.4161   ┆ 0.032     │
│ 18   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.4161   ┆ 0.032     │
│ 26   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.4299   ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴───────────┘

=== h = 15 min: validation ranking (28 models) ===
shape: (13, 6)
┌──────┬───────────────────┬─────────────────────────┬─────────┬──────────┬───────────┐
│ rank ┆ family            ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ skill_MAE │
│ ---  ┆ ---               ┆ ---                     ┆ ---     ┆ ---      ┆ ---       │
│ u32  ┆ str               ┆ str                     ┆ i64     ┆ f64      ┆ f64       │
╞══════╪═══════════════════╪═════════════════════════╪═════════╪══════════╪═══════════╡
│ 2    ┆ ML                ┆ lightgbm                ┆ 3       ┆ 0.5659   ┆ 0.065     │
│ 4    ┆ deep              ┆ patchtst                ┆ 3       ┆ 0.5693   ┆ 0.059     │
│ 6    ┆ deep              ┆ gru_uni                 ┆ 3       ┆ 0.5722   ┆ 0.054     │
│ 10   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.5786   ┆ 0.043     │
│ 11   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.5799   ┆ 0.041     │
│ 12   ┆ classical control ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.5799   ┆ 0.041     │
│ 13   ┆ classical control ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.585    ┆ 0.033     │
│ 15   ┆ quantum reservoir ┆ qrc_q8_d4               ┆ 3       ┆ 0.5866   ┆ 0.03      │
│ 17   ┆ classical control ┆ ctrl_rf_q8_f40          ┆ 3       ┆ 0.5867   ┆ 0.03      │
│ 19   ┆ quantum reservoir ┆ qrc_noent_q8_d4         ┆ 3       ┆ 0.5873   ┆ 0.029     │
│ 20   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.5876   ┆ 0.028     │
│ 22   ┆ classical control ┆ ctrl_raw_q8             ┆ 1       ┆ 0.592    ┆ 0.021     │
│ 26   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.6048   ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴───────────┘
```

## Cell 8: Diebold-Mariano (Holm per horizon) and ramp breakdown, seed 0; reference: ridge AR on the 60 lags
_execution count: 8_

```
references found: ['ar_lags', 'patchtst', 'persistence']
shape: (33, 9)
┌─────┬─────────────────┬─────────────────┬────────┬────────┬─────────┬────────────┬─────────────────┬──────────────────┐
│ h   ┆ model_a         ┆ model_b         ┆ MAE_a  ┆ MAE_b  ┆ DM_stat ┆ DM_p_holm  ┆ better          ┆ significant_5pct │
│ --- ┆ ---             ┆ ---             ┆ ---    ┆ ---    ┆ ---     ┆ ---        ┆ ---             ┆ ---              │
│ i64 ┆ str             ┆ str             ┆ f64    ┆ f64    ┆ f64     ┆ f64        ┆ str             ┆ bool             │
╞═════╪═════════════════╪═════════════════╪════════╪════════╪═════════╪════════════╪═════════════════╪══════════════════╡
│ 1   ┆ qrc_q8_d4       ┆ qrc_noent_q8_d4 ┆ 0.2659 ┆ 0.2649 ┆ 2.87    ┆ 0.024765   ┆ qrc_noent_q8_d4 ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ ctrl_rf_q8_f40  ┆ 0.2659 ┆ 0.2645 ┆ 3.83    ┆ 0.0009     ┆ ctrl_rf_q8_f40  ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ ctrl_raw_q8     ┆ 0.2659 ┆ 0.27   ┆ -6.68   ┆ 1.9655e-10 ┆ qrc_q8_d4       ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ ar_lags         ┆ 0.2659 ┆ 0.2673 ┆ -0.83   ┆ 0.813181   ┆ qrc_q8_d4       ┆ false            │
│ 1   ┆ qrc_q8_d4       ┆ patchtst        ┆ 0.2659 ┆ 0.2533 ┆ 11.85   ┆ 3.1052e-31 ┆ patchtst        ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ persistence     ┆ 0.2659 ┆ 0.2866 ┆ -12.17  ┆ 8.5237e-33 ┆ qrc_q8_d4       ┆ true             │
│ 1   ┆ qrc_noent_q8_d4 ┆ ctrl_rf_q8_f40  ┆ 0.2649 ┆ 0.2645 ┆ 0.83    ┆ 0.813181   ┆ ctrl_rf_q8_f40  ┆ false            │
│ 1   ┆ qrc_noent_q8_d4 ┆ ctrl_raw_q8     ┆ 0.2649 ┆ 0.27   ┆ -10.16  ┆ 3.3659e-23 ┆ qrc_noent_q8_d4 ┆ true             │
│ 1   ┆ qrc_noent_q8_d4 ┆ ar_lags         ┆ 0.2649 ┆ 0.2673 ┆ -1.53   ┆ 0.380012   ┆ qrc_noent_q8_d4 ┆ false            │
│ 1   ┆ qrc_noent_q8_d4 ┆ patchtst        ┆ 0.2649 ┆ 0.2533 ┆ 11.98   ┆ 7.6358e-32 ┆ patchtst        ┆ true             │
│ 1   ┆ qrc_noent_q8_d4 ┆ persistence     ┆ 0.2649 ┆ 0.2866 ┆ -12.78  ┆ 4.0346e-36 ┆ qrc_noent_q8_d4 ┆ true             │
│ 5   ┆ qrc_q8_d4       ┆ qrc_noent_q8_d4 ┆ 0.4127 ┆ 0.4124 ┆ 0.3     ┆ 1.0        ┆ qrc_noent_q8_d4 ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ ctrl_rf_q8_f40  ┆ 0.4127 ┆ 0.4117 ┆ 2.31    ┆ 0.146889   ┆ ctrl_rf_q8_f40  ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ ctrl_raw_q8     ┆ 0.4127 ┆ 0.4161 ┆ -2.39   ┆ 0.136629   ┆ qrc_q8_d4       ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ ar_lags         ┆ 0.4127 ┆ 0.4161 ┆ -1.36   ┆ 0.689372   ┆ qrc_q8_d4       ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ patchtst        ┆ 0.4127 ┆ 0.3982 ┆ 6.81    ┆ 1.3343e-10 ┆ patchtst        ┆ true             │
│ 5   ┆ qrc_q8_d4       ┆ persistence     ┆ 0.4127 ┆ 0.4299 ┆ -7.93   ┆ 3.6342e-14 ┆ qrc_q8_d4       ┆ true             │
│ 5   ┆ qrc_noent_q8_d4 ┆ ctrl_rf_q8_f40  ┆ 0.4124 ┆ 0.4117 ┆ 0.75    ┆ 1.0        ┆ ctrl_rf_q8_f40  ┆ false            │
│ 5   ┆ qrc_noent_q8_d4 ┆ ctrl_raw_q8     ┆ 0.4124 ┆ 0.4161 ┆ -4.47   ┆ 0.000078   ┆ qrc_noent_q8_d4 ┆ true             │
│ 5   ┆ qrc_noent_q8_d4 ┆ ar_lags         ┆ 0.4124 ┆ 0.4161 ┆ -1.64   ┆ 0.506913   ┆ qrc_noent_q8_d4 ┆ false            │
│ 5   ┆ qrc_noent_q8_d4 ┆ patchtst        ┆ 0.4124 ┆ 0.3982 ┆ 6.42    ┆ 1.6005e-9  ┆ patchtst        ┆ true             │
│ 5   ┆ qrc_noent_q8_d4 ┆ persistence     ┆ 0.4124 ┆ 0.4299 ┆ -10.87  ┆ 3.3343e-26 ┆ qrc_noent_q8_d4 ┆ true             │
│ 15  ┆ qrc_q8_d4       ┆ qrc_noent_q8_d4 ┆ 0.5864 ┆ 0.587  ┆ -0.65   ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 15  ┆ qrc_q8_d4       ┆ ctrl_rf_q8_f40  ┆ 0.5864 ┆ 0.5865 ┆ -0.04   ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 15  ┆ qrc_q8_d4       ┆ ctrl_raw_q8     ┆ 0.5864 ┆ 0.592  ┆ -2.99   ┆ 0.025018   ┆ qrc_q8_d4       ┆ true             │
│ 15  ┆ qrc_q8_d4       ┆ ar_lags         ┆ 0.5864 ┆ 0.5876 ┆ -0.39   ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 15  ┆ qrc_q8_d4       ┆ patchtst        ┆ 0.5864 ┆ 0.5688 ┆ 5.48    ┆ 5.8331e-7  ┆ patchtst        ┆ true             │
│ 15  ┆ qrc_q8_d4       ┆ persistence     ┆ 0.5864 ┆ 0.6048 ┆ -6.4    ┆ 2.8042e-9  ┆ qrc_q8_d4       ┆ true             │
│ 15  ┆ qrc_noent_q8_d4 ┆ ctrl_rf_q8_f40  ┆ 0.587  ┆ 0.5865 ┆ 0.46    ┆ 1.0        ┆ ctrl_rf_q8_f40  ┆ false            │
│ 15  ┆ qrc_noent_q8_d4 ┆ ctrl_raw_q8     ┆ 0.587  ┆ 0.592  ┆ -3.3    ┆ 0.009712   ┆ qrc_noent_q8_d4 ┆ true             │
│ 15  ┆ qrc_noent_q8_d4 ┆ ar_lags         ┆ 0.587  ┆ 0.5876 ┆ -0.21   ┆ 1.0        ┆ qrc_noent_q8_d4 ┆ false            │
│ 15  ┆ qrc_noent_q8_d4 ┆ patchtst        ┆ 0.587  ┆ 0.5688 ┆ 5.25    ┆ 0.000002   ┆ patchtst        ┆ true             │
│ 15  ┆ qrc_noent_q8_d4 ┆ persistence     ┆ 0.587  ┆ 0.6048 ┆ -7.19   ┆ 1.1903e-11 ┆ qrc_noent_q8_d4 ┆ true             │
└─────┴─────────────────┴─────────────────┴────────┴────────┴─────────┴────────────┴─────────────────┴──────────────────┘
shape: (21, 8)
┌─────┬─────────────────┬────────┬────────────┬──────────┬──────────────┬────────────┬─────────┐
│ h   ┆ model           ┆ n_ramp ┆ MAE_normal ┆ MAE_ramp ┆ skill_normal ┆ skill_ramp ┆ ramp_F1 │
│ --- ┆ ---             ┆ ---    ┆ ---        ┆ ---      ┆ ---          ┆ ---        ┆ ---     │
│ i64 ┆ str             ┆ i64    ┆ f64        ┆ f64      ┆ f64          ┆ f64        ┆ f64     │
╞═════╪═════════════════╪════════╪════════════╪══════════╪══════════════╪════════════╪═════════╡
│ 1   ┆ patchtst        ┆ 1903   ┆ 0.2046     ┆ 1.322    ┆ 0.096        ┆ 0.178      ┆ 0.169   │
│ 1   ┆ ar_lags         ┆ 1903   ┆ 0.2152     ┆ 1.407    ┆ 0.049        ┆ 0.125      ┆ 0.145   │
│ 1   ┆ ctrl_rf_q8_f40  ┆ 1903   ┆ 0.2123     ┆ 1.411    ┆ 0.062        ┆ 0.122      ┆ 0.055   │
│ 1   ┆ qrc_q8_d4       ┆ 1903   ┆ 0.2133     ┆ 1.42     ┆ 0.058        ┆ 0.117      ┆ 0.038   │
│ 1   ┆ qrc_noent_q8_d4 ┆ 1903   ┆ 0.2115     ┆ 1.434    ┆ 0.066        ┆ 0.108      ┆ 0.031   │
│ 1   ┆ ctrl_raw_q8     ┆ 1903   ┆ 0.2148     ┆ 1.479    ┆ 0.051        ┆ 0.08       ┆ 0.012   │
│ 1   ┆ persistence     ┆ 1903   ┆ 0.2264     ┆ 1.607    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 5   ┆ patchtst        ┆ 2444   ┆ 0.287      ┆ 2.236    ┆ 0.033        ┆ 0.15       ┆ 0.257   │
│ 5   ┆ ar_lags         ┆ 2444   ┆ 0.2971     ┆ 2.377    ┆ -0.001       ┆ 0.096      ┆ 0.194   │
│ 5   ┆ ctrl_rf_q8_f40  ┆ 2444   ┆ 0.291      ┆ 2.405    ┆ 0.019        ┆ 0.086      ┆ 0.135   │
│ 5   ┆ qrc_q8_d4       ┆ 2444   ┆ 0.2918     ┆ 2.409    ┆ 0.017        ┆ 0.084      ┆ 0.135   │
│ 5   ┆ qrc_noent_q8_d4 ┆ 2444   ┆ 0.2883     ┆ 2.46     ┆ 0.028        ┆ 0.065      ┆ 0.072   │
│ 5   ┆ ctrl_raw_q8     ┆ 2444   ┆ 0.2866     ┆ 2.554    ┆ 0.034        ┆ 0.029      ┆ 0.002   │
│ 5   ┆ persistence     ┆ 2444   ┆ 0.2967     ┆ 2.631    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 15  ┆ patchtst        ┆ 2740   ┆ 0.4098     ┆ 2.861    ┆ 0.024        ┆ 0.127      ┆ 0.201   │
│ 15  ┆ ar_lags         ┆ 2740   ┆ 0.4186     ┆ 3.031    ┆ 0.003        ┆ 0.075      ┆ 0.119   │
│ 15  ┆ ctrl_rf_q8_f40  ┆ 2740   ┆ 0.4138     ┆ 3.078    ┆ 0.014        ┆ 0.061      ┆ 0.079   │
│ 15  ┆ qrc_q8_d4       ┆ 2740   ┆ 0.4127     ┆ 3.094    ┆ 0.017        ┆ 0.056      ┆ 0.061   │
│ 15  ┆ qrc_noent_q8_d4 ┆ 2740   ┆ 0.4116     ┆ 3.12     ┆ 0.02         ┆ 0.049      ┆ 0.027   │
│ 15  ┆ ctrl_raw_q8     ┆ 2740   ┆ 0.4103     ┆ 3.217    ┆ 0.023        ┆ 0.019      ┆ 0.001   │
│ 15  ┆ persistence     ┆ 2740   ┆ 0.4199     ┆ 3.279    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
└─────┴─────────────────┴────────┴────────────┴──────────┴──────────────┴────────────┴─────────┘
```

## Cell 9: summary
_execution count: 9_

```
written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week5/qrc_val.json
```

## Cell 10: export outputs (save the notebook first with Cmd+S, then run this cell)
_execution count: None_

(no output)

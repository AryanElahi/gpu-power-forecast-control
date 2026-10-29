# Outputs of `19_week5_qrc.ipynb`
Exported 2026-10-11 16:07 from the saved notebook file.

## Cell 1: setup
_execution count: 11_

```
pennylane 0.45.1 | features per minute (qrc): 40 | window features: 160
```

## Cell 2: unit checks (no real data)
_execution count: 12_

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
_execution count: 13_

```
qrc_q4_d4        files ok True | finite True | skill +0.029 / +0.047 / +0.079
qrc_noent_q4_d4  files ok True | finite True | skill +0.034 / +0.048 / +0.079
ctrl_rf_q4_f20   files ok True | finite True | skill +0.035 / +0.063 / +0.112
ctrl_raw_q4      files ok True | finite True | skill +0.034 / +0.059 / +0.097
DRY RUN PASSED
```

## Cell 4: real data, inputs and reservoir timing (10,000 minutes per configuration)
_execution count: 14_

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
│ 4        ┆ true     ┆ 0.11      ┆ 0.06                │
│ 4        ┆ false    ┆ 0.1       ┆ 0.05                │
│ 6        ┆ true     ┆ 0.64      ┆ 0.32                │
│ 6        ┆ false    ┆ 0.55      ┆ 0.28                │
│ 8        ┆ true     ┆ 3.08      ┆ 1.56                │
│ 8        ┆ false    ┆ 2.54      ┆ 1.29                │
└──────────┴──────────┴───────────┴─────────────────────┘
```

## Cell 5: SCALE SCREEN, seed 0 (input scale a for qrc, qrc_noent and rf; raw has none). Resumable.
_execution count: 15_

```
[15:53:17] qrc_q8_d4_a0p098_s0: reservoir + readout
[15:54:55] qrc_q8_d4_a0p098_s0: features 160 | readout params 483 | reservoir 88 s | val MAE 0.2650 / 0.4122 / 0.5859 kW | skill +0.075 / +0.041 / +0.031
[15:54:55] qrc_q8_d4_a0p196_s0: reservoir + readout
[15:56:32] qrc_q8_d4_a0p196_s0: features 160 | readout params 483 | reservoir 88 s | val MAE 0.2653 / 0.4123 / 0.5858 kW | skill +0.074 / +0.041 / +0.031
[15:56:32] qrc_q8_d4_a0p393_s0: done earlier, skipped
[15:56:32] qrc_q8_d4_a0p785_s0: done earlier, skipped
[15:56:32] qrc_q8_d4_a1p571_s0: done earlier, skipped
[15:56:32] qrc_q8_d4_a3p142_s0: done earlier, skipped
[15:56:32] qrc_noent_q8_d4_a0p098_s0: reservoir + readout
[15:57:59] qrc_noent_q8_d4_a0p098_s0: features 160 | readout params 483 | reservoir 77 s | val MAE 0.2652 / 0.4126 / 0.5872 kW | skill +0.075 / +0.040 / +0.029
[15:57:59] qrc_noent_q8_d4_a0p196_s0: reservoir + readout
[15:59:30] qrc_noent_q8_d4_a0p196_s0: features 160 | readout params 483 | reservoir 77 s | val MAE 0.2649 / 0.4123 / 0.5870 kW | skill +0.076 / +0.041 / +0.030
[15:59:30] qrc_noent_q8_d4_a0p393_s0: done earlier, skipped
[15:59:30] qrc_noent_q8_d4_a0p785_s0: done earlier, skipped
[15:59:30] qrc_noent_q8_d4_a1p571_s0: done earlier, skipped
[15:59:30] qrc_noent_q8_d4_a3p142_s0: done earlier, skipped
[15:59:30] ctrl_rf_q8_f40_a0p098_s0: reservoir + readout
[15:59:47] ctrl_rf_q8_f40_a0p098_s0: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2642 / 0.4117 / 0.5865 kW | skill +0.078 / +0.042 / +0.030
[15:59:47] ctrl_rf_q8_f40_a0p196_s0: reservoir + readout
[15:59:58] ctrl_rf_q8_f40_a0p196_s0: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2642 / 0.4117 / 0.5865 kW | skill +0.078 / +0.042 / +0.030
[15:59:58] ctrl_rf_q8_f40_a0p393_s0: done earlier, skipped
[15:59:58] ctrl_rf_q8_f40_a0p785_s0: done earlier, skipped
[15:59:58] ctrl_rf_q8_f40_a1p571_s0: done earlier, skipped
[15:59:58] ctrl_rf_q8_f40_a3p142_s0: done earlier, skipped
[15:59:58] ctrl_raw_q8_s0: done earlier, skipped
shape: (19, 8)
┌───────────┬─────────────────┬────────┬──────────┬──────────┬───────────┬──────────────┬───────────────┐
│ kind      ┆ variant         ┆ scale  ┆ skill_h1 ┆ skill_h5 ┆ skill_h15 ┆ val_MAE_mean ┆ ridge_at_edge │
│ ---       ┆ ---             ┆ ---    ┆ ---      ┆ ---      ┆ ---       ┆ ---          ┆ ---           │
│ str       ┆ str             ┆ f64    ┆ f64      ┆ f64      ┆ f64       ┆ f64          ┆ bool          │
╞═══════════╪═════════════════╪════════╪══════════╪══════════╪═══════════╪══════════════╪═══════════════╡
│ qrc       ┆ qrc_q8_d4       ┆ 0.0982 ┆ 0.0751   ┆ 0.0413   ┆ 0.0314    ┆ 0.42103      ┆ true          │
│ qrc       ┆ qrc_q8_d4       ┆ 0.1963 ┆ 0.0741   ┆ 0.0411   ┆ 0.0314    ┆ 0.42115      ┆ true          │
│ qrc       ┆ qrc_q8_d4       ┆ 0.3927 ┆ 0.0721   ┆ 0.04     ┆ 0.0305    ┆ 0.42169      ┆ false         │
│ qrc       ┆ qrc_q8_d4       ┆ 0.7854 ┆ 0.0642   ┆ 0.0368   ┆ 0.0278    ┆ 0.42344      ┆ false         │
│ qrc       ┆ qrc_q8_d4       ┆ 1.5708 ┆ 0.05     ┆ 0.0287   ┆ 0.0214    ┆ 0.42725      ┆ false         │
│ qrc       ┆ qrc_q8_d4       ┆ 3.1416 ┆ 0.0194   ┆ 0.0075   ┆ 0.0044    ┆ 0.43663      ┆ true          │
│ qrc_noent ┆ qrc_noent_q8_d4 ┆ 0.1963 ┆ 0.0756   ┆ 0.041    ┆ 0.0296    ┆ 0.42139      ┆ true          │
│ qrc_noent ┆ qrc_noent_q8_d4 ┆ 0.3927 ┆ 0.0757   ┆ 0.0407   ┆ 0.0295    ┆ 0.42144      ┆ true          │
│ qrc_noent ┆ qrc_noent_q8_d4 ┆ 0.0982 ┆ 0.0745   ┆ 0.0404   ┆ 0.0292    ┆ 0.42165      ┆ true          │
│ qrc_noent ┆ qrc_noent_q8_d4 ┆ 0.7854 ┆ 0.0745   ┆ 0.0395   ┆ 0.0289    ┆ 0.42186      ┆ true          │
│ qrc_noent ┆ qrc_noent_q8_d4 ┆ 1.5708 ┆ 0.0687   ┆ 0.0343   ┆ 0.0258    ┆ 0.42378      ┆ true          │
│ qrc_noent ┆ qrc_noent_q8_d4 ┆ 3.1416 ┆ 0.0349   ┆ 0.0143   ┆ 0.0091    ┆ 0.43324      ┆ false         │
│ raw       ┆ ctrl_raw_q8     ┆ 1.5708 ┆ 0.0578   ┆ 0.0321   ┆ 0.0213    ┆ 0.42604      ┆ true          │
│ rf        ┆ ctrl_rf_q8_f40  ┆ 0.0982 ┆ 0.0783   ┆ 0.0424   ┆ 0.0304    ┆ 0.42078      ┆ true          │
│ rf        ┆ ctrl_rf_q8_f40  ┆ 0.1963 ┆ 0.0781   ┆ 0.0424   ┆ 0.0304    ┆ 0.42078      ┆ false         │
│ rf        ┆ ctrl_rf_q8_f40  ┆ 0.3927 ┆ 0.0769   ┆ 0.0424   ┆ 0.0304    ┆ 0.4209       ┆ false         │
│ rf        ┆ ctrl_rf_q8_f40  ┆ 0.7854 ┆ 0.0725   ┆ 0.0406   ┆ 0.0296    ┆ 0.42175      ┆ false         │
│ rf        ┆ ctrl_rf_q8_f40  ┆ 1.5708 ┆ 0.0651   ┆ 0.0336   ┆ 0.0245    ┆ 0.42447      ┆ false         │
│ rf        ┆ ctrl_rf_q8_f40  ┆ 3.1416 ┆ 0.0388   ┆ 0.013    ┆ 0.0095    ┆ 0.43296      ┆ false         │
└───────────┴─────────────────┴────────┴──────────┴──────────┴───────────┴──────────────┴───────────────┘
chosen scale at the edge of the grid: ['qrc', 'rf'] | rule: no further extension; a choice at the lower edge is reported as near-linear regime

chosen input scale per reservoir type (provisional, validation): {'qrc': 0.0982, 'qrc_noent': 0.1963, 'rf': 0.0982, 'raw': 1.5708} | 6.7 min
```

## Cell 6: THREE RESERVOIR SEEDS at the chosen scale (seed 0 reused from the screen). Resumable.
_execution count: 16_

```
[15:59:58] qrc_q8_d4_a0p098_s0: done earlier, skipped
[15:59:58] qrc_noent_q8_d4_a0p196_s0: done earlier, skipped
[15:59:58] ctrl_rf_q8_f40_a0p098_s0: done earlier, skipped
[15:59:58] ctrl_raw_q8_s0: done earlier, skipped
[15:59:58] qrc_q8_d4_a0p098_s1: reservoir + readout
[16:01:37] qrc_q8_d4_a0p098_s1: features 160 | readout params 483 | reservoir 87 s | val MAE 0.2650 / 0.4121 / 0.5861 kW | skill +0.075 / +0.041 / +0.031
[16:01:37] qrc_noent_q8_d4_a0p196_s1: reservoir + readout
[16:03:07] qrc_noent_q8_d4_a0p196_s1: features 160 | readout params 483 | reservoir 75 s | val MAE 0.2648 / 0.4122 / 0.5874 kW | skill +0.076 / +0.041 / +0.029
[16:03:07] ctrl_rf_q8_f40_a0p098_s1: reservoir + readout
[16:03:19] ctrl_rf_q8_f40_a0p098_s1: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2644 / 0.4116 / 0.5863 kW | skill +0.077 / +0.043 / +0.031
[16:03:19] qrc_q8_d4_a0p098_s2: reservoir + readout
[16:04:58] qrc_q8_d4_a0p098_s2: features 160 | readout params 483 | reservoir 88 s | val MAE 0.2656 / 0.4126 / 0.5870 kW | skill +0.073 / +0.040 / +0.030
[16:04:58] qrc_noent_q8_d4_a0p196_s2: reservoir + readout
[16:06:24] qrc_noent_q8_d4_a0p196_s2: features 160 | readout params 483 | reservoir 75 s | val MAE 0.2650 / 0.4122 / 0.5874 kW | skill +0.075 / +0.041 / +0.029
[16:06:24] ctrl_rf_q8_f40_a0p098_s2: reservoir + readout
[16:06:35] ctrl_rf_q8_f40_a0p098_s2: features 160 | readout params 483 | reservoir 0 s | val MAE 0.2651 / 0.4125 / 0.5871 kW | skill +0.075 / +0.041 / +0.029

all runs handled in 6.6 min
```

## Cell 7: table (mean +- sd over reservoir seeds), merged with the Week 4 benchmark and the hybrid QNN (if available)
_execution count: 17_

```
shape: (12, 11)
┌─────┬─────────────────┬─────────┬──────────┬────────┬───────────┬──────────┬────────────┬─────────┬──────────┬─────────────┐
│ h   ┆ model           ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 ┆ n_params ┆ reservoir_s │
│ --- ┆ ---             ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     ┆ ---      ┆ ---         │
│ i64 ┆ str             ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     ┆ i64      ┆ f64         │
╞═════╪═════════════════╪═════════╪══════════╪════════╪═══════════╪══════════╪════════════╪═════════╪══════════╪═════════════╡
│ 1   ┆ ctrl_rf_q8_f40  ┆ 3       ┆ 0.2645   ┆ 0.0005 ┆ 0.077     ┆ 1.415    ┆ 0.2121     ┆ 0.048   ┆ 483      ┆ 0.0         │
│ 1   ┆ qrc_noent_q8_d4 ┆ 3       ┆ 0.2649   ┆ 0.0001 ┆ 0.076     ┆ 1.434    ┆ 0.2116     ┆ 0.031   ┆ 483      ┆ 76.0        │
│ 1   ┆ qrc_q8_d4       ┆ 3       ┆ 0.2652   ┆ 0.0004 ┆ 0.075     ┆ 1.42     ┆ 0.2125     ┆ 0.041   ┆ 483      ┆ 88.0        │
│ 1   ┆ ctrl_raw_q8     ┆ 1       ┆ 0.27     ┆ null   ┆ 0.058     ┆ 1.479    ┆ 0.2148     ┆ 0.012   ┆ 99       ┆ 0.0         │
│ 5   ┆ ctrl_rf_q8_f40  ┆ 3       ┆ 0.4119   ┆ 0.0005 ┆ 0.042     ┆ 2.411    ┆ 0.2908     ┆ 0.131   ┆ 483      ┆ 0.0         │
│ 5   ┆ qrc_noent_q8_d4 ┆ 3       ┆ 0.4122   ┆ 0.0001 ┆ 0.041     ┆ 2.462    ┆ 0.288      ┆ 0.068   ┆ 483      ┆ 76.0        │
│ 5   ┆ qrc_q8_d4       ┆ 3       ┆ 0.4123   ┆ 0.0002 ┆ 0.041     ┆ 2.426    ┆ 0.2902     ┆ 0.119   ┆ 483      ┆ 88.0        │
│ 5   ┆ ctrl_raw_q8     ┆ 1       ┆ 0.4161   ┆ null   ┆ 0.032     ┆ 2.554    ┆ 0.2866     ┆ 0.002   ┆ 99       ┆ 0.0         │
│ 15  ┆ qrc_q8_d4       ┆ 3       ┆ 0.5863   ┆ 0.0006 ┆ 0.031     ┆ 3.094    ┆ 0.4125     ┆ 0.063   ┆ 483      ┆ 88.0        │
│ 15  ┆ ctrl_rf_q8_f40  ┆ 3       ┆ 0.5866   ┆ 0.0004 ┆ 0.03      ┆ 3.085    ┆ 0.4135     ┆ 0.078   ┆ 483      ┆ 0.0         │
│ 15  ┆ qrc_noent_q8_d4 ┆ 3       ┆ 0.5873   ┆ 0.0003 ┆ 0.029     ┆ 3.12     ┆ 0.4119     ┆ 0.028   ┆ 483      ┆ 76.0        │
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
│ 10   ┆ classical control ┆ ctrl_rf_q8_f40          ┆ 3       ┆ 0.2645   ┆ 0.077     │
│ 11   ┆ quantum reservoir ┆ qrc_noent_q8_d4         ┆ 3       ┆ 0.2649   ┆ 0.076     │
│ 12   ┆ quantum reservoir ┆ qrc_q8_d4               ┆ 3       ┆ 0.2652   ┆ 0.075     │
│ 13   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.2653   ┆ 0.074     │
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
│ 14   ┆ classical control ┆ ctrl_rf_q8_f40          ┆ 3       ┆ 0.4119   ┆ 0.042     │
│ 15   ┆ quantum reservoir ┆ qrc_noent_q8_d4         ┆ 3       ┆ 0.4122   ┆ 0.041     │
│ 16   ┆ quantum reservoir ┆ qrc_q8_d4               ┆ 3       ┆ 0.4123   ┆ 0.041     │
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
│ 14   ┆ quantum reservoir ┆ qrc_q8_d4               ┆ 3       ┆ 0.5863   ┆ 0.031     │
│ 16   ┆ classical control ┆ ctrl_rf_q8_f40          ┆ 3       ┆ 0.5866   ┆ 0.03      │
│ 19   ┆ quantum reservoir ┆ qrc_noent_q8_d4         ┆ 3       ┆ 0.5873   ┆ 0.029     │
│ 20   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.5876   ┆ 0.028     │
│ 22   ┆ classical control ┆ ctrl_raw_q8             ┆ 1       ┆ 0.592    ┆ 0.021     │
│ 26   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.6048   ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴───────────┘
```

## Cell 8: Diebold-Mariano (Holm per horizon) and ramp breakdown, seed 0; reference: ridge AR on the 60 lags
_execution count: 18_

```
references found: ['ar_lags', 'patchtst', 'persistence']
shape: (33, 9)
┌─────┬─────────────────┬─────────────────┬────────┬────────┬─────────┬────────────┬─────────────────┬──────────────────┐
│ h   ┆ model_a         ┆ model_b         ┆ MAE_a  ┆ MAE_b  ┆ DM_stat ┆ DM_p_holm  ┆ better          ┆ significant_5pct │
│ --- ┆ ---             ┆ ---             ┆ ---    ┆ ---    ┆ ---     ┆ ---        ┆ ---             ┆ ---              │
│ i64 ┆ str             ┆ str             ┆ f64    ┆ f64    ┆ f64     ┆ f64        ┆ str             ┆ bool             │
╞═════╪═════════════════╪═════════════════╪════════╪════════╪═════════╪════════════╪═════════════════╪══════════════════╡
│ 1   ┆ qrc_q8_d4       ┆ qrc_noent_q8_d4 ┆ 0.265  ┆ 0.2649 ┆ 0.41    ┆ 0.680975   ┆ qrc_noent_q8_d4 ┆ false            │
│ 1   ┆ qrc_q8_d4       ┆ ctrl_rf_q8_f40  ┆ 0.265  ┆ 0.2642 ┆ 3.54    ┆ 0.002835   ┆ ctrl_rf_q8_f40  ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ ctrl_raw_q8     ┆ 0.265  ┆ 0.27   ┆ -7.99   ┆ 1.2282e-14 ┆ qrc_q8_d4       ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ ar_lags         ┆ 0.265  ┆ 0.2673 ┆ -1.4    ┆ 0.387052   ┆ qrc_q8_d4       ┆ false            │
│ 1   ┆ qrc_q8_d4       ┆ patchtst        ┆ 0.265  ┆ 0.2533 ┆ 11.6    ┆ 5.9479e-30 ┆ patchtst        ┆ true             │
│ 1   ┆ qrc_q8_d4       ┆ persistence     ┆ 0.265  ┆ 0.2866 ┆ -12.29  ┆ 1.7032e-33 ┆ qrc_q8_d4       ┆ true             │
│ 1   ┆ qrc_noent_q8_d4 ┆ ctrl_rf_q8_f40  ┆ 0.2649 ┆ 0.2642 ┆ 2.21    ┆ 0.164018   ┆ ctrl_rf_q8_f40  ┆ false            │
│ 1   ┆ qrc_noent_q8_d4 ┆ ctrl_raw_q8     ┆ 0.2649 ┆ 0.27   ┆ -10.23  ┆ 1.6246e-23 ┆ qrc_noent_q8_d4 ┆ true             │
│ 1   ┆ qrc_noent_q8_d4 ┆ ar_lags         ┆ 0.2649 ┆ 0.2673 ┆ -1.52   ┆ 0.387052   ┆ qrc_noent_q8_d4 ┆ false            │
│ 1   ┆ qrc_noent_q8_d4 ┆ patchtst        ┆ 0.2649 ┆ 0.2533 ┆ 11.95   ┆ 1.0191e-31 ┆ patchtst        ┆ true             │
│ 1   ┆ qrc_noent_q8_d4 ┆ persistence     ┆ 0.2649 ┆ 0.2866 ┆ -12.78  ┆ 3.8171e-36 ┆ qrc_noent_q8_d4 ┆ true             │
│ 5   ┆ qrc_q8_d4       ┆ qrc_noent_q8_d4 ┆ 0.4122 ┆ 0.4123 ┆ -0.21   ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ ctrl_rf_q8_f40  ┆ 0.4122 ┆ 0.4117 ┆ 0.89    ┆ 1.0        ┆ ctrl_rf_q8_f40  ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ ctrl_raw_q8     ┆ 0.4122 ┆ 0.4161 ┆ -3.34   ┆ 0.007522   ┆ qrc_q8_d4       ┆ true             │
│ 5   ┆ qrc_q8_d4       ┆ ar_lags         ┆ 0.4122 ┆ 0.4161 ┆ -1.68   ┆ 0.550294   ┆ qrc_q8_d4       ┆ false            │
│ 5   ┆ qrc_q8_d4       ┆ patchtst        ┆ 0.4122 ┆ 0.3982 ┆ 6.84    ┆ 1.1367e-10 ┆ patchtst        ┆ true             │
│ 5   ┆ qrc_q8_d4       ┆ persistence     ┆ 0.4122 ┆ 0.4299 ┆ -8.94   ┆ 7.1413e-18 ┆ qrc_q8_d4       ┆ true             │
│ 5   ┆ qrc_noent_q8_d4 ┆ ctrl_rf_q8_f40  ┆ 0.4123 ┆ 0.4117 ┆ 0.6     ┆ 1.0        ┆ ctrl_rf_q8_f40  ┆ false            │
│ 5   ┆ qrc_noent_q8_d4 ┆ ctrl_raw_q8     ┆ 0.4123 ┆ 0.4161 ┆ -4.55   ┆ 0.000054   ┆ qrc_noent_q8_d4 ┆ true             │
│ 5   ┆ qrc_noent_q8_d4 ┆ ar_lags         ┆ 0.4123 ┆ 0.4161 ┆ -1.69   ┆ 0.550294   ┆ qrc_noent_q8_d4 ┆ false            │
│ 5   ┆ qrc_noent_q8_d4 ┆ patchtst        ┆ 0.4123 ┆ 0.3982 ┆ 6.39    ┆ 2.1400e-9  ┆ patchtst        ┆ true             │
│ 5   ┆ qrc_noent_q8_d4 ┆ persistence     ┆ 0.4123 ┆ 0.4299 ┆ -10.87  ┆ 3.2787e-26 ┆ qrc_noent_q8_d4 ┆ true             │
│ 15  ┆ qrc_q8_d4       ┆ qrc_noent_q8_d4 ┆ 0.5859 ┆ 0.587  ┆ -1.12   ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 15  ┆ qrc_q8_d4       ┆ ctrl_rf_q8_f40  ┆ 0.5859 ┆ 0.5865 ┆ -1.1    ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 15  ┆ qrc_q8_d4       ┆ ctrl_raw_q8     ┆ 0.5859 ┆ 0.592  ┆ -3.2    ┆ 0.013664   ┆ qrc_q8_d4       ┆ true             │
│ 15  ┆ qrc_q8_d4       ┆ ar_lags         ┆ 0.5859 ┆ 0.5876 ┆ -0.57   ┆ 1.0        ┆ qrc_q8_d4       ┆ false            │
│ 15  ┆ qrc_q8_d4       ┆ patchtst        ┆ 0.5859 ┆ 0.5688 ┆ 5.45    ┆ 7.2403e-7  ┆ patchtst        ┆ true             │
│ 15  ┆ qrc_q8_d4       ┆ persistence     ┆ 0.5859 ┆ 0.6048 ┆ -6.44   ┆ 2.0919e-9  ┆ qrc_q8_d4       ┆ true             │
│ 15  ┆ qrc_noent_q8_d4 ┆ ctrl_rf_q8_f40  ┆ 0.587  ┆ 0.5865 ┆ 0.43    ┆ 1.0        ┆ ctrl_rf_q8_f40  ┆ false            │
│ 15  ┆ qrc_noent_q8_d4 ┆ ctrl_raw_q8     ┆ 0.587  ┆ 0.592  ┆ -3.14   ┆ 0.01498    ┆ qrc_noent_q8_d4 ┆ true             │
│ 15  ┆ qrc_noent_q8_d4 ┆ ar_lags         ┆ 0.587  ┆ 0.5876 ┆ -0.22   ┆ 1.0        ┆ qrc_noent_q8_d4 ┆ false            │
│ 15  ┆ qrc_noent_q8_d4 ┆ patchtst        ┆ 0.587  ┆ 0.5688 ┆ 5.25    ┆ 0.000002   ┆ patchtst        ┆ true             │
│ 15  ┆ qrc_noent_q8_d4 ┆ persistence     ┆ 0.587  ┆ 0.6048 ┆ -7.0    ┆ 4.9738e-11 ┆ qrc_noent_q8_d4 ┆ true             │
└─────┴─────────────────┴─────────────────┴────────┴────────┴─────────┴────────────┴─────────────────┴──────────────────┘
shape: (21, 8)
┌─────┬─────────────────┬────────┬────────────┬──────────┬──────────────┬────────────┬─────────┐
│ h   ┆ model           ┆ n_ramp ┆ MAE_normal ┆ MAE_ramp ┆ skill_normal ┆ skill_ramp ┆ ramp_F1 │
│ --- ┆ ---             ┆ ---    ┆ ---        ┆ ---      ┆ ---          ┆ ---        ┆ ---     │
│ i64 ┆ str             ┆ i64    ┆ f64        ┆ f64      ┆ f64          ┆ f64        ┆ f64     │
╞═════╪═════════════════╪════════╪════════════╪══════════╪══════════════╪════════════╪═════════╡
│ 1   ┆ patchtst        ┆ 1903   ┆ 0.2046     ┆ 1.322    ┆ 0.096        ┆ 0.178      ┆ 0.169   │
│ 1   ┆ ar_lags         ┆ 1903   ┆ 0.2152     ┆ 1.407    ┆ 0.049        ┆ 0.125      ┆ 0.145   │
│ 1   ┆ ctrl_rf_q8_f40  ┆ 1903   ┆ 0.2118     ┆ 1.411    ┆ 0.064        ┆ 0.122      ┆ 0.048   │
│ 1   ┆ qrc_q8_d4       ┆ 1903   ┆ 0.2126     ┆ 1.416    ┆ 0.061        ┆ 0.119      ┆ 0.043   │
│ 1   ┆ qrc_noent_q8_d4 ┆ 1903   ┆ 0.2116     ┆ 1.434    ┆ 0.065        ┆ 0.108      ┆ 0.032   │
│ 1   ┆ ctrl_raw_q8     ┆ 1903   ┆ 0.2148     ┆ 1.479    ┆ 0.051        ┆ 0.08       ┆ 0.012   │
│ 1   ┆ persistence     ┆ 1903   ┆ 0.2264     ┆ 1.607    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 5   ┆ patchtst        ┆ 2444   ┆ 0.287      ┆ 2.236    ┆ 0.033        ┆ 0.15       ┆ 0.257   │
│ 5   ┆ ar_lags         ┆ 2444   ┆ 0.2971     ┆ 2.377    ┆ -0.001       ┆ 0.096      ┆ 0.194   │
│ 5   ┆ ctrl_rf_q8_f40  ┆ 2444   ┆ 0.2913     ┆ 2.4      ┆ 0.018        ┆ 0.088      ┆ 0.14    │
│ 5   ┆ qrc_q8_d4       ┆ 2444   ┆ 0.2901     ┆ 2.427    ┆ 0.022        ┆ 0.078      ┆ 0.119   │
│ 5   ┆ qrc_noent_q8_d4 ┆ 2444   ┆ 0.2883     ┆ 2.459    ┆ 0.028        ┆ 0.065      ┆ 0.071   │
│ 5   ┆ ctrl_raw_q8     ┆ 2444   ┆ 0.2866     ┆ 2.554    ┆ 0.034        ┆ 0.029      ┆ 0.002   │
│ 5   ┆ persistence     ┆ 2444   ┆ 0.2967     ┆ 2.631    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 15  ┆ patchtst        ┆ 2740   ┆ 0.4098     ┆ 2.861    ┆ 0.024        ┆ 0.127      ┆ 0.201   │
│ 15  ┆ ar_lags         ┆ 2740   ┆ 0.4186     ┆ 3.031    ┆ 0.003        ┆ 0.075      ┆ 0.119   │
│ 15  ┆ ctrl_rf_q8_f40  ┆ 2740   ┆ 0.4138     ┆ 3.079    ┆ 0.015        ┆ 0.061      ┆ 0.081   │
│ 15  ┆ qrc_q8_d4       ┆ 2740   ┆ 0.4124     ┆ 3.089    ┆ 0.018        ┆ 0.058      ┆ 0.071   │
│ 15  ┆ qrc_noent_q8_d4 ┆ 2740   ┆ 0.4119     ┆ 3.115    ┆ 0.019        ┆ 0.05       ┆ 0.034   │
│ 15  ┆ ctrl_raw_q8     ┆ 2740   ┆ 0.4103     ┆ 3.217    ┆ 0.023        ┆ 0.019      ┆ 0.001   │
│ 15  ┆ persistence     ┆ 2740   ┆ 0.4199     ┆ 3.279    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
└─────┴─────────────────┴────────┴────────────┴──────────┴──────────────┴────────────┴─────────┘
```

## Cell 9: summary
_execution count: 19_

```
written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week5/qrc_val.json
```

## Cell 10: export outputs (save the notebook first with Cmd+S, then run this cell)
_execution count: 20_

```
written 19_week5_qrc_outputs.md | code cells 11 | figures 0
```

## Cell 11
_execution count: None_

(no output)

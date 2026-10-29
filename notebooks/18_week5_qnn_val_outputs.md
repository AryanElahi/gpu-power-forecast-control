# Outputs of `18_week5_qnn_val.ipynb`
Exported 2026-10-10 16:20 from the saved notebook file.

## Cell 1: setup (reads the decisions of notebook 17)
_execution count: 1_

```
backend default.qubit | max_train_samples None | epoch cap 30 | unit checks of nb 17: True
shape: (4, 5)
┌───────────────────────┬──────────┬──────────┬───────────┬────────────┐
│ model                 ┆ skill_h1 ┆ skill_h5 ┆ skill_h15 ┆ best_epoch │
│ ---                   ┆ ---      ┆ ---      ┆ ---       ┆ ---        │
│ str                   ┆ f64      ┆ f64      ┆ f64       ┆ i64        │
╞═══════════════════════╪══════════╪══════════╪═══════════╪════════════╡
│ hqnn_q4_l2            ┆ 0.0421   ┆ 0.0512   ┆ 0.0377    ┆ 10         │
│ ctrl_mlp_q4_l2        ┆ 0.0359   ┆ 0.0554   ┆ 0.0434    ┆ 15         │
│ ctrl_bottleneck_q4_l2 ┆ 0.083    ┆ 0.0439   ┆ 0.0305    ┆ 5          │
│ hqnn_noent_q4_l2      ┆ 0.0784   ┆ 0.0526   ┆ 0.041     ┆ 15         │
└───────────────────────┴──────────┴──────────┴───────────┴────────────┘
runs of notebook 17 whose best epoch equals the cap of 30: none | epoch cap here: 30 | group: week5_qnn
```

## Cell 2: DRY RUN on synthetic data for the variants of this notebook (2 epochs; deleted afterwards)
_execution count: 2_

```
hqnn_q4_l4             params 311 (middle 48) | finite True
hqnn_noreup_q4_l2      params 287 (middle 24) | finite True
ctrl_mlp_q4_l4         params 312 (middle 49) | finite True
DRY RUN PASSED
```

## Cell 3: OVERFIT SANITY CHECK v2 (memorisation). The 1% check of notebook 17 (1,967 noisy samples, 60 epochs) gave
_execution count: 3_

```
shape: (3, 8)
┌─────────────────────┬──────────┬───────────┬─────────────┬───────────┬───────┬─────┬─────────────┐
│ variant             ┆ n_params ┆ n_samples ┆ loss_epoch1 ┆ loss_last ┆ ratio ┆ s   ┆ check       │
│ ---                 ┆ ---      ┆ ---       ┆ ---         ┆ ---       ┆ ---   ┆ --- ┆ ---         │
│ str                 ┆ i64      ┆ i64       ┆ f64         ┆ f64       ┆ f64   ┆ f64 ┆ str         │
╞═════════════════════╪══════════╪═══════════╪═════════════╪═══════════╪═══════╪═════╪═════════════╡
│ hqnn_q4_l2          ┆ 287      ┆ 64        ┆ 2.2506      ┆ 0.4681    ┆ 0.208 ┆ 5.9 ┆ PASS        │
│ ctrl_mlp_q4_l2      ┆ 285      ┆ 64        ┆ 2.0025      ┆ 0.244     ┆ 0.122 ┆ 0.9 ┆ PASS        │
│ ctrl_bottleneck_q4_ ┆ 263      ┆ 64        ┆ 2.2155      ┆ 0.2075    ┆ 0.094 ┆ 0.9 ┆ report only │
│ l2                  ┆          ┆           ┆             ┆           ┆       ┆     ┆             │
└─────────────────────┴──────────┴───────────┴─────────────┴───────────┴───────┴─────┴─────────────┘
MEMORISATION CHECK PASSED (the training pipeline can fit data)
```

## Cell 4: DESIGN SCREEN, seed 0, validation (4 qubits; layers x re-uploading). Resumable; about 10-30 min.
_execution count: 4_

```
[15:54:50] hqnn_q4_l2_s0: done earlier, skipped
[15:54:50] hqnn_q4_l4_s0: training
```

```stderr
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /Users/aryaneuse/.netrc.
```

```
[15:57:57] hqnn_q4_l4_s0: params 311 (middle 48) | train 167 s | best epoch 16/19 | val MAE 0.2756 / 0.4062 / 0.5778 kW | skill +0.038 / +0.055 / +0.045
[15:57:57] hqnn_noreup_q4_l2_s0: training
[15:59:43] hqnn_noreup_q4_l2_s0: params 287 (middle 24) | train 97 s | best epoch 19/22 | val MAE 0.2697 / 0.4056 / 0.5770 kW | skill +0.059 / +0.057 / +0.046
[15:59:43] hqnn_noreup_q4_l4_s0: training
[16:00:44] hqnn_noreup_q4_l4_s0: params 311 (middle 48) | train 48 s | best epoch 4/7 | val MAE 0.2734 / 0.4086 / 0.5827 kW | skill +0.046 / +0.050 / +0.037
shape: (4, 8)
┌─────────────┬──────────┬─────────────┬────────────┬──────────┬──────────┬───────────┬────────────┐
│ variant     ┆ n_params ┆ n_params_mi ┆ val_MAE_me ┆ skill_h1 ┆ skill_h5 ┆ skill_h15 ┆ best_epoch │
│ ---         ┆ ---      ┆ ddle        ┆ an         ┆ ---      ┆ ---      ┆ ---       ┆ ---        │
│ str         ┆ i64      ┆ ---         ┆ ---        ┆ f64      ┆ f64      ┆ f64       ┆ i64        │
│             ┆          ┆ i64         ┆ f64        ┆          ┆          ┆           ┆            │
╞═════════════╪══════════╪═════════════╪════════════╪══════════╪══════════╪═══════════╪════════════╡
│ hqnn_noreup ┆ 287      ┆ 24          ┆ 0.41744    ┆ 0.0589   ┆ 0.0566   ┆ 0.046     ┆ 19         │
│ _q4_l2      ┆          ┆             ┆            ┆          ┆          ┆           ┆            │
│ hqnn_q4_l4  ┆ 311      ┆ 48          ┆ 0.41986    ┆ 0.0382   ┆ 0.0553   ┆ 0.0447    ┆ 16         │
│ hqnn_q4_l2  ┆ 287      ┆ 24          ┆ 0.4215     ┆ 0.0421   ┆ 0.0512   ┆ 0.0377    ┆ 10         │
│ hqnn_noreup ┆ 311      ┆ 48          ┆ 0.42158    ┆ 0.0459   ┆ 0.0496   ┆ 0.0366    ┆ 4          │
│ _q4_l4      ┆          ┆             ┆            ┆          ┆          ┆           ┆            │
└─────────────┴──────────┴─────────────┴────────────┴──────────┴──────────┴───────────┴────────────┘

MAIN hybrid QNN (provisional, chosen on validation): hqnn_noreup_q4_l2 | margin to the second variant 0.00242 kW (mean MAE); a margin this small is within seed noise if it is below ~0.002 kW
variants at the epoch cap: none
```

## Cell 5: THREE SEEDS: main hybrid QNN, entanglement ablation, MLP control (matched to the main circuit), bottleneck.
_execution count: 5_

```
[16:00:44] hqnn_noreup_q4_l2_s0: done earlier, skipped
[16:00:44] hqnn_noent_noreup_q4_l2_s0: training
[16:02:08] hqnn_noent_noreup_q4_l2_s0: params 287 (middle 24) | train 67 s | best epoch 15/18 | val MAE 0.2639 / 0.4053 / 0.5783 kW | skill +0.079 / +0.057 / +0.044
[16:02:08] ctrl_mlp_q4_l2_s0: done earlier, skipped
[16:02:08] ctrl_bottleneck_q4_l2_s0: done earlier, skipped
[16:02:08] hqnn_noreup_q4_l2_s1: training
[16:03:15] hqnn_noreup_q4_l2_s1: params 287 (middle 24) | train 53 s | best epoch 10/13 | val MAE 0.2626 / 0.4060 / 0.5792 kW | skill +0.084 / +0.056 / +0.042
[16:03:15] hqnn_noent_noreup_q4_l2_s1: training
[16:04:35] hqnn_noent_noreup_q4_l2_s1: params 287 (middle 24) | train 66 s | best epoch 15/18 | val MAE 0.2609 / 0.4072 / 0.5812 kW | skill +0.089 / +0.053 / +0.039
[16:04:35] ctrl_mlp_q4_l2_s1: training
[16:04:55] ctrl_mlp_q4_l2_s1: params 285 (middle 22) | train 4 s | best epoch 10/13 | val MAE 0.2657 / 0.4078 / 0.5812 kW | skill +0.073 / +0.051 / +0.039
[16:04:55] ctrl_bottleneck_q4_l2_s1: training
[16:05:17] ctrl_bottleneck_q4_l2_s1: params 263 (middle 0) | train 3 s | best epoch 12/15 | val MAE 0.2674 / 0.4086 / 0.5827 kW | skill +0.067 / +0.050 / +0.037
[16:05:17] hqnn_noreup_q4_l2_s2: training
[16:06:33] hqnn_noreup_q4_l2_s2: params 287 (middle 24) | train 59 s | best epoch 10/13 | val MAE 0.2636 / 0.4075 / 0.5797 kW | skill +0.080 / +0.052 / +0.042
[16:06:33] hqnn_noent_noreup_q4_l2_s2: training
[16:07:37] hqnn_noent_noreup_q4_l2_s2: params 287 (middle 24) | train 51 s | best epoch 11/14 | val MAE 0.2607 / 0.4081 / 0.5801 kW | skill +0.090 / +0.051 / +0.041
[16:07:37] ctrl_mlp_q4_l2_s2: training
[16:07:55] ctrl_mlp_q4_l2_s2: params 285 (middle 22) | train 4 s | best epoch 11/14 | val MAE 0.2698 / 0.4080 / 0.5798 kW | skill +0.058 / +0.051 / +0.041
[16:07:55] ctrl_bottleneck_q4_l2_s2: training
[16:08:10] ctrl_bottleneck_q4_l2_s2: params 263 (middle 0) | train 2 s | best epoch 6/9 | val MAE 0.2724 / 0.4115 / 0.5858 kW | skill +0.049 / +0.043 / +0.032

all runs handled in 7.4 min
models: ['hqnn_noreup_q4_l2', 'hqnn_noent_noreup_q4_l2', 'ctrl_mlp_q4_l2', 'ctrl_bottleneck_q4_l2']
```

## Cell 6: table, mean +- sd over seeds (validation), merged with the frozen Week 4 benchmark
_execution count: 6_

```
shape: (12, 15)
┌─────┬─────────────────────────┬─────────┬──────────┬────────┬───────────┬──────────┬──────────┬────────────┬─────────┬──────────┬─────────────────┬─────────────┬────────────┬─────────┐
│ h   ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ skill_sd ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 ┆ n_params ┆ n_params_middle ┆ s_per_epoch ┆ best_epoch ┆ cap_hit │
│ --- ┆ ---                     ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---      ┆ ---        ┆ ---     ┆ ---      ┆ ---             ┆ ---         ┆ ---        ┆ ---     │
│ i64 ┆ str                     ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64      ┆ f64        ┆ f64     ┆ i64      ┆ i64             ┆ f64         ┆ f64        ┆ bool    │
╞═════╪═════════════════════════╪═════════╪══════════╪════════╪═══════════╪══════════╪══════════╪════════════╪═════════╪══════════╪═════════════════╪═════════════╪════════════╪═════════╡
│ 1   ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.2618   ┆ 0.0018 ┆ 0.086     ┆ 0.006    ┆ 1.391    ┆ 0.2103     ┆ 0.109   ┆ 287      ┆ 24              ┆ 3.5         ┆ 13.7       ┆ false   │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.2653   ┆ 0.0039 ┆ 0.074     ┆ 0.013    ┆ 1.416    ┆ 0.2128     ┆ 0.115   ┆ 287      ┆ 24              ┆ 4.2         ┆ 13.0       ┆ false   │
│ 1   ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.2676   ┆ 0.0048 ┆ 0.066     ┆ 0.017    ┆ 1.476    ┆ 0.2124     ┆ 0.013   ┆ 263      ┆ 0               ┆ 0.2         ┆ 7.7        ┆ false   │
│ 1   ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.2706   ┆ 0.0054 ┆ 0.056     ┆ 0.019    ┆ 1.432    ┆ 0.2176     ┆ 0.127   ┆ 285      ┆ 22              ┆ 0.3         ┆ 12.0       ┆ false   │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.4064   ┆ 0.001  ┆ 0.055     ┆ 0.002    ┆ 2.404    ┆ 0.2853     ┆ 0.154   ┆ 287      ┆ 24              ┆ 4.2         ┆ 13.0       ┆ false   │
│ 5   ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.4069   ┆ 0.0015 ┆ 0.054     ┆ 0.003    ┆ 2.393    ┆ 0.2864     ┆ 0.151   ┆ 287      ┆ 24              ┆ 3.5         ┆ 13.7       ┆ false   │
│ 5   ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.4073   ┆ 0.0011 ┆ 0.053     ┆ 0.002    ┆ 2.384    ┆ 0.2875     ┆ 0.17    ┆ 285      ┆ 22              ┆ 0.3         ┆ 12.0       ┆ false   │
│ 5   ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.4104   ┆ 0.0015 ┆ 0.045     ┆ 0.004    ┆ 2.494    ┆ 0.2841     ┆ 0.012   ┆ 263      ┆ 0               ┆ 0.2         ┆ 7.7        ┆ false   │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.5786   ┆ 0.0014 ┆ 0.043     ┆ 0.002    ┆ 3.084    ┆ 0.405      ┆ 0.057   ┆ 287      ┆ 24              ┆ 4.2         ┆ 13.0       ┆ false   │
│ 15  ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.5799   ┆ 0.0015 ┆ 0.041     ┆ 0.002    ┆ 3.077    ┆ 0.4068     ┆ 0.05    ┆ 287      ┆ 24              ┆ 3.5         ┆ 13.7       ┆ false   │
│ 15  ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.5799   ┆ 0.0013 ┆ 0.041     ┆ 0.002    ┆ 3.065    ┆ 0.4076     ┆ 0.064   ┆ 285      ┆ 22              ┆ 0.3         ┆ 12.0       ┆ false   │
│ 15  ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.585    ┆ 0.002  ┆ 0.033     ┆ 0.003    ┆ 3.169    ┆ 0.4058     ┆ 0.003   ┆ 263      ┆ 0               ┆ 0.2         ┆ 7.7        ┆ false   │
└─────┴─────────────────────────┴─────────┴──────────┴────────┴───────────┴──────────┴──────────┴────────────┴─────────┴──────────┴─────────────────┴─────────────┴────────────┴─────────┘

=== h = 1 min: validation ranking (24 models) ===
shape: (11, 7)
┌──────┬───────────────────┬─────────────────────────┬─────────┬──────────┬────────┬───────────┐
│ rank ┆ family            ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE │
│ ---  ┆ ---               ┆ ---                     ┆ ---     ┆ ---      ┆ ---    ┆ ---       │
│ u32  ┆ str               ┆ str                     ┆ i64     ┆ f64      ┆ f64    ┆ f64       │
╞══════╪═══════════════════╪═════════════════════════╪═════════╪══════════╪════════╪═══════════╡
│ 2    ┆ ML                ┆ lightgbm                ┆ 3       ┆ 0.2491   ┆ 0.0001 ┆ 0.131     │
│ 3    ┆ deep              ┆ patchtst                ┆ 3       ┆ 0.2543   ┆ 0.0015 ┆ 0.112     │
│ 7    ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.2618   ┆ 0.0018 ┆ 0.086     │
│ 10   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.2653   ┆ 0.0039 ┆ 0.074     │
│ 11   ┆ deep              ┆ nlinear                 ┆ 3       ┆ 0.2666   ┆ 0.0007 ┆ 0.07      │
│ 12   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.2673   ┆ 0.0    ┆ 0.067     │
│ 13   ┆ classical control ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.2676   ┆ 0.0048 ┆ 0.066     │
│ 15   ┆ deep              ┆ dlinear                 ┆ 3       ┆ 0.268    ┆ 0.0018 ┆ 0.065     │
│ 16   ┆ deep              ┆ gru_uni                 ┆ 3       ┆ 0.27     ┆ 0.0005 ┆ 0.058     │
│ 17   ┆ classical control ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.2706   ┆ 0.0054 ┆ 0.056     │
│ 23   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.2866   ┆ 0.0    ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴────────┴───────────┘

=== h = 5 min: validation ranking (24 models) ===
shape: (11, 7)
┌──────┬───────────────────┬─────────────────────────┬─────────┬──────────┬────────┬───────────┐
│ rank ┆ family            ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE │
│ ---  ┆ ---               ┆ ---                     ┆ ---     ┆ ---      ┆ ---    ┆ ---       │
│ u32  ┆ str               ┆ str                     ┆ i64     ┆ f64      ┆ f64    ┆ f64       │
╞══════╪═══════════════════╪═════════════════════════╪═════════╪══════════╪════════╪═══════════╡
│ 2    ┆ ML                ┆ lightgbm                ┆ 3       ┆ 0.3934   ┆ 0.0001 ┆ 0.085     │
│ 4    ┆ deep              ┆ patchtst                ┆ 3       ┆ 0.3989   ┆ 0.0018 ┆ 0.072     │
│ 8    ┆ deep              ┆ gru_uni                 ┆ 3       ┆ 0.4016   ┆ 0.0005 ┆ 0.066     │
│ 9    ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.4064   ┆ 0.001  ┆ 0.055     │
│ 10   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.4069   ┆ 0.0015 ┆ 0.054     │
│ 11   ┆ classical control ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.4073   ┆ 0.0011 ┆ 0.053     │
│ 13   ┆ classical control ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.4104   ┆ 0.0015 ┆ 0.045     │
│ 14   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.4161   ┆ 0.0    ┆ 0.032     │
│ 17   ┆ deep              ┆ dlinear                 ┆ 3       ┆ 0.4168   ┆ 0.0034 ┆ 0.03      │
│ 18   ┆ deep              ┆ nlinear                 ┆ 3       ┆ 0.4176   ┆ 0.0041 ┆ 0.029     │
│ 22   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.4299   ┆ 0.0    ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴────────┴───────────┘

=== h = 15 min: validation ranking (24 models) ===
shape: (11, 7)
┌──────┬───────────────────┬─────────────────────────┬─────────┬──────────┬────────┬───────────┐
│ rank ┆ family            ┆ model                   ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE │
│ ---  ┆ ---               ┆ ---                     ┆ ---     ┆ ---      ┆ ---    ┆ ---       │
│ u32  ┆ str               ┆ str                     ┆ i64     ┆ f64      ┆ f64    ┆ f64       │
╞══════╪═══════════════════╪═════════════════════════╪═════════╪══════════╪════════╪═══════════╡
│ 2    ┆ ML                ┆ lightgbm                ┆ 3       ┆ 0.5659   ┆ 0.0004 ┆ 0.065     │
│ 4    ┆ deep              ┆ patchtst                ┆ 3       ┆ 0.5693   ┆ 0.0007 ┆ 0.059     │
│ 6    ┆ deep              ┆ gru_uni                 ┆ 3       ┆ 0.5722   ┆ 0.0013 ┆ 0.054     │
│ 10   ┆ hybrid quantum    ┆ hqnn_noreup_q4_l2       ┆ 3       ┆ 0.5786   ┆ 0.0014 ┆ 0.043     │
│ 11   ┆ hybrid quantum    ┆ hqnn_noent_noreup_q4_l2 ┆ 3       ┆ 0.5799   ┆ 0.0015 ┆ 0.041     │
│ 12   ┆ classical control ┆ ctrl_mlp_q4_l2          ┆ 3       ┆ 0.5799   ┆ 0.0013 ┆ 0.041     │
│ 13   ┆ classical control ┆ ctrl_bottleneck_q4_l2   ┆ 3       ┆ 0.585    ┆ 0.002  ┆ 0.033     │
│ 14   ┆ deep              ┆ dlinear                 ┆ 3       ┆ 0.5865   ┆ 0.0007 ┆ 0.03      │
│ 15   ┆ deep              ┆ nlinear                 ┆ 3       ┆ 0.5867   ┆ 0.0012 ┆ 0.03      │
│ 17   ┆ linear            ┆ ar_lags                 ┆ 1       ┆ 0.5876   ┆ 0.0    ┆ 0.028     │
│ 22   ┆ naive             ┆ persistence             ┆ 1       ┆ 0.6048   ┆ 0.0    ┆ 0.0       │
└──────┴───────────────────┴─────────────────────────┴─────────┴──────────┴────────┴───────────┘
```

## Cell 7: Diebold-Mariano (absolute errors, Newey-West lag 60, Holm per horizon) and ramp breakdown, seed 0
_execution count: 7_

```
references found: ['patchtst', 'gru_uni', 'persistence'] | missing: []
shape: (3, 3)
┌─────┬───────┬──────────┐
│ h   ┆ n_min ┆ n_common │
│ --- ┆ ---   ┆ ---      │
│ i64 ┆ u32   ┆ u32      │
╞═════╪═══════╪══════════╡
│ 1   ┆ 42575 ┆ 42575    │
│ 5   ┆ 42567 ┆ 42567    │
│ 15  ┆ 42547 ┆ 42547    │
└─────┴───────┴──────────┘
shape: (18, 9)
┌─────┬───────────────────┬─────────────────────────┬────────┬────────┬─────────┬────────────┬─────────────────────────┬──────────────────┐
│ h   ┆ model_a           ┆ model_b                 ┆ MAE_a  ┆ MAE_b  ┆ DM_stat ┆ DM_p_holm  ┆ better                  ┆ significant_5pct │
│ --- ┆ ---               ┆ ---                     ┆ ---    ┆ ---    ┆ ---     ┆ ---        ┆ ---                     ┆ ---              │
│ i64 ┆ str               ┆ str                     ┆ f64    ┆ f64    ┆ f64     ┆ f64        ┆ str                     ┆ bool             │
╞═════╪═══════════════════╪═════════════════════════╪════════╪════════╪═════════╪════════════╪═════════════════════════╪══════════════════╡
│ 1   ┆ hqnn_noreup_q4_l2 ┆ hqnn_noent_noreup_q4_l2 ┆ 0.2697 ┆ 0.2639 ┆ 9.83    ┆ 7.9738e-22 ┆ hqnn_noent_noreup_q4_l2 ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2 ┆ ctrl_mlp_q4_l2          ┆ 0.2697 ┆ 0.2763 ┆ -10.02  ┆ 1.3559e-22 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2 ┆ ctrl_bottleneck_q4_l2   ┆ 0.2697 ┆ 0.2628 ┆ 8.25    ┆ 1.2649e-15 ┆ ctrl_bottleneck_q4_l2   ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2 ┆ patchtst                ┆ 0.2697 ┆ 0.2533 ┆ 16.29   ┆ 2.4691e-58 ┆ patchtst                ┆ true             │
│ 1   ┆ hqnn_noreup_q4_l2 ┆ gru_uni                 ┆ 0.2697 ┆ 0.2702 ┆ -0.49   ┆ 0.623234   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 1   ┆ hqnn_noreup_q4_l2 ┆ persistence             ┆ 0.2697 ┆ 0.2866 ┆ -9.15   ┆ 5.3170e-19 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2 ┆ hqnn_noent_noreup_q4_l2 ┆ 0.4056 ┆ 0.4053 ┆ 0.58    ┆ 1.0        ┆ hqnn_noent_noreup_q4_l2 ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2 ┆ ctrl_mlp_q4_l2          ┆ 0.4056 ┆ 0.4061 ┆ -0.52   ┆ 1.0        ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2 ┆ ctrl_bottleneck_q4_l2   ┆ 0.4056 ┆ 0.4111 ┆ -4.81   ┆ 0.00002    ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2 ┆ patchtst                ┆ 0.4056 ┆ 0.3982 ┆ 4.12    ┆ 0.000376   ┆ patchtst                ┆ true             │
│ 5   ┆ hqnn_noreup_q4_l2 ┆ gru_uni                 ┆ 0.4056 ┆ 0.402  ┆ 0.89    ┆ 1.0        ┆ gru_uni                 ┆ false            │
│ 5   ┆ hqnn_noreup_q4_l2 ┆ persistence             ┆ 0.4056 ┆ 0.4299 ┆ -12.2   ┆ 6.3798e-33 ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2 ┆ hqnn_noent_noreup_q4_l2 ┆ 0.577  ┆ 0.5783 ┆ -1.92   ┆ 0.443022   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2 ┆ ctrl_mlp_q4_l2          ┆ 0.577  ┆ 0.5786 ┆ -1.48   ┆ 0.832389   ┆ hqnn_noreup_q4_l2       ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2 ┆ ctrl_bottleneck_q4_l2   ┆ 0.577  ┆ 0.5864 ┆ -6.17   ┆ 1.1023e-8  ┆ hqnn_noreup_q4_l2       ┆ true             │
│ 15  ┆ hqnn_noreup_q4_l2 ┆ patchtst                ┆ 0.577  ┆ 0.5688 ┆ 2.64    ┆ 0.073639   ┆ patchtst                ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2 ┆ gru_uni                 ┆ 0.577  ┆ 0.5731 ┆ 0.61    ┆ 1.0        ┆ gru_uni                 ┆ false            │
│ 15  ┆ hqnn_noreup_q4_l2 ┆ persistence             ┆ 0.577  ┆ 0.6048 ┆ -11.86  ┆ 3.6703e-31 ┆ hqnn_noreup_q4_l2       ┆ true             │
└─────┴───────────────────┴─────────────────────────┴────────┴────────┴─────────┴────────────┴─────────────────────────┴──────────────────┘
shape: (21, 8)
┌─────┬─────────────────────────┬────────┬────────────┬──────────┬──────────────┬────────────┬─────────┐
│ h   ┆ model                   ┆ n_ramp ┆ MAE_normal ┆ MAE_ramp ┆ skill_normal ┆ skill_ramp ┆ ramp_F1 │
│ --- ┆ ---                     ┆ ---    ┆ ---        ┆ ---      ┆ ---          ┆ ---        ┆ ---     │
│ i64 ┆ str                     ┆ i64    ┆ f64        ┆ f64      ┆ f64          ┆ f64        ┆ f64     │
╞═════╪═════════════════════════╪════════╪════════════╪══════════╪══════════════╪════════════╪═════════╡
│ 1   ┆ patchtst                ┆ 1903   ┆ 0.2046     ┆ 1.322    ┆ 0.096        ┆ 0.178      ┆ 0.169   │
│ 1   ┆ hqnn_noent_noreup_q4_l2 ┆ 1903   ┆ 0.2127     ┆ 1.385    ┆ 0.06         ┆ 0.138      ┆ 0.152   │
│ 1   ┆ gru_uni                 ┆ 1903   ┆ 0.2193     ┆ 1.392    ┆ 0.032        ┆ 0.134      ┆ 0.182   │
│ 1   ┆ hqnn_noreup_q4_l2       ┆ 1903   ┆ 0.2172     ┆ 1.422    ┆ 0.041        ┆ 0.116      ┆ 0.168   │
│ 1   ┆ ctrl_mlp_q4_l2          ┆ 1903   ┆ 0.2236     ┆ 1.432    ┆ 0.012        ┆ 0.109      ┆ 0.209   │
│ 1   ┆ ctrl_bottleneck_q4_l2   ┆ 1903   ┆ 0.2088     ┆ 1.444    ┆ 0.078        ┆ 0.102      ┆ 0.005   │
│ 1   ┆ persistence             ┆ 1903   ┆ 0.2264     ┆ 1.607    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 5   ┆ gru_uni                 ┆ 2444   ┆ 0.2912     ┆ 2.234    ┆ 0.019        ┆ 0.151      ┆ 0.23    │
│ 5   ┆ patchtst                ┆ 2444   ┆ 0.287      ┆ 2.236    ┆ 0.033        ┆ 0.15       ┆ 0.257   │
│ 5   ┆ ctrl_mlp_q4_l2          ┆ 2444   ┆ 0.2888     ┆ 2.341    ┆ 0.027        ┆ 0.11       ┆ 0.258   │
│ 5   ┆ hqnn_noent_noreup_q4_l2 ┆ 2444   ┆ 0.2864     ┆ 2.363    ┆ 0.035        ┆ 0.102      ┆ 0.227   │
│ 5   ┆ hqnn_noreup_q4_l2       ┆ 2444   ┆ 0.2864     ┆ 2.372    ┆ 0.035        ┆ 0.098      ┆ 0.217   │
│ 5   ┆ ctrl_bottleneck_q4_l2   ┆ 2444   ┆ 0.2839     ┆ 2.509    ┆ 0.043        ┆ 0.046      ┆ 0.003   │
│ 5   ┆ persistence             ┆ 2444   ┆ 0.2967     ┆ 2.631    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
│ 15  ┆ gru_uni                 ┆ 2740   ┆ 0.4159     ┆ 2.843    ┆ 0.009        ┆ 0.133      ┆ 0.203   │
│ 15  ┆ patchtst                ┆ 2740   ┆ 0.4098     ┆ 2.861    ┆ 0.024        ┆ 0.127      ┆ 0.201   │
│ 15  ┆ ctrl_mlp_q4_l2          ┆ 2740   ┆ 0.4089     ┆ 3.027    ┆ 0.026        ┆ 0.077      ┆ 0.12    │
│ 15  ┆ hqnn_noreup_q4_l2       ┆ 2740   ┆ 0.4055     ┆ 3.052    ┆ 0.034        ┆ 0.069      ┆ 0.097   │
│ 15  ┆ hqnn_noent_noreup_q4_l2 ┆ 2740   ┆ 0.4067     ┆ 3.053    ┆ 0.031        ┆ 0.069      ┆ 0.091   │
│ 15  ┆ ctrl_bottleneck_q4_l2   ┆ 2740   ┆ 0.4058     ┆ 3.192    ┆ 0.033        ┆ 0.026      ┆ 0.0     │
│ 15  ┆ persistence             ┆ 2740   ┆ 0.4199     ┆ 3.279    ┆ 0.0          ┆ 0.0        ┆ 0.0     │
└─────┴─────────────────────────┴────────┴────────────┴──────────┴──────────────┴────────────┴─────────┘
```

## Cell 8: summary
_execution count: 8_

```
written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week5/qnn_val.json
h= 1: hqnn_noreup_q4_l2 +0.074 | hqnn_noent_noreup_q4_l2 +0.086 | ctrl_mlp_q4_l2 +0.056 | ctrl_bottleneck_q4_l2 +0.066
h= 5: hqnn_noreup_q4_l2 +0.055 | hqnn_noent_noreup_q4_l2 +0.054 | ctrl_mlp_q4_l2 +0.053 | ctrl_bottleneck_q4_l2 +0.045
h=15: hqnn_noreup_q4_l2 +0.043 | hqnn_noent_noreup_q4_l2 +0.041 | ctrl_mlp_q4_l2 +0.041 | ctrl_bottleneck_q4_l2 +0.033
```

## Cell 9: export outputs (save the notebook first with Cmd+S, then run this cell)
_execution count: None_

(no output)

# Outputs of `10_week4_nf_smoke.ipynb`
Exported 2026-10-09 14:47 from the saved notebook file.

## Cell 1: setup
_execution count: 1_

```stderr
2026-10-09 14:46:05,941	INFO util.py:155 -- Missing packages: ['ipywidgets']. Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.
2026-10-09 14:46:06,428	INFO util.py:155 -- Missing packages: ['ipywidgets']. Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.
```

```
{
 "versions": {
  "python": "3.11.16",
  "torch": "2.14.1",
  "neuralforecast": "3.3.0",
  "lightning": "2.6.6",
  "numpy": "2.4.6",
  "polars": "1.44.2"
 },
 "mps_available": true,
 "torch_threads": 4
}
project root: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting | models: ['nhits', 'patchtst', 'dlinear', 'nlinear'] | devices: ['cpu', 'mps']
```

## Cell 2: probe each architecture on a random batch (shape check, parameter count, CPU and MPS forward)
_execution count: 2_

```stderr
Seed set to 0
Seed set to 0
```

```
{'model': 'nhits', 'device': 'cpu', 'ok': True, 'n_params': 676301, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
```

```stderr
Seed set to 0
Seed set to 0
```

```
{'model': 'nhits', 'device': 'mps', 'ok': True, 'n_params': 676301, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
{'model': 'patchtst', 'device': 'cpu', 'ok': True, 'n_params': 75215, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
```

```stderr
Seed set to 0
Seed set to 0
Seed set to 0
Seed set to 0
```

```
{'model': 'patchtst', 'device': 'mps', 'ok': True, 'n_params': 75215, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
{'model': 'dlinear', 'device': 'cpu', 'ok': True, 'n_params': 1830, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
{'model': 'dlinear', 'device': 'mps', 'ok': True, 'n_params': 1830, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
{'model': 'nlinear', 'device': 'cpu', 'ok': True, 'n_params': 915, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
{'model': 'nlinear', 'device': 'mps', 'ok': True, 'n_params': 915, 'out_shape': '(8, 3)', 'layout': 'BL1', 'finite': True}
{'model': 'gru', 'device': 'cpu', 'ok': True, 'n_params': 38019, 'out_shape': '(8, 3)', 'layout': 'rnn', 'finite': True}
{'model': 'gru', 'device': 'mps', 'ok': True, 'n_params': 38019, 'out_shape': '(8, 3)', 'layout': 'rnn', 'finite': True}

ALL PROBES OK
```

## Cell 3: end-to-end training on SYNTHETIC data (no real data); every model should beat persistence clearly
_execution count: 3_

```stderr
Seed set to 0
```

```
synthetic rows: 30000 | valid share: 0.97 | usable train/val origins h1: 20100 4289
```

```stderr
Seed set to 0
```

```
{'model': 'nhits', 'ok': True, 'n_params': 676301, 's': 0.6, 'skill_h1': 0.068, 'skill_h5': 0.081, 'skill_h15': 0.16, 'finite': True}
```

```stderr
Seed set to 0
Seed set to 0
```

```
{'model': 'patchtst', 'ok': True, 'n_params': 75215, 's': 3.5, 'skill_h1': 0.076, 'skill_h5': 0.081, 'skill_h15': 0.16, 'finite': True}
{'model': 'dlinear', 'ok': True, 'n_params': 1830, 's': 0.1, 'skill_h1': -0.021, 'skill_h5': 0.068, 'skill_h15': 0.155, 'finite': True}
{'model': 'nlinear', 'ok': True, 'n_params': 915, 's': 0.0, 'skill_h1': -0.118, 'skill_h5': 0.052, 'skill_h15': 0.145, 'finite': True}
{'model': 'gru', 'ok': True, 'n_params': 38019, 's': 6.7, 'skill_h1': 0.021, 'skill_h5': 0.065, 'skill_h15': 0.114, 'finite': True}

SYNTHETIC CHECK NOT PASSED: send this output
```

## Cell 4: timing on the REAL modelling dataset (a few batches per model and device; no full training)
_execution count: 4_

```stderr
Seed set to 0
```

```
rows 303,840 | train origins 196,700 | val origins 42,575 | scale s = 0.4340 kW
```

```stderr
Seed set to 0
```

```
{'model': 'nhits', 'device': 'cpu', 'n_params': 676301, 'ms_per_train_sample': 0.021, 'ms_per_pred_sample': 0.0042, 'est_epoch_min': 0.0719}
```

```stderr
Seed set to 0
```

```
{'model': 'nhits', 'device': 'mps', 'n_params': 676301, 'ms_per_train_sample': 0.0444, 'ms_per_pred_sample': 0.0339, 'est_epoch_min': 0.1695}
```

```stderr
Seed set to 0
```

```
{'model': 'patchtst', 'device': 'cpu', 'n_params': 75215, 'ms_per_train_sample': 0.1185, 'ms_per_pred_sample': 0.0323, 'est_epoch_min': 0.4115}
```

```stderr
Seed set to 0
Seed set to 0
```

```
{'model': 'patchtst', 'device': 'mps', 'n_params': 75215, 'ms_per_train_sample': 0.0583, 'ms_per_pred_sample': 0.0953, 'est_epoch_min': 0.2588}
{'model': 'dlinear', 'device': 'cpu', 'n_params': 1830, 'ms_per_train_sample': 0.0039, 'ms_per_pred_sample': 0.0005, 'est_epoch_min': 0.0133}
```

```stderr
Seed set to 0
Seed set to 0
```

```
{'model': 'dlinear', 'device': 'mps', 'n_params': 1830, 'ms_per_train_sample': 0.0093, 'ms_per_pred_sample': 0.0184, 'est_epoch_min': 0.0437}
{'model': 'nlinear', 'device': 'cpu', 'n_params': 915, 'ms_per_train_sample': 0.0042, 'ms_per_pred_sample': 0.0006, 'est_epoch_min': 0.0141}
{'model': 'nlinear', 'device': 'mps', 'n_params': 915, 'ms_per_train_sample': 0.0083, 'ms_per_pred_sample': 0.0006, 'est_epoch_min': 0.0276}
{'model': 'gru', 'device': 'cpu', 'n_params': 38019, 'ms_per_train_sample': 0.2328, 'ms_per_pred_sample': 0.0765, 'est_epoch_min': 0.8176}
{'model': 'gru', 'device': 'mps', 'n_params': 38019, 'ms_per_train_sample': 0.1697, 'ms_per_pred_sample': 0.0436, 'est_epoch_min': 0.5872}

fastest device per model (estimated minutes per full epoch incl. validation):
shape: (5, 4)
┌──────────┬────────┬──────────┬───────────────┐
│ model    ┆ device ┆ n_params ┆ est_epoch_min │
│ ---      ┆ ---    ┆ ---      ┆ ---           │
│ str      ┆ str    ┆ i64      ┆ f64           │
╞══════════╪════════╪══════════╪═══════════════╡
│ dlinear  ┆ cpu    ┆ 1830     ┆ 0.0133        │
│ nlinear  ┆ cpu    ┆ 915      ┆ 0.0141        │
│ nhits    ┆ cpu    ┆ 676301   ┆ 0.0719        │
│ patchtst ┆ mps    ┆ 75215    ┆ 0.2588        │
│ gru      ┆ mps    ┆ 38019    ┆ 0.5872        │
└──────────┴────────┴──────────┴───────────────┘
```

## Cell 5: one short REAL-data check per model: 1 epoch on 20,000 train origins, scored on 5,000 val origins
_execution count: 5_

```stderr
Seed set to 0
Seed set to 0
```

```
  epoch   1 | train loss 1.90712 | val MAE 0.2775 / 0.4152 / 0.5885 kW | 0 s
{'model': 'nhits', 'device': 'cpu', 's': 0.7, 'MAE_h1': 0.2775, 'pers_h1': 0.2861, 'MAE_h5': 0.4152, 'pers_h5': 0.4313, 'MAE_h15': 0.5885, 'pers_h15': 0.6089, 'finite': True}
```

```stderr
Seed set to 0
Seed set to 0
```

```
  epoch   1 | train loss 2.01382 | val MAE 0.3069 / 0.4398 / 0.5954 kW | 2 s
{'model': 'patchtst', 'device': 'mps', 's': 2.5, 'MAE_h1': 0.3069, 'pers_h1': 0.2861, 'MAE_h5': 0.4398, 'pers_h5': 0.4313, 'MAE_h15': 0.5954, 'pers_h15': 0.6089, 'finite': True}
  epoch   1 | train loss 2.32446 | val MAE 0.3197 / 0.4557 / 0.6195 kW | 0 s
{'model': 'dlinear', 'device': 'cpu', 's': 0.1, 'MAE_h1': 0.3197, 'pers_h1': 0.2861, 'MAE_h5': 0.4557, 'pers_h5': 0.4313, 'MAE_h15': 0.6195, 'pers_h15': 0.6089, 'finite': True}
  epoch   1 | train loss 2.40600 | val MAE 0.3629 / 0.4783 / 0.6349 kW | 0 s
{'model': 'nlinear', 'device': 'cpu', 's': 0.1, 'MAE_h1': 0.3629, 'pers_h1': 0.2861, 'MAE_h5': 0.4783, 'pers_h5': 0.4313, 'MAE_h15': 0.6349, 'pers_h15': 0.6089, 'finite': True}
```

## Cell 6: save and summarise
_execution count: 6_

```
written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/nf_smoke.json
dlinear   cpu  params    1,830 | ~  0.0 min/epoch | ~ 0.0 h for 3 seeds x 10 epochs
nlinear   cpu  params      915 | ~  0.0 min/epoch | ~ 0.0 h for 3 seeds x 10 epochs
nhits     cpu  params  676,301 | ~  0.1 min/epoch | ~ 0.0 h for 3 seeds x 10 epochs
patchtst  mps  params   75,215 | ~  0.3 min/epoch | ~ 0.1 h for 3 seeds x 10 epochs
gru       mps  params   38,019 | ~  0.6 min/epoch | ~ 0.3 h for 3 seeds x 10 epochs
```

## %run ../src/tools/nb_outputs_to_md.py 10_week4_nf_smoke.ipynb
_execution count: 7_

```
written 10_week4_nf_smoke_outputs.md | code cells 6 | figures 0
```

## Cell 8
_execution count: None_

(no output)

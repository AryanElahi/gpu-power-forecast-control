# Outputs of `13_week4_chronos_smoke.ipynb`
Exported 2026-10-09 16:18 from the saved notebook file.

## Cell 1: setup (offline: weights already in the cache)
_execution count: 1_

```stderr
Loading weights: 100%|█████████████████████| 143/143 [00:00<00:00, 6364.24it/s]
```

```
loaded on cpu in 0.1 s
```

```stderr
Loading weights: 100%|████████████████████| 143/143 [00:00<00:00, 41823.13it/s]
```

```
loaded on mps in 0.7 s
{
 "model": "amazon/chronos-bolt-small",
 "chronos": "2.3.2",
 "transformers": "5.19.0",
 "torch": "2.14.1",
 "mps": true,
 "n_params": 47718016
} | params 47,718,016
```

## Cell 2: SYNTHETIC check (shapes, NaN handling, sensible accuracy on a smooth series)
_execution count: 2_

```
{'L': 60, 'shape_ok': True, 'nan_in_context_rows': 0, 'skill_h1': np.float32(0.116), 'skill_h5': np.float32(0.125), 'skill_h15': np.float32(0.298), 'quantiles_ordered': True}
{'L': 512, 'shape_ok': True, 'nan_in_context_rows': 12, 'skill_h1': np.float32(0.085), 'skill_h5': np.float32(0.117), 'skill_h15': np.float32(0.44), 'quantiles_ordered': True}

SYNTHETIC CHECK PASSED
```

## Cell 3: REAL data, 2,000 validation origins: timing (CPU vs MPS) and accuracy vs persistence
_execution count: 3_

```
{'L': 60, 'device': 'cpu', 'ms_per_origin': 0.681, 'est_min_full_val': 0.5, 'nan_share_context': 0.0, 'MAE_h1': 0.2721, 'skill_h1': 0.019, 'MAE_h5': 0.438, 'skill_h5': -0.001, 'MAE_h15': 0.6363, 'skill_h15': -0.082}
{'L': 60, 'device': 'mps', 'ms_per_origin': 0.646, 'est_min_full_val': 0.5, 'nan_share_context': 0.0, 'MAE_h1': 0.2721, 'skill_h1': 0.019, 'MAE_h5': 0.438, 'skill_h5': -0.001, 'MAE_h15': 0.6363, 'skill_h15': -0.082}
{'L': 512, 'device': 'cpu', 'ms_per_origin': 3.05, 'est_min_full_val': 2.2, 'nan_share_context': 0.0054, 'MAE_h1': 0.2776, 'skill_h1': -0.001, 'MAE_h5': 0.4224, 'skill_h5': 0.035, 'MAE_h15': 0.5721, 'skill_h15': 0.027}
{'L': 512, 'device': 'mps', 'ms_per_origin': 1.773, 'est_min_full_val': 1.3, 'nan_share_context': 0.0054, 'MAE_h1': 0.2776, 'skill_h1': -0.001, 'MAE_h5': 0.4224, 'skill_h5': 0.035, 'MAE_h15': 0.5721, 'skill_h15': 0.027}
```

## Cell 4: save
_execution count: 4_

```
written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/chronos_smoke.json
```

## Cell 5: export outputs (save the notebook first with Cmd+S)
_execution count: 5_

```
written 13_week4_chronos_smoke_outputs.md | code cells 5 | figures 0
```

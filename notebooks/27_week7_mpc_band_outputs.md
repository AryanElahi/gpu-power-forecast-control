# Outputs of `27_week7_mpc_band.ipynb`
Exported 2026-10-14 12:08 from the saved notebook file.

## Cell 1: setup
_execution count: 1_

```
{'lightgbm_cov|1|aci': 'aci|abs|W10080|g0.01', 'lightgbm_cov|1|mondrian_pc': 'mondrian_pc|res60|W1440', 'lightgbm_cov|5|aci': 'aci|abs|W10080|g0.005', 'lightgbm_cov|5|mondrian_pc': 'mondrian_pc|res60|W1440', 'lightgbm_cov|15|aci': 'aci|vol60|W10080|g0.001', 'lightgbm_cov|15|mondrian_pc': 'mondrian_pc|abs|W1440', 'persistence|1|aci': 'aci|abs|W1440|g0.01', 'persistence|1|mondrian_pc': 'mondrian_pc|res60|W10080', 'persistence|5|aci': 'aci|vol60|W10080|g0.001', 'persistence|5|mondrian_pc': 'mondrian_pc|vol60|W10080', 'persistence|15|aci': 'aci|vol60|W4320|g0.001', 'persistence|15|mondrian_pc': 'mondrian_pc|vol60|W4320'}
notebook-26 references loaded: 21 validation rows, 31 test rows
freeze record exists: False
```

## Cell 2: unit checks of the scenario MPC (synthetic)
_execution count: 2_

```
PASS  three identical flat scenarios = notebook-26 persistence MPC
PASS  non-anticipativity: one first move for all scenarios  [(3.5111897955016386e-10, 3.684034850867631e-10)]
PASS  an upper scenario with a jump larger than P_max (8 kW > 4 kW) makes the controller act before any load change (mean |g - P| > 0.1 kW)  [0.39]

ALL UNIT CHECKS PASSED
```

## Cell 3: validation: conformal bands (as in notebook 24's validation protocol) and band-MPC runs (~30 min, cached)
_execution count: 3_

```
  bands lightgbm_cov aci: 42,483 minutes
  bands lightgbm_cov mondrian_pc: 42,231 minutes
  bands persistence aci: 42,483 minutes
  bands persistence mondrian_pc: 42,483 minutes
  val lightgbm_cov aci         E=1 R=1:    87 s
  val lightgbm_cov aci         E=2 R=1:    86 s
  val lightgbm_cov mondrian_pc E=1 R=1:    83 s
  val lightgbm_cov mondrian_pc E=2 R=1:    84 s
  val persistence  aci         E=1 R=1:    91 s
  val persistence  aci         E=2 R=1:    95 s
  val persistence  mondrian_pc E=1 R=1:    88 s
  val persistence  mondrian_pc E=2 R=1:    86 s
```

## Cell 4: validation table and selection of the conformal method per forecaster (rule in the header)
_execution count: 4_

```
shape: (8, 12)
┌──────────────┬─────────────┬───────┬────────────┬───┬─────────┬─────────┬─────────────┬──────────┐
│ forecaster   ┆ method      ┆ E_kWh ┆ viol_steps ┆ … ┆ soc_min ┆ soc_max ┆ solver_fail ┆ fallback │
│ ---          ┆ ---         ┆ ---   ┆ ---        ┆   ┆ ---     ┆ ---     ┆ ---         ┆ ---      │
│ str          ┆ str         ┆ f64   ┆ i64        ┆   ┆ f64     ┆ f64     ┆ i64         ┆ i64      │
╞══════════════╪═════════════╪═══════╪════════════╪═══╪═════════╪═════════╪═════════════╪══════════╡
│ lightgbm_cov ┆ aci         ┆ 1.0   ┆ 190        ┆ … ┆ 0.3806  ┆ 0.6965  ┆ 0           ┆ 153      │
│ lightgbm_cov ┆ aci         ┆ 2.0   ┆ 66         ┆ … ┆ 0.3553  ┆ 0.739   ┆ 0           ┆ 153      │
│ lightgbm_cov ┆ mondrian_pc ┆ 1.0   ┆ 196        ┆ … ┆ 0.3806  ┆ 0.7262  ┆ 0           ┆ 405      │
│ lightgbm_cov ┆ mondrian_pc ┆ 2.0   ┆ 62         ┆ … ┆ 0.3553  ┆ 0.6941  ┆ 0           ┆ 405      │
│ persistence  ┆ aci         ┆ 1.0   ┆ 197        ┆ … ┆ 0.3669  ┆ 0.7036  ┆ 0           ┆ 153      │
│ persistence  ┆ aci         ┆ 2.0   ┆ 65         ┆ … ┆ 0.3377  ┆ 0.6845  ┆ 0           ┆ 153      │
│ persistence  ┆ mondrian_pc ┆ 1.0   ┆ 198        ┆ … ┆ 0.3661  ┆ 0.7023  ┆ 0           ┆ 153      │
│ persistence  ┆ mondrian_pc ┆ 2.0   ┆ 66         ┆ … ┆ 0.3364  ┆ 0.6734  ┆ 0           ┆ 153      │
└──────────────┴─────────────┴───────┴────────────┴───┴─────────┴─────────┴─────────────┴──────────┘

notebook-26 references on validation (R = 1):
shape: (8, 4)
┌───────┬─────────────┬────────────┬────────────────┐
│ E_kWh ┆ scenario    ┆ viol_steps ┆ viol_excess_kW │
│ ---   ┆ ---         ┆ ---        ┆ ---            │
│ f64   ┆ str         ┆ i64        ┆ f64            │
╞═══════╪═════════════╪════════════╪════════════════╡
│ 1.0   ┆ model       ┆ 199        ┆ 276.79         │
│ 1.0   ┆ myopic      ┆ 194        ┆ 246.56         │
│ 1.0   ┆ oracle      ┆ 96         ┆ 117.78         │
│ 1.0   ┆ persistence ┆ 202        ┆ 277.76         │
│ 2.0   ┆ model       ┆ 68         ┆ 93.49          │
│ 2.0   ┆ myopic      ┆ 58         ┆ 81.82          │
│ 2.0   ┆ oracle      ┆ 7          ┆ 11.51          │
│ 2.0   ┆ persistence ┆ 68         ┆ 94.54          │
└───────┴─────────────┴────────────┴────────────────┘
selected for lightgbm_cov: mondrian_pc  (excess at E = 2: {'aci': 87.91648969927516, 'mondrian_pc': 85.0122161674571})
selected for persistence: aci  (excess at E = 2: {'aci': 87.98150690784658, 'mondrian_pc': 88.89483603005954})
```

## Cell 5: test bands rebuilt exactly as in notebook 24 (continuous validation -> test run); check vs stored intervals
_execution count: 5_

```
shape: (6, 5)
┌──────────────┬─────────────┬─────┬───────┬─────────────────────┐
│ forecaster   ┆ method      ┆ h   ┆ n     ┆ matches_notebook_24 │
│ ---          ┆ ---         ┆ --- ┆ ---   ┆ ---                 │
│ str          ┆ str         ┆ i64 ┆ i64   ┆ bool                │
╞══════════════╪═════════════╪═════╪═══════╪═════════════════════╡
│ lightgbm_cov ┆ mondrian_pc ┆ 1   ┆ 42575 ┆ true                │
│ lightgbm_cov ┆ mondrian_pc ┆ 5   ┆ 42567 ┆ true                │
│ lightgbm_cov ┆ mondrian_pc ┆ 15  ┆ 42547 ┆ true                │
│ persistence  ┆ aci         ┆ 1   ┆ 42575 ┆ true                │
│ persistence  ┆ aci         ┆ 5   ┆ 42567 ┆ true                │
│ persistence  ┆ aci         ┆ 15  ┆ 42547 ┆ true                │
└──────────────┴─────────────┴─────┴───────┴─────────────────────┘
REPRODUCTION PASSED: test bands equal the stored notebook-24 intervals
```

## Cell 6: freeze record (BEFORE the test month is simulated)
_execution count: 6_

```
freeze record mpc_band_test_freeze.json | selected {'lightgbm_cov': 'mondrian_pc', 'persistence': 'aci'}
```

## Cell 7: TEST month (once): band MPC for both forecasters (~30 min, cached)
_execution count: None_

```
  test lightgbm_cov mondrian_pc E=1 R=1:    82 s
  test lightgbm_cov mondrian_pc E=2 R=1:    83 s
  test lightgbm_cov mondrian_pc E=5 R=1:    92 s
  test lightgbm_cov mondrian_pc E=5 R=0.5:   103 s
  test persistence  aci         E=1 R=1:    97 s
  test persistence  aci         E=2 R=1:   102 s
  test persistence  aci         E=5 R=1:   106 s
```

## Cell 8: test comparison with the notebook-26 controllers on the same minutes; figures
_execution count: None_

(no output)

## Cell 9: summary
_execution count: None_

(no output)

## Cell 10: export outputs (save the notebook first with Cmd+S after cell 9 has finished, then run this cell)
_execution count: None_

(no output)

## Cell 11
_execution count: None_

(no output)

## Cell 12
_execution count: None_

(no output)

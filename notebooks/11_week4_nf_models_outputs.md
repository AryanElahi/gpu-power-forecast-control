# Outputs of `11_week4_nf_models.ipynb`
Exported 2026-10-09 15:27 from the saved notebook file.

## Cell 1: setup
_execution count: 1_

```
torch 2.14.1 | mps True | models ['nhits', 'patchtst', 'dlinear', 'nlinear', 'gru_uni'] | seeds [0, 1, 2]
```

## Cell 2: DRY RUN of the whole pipeline on SYNTHETIC data (2 epochs, small samples, no tracking; deleted after)
_execution count: 2_

```stderr
2026-10-09 14:55:07,571	INFO util.py:155 -- Missing packages: ['ipywidgets']. Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.
2026-10-09 14:55:08,167	INFO util.py:155 -- Missing packages: ['ipywidgets']. Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.
Seed set to 0
Seed set to 0
```

```
nhits     files ok: True | prediction rows 12,831 | metrics finite: True | skill h1/h5/h15: +0.026 / +0.081 / +0.160 | ramp F1 h15 0.000
```

```stderr
Seed set to 0
Seed set to 0
```

```
patchtst  files ok: True | prediction rows 12,831 | metrics finite: True | skill h1/h5/h15: +0.071 / +0.088 / +0.148 | ramp F1 h15 0.048
dlinear   files ok: True | prediction rows 12,831 | metrics finite: True | skill h1/h5/h15: -0.126 / +0.048 / +0.136 | ramp F1 h15 0.010
nlinear   files ok: True | prediction rows 12,831 | metrics finite: True | skill h1/h5/h15: -0.256 / +0.015 / +0.133 | ramp F1 h15 0.000
gru_uni   files ok: True | prediction rows 12,831 | metrics finite: True | skill h1/h5/h15: +0.018 / +0.044 / +0.072 | ramp F1 h15 0.000

DRY RUN PASSED (pipeline works end to end; skills on 2 epochs are not meaningful)
```

## Cell 3: REAL runs, validation only (resumable). Expected total about 45-60 min; W&B mode "auto".
_execution count: 3_

```stderr
Seed set to 0
```

```
rows 303,840 | train origins 196,700 | val origins 42,575
[14:55:19] nhits_s0: training on cpu
  epoch   1 | train loss 1.88721 | val MAE 0.2606 / 0.4006 / 0.5736 kW | 4 s
  epoch   2 | train loss 1.84814 | val MAE 0.2599 / 0.4011 / 0.5701 kW | 7 s
  epoch   3 | train loss 1.83291 | val MAE 0.2658 / 0.3988 / 0.5698 kW | 11 s
  epoch   4 | train loss 1.82858 | val MAE 0.2629 / 0.4160 / 0.5783 kW | 15 s
  epoch   5 | train loss 1.81504 | val MAE 0.2617 / 0.4022 / 0.5700 kW | 20 s
```

```stderr
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /Users/aryaneuse/.netrc.
Seed set to 1
```

```
[14:55:48] nhits_s0: params 676,301 | train 20 s | best epoch 2/5 | val MAE 0.2599 / 0.4011 / 0.5701 kW | skill +0.093 / +0.067 / +0.058
[14:55:48] nhits_s1: training on cpu
  epoch   1 | train loss 1.89168 | val MAE 0.2635 / 0.3991 / 0.5682 kW | 5 s
  epoch   2 | train loss 1.84885 | val MAE 0.2722 / 0.4081 / 0.5860 kW | 10 s
  epoch   3 | train loss 1.83865 | val MAE 0.2614 / 0.4028 / 0.5729 kW | 16 s
  epoch   4 | train loss 1.82689 | val MAE 0.2668 / 0.4060 / 0.5688 kW | 21 s
```

```stderr
Seed set to 2
```

```
[14:56:20] nhits_s1: params 676,301 | train 21 s | best epoch 1/4 | val MAE 0.2635 / 0.3991 / 0.5682 kW | skill +0.080 / +0.072 / +0.061
[14:56:20] nhits_s2: training on cpu
  epoch   1 | train loss 1.88532 | val MAE 0.2590 / 0.3996 / 0.5668 kW | 4 s
  epoch   2 | train loss 1.85060 | val MAE 0.2568 / 0.4013 / 0.5686 kW | 9 s
  epoch   3 | train loss 1.83588 | val MAE 0.2682 / 0.4054 / 0.5741 kW | 14 s
  epoch   4 | train loss 1.82480 | val MAE 0.2667 / 0.3995 / 0.5683 kW | 18 s
```

```stderr
Seed set to 0
```

```
[14:56:49] nhits_s2: params 676,301 | train 18 s | best epoch 1/4 | val MAE 0.2590 / 0.3996 / 0.5668 kW | skill +0.096 / +0.071 / +0.063
[14:56:49] patchtst_s0: training on mps
  epoch   1 | train loss 1.95612 | val MAE 0.2641 / 0.4094 / 0.5785 kW | 15 s
  epoch   2 | train loss 1.90539 | val MAE 0.2591 / 0.4041 / 0.5701 kW | 28 s
  epoch   3 | train loss 1.88245 | val MAE 0.2648 / 0.4006 / 0.5913 kW | 41 s
  epoch   4 | train loss 1.87303 | val MAE 0.2552 / 0.4033 / 0.5722 kW | 53 s
  epoch   5 | train loss 1.86577 | val MAE 0.2566 / 0.4019 / 0.5705 kW | 66 s
  epoch   6 | train loss 1.85805 | val MAE 0.2554 / 0.4109 / 0.5751 kW | 79 s
  epoch   7 | train loss 1.85210 | val MAE 0.2547 / 0.4025 / 0.5697 kW | 92 s
  epoch   8 | train loss 1.84853 | val MAE 0.2556 / 0.4009 / 0.5726 kW | 109 s
  epoch   9 | train loss 1.84288 | val MAE 0.2551 / 0.4010 / 0.5701 kW | 123 s
  epoch  10 | train loss 1.83793 | val MAE 0.2541 / 0.3989 / 0.5711 kW | 136 s
  epoch  11 | train loss 1.83339 | val MAE 0.2550 / 0.3985 / 0.5695 kW | 149 s
  epoch  12 | train loss 1.82895 | val MAE 0.2532 / 0.4004 / 0.5717 kW | 162 s
  epoch  13 | train loss 1.82585 | val MAE 0.2563 / 0.3970 / 0.5672 kW | 176 s
  epoch  14 | train loss 1.81728 | val MAE 0.2528 / 0.4001 / 0.5711 kW | 191 s
  epoch  15 | train loss 1.81472 | val MAE 0.2533 / 0.3982 / 0.5688 kW | 204 s
  epoch  16 | train loss 1.81203 | val MAE 0.2512 / 0.3996 / 0.5703 kW | 216 s
  epoch  17 | train loss 1.80471 | val MAE 0.2583 / 0.4013 / 0.5699 kW | 229 s
  epoch  18 | train loss 1.80555 | val MAE 0.2570 / 0.4000 / 0.5714 kW | 242 s
```

```stderr
Seed set to 1
```

```
[15:01:12] patchtst_s0: params 75,215 | train 242 s | best epoch 15/18 | val MAE 0.2533 / 0.3982 / 0.5688 kW | skill +0.116 / +0.074 / +0.060
[15:01:12] patchtst_s1: training on mps
  epoch   1 | train loss 1.97239 | val MAE 0.2813 / 0.4152 / 0.6092 kW | 14 s
  epoch   2 | train loss 1.92271 | val MAE 0.2866 / 0.4425 / 0.5705 kW | 27 s
  epoch   3 | train loss 1.88697 | val MAE 0.2610 / 0.4032 / 0.5709 kW | 40 s
  epoch   4 | train loss 1.87606 | val MAE 0.2735 / 0.4123 / 0.5785 kW | 54 s
  epoch   5 | train loss 1.86840 | val MAE 0.2552 / 0.4032 / 0.5696 kW | 67 s
  epoch   6 | train loss 1.85553 | val MAE 0.2533 / 0.4015 / 0.5728 kW | 81 s
  epoch   7 | train loss 1.85503 | val MAE 0.2560 / 0.4010 / 0.5701 kW | 95 s
  epoch   8 | train loss 1.85120 | val MAE 0.2579 / 0.4001 / 0.5756 kW | 108 s
  epoch   9 | train loss 1.84409 | val MAE 0.2556 / 0.4012 / 0.5711 kW | 121 s
  epoch  10 | train loss 1.83780 | val MAE 0.2556 / 0.4004 / 0.5733 kW | 135 s
```

```stderr
Seed set to 2
```

```
[15:03:41] patchtst_s1: params 75,215 | train 135 s | best epoch 7/10 | val MAE 0.2560 / 0.4010 / 0.5701 kW | skill +0.107 / +0.067 / +0.057
[15:03:41] patchtst_s2: training on mps
  epoch   1 | train loss 1.95258 | val MAE 0.3150 / 0.4167 / 0.5932 kW | 14 s
  epoch   2 | train loss 1.89929 | val MAE 0.2631 / 0.4034 / 0.5758 kW | 28 s
  epoch   3 | train loss 1.87956 | val MAE 0.2608 / 0.4029 / 0.5736 kW | 41 s
  epoch   4 | train loss 1.87358 | val MAE 0.2646 / 0.4045 / 0.5753 kW | 54 s
  epoch   5 | train loss 1.86676 | val MAE 0.2680 / 0.4022 / 0.5705 kW | 67 s
  epoch   6 | train loss 1.86013 | val MAE 0.2574 / 0.4020 / 0.5723 kW | 80 s
  epoch   7 | train loss 1.85434 | val MAE 0.2537 / 0.4029 / 0.5728 kW | 93 s
  epoch   8 | train loss 1.84765 | val MAE 0.2566 / 0.4006 / 0.5744 kW | 106 s
  epoch   9 | train loss 1.84891 | val MAE 0.2579 / 0.3987 / 0.5678 kW | 119 s
  epoch  10 | train loss 1.83881 | val MAE 0.2621 / 0.4000 / 0.5703 kW | 132 s
  epoch  11 | train loss 1.83459 | val MAE 0.2579 / 0.3980 / 0.5713 kW | 144 s
  epoch  12 | train loss 1.83047 | val MAE 0.2538 / 0.3993 / 0.5709 kW | 157 s
  epoch  13 | train loss 1.82485 | val MAE 0.2556 / 0.4013 / 0.5723 kW | 169 s
  epoch  14 | train loss 1.82229 | val MAE 0.2535 / 0.3982 / 0.5708 kW | 182 s
  epoch  15 | train loss 1.81735 | val MAE 0.2537 / 0.3975 / 0.5692 kW | 195 s
  epoch  16 | train loss 1.81542 | val MAE 0.2585 / 0.4005 / 0.5711 kW | 208 s
  epoch  17 | train loss 1.80958 | val MAE 0.2530 / 0.4017 / 0.5745 kW | 220 s
  epoch  18 | train loss 1.80422 | val MAE 0.2545 / 0.3993 / 0.5716 kW | 233 s
```

```stderr
Seed set to 0
```

```
[15:07:54] patchtst_s2: params 75,215 | train 233 s | best epoch 15/18 | val MAE 0.2537 / 0.3975 / 0.5692 kW | skill +0.115 / +0.075 / +0.059
[15:07:54] dlinear_s0: training on cpu
  epoch   1 | train loss 2.02886 | val MAE 0.2683 / 0.4176 / 0.5886 kW | 0 s
  epoch   2 | train loss 1.95088 | val MAE 0.2690 / 0.4232 / 0.5862 kW | 1 s
  epoch   3 | train loss 1.94706 | val MAE 0.2677 / 0.4148 / 0.5857 kW | 1 s
  epoch   4 | train loss 1.94716 | val MAE 0.2693 / 0.4215 / 0.5874 kW | 1 s
  epoch   5 | train loss 1.94776 | val MAE 0.2676 / 0.4210 / 0.5895 kW | 2 s
  epoch   6 | train loss 1.94739 | val MAE 0.2684 / 0.4166 / 0.5888 kW | 2 s
```

```stderr
Seed set to 1
```

```
[15:08:06] dlinear_s0: params 1,830 | train 2 s | best epoch 3/6 | val MAE 0.2677 / 0.4148 / 0.5857 kW | skill +0.066 / +0.035 / +0.032
[15:08:06] dlinear_s1: training on cpu
  epoch   1 | train loss 2.08105 | val MAE 0.2699 / 0.4149 / 0.5870 kW | 0 s
  epoch   2 | train loss 1.94799 | val MAE 0.2922 / 0.4408 / 0.5944 kW | 1 s
  epoch   3 | train loss 1.94799 | val MAE 0.2674 / 0.4180 / 0.5882 kW | 1 s
  epoch   4 | train loss 1.94849 | val MAE 0.2822 / 0.4245 / 0.5939 kW | 1 s
```

```stderr
Seed set to 2
```

```
[15:08:22] dlinear_s1: params 1,830 | train 1 s | best epoch 1/4 | val MAE 0.2699 / 0.4149 / 0.5870 kW | skill +0.058 / +0.035 / +0.030
[15:08:22] dlinear_s2: training on cpu
  epoch   1 | train loss 2.01946 | val MAE 0.2742 / 0.4174 / 0.5889 kW | 0 s
  epoch   2 | train loss 1.94877 | val MAE 0.2665 / 0.4208 / 0.5868 kW | 1 s
  epoch   3 | train loss 1.94837 | val MAE 0.2735 / 0.4166 / 0.5894 kW | 1 s
  epoch   4 | train loss 1.94670 | val MAE 0.2802 / 0.4153 / 0.5869 kW | 1 s
  epoch   5 | train loss 1.94666 | val MAE 0.2707 / 0.4171 / 0.5930 kW | 1 s
```

```stderr
Seed set to 0
```

```
[15:08:45] dlinear_s2: params 1,830 | train 1 s | best epoch 2/5 | val MAE 0.2665 / 0.4208 / 0.5868 kW | skill +0.070 / +0.021 / +0.030
[15:08:45] nlinear_s0: training on cpu
  epoch   1 | train loss 2.08493 | val MAE 0.2735 / 0.4227 / 0.5916 kW | 0 s
  epoch   2 | train loss 1.96032 | val MAE 0.2694 / 0.4223 / 0.5869 kW | 0 s
  epoch   3 | train loss 1.94847 | val MAE 0.2671 / 0.4149 / 0.5858 kW | 1 s
  epoch   4 | train loss 1.94585 | val MAE 0.2699 / 0.4213 / 0.5875 kW | 1 s
  epoch   5 | train loss 1.94562 | val MAE 0.2671 / 0.4201 / 0.5889 kW | 1 s
  epoch   6 | train loss 1.94492 | val MAE 0.2680 / 0.4156 / 0.5877 kW | 1 s
```

```stderr
Seed set to 1
```

```
[15:08:57] nlinear_s0: params 915 | train 1 s | best epoch 3/6 | val MAE 0.2671 / 0.4149 / 0.5858 kW | skill +0.068 / +0.035 / +0.032
[15:08:57] nlinear_s1: training on cpu
  epoch   1 | train loss 2.11615 | val MAE 0.2747 / 0.4186 / 0.5918 kW | 0 s
  epoch   2 | train loss 1.96092 | val MAE 0.2918 / 0.4383 / 0.5951 kW | 0 s
  epoch   3 | train loss 1.95013 | val MAE 0.2677 / 0.4175 / 0.5886 kW | 1 s
  epoch   4 | train loss 1.94738 | val MAE 0.2818 / 0.4216 / 0.5923 kW | 1 s
  epoch   5 | train loss 1.94664 | val MAE 0.2658 / 0.4156 / 0.5864 kW | 1 s
  epoch   6 | train loss 1.94623 | val MAE 0.2702 / 0.4164 / 0.5873 kW | 1 s
  epoch   7 | train loss 1.94661 | val MAE 0.2685 / 0.4140 / 0.5898 kW | 2 s
  epoch   8 | train loss 1.94364 | val MAE 0.2695 / 0.4153 / 0.5861 kW | 2 s
```

```stderr
Seed set to 2
```

```
[15:09:18] nlinear_s1: params 915 | train 2 s | best epoch 5/8 | val MAE 0.2658 / 0.4156 / 0.5864 kW | skill +0.072 / +0.033 / +0.031
[15:09:18] nlinear_s2: training on cpu
  epoch   1 | train loss 2.05271 | val MAE 0.2742 / 0.4230 / 0.5922 kW | 0 s
  epoch   2 | train loss 1.95995 | val MAE 0.2669 / 0.4223 / 0.5881 kW | 0 s
  epoch   3 | train loss 1.95046 | val MAE 0.2716 / 0.4168 / 0.5892 kW | 1 s
  epoch   4 | train loss 1.94563 | val MAE 0.2796 / 0.4157 / 0.5866 kW | 1 s
  epoch   5 | train loss 1.94476 | val MAE 0.2698 / 0.4170 / 0.5922 kW | 1 s
[15:09:32] nlinear_s2: params 915 | train 1 s | best epoch 2/5 | val MAE 0.2669 / 0.4223 / 0.5881 kW | skill +0.069 / +0.018 / +0.028
[15:09:32] gru_uni_s0: training on mps
  epoch   1 | train loss 1.94266 | val MAE 0.2752 / 0.4071 / 0.5794 kW | 37 s
  epoch   2 | train loss 1.88743 | val MAE 0.2729 / 0.4053 / 0.5760 kW | 73 s
  epoch   3 | train loss 1.86571 | val MAE 0.2757 / 0.4081 / 0.5772 kW | 109 s
  epoch   4 | train loss 1.85011 | val MAE 0.2714 / 0.4067 / 0.5757 kW | 145 s
  epoch   5 | train loss 1.84111 | val MAE 0.2702 / 0.4020 / 0.5731 kW | 182 s
  epoch   6 | train loss 1.83258 | val MAE 0.2694 / 0.4038 / 0.5762 kW | 218 s
  epoch   7 | train loss 1.82387 | val MAE 0.2702 / 0.4060 / 0.5765 kW | 254 s
  epoch   8 | train loss 1.81748 | val MAE 0.2718 / 0.4076 / 0.5780 kW | 290 s
[15:14:40] gru_uni_s0: params 38,019 | train 290 s | best epoch 5/8 | val MAE 0.2702 / 0.4020 / 0.5731 kW | skill +0.057 / +0.065 / +0.052
[15:14:40] gru_uni_s1: training on mps
  epoch   1 | train loss 1.94395 | val MAE 0.2748 / 0.4055 / 0.5759 kW | 36 s
  epoch   2 | train loss 1.88712 | val MAE 0.2768 / 0.4072 / 0.5785 kW | 72 s
  epoch   3 | train loss 1.86220 | val MAE 0.2704 / 0.4018 / 0.5708 kW | 108 s
  epoch   4 | train loss 1.85238 | val MAE 0.2745 / 0.4033 / 0.5721 kW | 145 s
  epoch   5 | train loss 1.84186 | val MAE 0.2685 / 0.4026 / 0.5730 kW | 181 s
  epoch   6 | train loss 1.83443 | val MAE 0.2718 / 0.4052 / 0.5746 kW | 217 s
[15:18:31] gru_uni_s1: params 38,019 | train 217 s | best epoch 3/6 | val MAE 0.2704 / 0.4018 / 0.5708 kW | skill +0.057 / +0.065 / +0.056
[15:18:31] gru_uni_s2: training on mps
  epoch   1 | train loss 1.94495 | val MAE 0.2772 / 0.4077 / 0.5770 kW | 36 s
  epoch   2 | train loss 1.88506 | val MAE 0.2734 / 0.4083 / 0.5762 kW | 72 s
  epoch   3 | train loss 1.86180 | val MAE 0.2726 / 0.4030 / 0.5724 kW | 109 s
  epoch   4 | train loss 1.84776 | val MAE 0.2695 / 0.4010 / 0.5727 kW | 145 s
  epoch   5 | train loss 1.83796 | val MAE 0.2698 / 0.4029 / 0.5734 kW | 181 s
  epoch   6 | train loss 1.83233 | val MAE 0.2720 / 0.4065 / 0.5791 kW | 218 s
  epoch   7 | train loss 1.82626 | val MAE 0.2706 / 0.4073 / 0.5748 kW | 256 s
[15:23:22] gru_uni_s2: params 38,019 | train 256 s | best epoch 4/7 | val MAE 0.2695 / 0.4010 / 0.5727 kW | skill +0.060 / +0.067 / +0.053

all runs handled in 28.0 min | finished: 15 of 15
```

## Cell 4: per-run table and mean +- sd over seeds (validation)
_execution count: 4_

```
shape: (15, 15)
┌─────┬──────────┬─────────┬──────────┬────────┬───────────┬──────────┬────────────┬─────────┬────────────┬──────────┬─────────┬────────────┬─────────┬────────┐
│ h   ┆ model    ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 ┆ DM_p_max   ┆ n_params ┆ train_s ┆ best_epoch ┆ cap_hit ┆ device │
│ --- ┆ ---      ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     ┆ ---        ┆ ---      ┆ ---     ┆ ---        ┆ ---     ┆ ---    │
│ i64 ┆ str      ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     ┆ f64        ┆ i64      ┆ f64     ┆ f64        ┆ bool    ┆ str    │
╞═════╪══════════╪═════════╪══════════╪════════╪═══════════╪══════════╪════════════╪═════════╪════════════╪══════════╪═════════╪════════════╪═════════╪════════╡
│ 1   ┆ patchtst ┆ 3       ┆ 0.2543   ┆ 0.0015 ┆ 0.112     ┆ 1.324    ┆ 0.2056     ┆ 0.175   ┆ 2.7085e-37 ┆ 75215    ┆ 203.0   ┆ 12.3       ┆ false   ┆ mps    │
│ 1   ┆ nhits    ┆ 3       ┆ 0.2608   ┆ 0.0024 ┆ 0.09      ┆ 1.33     ┆ 0.2121     ┆ 0.183   ┆ 7.8259e-15 ┆ 676301   ┆ 20.0    ┆ 1.3        ┆ false   ┆ cpu    │
│ 1   ┆ nlinear  ┆ 3       ┆ 0.2666   ┆ 0.0007 ┆ 0.07      ┆ 1.416    ┆ 0.2141     ┆ 0.134   ┆ 1.0172e-13 ┆ 915      ┆ 1.0     ┆ 3.3        ┆ false   ┆ cpu    │
│ 1   ┆ dlinear  ┆ 3       ┆ 0.268    ┆ 0.0018 ┆ 0.065     ┆ 1.405    ┆ 0.2161     ┆ 0.148   ┆ 1.0782e-8  ┆ 1830     ┆ 1.0     ┆ 2.0        ┆ false   ┆ cpu    │
│ 1   ┆ gru_uni  ┆ 3       ┆ 0.27     ┆ 0.0005 ┆ 0.058     ┆ 1.396    ┆ 0.2189     ┆ 0.189   ┆ 8.5318e-15 ┆ 38019    ┆ 255.0   ┆ 4.0        ┆ false   ┆ mps    │
│ 5   ┆ patchtst ┆ 3       ┆ 0.3989   ┆ 0.0018 ┆ 0.072     ┆ 2.283    ┆ 0.2848     ┆ 0.229   ┆ 3.3899e-24 ┆ 75215    ┆ 203.0   ┆ 12.3       ┆ false   ┆ mps    │
│ 5   ┆ nhits    ┆ 3       ┆ 0.4      ┆ 0.001  ┆ 0.07      ┆ 2.198    ┆ 0.2911     ┆ 0.278   ┆ 7.6927e-15 ┆ 676301   ┆ 20.0    ┆ 1.3        ┆ false   ┆ cpu    │
│ 5   ┆ gru_uni  ┆ 3       ┆ 0.4016   ┆ 0.0005 ┆ 0.066     ┆ 2.228    ┆ 0.2911     ┆ 0.249   ┆ 1.7851e-8  ┆ 38019    ┆ 255.0   ┆ 4.0        ┆ false   ┆ mps    │
│ 5   ┆ dlinear  ┆ 3       ┆ 0.4168   ┆ 0.0034 ┆ 0.03      ┆ 2.389    ┆ 0.2972     ┆ 0.182   ┆ 0.009624   ┆ 1830     ┆ 1.0     ┆ 2.0        ┆ false   ┆ cpu    │
│ 5   ┆ nlinear  ┆ 3       ┆ 0.4176   ┆ 0.0041 ┆ 0.029     ┆ 2.38     ┆ 0.2986     ┆ 0.196   ┆ 0.030413   ┆ 915      ┆ 1.0     ┆ 3.3        ┆ false   ┆ cpu    │
│ 15  ┆ nhits    ┆ 3       ┆ 0.5684   ┆ 0.0016 ┆ 0.06      ┆ 2.817    ┆ 0.4123     ┆ 0.229   ┆ 4.6601e-10 ┆ 676301   ┆ 20.0    ┆ 1.3        ┆ false   ┆ cpu    │
│ 15  ┆ patchtst ┆ 3       ┆ 0.5693   ┆ 0.0007 ┆ 0.059     ┆ 2.89     ┆ 0.4085     ┆ 0.184   ┆ 3.2601e-14 ┆ 75215    ┆ 203.0   ┆ 12.3       ┆ false   ┆ mps    │
│ 15  ┆ gru_uni  ┆ 3       ┆ 0.5722   ┆ 0.0013 ┆ 0.054     ┆ 2.845    ┆ 0.4147     ┆ 0.215   ┆ 0.000033   ┆ 38019    ┆ 255.0   ┆ 4.0        ┆ false   ┆ mps    │
│ 15  ┆ dlinear  ┆ 3       ┆ 0.5865   ┆ 0.0007 ┆ 0.03      ┆ 3.067    ┆ 0.4148     ┆ 0.088   ┆ 6.8623e-11 ┆ 1830     ┆ 1.0     ┆ 2.0        ┆ false   ┆ cpu    │
│ 15  ┆ nlinear  ┆ 3       ┆ 0.5867   ┆ 0.0012 ┆ 0.03      ┆ 3.061    ┆ 0.4155     ┆ 0.093   ┆ 5.6715e-9  ┆ 915      ┆ 1.0     ┆ 3.3        ┆ false   ┆ cpu    │
└─────┴──────────┴─────────┴──────────┴────────┴───────────┴──────────┴────────────┴─────────┴────────────┴──────────┴─────────┴────────────┴─────────┴────────┘

runs with best epoch at the cap: []
```

## Cell 5: merge with the Week 3 benchmark (validation) and print the ranking per horizon
_execution count: 5_

```

=== h = 1 min (validation; sorted by MAE) ===
shape: (18, 10)
┌─────────────┬───────────────────┬────────────┬─────────┬──────────┬────────┬───────────┬──────────┬────────────┬─────────┐
│ family      ┆ model             ┆ covariates ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 │
│ ---         ┆ ---               ┆ ---        ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     │
│ str         ┆ str               ┆ bool       ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     │
╞═════════════╪═══════════════════╪════════════╪═════════╪══════════╪════════╪═══════════╪══════════╪════════════╪═════════╡
│ ML          ┆ lightgbm_cov      ┆ true       ┆ 3       ┆ 0.2464   ┆ 0.0003 ┆ 0.14      ┆ 1.276    ┆ 0.1998     ┆ 0.266   │
│ ML          ┆ lightgbm          ┆ false      ┆ 3       ┆ 0.2491   ┆ 0.0001 ┆ 0.131     ┆ 1.341    ┆ 0.1993     ┆ 0.191   │
│ deep        ┆ patchtst          ┆ false      ┆ 3       ┆ 0.2543   ┆ 0.0015 ┆ 0.112     ┆ 1.324    ┆ 0.2056     ┆ 0.175   │
│ deep        ┆ gru               ┆ false      ┆ 3       ┆ 0.2583   ┆ 0.0034 ┆ 0.099     ┆ 1.313    ┆ 0.2103     ┆ 0.199   │
│ deep        ┆ nhits             ┆ false      ┆ 3       ┆ 0.2608   ┆ 0.0024 ┆ 0.09      ┆ 1.33     ┆ 0.2121     ┆ 0.183   │
│ deep        ┆ lstm              ┆ false      ┆ 3       ┆ 0.2631   ┆ 0.0056 ┆ 0.082     ┆ 1.323    ┆ 0.2149     ┆ 0.233   │
│ linear      ┆ ar_lags_cal_cov   ┆ true       ┆ 1       ┆ 0.2637   ┆ 0.0    ┆ 0.08      ┆ 1.333    ┆ 0.2151     ┆ 0.18    │
│ deep        ┆ nlinear           ┆ false      ┆ 3       ┆ 0.2666   ┆ 0.0007 ┆ 0.07      ┆ 1.416    ┆ 0.2141     ┆ 0.134   │
│ linear      ┆ ar_lags           ┆ false      ┆ 1       ┆ 0.2673   ┆ 0.0    ┆ 0.067     ┆ 1.407    ┆ 0.2152     ┆ 0.145   │
│ linear      ┆ ar_lags_cal       ┆ false      ┆ 1       ┆ 0.2677   ┆ 0.0    ┆ 0.066     ┆ 1.401    ┆ 0.2159     ┆ 0.143   │
│ deep        ┆ dlinear           ┆ false      ┆ 3       ┆ 0.268    ┆ 0.0018 ┆ 0.065     ┆ 1.405    ┆ 0.2161     ┆ 0.148   │
│ deep        ┆ gru_uni           ┆ false      ┆ 3       ┆ 0.27     ┆ 0.0005 ┆ 0.058     ┆ 1.396    ┆ 0.2189     ┆ 0.189   │
│ naive       ┆ pers_mod5_rolling ┆ false      ┆ 1       ┆ 0.2707   ┆ 0.0    ┆ 0.056     ┆ 1.445    ┆ 0.2157     ┆ 0.0     │
│ statistical ┆ auto_ets          ┆ false      ┆ 1       ┆ 0.2778   ┆ 0.0    ┆ 0.031     ┆ 1.558    ┆ 0.2194     ┆ 0.068   │
│ statistical ┆ auto_arima        ┆ false      ┆ 1       ┆ 0.2838   ┆ 0.0    ┆ 0.01      ┆ 1.465    ┆ 0.2299     ┆ 0.091   │
│ naive       ┆ pers_mod5_static  ┆ false      ┆ 1       ┆ 0.2839   ┆ 0.0    ┆ 0.009     ┆ 1.557    ┆ 0.2257     ┆ 0.0     │
│ naive       ┆ persistence       ┆ false      ┆ 1       ┆ 0.2866   ┆ 0.0    ┆ 0.0       ┆ 1.607    ┆ 0.2264     ┆ 0.0     │
│ naive       ┆ seasonal_naive_1d ┆ false      ┆ 1       ┆ 4.8896   ┆ 0.0    ┆ -16.062   ┆ 4.973    ┆ 4.9642     ┆ 0.087   │
└─────────────┴───────────────────┴────────────┴─────────┴──────────┴────────┴───────────┴──────────┴────────────┴─────────┘

=== h = 5 min (validation; sorted by MAE) ===
shape: (18, 10)
┌─────────────┬───────────────────┬────────────┬─────────┬──────────┬────────┬───────────┬──────────┬────────────┬─────────┐
│ family      ┆ model             ┆ covariates ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 │
│ ---         ┆ ---               ┆ ---        ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     │
│ str         ┆ str               ┆ bool       ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     │
╞═════════════╪═══════════════════╪════════════╪═════════╪══════════╪════════╪═══════════╪══════════╪════════════╪═════════╡
│ ML          ┆ lightgbm_cov      ┆ true       ┆ 3       ┆ 0.385    ┆ 0.0005 ┆ 0.104     ┆ 2.19     ┆ 0.2761     ┆ 0.299   │
│ ML          ┆ lightgbm          ┆ false      ┆ 3       ┆ 0.3934   ┆ 0.0001 ┆ 0.085     ┆ 2.276    ┆ 0.2793     ┆ 0.249   │
│ deep        ┆ patchtst          ┆ false      ┆ 3       ┆ 0.3989   ┆ 0.0018 ┆ 0.072     ┆ 2.283    ┆ 0.2848     ┆ 0.229   │
│ deep        ┆ lstm              ┆ false      ┆ 3       ┆ 0.3998   ┆ 0.001  ┆ 0.07      ┆ 2.182    ┆ 0.292      ┆ 0.27    │
│ deep        ┆ nhits             ┆ false      ┆ 3       ┆ 0.4      ┆ 0.001  ┆ 0.07      ┆ 2.198    ┆ 0.2911     ┆ 0.278   │
│ deep        ┆ gru               ┆ false      ┆ 3       ┆ 0.4002   ┆ 0.0025 ┆ 0.069     ┆ 2.197    ┆ 0.2915     ┆ 0.246   │
│ deep        ┆ gru_uni           ┆ false      ┆ 3       ┆ 0.4016   ┆ 0.0005 ┆ 0.066     ┆ 2.228    ┆ 0.2911     ┆ 0.249   │
│ linear      ┆ ar_lags_cal_cov   ┆ true       ┆ 1       ┆ 0.409    ┆ 0.0    ┆ 0.049     ┆ 2.249    ┆ 0.298      ┆ 0.255   │
│ linear      ┆ ar_lags           ┆ false      ┆ 1       ┆ 0.4161   ┆ 0.0    ┆ 0.032     ┆ 2.377    ┆ 0.2971     ┆ 0.194   │
│ linear      ┆ ar_lags_cal       ┆ false      ┆ 1       ┆ 0.4163   ┆ 0.0    ┆ 0.032     ┆ 2.374    ┆ 0.2975     ┆ 0.193   │
│ deep        ┆ dlinear           ┆ false      ┆ 3       ┆ 0.4168   ┆ 0.0034 ┆ 0.03      ┆ 2.389    ┆ 0.2972     ┆ 0.182   │
│ deep        ┆ nlinear           ┆ false      ┆ 3       ┆ 0.4176   ┆ 0.0041 ┆ 0.029     ┆ 2.38     ┆ 0.2986     ┆ 0.196   │
│ statistical ┆ auto_ets          ┆ false      ┆ 1       ┆ 0.4178   ┆ 0.0    ┆ 0.028     ┆ 2.555    ┆ 0.2883     ┆ 0.033   │
│ statistical ┆ auto_arima        ┆ false      ┆ 1       ┆ 0.418    ┆ 0.0    ┆ 0.028     ┆ 2.382    ┆ 0.2989     ┆ 0.185   │
│ naive       ┆ pers_mod5_static  ┆ false      ┆ 1       ┆ 0.4299   ┆ 0.0    ┆ 0.0       ┆ 2.631    ┆ 0.2967     ┆ 0.0     │
│ naive       ┆ persistence       ┆ false      ┆ 1       ┆ 0.4299   ┆ 0.0    ┆ 0.0       ┆ 2.631    ┆ 0.2967     ┆ 0.0     │
│ naive       ┆ pers_mod5_rolling ┆ false      ┆ 1       ┆ 0.4302   ┆ 0.0    ┆ -0.001    ┆ 2.632    ┆ 0.2969     ┆ 0.0     │
│ naive       ┆ seasonal_naive_1d ┆ false      ┆ 1       ┆ 4.8958   ┆ 0.0    ┆ -10.388   ┆ 5.032    ┆ 4.9652     ┆ 0.11    │
└─────────────┴───────────────────┴────────────┴─────────┴──────────┴────────┴───────────┴──────────┴────────────┴─────────┘

=== h = 15 min (validation; sorted by MAE) ===
shape: (18, 10)
┌─────────────┬───────────────────┬────────────┬─────────┬──────────┬────────┬───────────┬──────────┬────────────┬─────────┐
│ family      ┆ model             ┆ covariates ┆ n_seeds ┆ MAE_mean ┆ MAE_sd ┆ skill_MAE ┆ MAE_ramp ┆ MAE_normal ┆ ramp_F1 │
│ ---         ┆ ---               ┆ ---        ┆ ---     ┆ ---      ┆ ---    ┆ ---       ┆ ---      ┆ ---        ┆ ---     │
│ str         ┆ str               ┆ bool       ┆ u32     ┆ f64      ┆ f64    ┆ f64       ┆ f64      ┆ f64        ┆ f64     │
╞═════════════╪═══════════════════╪════════════╪═════════╪══════════╪════════╪═══════════╪══════════╪════════════╪═════════╡
│ ML          ┆ lightgbm_cov      ┆ true       ┆ 3       ┆ 0.5559   ┆ 0.0008 ┆ 0.081     ┆ 2.805    ┆ 0.3995     ┆ 0.234   │
│ ML          ┆ lightgbm          ┆ false      ┆ 3       ┆ 0.5659   ┆ 0.0004 ┆ 0.065     ┆ 2.91     ┆ 0.4027     ┆ 0.198   │
│ deep        ┆ nhits             ┆ false      ┆ 3       ┆ 0.5684   ┆ 0.0016 ┆ 0.06      ┆ 2.817    ┆ 0.4123     ┆ 0.229   │
│ deep        ┆ patchtst          ┆ false      ┆ 3       ┆ 0.5693   ┆ 0.0007 ┆ 0.059     ┆ 2.89     ┆ 0.4085     ┆ 0.184   │
│ deep        ┆ gru               ┆ false      ┆ 3       ┆ 0.5704   ┆ 0.0036 ┆ 0.057     ┆ 2.811    ┆ 0.4152     ┆ 0.221   │
│ deep        ┆ gru_uni           ┆ false      ┆ 3       ┆ 0.5722   ┆ 0.0013 ┆ 0.054     ┆ 2.845    ┆ 0.4147     ┆ 0.215   │
│ deep        ┆ lstm              ┆ false      ┆ 3       ┆ 0.5724   ┆ 0.0043 ┆ 0.054     ┆ 2.816    ┆ 0.417      ┆ 0.223   │
│ linear      ┆ ar_lags_cal_cov   ┆ true       ┆ 1       ┆ 0.5786   ┆ 0.0    ┆ 0.043     ┆ 2.891    ┆ 0.419      ┆ 0.175   │
│ deep        ┆ dlinear           ┆ false      ┆ 3       ┆ 0.5865   ┆ 0.0007 ┆ 0.03      ┆ 3.067    ┆ 0.4148     ┆ 0.088   │
│ deep        ┆ nlinear           ┆ false      ┆ 3       ┆ 0.5867   ┆ 0.0012 ┆ 0.03      ┆ 3.061    ┆ 0.4155     ┆ 0.093   │
│ linear      ┆ ar_lags_cal       ┆ false      ┆ 1       ┆ 0.5869   ┆ 0.0    ┆ 0.03      ┆ 3.023    ┆ 0.4185     ┆ 0.122   │
│ linear      ┆ ar_lags           ┆ false      ┆ 1       ┆ 0.5876   ┆ 0.0    ┆ 0.028     ┆ 3.031    ┆ 0.4186     ┆ 0.119   │
│ statistical ┆ auto_arima        ┆ false      ┆ 1       ┆ 0.5885   ┆ 0.0    ┆ 0.027     ┆ 3.057    ┆ 0.4177     ┆ 0.104   │
│ statistical ┆ auto_ets          ┆ false      ┆ 1       ┆ 0.593    ┆ 0.0    ┆ 0.02      ┆ 3.211    ┆ 0.4119     ┆ 0.019   │
│ naive       ┆ pers_mod5_static  ┆ false      ┆ 1       ┆ 0.6047   ┆ 0.0    ┆ 0.0       ┆ 3.279    ┆ 0.4197     ┆ 0.0     │
│ naive       ┆ persistence       ┆ false      ┆ 1       ┆ 0.6048   ┆ 0.0    ┆ 0.0       ┆ 3.279    ┆ 0.4199     ┆ 0.0     │
│ naive       ┆ pers_mod5_rolling ┆ false      ┆ 1       ┆ 0.6059   ┆ 0.0    ┆ -0.002    ┆ 3.28     ┆ 0.4209     ┆ 0.0     │
│ naive       ┆ seasonal_naive_1d ┆ false      ┆ 1       ┆ 4.9088   ┆ 0.0    ┆ -7.116    ┆ 5.136    ┆ 4.9676     ┆ 0.123   │
└─────────────┴───────────────────┴────────────┴─────────┴──────────┴────────┴───────────┴──────────┴────────────┴─────────┘

written: /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/nf_val_runs.csv | /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/nf_val_summary.csv | /Users/aryaneuse/Desktop/Uni Project/datacenter-power-forecasting/reports/week4/benchmark_val_week4_nf.csv
note: DM_p for the new models = largest p-value over the three seeds (conservative), vs persistence
```

## Cell 6: export outputs (save the notebook first with Cmd+S, then run this cell)
_execution count: 6_

```
written 11_week4_nf_models_outputs.md | code cells 6 | figures 0
```

"""tabular_features.py: per-origin feature matrices for direct tabular models (linear AR, LightGBM) (Week 3).

All features use minutes <= t only (t = forecast origin). Usable origins (ok_h{h} in the modelling dataset) have
all 60 input minutes valid, so the lag and covariate-change features are always defined there; L24 can be
undefined (NaN) after maintenance windows and is only used by the "gbm" kinds (LightGBM handles NaN).

Kinds:
  lags     P(t - k) - P(t), k = 1..59                                   (59)
  cal      lags + minute-mod-5 one-hot (classes 1..4) + hour sin/cos    (65)   = notebook 07 ar_lags_cal
  cal_cov  cal + d1_/d5_ covariate changes + allocation gap            (90)   = notebook 07 ar_lags_cal_cov
  gbm      cal + level features: P(t) / L24(t) - 1 and L24(t)          (67)   univariate arm
  gbm_cov  gbm + covariates of cal_cov                                 (92)   covariate arm
"""
import numpy as np
import polars as pl

INPUT_LEN = 60
COVARIATES = ["util_gpu_mean", "util_mem_mean", "temp_gpu_mean", "temp_mem_mean", "mem_used_sum", "mem_free_sum",
              "n_gpu_equiv", "gpus_alloc", "active_gpu_jobs", "active_jobs", "queued_gpu_jobs", "queued_jobs"]


def arrays(md: pl.DataFrame) -> dict:
    """Numpy arrays needed by `features`, from the modelling dataset."""
    A = {"P": md["P_kW"].to_numpy().astype(float), "MOD5": md["mod5"].to_numpy().astype(int),
         "HSIN": md["hour_sin"].to_numpy(), "HCOS": md["hour_cos"].to_numpy(),
         "L24": md["L24_kW"].to_numpy().astype(float),
         "ALLOC_GAP": md["gpus_alloc"].to_numpy().astype(float) - md["n_gpu_equiv"].to_numpy().astype(float)}
    A["DCOV"] = {f"d{k}_{c}": md[f"d{k}_{c}"].to_numpy().astype(float) for c in COVARIATES for k in (1, 5)}
    return A


def features(A: dict, t: np.ndarray, kind: str):
    """(X, names) for origins t."""
    if kind not in ("lags", "cal", "cal_cov", "gbm", "gbm_cov"):
        raise ValueError(kind)
    P = A["P"]
    cols = [np.stack([P[t - k] - P[t] for k in range(1, INPUT_LEN)], axis=1)]
    names = [f"lag{k}" for k in range(1, INPUT_LEN)]
    if kind != "lags":
        cols.append(np.stack([(A["MOD5"][t] == m).astype(float) for m in range(1, 5)] + [A["HSIN"][t], A["HCOS"][t]], axis=1))
        names += [f"mod5_{m}" for m in range(1, 5)] + ["hour_sin", "hour_cos"]
    n_strict = sum(c.shape[1] for c in cols)
    if kind in ("cal_cov", "gbm_cov"):
        cols.append(np.stack([A["DCOV"][c][t] for c in A["DCOV"]] + [A["ALLOC_GAP"][t]], axis=1))
        names += list(A["DCOV"]) + ["alloc_gap"]
        n_strict += len(A["DCOV"]) + 1
    if kind in ("gbm", "gbm_cov"):
        L = A["L24"][t]
        cols.append(np.stack([P[t] / L - 1.0, L], axis=1))
        names += ["level_rel_L24", "L24"]
    X = np.concatenate(cols, axis=1)
    strict = [i for i, nm in enumerate(names) if nm not in ("level_rel_L24", "L24")]
    if not np.isfinite(X[:, strict]).all():
        raise ValueError(f"non-finite features ({kind})")
    return X, names

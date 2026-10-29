"""metrics.py: shared evaluation module for all model families (Week 3).

Every model is scored with the same functions on the same usable samples (ok_h{h} and split_h{h} from
data/processed/model_dataset.parquet). Inputs are plain numpy arrays in kW, one entry per forecast origin t:
  y      : true power P(t + h)
  yhat   : forecast of P(t + h)
  p0     : last observed power P(t) (needed for skill, change-based metrics and ramp detection)
  l24    : causal 24-h mean L24(t) (ramp detection; NaN where undefined)
  ramp   : ramp label at the origin (1 ramp, 0 normal, NaN undefined), from ramp_h{h} of the dataset
  thr    : relative ramp threshold of that horizon (meta["thresholds"][h]["rel_p95"])

Definitions:
  MAE, RMSE                in kW
  sMAPE                    mean of 200 |e| / (|y| + |yhat|), in percent (pairs with y = yhat = 0 count as 0)
  R2                       on the level P(t + h)
  R2_delta                 on the change P(t + h) - P(t) (1 - SSE / SS of the true change about its mean);
                           persistence has R2_delta <= 0, so this shows how much of the change is explained
  skill_MAE                1 - MAE(model) / MAE(persistence) on the same samples (0 = persistence, 1 = perfect)
  MAE_ramp, MAE_normal     MAE on samples labelled ramp / normal (label NaN excluded from both)
  ramp precision / recall / F1
                           a forecast "predicts a ramp" when |yhat - P(t)| / L24(t) > thr; scored against the
                           label; persistence never predicts a ramp (recall 0); F1 = 0 when undefined
  Diebold-Mariano          absolute-error loss differential d = |e_model| - |e_ref|, Newey-West (Bartlett)
                           long-run variance with `lag` lags, two-sided normal p-value. Samples are treated as one
                           sequence although gaps exist (approximation; noted in the report).
"""
import math

import numpy as np


def _f(x):
    return np.asarray(x, dtype=np.float64)


def mae(y, yhat):
    return float(np.mean(np.abs(_f(yhat) - _f(y))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((_f(yhat) - _f(y)) ** 2)))


def smape(y, yhat):
    y, yhat = _f(y), _f(yhat)
    den = np.abs(y) + np.abs(yhat)
    r = np.where(den > 0, 2.0 * np.abs(yhat - y) / np.where(den > 0, den, 1.0), 0.0)
    return float(100.0 * r.mean())


def r2(y, yhat):
    y, yhat = _f(y), _f(yhat)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else float("nan")


def r2_delta(y, yhat, p0):
    return r2(_f(y) - _f(p0), _f(yhat) - _f(p0))


def skill(mae_model, mae_ref):
    return float(1.0 - mae_model / mae_ref) if mae_ref > 0 else float("nan")


def ramp_scores(y, yhat, p0, l24, ramp, thr):
    """MAE on ramp / normal samples and ramp-detection precision, recall, F1."""
    y, yhat, p0, l24, ramp = _f(y), _f(yhat), _f(p0), _f(l24), _f(ramp)
    lab = np.isfinite(ramp) & np.isfinite(l24) & (l24 > 0)
    r, nr = lab & (ramp == 1), lab & (ramp == 0)
    e = np.abs(yhat - y)
    pred = np.zeros(len(y), dtype=bool)
    pred[lab] = np.abs(yhat[lab] - p0[lab]) / l24[lab] > thr
    tp = int((pred & r).sum())
    fp = int((pred & nr).sum())
    fn = int((~pred & r).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n_ramp": int(r.sum()), "n_normal": int(nr.sum()),
            "MAE_ramp": float(e[r].mean()) if r.any() else float("nan"),
            "MAE_normal": float(e[nr].mean()) if nr.any() else float("nan"),
            "ramp_precision": float(prec), "ramp_recall": float(rec), "ramp_F1": float(f1)}


def dm_test(y, yhat_model, yhat_ref, lag=60):
    """Diebold-Mariano test on absolute errors. Negative stat = model better than ref."""
    d = np.abs(_f(yhat_model) - _f(y)) - np.abs(_f(yhat_ref) - _f(y))
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        return {"DM_stat": float("nan"), "DM_p": float("nan"), "mean_loss_diff": float(d.mean())}
    stat = d.mean() / math.sqrt(lrv / n)
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return {"DM_stat": float(stat), "DM_p": float(p), "mean_loss_diff": float(d.mean())}


def evaluate(y, yhat, p0, l24=None, ramp=None, thr=None, yhat_ref=None, mod5_target=None, dm_lag=60):
    """All metrics for one model, one horizon, one split. yhat_ref = persistence forecasts on the same samples."""
    y, yhat, p0 = _f(y), _f(yhat), _f(p0)
    if not (len(y) == len(yhat) == len(p0)) or len(y) == 0:
        raise ValueError("y, yhat and p0 must be non-empty and of equal length")
    if not (np.isfinite(y).all() and np.isfinite(yhat).all() and np.isfinite(p0).all()):
        raise ValueError("non-finite values in y, yhat or p0")
    out = {"n": int(len(y)), "MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "sMAPE": smape(y, yhat),
           "R2": r2(y, yhat), "R2_delta": r2_delta(y, yhat, p0)}
    ref = p0 if yhat_ref is None else _f(yhat_ref)
    out["MAE_persistence"] = mae(y, ref)
    out["skill_MAE"] = skill(out["MAE"], out["MAE_persistence"])
    out["skill_RMSE"] = skill(out["RMSE"], rmse(y, ref))
    if l24 is not None and ramp is not None and thr is not None:
        out.update(ramp_scores(y, yhat, p0, l24, ramp, thr))
    if mod5_target is not None:
        m = _f(mod5_target) != 0
        out["MAE_no_mod0_target"] = mae(y[m], yhat[m]) if m.any() else float("nan")
    if not np.allclose(yhat, ref):
        out.update(dm_test(y, yhat, ref, lag=dm_lag))
    else:
        out.update({"DM_stat": float("nan"), "DM_p": float("nan"), "mean_loss_diff": 0.0})
    return out

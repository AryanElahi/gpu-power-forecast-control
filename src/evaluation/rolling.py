"""rolling.py: rolling-origin evaluation (Week 3).

Two parts:
  per_block(...)        metrics of a fixed model per calendar block (day or week): every minute is a forecast origin,
                        so a fixed model is already evaluated at all origins; the block table shows how stable the
                        skill is over time (used on validation now, on test at the end of Sprint 2).
  rolling_refit(...)    rolling-origin evaluation with periodic re-fitting on an expanding window: the evaluation
                        span is cut into blocks; before each block the model is re-fitted on every sample whose
                        target is observed before the block starts (t + h < block start), then the block is
                        predicted. Model-agnostic through two callbacks.
"""
import numpy as np
import polars as pl

from . import metrics as M


def block_ids(minute_epoch, block="week"):
    """Integer block id per sample: days since epoch ("day") or weeks starting Monday ("week")."""
    d = np.asarray(minute_epoch) // 86400
    if block == "day":
        return d
    if block == "week":
        return (d + 3) // 7          # 1970-01-01 was a Thursday -> +3 makes weeks start on Monday
    raise ValueError(block)


def per_block(minute_epoch, y, yhat, p0, block="week", min_n=500):
    """Polars table: block start (UTC-naive file clock), n, MAE, MAE of persistence, skill."""
    b = block_ids(minute_epoch, block)
    rows = []
    for k in np.unique(b):
        m = b == k
        if m.sum() < min_n:
            continue
        mae, mp = M.mae(y[m], yhat[m]), M.mae(y[m], p0[m])
        start = int(k * 7 - 3 if block == "week" else k) * 86400
        rows.append({"block_start_epoch": start, "n": int(m.sum()), "MAE": mae, "MAE_persistence": mp,
                     "skill_MAE": M.skill(mae, mp)})
    return (pl.DataFrame(rows).with_columns(pl.from_epoch("block_start_epoch", time_unit="s").alias("block_start"))
              .drop("block_start_epoch").select("block_start", "n", "MAE", "MAE_persistence", "skill_MAE"))


def rolling_refit(origins, h, block_edges, fit_fn, predict_fn, first_fit_origin=0):
    """Rolling-origin evaluation with re-fitting before each block.

    origins     : sorted array of usable origin indices (all splits, already filtered by the caller)
    block_edges : sorted minute indices [e0, e1, ..., eK]; block k covers origins in [e_k, e_{k+1})
    fit_fn(train_origins) -> model ; predict_fn(model, block_origins) -> forecasts of P(t + h)
    Returns (block_origins_concatenated, forecasts, per-block info list).
    """
    origins = np.asarray(origins)
    outs, preds, info = [], [], []
    for k in range(len(block_edges) - 1):
        lo, hi = block_edges[k], block_edges[k + 1]
        blk = origins[(origins >= lo) & (origins < hi)]
        if len(blk) == 0:
            continue
        tr = origins[(origins >= first_fit_origin) & (origins + h < lo)]
        model = fit_fn(tr)
        outs.append(blk)
        preds.append(np.asarray(predict_fn(model, blk), dtype=float))
        info.append({"block": k, "start_index": int(lo), "end_index": int(hi), "n_fit": int(len(tr)),
                     "n_pred": int(len(blk))})
    return np.concatenate(outs), np.concatenate(preds), info

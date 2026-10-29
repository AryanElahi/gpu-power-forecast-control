"""compare.py: cross-model analysis on validation (Week 4): one prediction table, ramp errors, Diebold-Mariano with
Holm correction, a ramp-event window, weekly skill with and without five-minute-dip targets.

Prediction sources (all validation only; columns minute_epoch, h, y, p0, yhat):
  reports/baselines/val_predictions.parquet          notebook 07 (column `model`)
  reports/classical/val_predictions.parquet          notebook 08 (column `model`; `seed` if present -> seed 0 kept)
  experiments/week3_rnn/<cell>_local_s<seed>/        notebook 09 (LSTM, GRU)
  experiments/week4_nf/<model>_s<seed>/              notebook 11 (N-HiTS, PatchTST, DLinear, NLinear, gru_uni)
Multi-seed models are represented by seed 0 in this module (DM tests, figures, weekly tables); the benchmark table
keeps the mean +- sd over seeds.
"""
import math
from pathlib import Path

import numpy as np
import polars as pl

from . import metrics as M

COLS = ["model", "h", "minute_epoch", "y", "p0", "yhat"]
RNN_W3 = {"lstm": "lstm_local_s{s}", "gru": "gru_local_s{s}"}
NF_W4 = ["nhits", "patchtst", "dlinear", "nlinear", "gru_uni"]


def _std(df, model=None):
    if model is not None:
        df = df.with_columns(pl.lit(model).alias("model"))
    return df.select(pl.col("model").cast(pl.Utf8), pl.col("h").cast(pl.Int64), pl.col("minute_epoch").cast(pl.Int64),
                     pl.col("y").cast(pl.Float64), pl.col("p0").cast(pl.Float64), pl.col("yhat").cast(pl.Float64))


def load_predictions(root, seed=0, log=print):
    """All validation predictions in one long table (model, h, minute_epoch, y, p0, yhat)."""
    root = Path(root)
    parts, seen = [], set()
    for src in ("baselines", "classical"):
        p = root / "reports" / src / "val_predictions.parquet"
        if not p.exists():
            log(f"missing {p}")
            continue
        d = pl.read_parquet(p)
        if "seed" in d.columns:
            d = d.filter((pl.col("seed") == seed) | pl.col("seed").is_null())
        d = d.filter(~pl.col("model").is_in(sorted(seen)))
        seen |= set(d["model"].unique().to_list())
        parts.append(_std(d))
    for m, pat in RNN_W3.items():
        p = root / "experiments" / "week3_rnn" / pat.format(s=seed) / "predictions_val.parquet"
        if p.exists():
            parts.append(_std(pl.read_parquet(p), m))
        else:
            log(f"missing {p}")
    for m in NF_W4:
        p = root / "experiments" / "week4_nf" / f"{m}_s{seed}" / "predictions_val.parquet"
        if p.exists():
            parts.append(_std(pl.read_parquet(p), m))
        else:
            log(f"missing {p}")
    return pl.concat(parts)


def align(pred):
    """Keep, per horizon, only origins that every model has. Returns (aligned table, coverage table)."""
    cov = pred.group_by("h", "model").agg(pl.len().alias("n")).sort("h", "model")
    n_models = pred.group_by("h").agg(pl.col("model").n_unique().alias("k"))
    common = (pred.group_by("h", "minute_epoch").agg(pl.len().alias("c"))
                  .join(n_models, on="h").filter(pl.col("c") == pl.col("k")).select("h", "minute_epoch"))
    out = pred.join(common, on=["h", "minute_epoch"], how="inner").sort("h", "model", "minute_epoch")
    cov = cov.join(common.group_by("h").agg(pl.len().alias("n_common")), on="h")
    return out, cov


def add_labels(pred, md):
    """Join ramp labels, severe-ramp labels, L24 and the target's minute-mod-5 from the modelling dataset."""
    lab = []
    for h in sorted(pred["h"].unique().to_list()):
        lab.append(md.select(pl.col("minute_epoch").cast(pl.Int64), pl.col(f"ramp_h{h}").cast(pl.Float64).alias("ramp"),
                             pl.col(f"ramp99_h{h}").cast(pl.Float64).alias("ramp99"),
                             pl.col("L24_kW").cast(pl.Float64).alias("L24"))
                     .with_columns(pl.lit(h).cast(pl.Int64).alias("h")))
    lab = pl.concat(lab)
    return (pred.join(lab, on=["h", "minute_epoch"], how="left")
                .with_columns((((pl.col("minute_epoch") // 60) + pl.col("h")) % 5).alias("mod5_target")))


def ramp_breakdown(pl_pred, thresholds):
    """Per model and horizon: errors on normal / ramp / severe-ramp steps, up vs down ramps, skill on ramps."""
    rows = []
    for (m, h), g in pl_pred.group_by("model", "h", maintain_order=True):
        y, yh, p0 = g["y"].to_numpy(), g["yhat"].to_numpy(), g["p0"].to_numpy()
        r = g["ramp"].to_numpy()
        r99 = g["ramp99"].to_numpy()
        e, ep = np.abs(yh - y), np.abs(p0 - y)
        up, rm, sv, nr = (y - p0) > 0, r == 1, r99 == 1, r == 0
        sc = M.ramp_scores(y, yh, p0, g["L24"].fill_null(np.nan).to_numpy(), r, thresholds[str(h)]["rel_p95"])
        f = lambda mask: float(e[mask].mean()) if mask.any() else float("nan")  # noqa: E731
        fp = lambda mask: float(ep[mask].mean()) if mask.any() else float("nan")  # noqa: E731
        rows.append({"model": m, "h": h, "n": len(y), "n_ramp": int(rm.sum()), "n_severe": int(sv.sum()),
                     "MAE_normal": f(nr), "MAE_ramp": f(rm), "MAE_ramp_up": f(rm & up), "MAE_ramp_down": f(rm & ~up),
                     "MAE_severe": f(sv), "ramp_to_normal": f(rm) / f(nr) if f(nr) > 0 else float("nan"),
                     "skill_normal": 1 - f(nr) / fp(nr), "skill_ramp": 1 - f(rm) / fp(rm),
                     "skill_severe": 1 - f(sv) / fp(sv), "ramp_recall": sc["ramp_recall"],
                     "ramp_precision": sc["ramp_precision"], "ramp_F1": sc["ramp_F1"]})
    return pl.DataFrame(rows).sort("h", "MAE_ramp")


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p, dtype=float)
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (k - i) * p[j]))
        adj[j] = running
    return adj


def dm_pairs(pred, models_by_h, lag=60):
    """Pairwise DM tests (absolute errors, Newey-West) between the listed models per horizon, Holm per horizon."""
    rows = []
    for h, models in models_by_h.items():
        g = pred.filter(pl.col("h") == h)
        wide = g.pivot(on="model", index=["minute_epoch", "y"], values="yhat").sort("minute_epoch")
        y = wide["y"].to_numpy()
        part = []
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                d = M.dm_test(y, wide[a].to_numpy(), wide[b].to_numpy(), lag=lag)
                part.append({"h": h, "model_a": a, "model_b": b, "MAE_a": M.mae(y, wide[a].to_numpy()),
                             "MAE_b": M.mae(y, wide[b].to_numpy()), **d})
        adj = holm([r["DM_p"] if np.isfinite(r["DM_p"]) else 1.0 for r in part])
        for r, q in zip(part, adj):
            r["DM_p_holm"] = float(q)
            r["better"] = r["model_a"] if r["mean_loss_diff"] < 0 else r["model_b"]
            r["significant_5pct"] = bool(q < 0.05)
        rows += part
    return pl.DataFrame(rows)


def pick_event(md, h=5, half=45, min_valid=None):
    """Validation origin with the largest relative change among severe ramps, with a mostly valid window around it.
    Returns the minute index of the origin."""
    min_valid = min_valid or int(1.8 * half)
    ok = md[f"ok_h{h}"].to_numpy()
    sp = np.array(md[f"split_h{h}"].to_list(), dtype=object)
    r99 = md[f"ramp99_h{h}"].fill_null(0).to_numpy() == 1
    P = md["P_kW"].fill_null(np.nan).to_numpy().astype(float)
    L = md["L24_kW"].fill_null(np.nan).to_numpy().astype(float)
    valid = md["valid"].to_numpy()
    n = len(P)
    cand = np.flatnonzero(ok & (sp == "val") & r99)
    best, best_t = -1.0, None
    for t in cand:
        lo, hi = t - half, t + half + h
        if lo < 0 or hi >= n or valid[lo:hi].sum() < min_valid:
            continue
        rel = abs(P[t + h] - P[t]) / L[t] if L[t] > 0 else 0.0
        if rel > best:
            best, best_t = rel, int(t)
    return best_t, best


def weekly_skill_mod5(pred, models, h=1, min_n=500):
    """Per-week skill vs persistence on all targets and without targets at minute-mod-5 = 0."""
    from .rolling import per_block
    rows = []
    for m in models:
        g = pred.filter((pl.col("model") == m) & (pl.col("h") == h))
        for label, sub in (("all", g), ("no_mod0", g.filter(pl.col("mod5_target") != 0))):
            t = per_block(sub["minute_epoch"].to_numpy(), sub["y"].to_numpy(), sub["yhat"].to_numpy(),
                          sub["p0"].to_numpy(), "week", min_n=min_n)
            rows.append(t.with_columns(pl.lit(m).alias("model"), pl.lit(label).alias("targets")))
    return pl.concat(rows)


def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

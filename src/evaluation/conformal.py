"""conformal.py: conformal prediction intervals on stored point forecasts (Sprint 4, Week 7, RQ3).

No model is refit: every method only uses the frozen forecasts (validation: notebooks 07-11; test: the Week 4 test
run, reports/week4/test_predictions.parquet) and the residuals they leave behind. Target miscoverage alpha = 0.1.

Origins of one (model, horizon) are processed in time order. The residual of origin s, e_s = y_s - yhat_s, is only
known once its target is observed, i.e. at origins t with minute(t) >= minute(s) + h (delayed feedback; for h = 1
this is the usual one-step setting).

Scale (makes the score |e| / scale(t) comparable over regimes; always known at the origin t):
  abs     scale = 1
  vol60   mean |P(k) - P(k-1)| over the 60 minutes up to t (both minutes valid), floor FLOOR kW
  res60   mean |e| of the last 60 residuals known at t, floor FLOOR kW (undefined before 10 are known)

Methods (interval yhat_t +- q_t * scale_t):
  static    split conformal: q = conformal quantile of the scores of a fixed calibration set
  rolling   q = conformal quantile of the last W scores known at t
  aci       rolling + adaptive conformal inference (Gibbs and Candes 2021):
            alpha_{k+1} = alpha_k + gamma (alpha - err_k), applied when err_k becomes known (delayed for h > 1)
  mondrian  rolling, but the window holds only scores of origins in the same volatility bin as t
            (bins = N_BINS quantile bins of vol60 on the calibration part; vol60 is known at t)
  mondrian_pc  same, binned by the model's own predicted change |yhat(t) - P(t)| (known at t): a regime proxy
            for expected ramps (the ramp label itself uses the future and is never used to build intervals)

Conformal quantile of n scores at level 1 - a: k = ceil((n + 1)(1 - a)); a <= 0 or k > n -> infinite interval
(always covers; width reported as capped at the largest window score); a >= 1 -> empty interval (never covers,
width 0). With these rules alpha_t stays in [-gamma, 1 + gamma], so for h = 1 the ACI average miscoverage satisfies
|mean(err) - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T) (Gibbs and Candes 2021, Prop. 4.1).
For h > 1 the published guarantee does not cover delayed updates: empirical coverage only.
"""
import bisect
import math
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

HORIZONS = (1, 5, 15)
ALPHA = 0.1
FLOOR = 0.01          # kW, lower bound of the scale
K_SCALE = 60          # window of vol60 / res60
MIN_SCORES = 50       # no interval before this many scores are in the window
SCALES = ("abs", "vol60", "res60")
WINDOWS = (1440, 4320, 10080)
GAMMAS = (0.001, 0.005, 0.01)
N_BINS = 5


# ------------------------------------------------------------------ inputs
def vol60_series(md):
    """vol60 for every minute of the modelling dataset (NaN if fewer than 10 valid 1-minute changes in the hour)."""
    P = md["P_kW"].fill_null(np.nan).to_numpy().astype(float)
    V = md["valid"].to_numpy().astype(bool)
    ok = np.zeros(len(P), dtype=bool)
    ok[1:] = V[1:] & V[:-1]
    d = np.where(ok, np.abs(np.diff(P, prepend=np.nan)), 0.0)
    d = np.nan_to_num(d)
    cs, cn = np.concatenate([[0.0], np.cumsum(d)]), np.concatenate([[0], np.cumsum(ok.astype(int))])
    i = np.arange(len(P))
    lo = np.maximum(i - K_SCALE + 1, 0)
    s, n = cs[i + 1] - cs[lo], cn[i + 1] - cn[lo]
    return np.where(n >= 10, s / np.maximum(n, 1), np.nan)


def load_val(root, models, seed=0):
    """Stored validation forecasts: long table model, h, minute_epoch, y, p0, yhat."""
    root = Path(root)
    parts = []
    for src in ("baselines", "classical"):
        f = root / "reports" / src / "val_predictions.parquet"
        if f.exists():
            d = pl.read_parquet(f)
            if "seed" in d.columns:
                d = d.filter((pl.col("seed") == seed) | pl.col("seed").is_null())
            parts.append(d.filter(pl.col("model").is_in(list(models))))
    for m in models:
        for group, pat in (("week4_nf", f"{m}_s{seed}"), ("week3_rnn", f"{m}_local_s{seed}")):
            f = root / "experiments" / group / pat / "predictions_val.parquet"
            if f.exists():
                parts.append(pl.read_parquet(f).with_columns(pl.lit(m).alias("model")))
    out = pl.concat([p.select(pl.col("model").cast(pl.Utf8), pl.col("h").cast(pl.Int64),
                              pl.col("minute_epoch").cast(pl.Int64), pl.col("y").cast(pl.Float64),
                              pl.col("p0").cast(pl.Float64), pl.col("yhat").cast(pl.Float64)) for p in parts])
    return out.unique(["model", "h", "minute_epoch"], keep="first").sort("model", "h", "minute_epoch")


def load_test(root, models, seed=0):
    """Frozen Week 4 test forecasts (seed `seed`; single-seed models have seed 0)."""
    d = pl.read_parquet(Path(root) / "reports" / "week4" / "test_predictions.parquet")
    d = d.filter(pl.col("model").is_in(list(models)) & (pl.col("seed") == seed))
    return d.select(pl.col("model").cast(pl.Utf8), pl.col("h").cast(pl.Int64), pl.col("minute_epoch").cast(pl.Int64),
                    pl.col("y").cast(pl.Float64), pl.col("p0").cast(pl.Float64),
                    pl.col("yhat").cast(pl.Float64)).sort("model", "h", "minute_epoch")


def frame(pred, md, model, h, vol60):
    """Arrays of one (model, h), sorted by time, with labels: dict of numpy arrays."""
    g = pred.filter((pl.col("model") == model) & (pl.col("h") == h)).sort("minute_epoch")
    ep = md["minute_epoch"].to_numpy()
    me = g["minute_epoch"].to_numpy()
    t = np.searchsorted(ep, me)
    if not np.array_equal(ep[np.clip(t, 0, len(ep) - 1)], me):
        raise ValueError("forecast origins not found in the modelling dataset")
    return {"minute": t.astype(np.int64), "epoch": me, "y": g["y"].to_numpy(), "yhat": g["yhat"].to_numpy(),
            "p0": g["p0"].to_numpy(), "vol60": vol60[t],
            "ramp": md[f"ramp_h{h}"].cast(pl.Float64).to_numpy()[t],
            "ramp99": md[f"ramp99_h{h}"].cast(pl.Float64).to_numpy()[t]}


def concat_frames(a, b):
    """Validation frame followed by the test frame (one continuous run); returns the joined frame and test mask."""
    out = {k: np.concatenate([a[k], b[k]]) for k in a}
    mask = np.concatenate([np.zeros(len(a["y"]), bool), np.ones(len(b["y"]), bool)])
    if np.any(np.diff(out["minute"]) <= 0):
        raise ValueError("origins are not strictly increasing")
    return out, mask


# ------------------------------------------------------------------ scale and scores
def known_count(minute, h):
    """m[i] = number of origins whose residual is known at origin i (minute[s] + h <= minute[i])."""
    return np.searchsorted(minute + h, minute, side="right")


def scale_series(f, h, kind):
    n = len(f["y"])
    if kind == "abs":
        return np.ones(n)
    if kind == "vol60":
        v = f["vol60"].copy()
        return np.where(np.isfinite(v), np.maximum(v, FLOOR), np.nan)
    if kind == "res60":
        ae = np.abs(f["y"] - f["yhat"])
        cs = np.concatenate([[0.0], np.cumsum(ae)])
        m = known_count(f["minute"], h)
        lo = np.maximum(m - K_SCALE, 0)
        cnt = m - lo
        s = (cs[m] - cs[lo]) / np.maximum(cnt, 1)
        return np.where(cnt >= 10, np.maximum(s, FLOOR), np.nan)
    raise ValueError(kind)


def conformal_q(sorted_scores, a):
    """(q, flag): flag 0 finite, 1 infinite, -1 empty, None if too few scores."""
    n = len(sorted_scores)
    if n < MIN_SCORES:
        return np.nan, None
    if a >= 1.0:
        return 0.0, -1
    if a <= 0.0:
        return np.inf, 1
    k = math.ceil((n + 1) * (1.0 - a))
    if k > n:
        return np.inf, 1
    if k <= 0:
        return 0.0, -1
    return sorted_scores[k - 1], 0


# ------------------------------------------------------------------ methods
def run_static(f, h, scale, calib_mask, alpha=ALPHA):
    """Split conformal with a fixed calibration set (calib_mask) and a fixed conformal quantile."""
    sc = np.abs(f["y"] - f["yhat"]) / scale
    cal = np.sort(sc[calib_mask & np.isfinite(sc)])
    q, flag = conformal_q(cal, alpha)
    n = len(sc)
    return {"q": np.full(n, q), "flag": np.full(n, -9 if flag is None else flag), "alpha_t": np.full(n, alpha),
            "cap": np.full(n, cal[-1] if len(cal) else np.nan), "n_cal": len(cal)}


def proxy(f, method):
    """Binning variable known at the origin."""
    return f["vol60"] if method == "mondrian" else np.abs(f["yhat"] - f["p0"])


def run_sequential(f, h, scale, method, W, gamma=0.0, alpha=ALPHA, bin_edges=None):
    """rolling / aci / mondrian with delayed feedback. Returns per-origin q (in score units), flag, alpha_t, cap."""
    n = len(f["y"])
    sc = np.abs(f["y"] - f["yhat"]) / scale
    m = known_count(f["minute"], h)
    if method in ("mondrian", "mondrian_pc"):
        b = np.digitize(np.nan_to_num(proxy(f, method), nan=0.0), bin_edges)
        nb = len(bin_edges) + 1
    else:
        b = np.zeros(n, dtype=int)
        nb = 1
    srt = [[] for _ in range(nb)]
    win = [deque() for _ in range(nb)]
    q = np.full(n, np.nan)
    flag = np.full(n, -9, dtype=np.int8)
    alpha_t = np.full(n, np.nan)
    cap = np.full(n, np.nan)
    err = np.full(n, -1, dtype=np.int8)        # -1 = no interval produced
    a_t = alpha
    j = 0
    use_aci = method == "aci"
    for i in range(n):
        while j < m[i]:                       # residuals that became known before origin i
            s = sc[j]
            if np.isfinite(s):
                k = b[j]
                bisect.insort(srt[k], s)
                win[k].append(s)
                if len(win[k]) > W:
                    old = win[k].popleft()
                    del srt[k][bisect.bisect_left(srt[k], old)]
            if use_aci and err[j] >= 0:
                a_t = a_t + gamma * (alpha - err[j])
            j += 1
        if not np.isfinite(scale[i]):
            continue
        k = b[i]
        qi, fl = conformal_q(srt[k], a_t if use_aci else alpha)
        if fl is None:
            continue
        q[i], flag[i], alpha_t[i] = qi, fl, a_t
        cap[i] = srt[k][-1]
        if fl == 1:
            err[i] = 0
        elif fl == -1:
            err[i] = 1
        else:
            err[i] = int(abs(f["y"][i] - f["yhat"][i]) > qi * scale[i])
    return {"q": q, "flag": flag, "alpha_t": alpha_t, "cap": cap, "err": err, "n_updates": int((err[:j] >= 0).sum())}


def aci_bound(alpha, gamma, T, alpha1=ALPHA):
    return (max(alpha1, 1 - alpha1) + gamma) / (gamma * T)


def intervals(f, scale, res, alpha=ALPHA):
    """Lower/upper bounds (infinite -> capped at the largest window score), covered, width, interval score."""
    q = res["q"]
    fl = res["flag"]
    have = fl != -9
    qc = np.where(fl == 1, res["cap"], q)                    # capped half-width in score units
    half = np.where(fl == -1, 0.0, qc * scale)
    lo, hi = f["yhat"] - half, f["yhat"] + half
    e = np.abs(f["y"] - f["yhat"])
    cov = np.where(fl == 1, True, np.where(fl == -1, False, e <= q * scale))
    y = f["y"]
    width = hi - lo
    iscore = width + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return {"have": have, "lo": lo, "hi": hi, "covered": cov, "width": width, "iscore": iscore,
            "infinite": fl == 1, "empty": fl == -1}


def summarise(f, iv, mask, day_min_n=600):
    """Coverage and width overall and by ramp class on origins in mask (and with an interval)."""
    k = mask & iv["have"]
    r, r99 = f["ramp"], f["ramp99"]
    out = {"n": int(mask.sum()), "n_interval": int(k.sum())}
    sub = {"all": k, "normal": k & (r == 0), "ramp": k & (r == 1), "severe": k & (r99 == 1)}
    for name, s in sub.items():
        out[f"n_{name}"] = int(s.sum())
        out[f"cov_{name}"] = float(iv["covered"][s].mean()) if s.any() else float("nan")
        out[f"width_{name}"] = float(iv["width"][s].mean()) if s.any() else float("nan")
    out["width_median"] = float(np.median(iv["width"][k])) if k.any() else float("nan")
    out["iscore"] = float(iv["iscore"][k].mean()) if k.any() else float("nan")
    out["n_infinite"] = int((iv["infinite"] & k).sum())
    out["n_empty"] = int((iv["empty"] & k).sum())
    day = f["epoch"][k] // 86400
    if len(day):
        u, inv, cnt = np.unique(day, return_inverse=True, return_counts=True)
        dc = np.bincount(inv, weights=iv["covered"][k].astype(float)) / cnt
        dc = dc[cnt >= day_min_n]
        out["n_days"] = int(len(dc))
        out["cov_day_min"] = float(dc.min()) if len(dc) else float("nan")
        out["cov_day_p10"] = float(np.quantile(dc, 0.1)) if len(dc) else float("nan")
        out["share_days_below_0p8"] = float((dc < 0.8).mean()) if len(dc) else float("nan")
    return out


# ------------------------------------------------------------------ one configuration
def configs():
    out = [{"method": "static", "scale": s} for s in SCALES]
    out += [{"method": "rolling", "scale": s, "W": w} for s in SCALES for w in WINDOWS]
    out += [{"method": "aci", "scale": s, "W": w, "gamma": g} for s in SCALES for w in WINDOWS for g in GAMMAS]
    out += [{"method": mm, "scale": s, "W": w} for mm in ("mondrian", "mondrian_pc") for s in SCALES for w in WINDOWS]
    return out


def cfg_name(c):
    s = f"{c['method']}|{c['scale']}"
    if "W" in c:
        s += f"|W{c['W']}"
    if "gamma" in c:
        s += f"|g{c['gamma']}"
    return s


def run_config(f, h, c, calib_mask, alpha=ALPHA):
    """Run one configuration on the frame f. calib_mask = calibration part (static set; mondrian bin edges)."""
    scale = scale_series(f, h, c["scale"])
    if c["method"] == "static":
        res = run_static(f, h, scale, calib_mask, alpha)
    else:
        edges = None
        if c["method"] in ("mondrian", "mondrian_pc"):
            v = proxy(f, c["method"])[calib_mask]
            edges = np.quantile(v[np.isfinite(v)], np.linspace(0, 1, N_BINS + 1)[1:-1])
        res = run_sequential(f, h, scale, c["method"], c["W"], c.get("gamma", 0.0), alpha, edges)
    return scale, res, intervals(f, scale, res, alpha)

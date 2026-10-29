#!/usr/bin/env python
"""build_model_dataset.py: per-minute modelling dataset (validity rules, features, split, ramp labels).

Input : the analysis-window series with log features (output of build_log_features.py, 20 columns),
        default data/processed/series_full_features.parquet (2021-03-04 .. 2021-09-30, New York wall clock).
Output: data/processed/model_dataset.parquet (one row per minute, same grid as the input)
        data/processed/model_dataset_meta.json (rules, counts, checks, split, thresholds, persistence, train stats)

Input windows (60 minutes) are NOT materialised here; a loader builds them from this table.
Scaling is NOT applied here; train statistics for scaling are stored in the meta file.

Validity rules (Meeting 1 decisions, Week 3; notebook 05 v2 cell 6 plus the unscheduled-outage rule):
  outage    : from build_log_features.py (cluster-wide log gap without GPU telemetry; power null)
  mwin      : maintenance event window = 23:00-23:59 of the day before + maintenance day + whole day after
              (maintenance days = days with maint_day = True in the input)
  dst_gap   : wall-clock hour that does not exist in America/New_York (2021-03-14 02:00-02:59)
  susp_zero : power == 0, outside mwin and dst_gap, gpus_alloc > 0 (telemetry gap while GPUs are allocated)
  uwin      : unscheduled outage window: for every outage run outside mwin, from the earlier of
              (first susp_zero minute of that day, run start) to 23:59 of that day
  no_gpu    : not outage and n_gpu_streams == 0 (cluster up, no GPU stream, power 0 by the empty-minute rule);
              util/temp means are null there and are filled with 0
  valid     : not outage, not mwin, not uwin, not dst_gap, power not null, not susp_zero

Features (all causal: they use only minutes <= t):
  P_kW                 target, power_W / 1000
  mod5                 (minute_epoch // 60) mod 5 (five-minute dip)
  hour_sin, hour_cos   time of day (New York local hour + minute / 60) on a 24-h circle
  L24_kW               mean of valid P over the 1,440 minutes ending at t (t included); null if < 720 valid
  n_valid_in60         number of valid minutes among the 60 minutes ending at t (t included)
  d1_<cov>, d5_<cov>   change of each covariate over 1 and 5 minutes; null unless both ends are valid

Per horizon h in (1, 5, 15), indexed by the forecast origin t:
  ok_h{h}       sample usable: all 60 input minutes valid (n_valid_in60 == 60) and the target minute t + h valid
  split_h{h}    pair split: train if t + h < i_tr; val if t >= i_tr and t + h < i_va; test if t >= i_va; else null
  ramp_h{h}     1 if |P(t+h) - P(t)| / L24(t) > train p95 (main ramp definition), 0 if not, null if undefined
  ramp99_h{h}   same with the train p99 (severe tier, sensitivity)
Thresholds are computed on train pairs with both ends valid and L24(t) defined (no 60-minute input rule).

Split: chronological 70/15/15 by minute index, i_tr = int(0.7 n), i_va = int(0.85 n) (as notebook 05).
"""
import argparse
import calendar
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

VERSION = "1.1 (2026-10-08)"
HORIZONS = (1, 5, 15)
INPUT_LEN = 60
L24_LEN = 1440
L24_MIN_VALID = 720
TZ = ZoneInfo("America/New_York")
UTIL_TEMP = ["util_gpu_mean", "util_mem_mean", "temp_gpu_mean", "temp_mem_mean"]
COVARIATES = UTIL_TEMP + ["mem_used_sum", "mem_free_sum", "n_gpu_equiv", "gpus_alloc", "active_gpu_jobs",
                          "active_jobs", "queued_gpu_jobs", "queued_jobs"]
REQUIRED = ["minute_epoch", "minute_local", "power_W", "n_gpu_streams", "n_edge_streams", "cluster_gap",
            "outage", "maint_day"] + COVARIATES

# Reference values of the v2 window (notebook 05 v2, HANDOFF_week2_session5.md Sections 7 and 8.4).
# Checked unless --no-expected. valid_before_uwin uses the notebook-05 rule (without uwin).
EXPECTED = {
    "rows": 303_840,
    "first_minute": "2021-03-04 00:00",
    "last_minute": "2021-09-30 23:59",
    "outage": 5_548,
    "mwin": 20_580,
    "dst_gap": 60,
    "susp_zero": 49,
    "no_gpu": 585,
    "valid_before_uwin": 282_988,
    "uwin_days": 1,          # days with an unscheduled outage (one outage can appear as several runs)
    "uwin_first_start": "2021-03-29 11:32",
    "val_start": "2021-07-29 16:48",
    "test_start": "2021-08-30 08:24",
}
# Persistence MAE on test pairs (both ends valid). 29 March lies in train, so test must not change.
EXPECTED_PERSIST_TEST = {1: 0.2407, 5: 0.4000, 15: 0.6221}
PERSIST_TOL = 6e-5   # the reference values are rounded to 4 decimals


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nonexistent_hours(day):
    """Wall-clock hours of `day` that do not exist in America/New_York (DST start)."""
    out = []
    for hr in range(24):
        naive = datetime(day.year, day.month, day.day, hr, 30)
        back = naive.replace(tzinfo=TZ).astimezone(timezone.utc).astimezone(TZ).replace(tzinfo=None)
        if back != naive:
            out.append(hr)
    return out


def window_count(x, length):
    """Sum of x over the `length` entries ending at each index (index included; shorter at the start)."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    i = np.arange(len(x))
    lo = np.maximum(0, i - length + 1)
    return c[i + 1] - c[lo]


def runs(mask):
    """(start, end) index pairs (end inclusive) of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return [(int(s), int(e)) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1)]


def shift_back(x, k):
    """x(t - k); NaN for t < k."""
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def shift_fwd(x, k):
    """x(t + k); NaN for t >= n - k."""
    out = np.full(len(x), np.nan)
    out[:-k] = x[k:]
    return out


def nullable(arr, defined, dtype):
    """polars Series from a numpy array with nulls where `defined` is False."""
    return pl.Series(np.where(defined, arr, np.nan).astype(np.float64)).fill_nan(None).cast(dtype)


def stats(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="data/processed/series_full_features.parquet")
    ap.add_argument("--out", default="data/processed/model_dataset.parquet")
    ap.add_argument("--meta", default="data/processed/model_dataset_meta.json")
    ap.add_argument("--no-expected", action="store_true",
                    help="skip the comparison with the v2-window reference values (other windows, tests)")
    ap.add_argument("--strict", action="store_true", help="exit with code 3 if any reference check fails")
    a = ap.parse_args()
    T = time.time()

    # ---------------------------------------------------------------- load and basic checks
    log(f"build_model_dataset.py {VERSION}")
    log(f"input: {a.input}")
    df = pl.read_parquet(a.input)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        log(f"ABORT: missing input columns {missing}")
        sys.exit(2)
    df = df.sort("minute_epoch")
    n = df.height
    ep = df["minute_epoch"].to_numpy()
    steps = np.unique(np.diff(ep))
    if steps.size != 1 or steps[0] != 60:
        log(f"ABORT: minute grid is not complete (steps found: {steps[:10]})")
        sys.exit(2)
    loc = df["minute_local"]
    locl = loc.to_list()                     # python datetimes, indexed with plain ints
    first, last = locl[0], locl[-1]
    log(f"rows {n:,} | {fmt(first)} .. {fmt(last)} | step 60 s | columns {df.width}")

    dates = loc.dt.date()
    hour = loc.dt.hour().to_numpy()
    minute = loc.dt.minute().to_numpy()
    day_ord = np.array([d.toordinal() for d in dates.to_list()])

    power = df["power_W"].to_numpy().astype(np.float64)          # NaN where null
    outage = df["outage"].fill_null(False).to_numpy().astype(bool)
    maint_day = df["maint_day"].fill_null(False).to_numpy().astype(bool)
    n_streams = df["n_gpu_streams"].to_numpy()
    gpus_alloc = df["gpus_alloc"].to_numpy().astype(np.float64)

    # ---------------------------------------------------------------- validity flags
    mdays = sorted({d for d, m in zip(dates.to_list(), maint_day) if m})
    m_ord = {d.toordinal() for d in mdays}
    after_ord = {o + 1 for o in m_ord}
    before_ord = {o - 1 for o in m_ord}
    mwin = (np.isin(day_ord, list(m_ord)) | np.isin(day_ord, list(after_ord))
            | (np.isin(day_ord, list(before_ord)) & (hour == 23)))

    dst_gap = np.zeros(n, dtype=bool)
    dst_days = []
    for d in sorted(set(dates.to_list())):
        hrs = nonexistent_hours(d)
        if hrs:
            dst_days.append({"date": d.isoformat(), "hours": hrs})
            dst_gap |= (day_ord == d.toordinal()) & np.isin(hour, hrs)

    pzero = np.isfinite(power) & (power == 0)
    susp_zero = pzero & ~mwin & ~dst_gap & (gpus_alloc > 0)
    no_gpu = ~outage & (n_streams == 0)
    valid_before_uwin = ~outage & ~mwin & ~dst_gap & np.isfinite(power) & ~susp_zero

    uwin = np.zeros(n, dtype=bool)
    uwin_runs = []
    for s, e in runs(outage & ~mwin):
        d = day_ord[s]
        day_lo = int(np.searchsorted(day_ord, d, side="left"))
        day_hi = int(np.searchsorted(day_ord, d, side="right")) - 1
        sz = np.flatnonzero(susp_zero[day_lo:day_hi + 1])
        start = min(s, day_lo + int(sz[0])) if sz.size else s
        uwin[start:day_hi + 1] = True
        uwin_runs.append({
            "outage_start": fmt(locl[s]), "outage_end": fmt(locl[e]), "outage_minutes": int(e - s + 1),
            "first_susp_zero": fmt(locl[day_lo + int(sz[0])]) if sz.size else None,
            "uwin_start": fmt(locl[start]), "uwin_end": fmt(locl[day_hi]), "uwin_minutes": int(day_hi - start + 1)})
    valid = valid_before_uwin & ~uwin

    log(f"maintenance days in input: {[d.isoformat() for d in mdays]}")
    log(f"DST gap days: {dst_days}")
    log(f"unscheduled outage runs (outside mwin): {len(uwin_runs)}")
    for r in uwin_runs:
        log(f"  outage {r['outage_start']} .. {r['outage_end']} ({r['outage_minutes']} min) | "
            f"first susp_zero {r['first_susp_zero']} | uwin {r['uwin_start']} .. {r['uwin_end']} "
            f"({r['uwin_minutes']} min)")
    counts = {"rows": n, "outage": int(outage.sum()), "mwin": int(mwin.sum()), "dst_gap": int(dst_gap.sum()),
              "susp_zero": int(susp_zero.sum()), "no_gpu": int(no_gpu.sum()), "uwin": int(uwin.sum()),
              "uwin_removed_from_valid": int((valid_before_uwin & uwin).sum()),
              "zero_power": int(pzero.sum()), "zero_power_valid": int((pzero & valid).sum()),
              "valid_before_uwin": int(valid_before_uwin.sum()), "valid": int(valid.sum())}
    log("counts: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    log(f"valid share: {valid.mean():.4%}")

    # ---------------------------------------------------------------- fill rule
    fill_before = {c: int(df[c].null_count()) for c in UTIL_TEMP}
    df = (df.with_columns(pl.Series("_no_gpu", no_gpu))
            .with_columns([pl.when(pl.col("_no_gpu") & pl.col(c).is_null()).then(0.0).otherwise(pl.col(c)).alias(c)
                           for c in UTIL_TEMP])
            .drop("_no_gpu"))
    cov_null_outside_outage = {c: int((df[c].is_null().to_numpy() & ~outage).sum()) for c in COVARIATES}
    bad = {c: v for c, v in cov_null_outside_outage.items() if v}
    log(f"util/temp nulls before fill {fill_before}; covariate nulls outside outage after fill: "
        f"{bad if bad else 'none'}")

    # ---------------------------------------------------------------- features
    P = power / 1000.0
    Pv = np.where(valid, P, 0.0)
    cnt24 = window_count(valid.astype(np.float64), L24_LEN)
    sum24 = window_count(Pv, L24_LEN)
    L24 = np.where(cnt24 >= L24_MIN_VALID, sum24 / np.maximum(cnt24, 1), np.nan)
    nvi60 = window_count(valid.astype(np.float64), INPUT_LEN).round().astype(np.int16)
    mod5 = ((ep // 60) % 5).astype(np.int8)
    frac = (hour + minute / 60.0) / 24.0
    hour_sin, hour_cos = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)

    i_tr, i_va = int(0.7 * n), int(0.85 * n)
    idx = np.arange(n)
    split = np.where(idx < i_tr, "train", np.where(idx < i_va, "val", "test"))
    log(f"split: i_tr {i_tr:,} ({fmt(locl[i_tr])}) | i_va {i_va:,} ({fmt(locl[i_va])}) | "
        f"valid minutes train {int((valid & (split == 'train')).sum()):,} / val "
        f"{int((valid & (split == 'val')).sum()):,} / test {int((valid & (split == 'test')).sum()):,}")

    cols = {}
    for c in COVARIATES:
        x = df[c].to_numpy().astype(np.float64)
        for k in (1, 5):
            ok = valid & np.concatenate([np.zeros(k, bool), valid[:-k]])
            dx = x - shift_back(x, k)
            cols[f"d{k}_{c}"] = nullable(dx, ok & np.isfinite(dx), pl.Float64)

    # ---------------------------------------------------------------- per-horizon samples, ramps, persistence
    thresholds, ramp_rates, persist_pairs, persist_samples, sample_counts = {}, {}, {}, {}, {}
    for h in HORIZONS:
        Pf = shift_fwd(P, h)
        vf = np.concatenate([valid[h:], np.zeros(h, bool)])
        pair = valid & vf
        ok = pair & (nvi60 == INPUT_LEN)
        ps = np.full(n, None, dtype=object)
        ps[(idx + h) < i_tr] = "train"
        ps[(idx >= i_tr) & ((idx + h) < i_va)] = "val"
        ps[idx >= i_va] = "test"
        ps[idx + h >= n] = None
        dP = np.abs(Pf - P)
        rel = dP / L24
        rdef = pair & np.isfinite(L24) & (L24 > 0)
        tr = rdef & (ps == "train")
        p95, p99 = (float(v) for v in np.quantile(rel[tr], [0.95, 0.99]))
        tra = pair & (ps == "train")
        a95, a99 = (float(v) for v in np.quantile(dP[tra], [0.95, 0.99]))
        thresholds[h] = {"rel_p95": p95, "rel_p99": p99, "n_train_rel": int(tr.sum()),
                         "abs_p95_kW": a95, "abs_p99_kW": a99, "n_train_abs": int(tra.sum())}
        r95 = (rel > p95) & rdef
        r99 = (rel > p99) & rdef
        cols[f"ok_h{h}"] = pl.Series(ok)
        cols[f"split_h{h}"] = pl.Series(ps.tolist(), dtype=pl.Utf8)
        cols[f"ramp_h{h}"] = nullable(r95.astype(np.int8), rdef, pl.Int8)
        cols[f"ramp99_h{h}"] = nullable(r99.astype(np.int8), rdef, pl.Int8)
        ramp_rates[h], persist_pairs[h], persist_samples[h], sample_counts[h] = {}, {}, {}, {}
        for sp in ("train", "val", "test"):
            m = ps == sp
            mr = rdef & m
            ramp_rates[h][sp] = {"n": int(mr.sum()), "rate_p95": float(r95[mr].mean()) if mr.any() else None,
                                 "rate_p99": float(r99[mr].mean()) if mr.any() else None,
                                 "n_ramps_p95": int(r95[mr].sum()), "n_ramps_p99": int(r99[mr].sum())}
            for name, mask, store in (("pairs", pair & m, persist_pairs), ("samples", ok & m, persist_samples)):
                e = (Pf - P)[mask]
                store[h][sp] = {"n": int(mask.sum()), "MAE": float(np.abs(e).mean()) if e.size else None,
                                "RMSE": float(np.sqrt((e ** 2).mean())) if e.size else None,
                                "mean_P": float(P[mask].mean()) if e.size else None}
            if h == 1:
                mm = ok & m & (shift_fwd(mod5.astype(float), 1) != 0)
                e = (Pf - P)[mm]
                persist_samples[h][sp + "_no_mod0_target"] = {"n": int(mm.sum()), "MAE": float(np.abs(e).mean()),
                                                              "RMSE": float(np.sqrt((e ** 2).mean()))}
            sample_counts[h][sp] = {"pairs": int((pair & m).sum()), "samples": int((ok & m).sum())}
        log(f"h={h:>2}: rel p95 {p95:.4f} p99 {p99:.4f} | abs p95 {a95:.3f} p99 {a99:.3f} kW | "
            f"ramp rate p95 val {ramp_rates[h]['val']['rate_p95']:.3f} test {ramp_rates[h]['test']['rate_p95']:.3f}")
        log(f"      persistence MAE pairs train/val/test "
            + " / ".join(f"{persist_pairs[h][s]['MAE']:.4f}" for s in ("train", "val", "test"))
            + " | samples " + " / ".join(f"{persist_samples[h][s]['MAE']:.4f}" for s in ("train", "val", "test"))
            + " | samples n " + " / ".join(f"{sample_counts[h][s]['samples']:,}" for s in ("train", "val", "test")))

    # ---------------------------------------------------------------- assemble output
    flags = {"valid": valid, "outage": outage, "mwin": mwin, "uwin": uwin, "dst_gap": dst_gap,
             "susp_zero": susp_zero, "no_gpu": no_gpu, "maint_day": maint_day}
    out = pl.DataFrame({
        "minute_epoch": df["minute_epoch"], "minute_local": df["minute_local"],
        "split": pl.Series(split.tolist(), dtype=pl.Utf8),
        "P_kW": nullable(P, np.isfinite(P), pl.Float64),
        **{k: pl.Series(v) for k, v in flags.items()},
        **{c: df[c] for c in COVARIATES},
        "n_gpu_streams": df["n_gpu_streams"], "n_edge_streams": df["n_edge_streams"], "cluster_gap": df["cluster_gap"],
        "mod5": pl.Series(mod5), "hour_sin": pl.Series(hour_sin), "hour_cos": pl.Series(hour_cos),
        "L24_kW": nullable(L24, np.isfinite(L24), pl.Float64), "n_valid_in60": pl.Series(nvi60),
        **cols})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    log(f"written {a.out}: {out.height:,} rows x {out.width} columns, {Path(a.out).stat().st_size / 2**20:.1f} MiB")

    # ---------------------------------------------------------------- train statistics (for scaling)
    trv = valid & (split == "train")
    train_stats = {"P_kW": stats(P[trv]), "L24_kW": stats(L24[trv])}
    for c in COVARIATES:
        train_stats[c] = stats(df[c].to_numpy().astype(np.float64)[trv])
    for name, s in cols.items():
        if name.startswith("d"):
            train_stats[name] = stats(s.to_numpy().astype(np.float64)[trv])

    # ---------------------------------------------------------------- reference checks
    checks = []
    if not a.no_expected:
        obs = {"rows": n, "first_minute": fmt(first), "last_minute": fmt(last), "outage": counts["outage"],
               "mwin": counts["mwin"], "dst_gap": counts["dst_gap"], "susp_zero": counts["susp_zero"],
               "no_gpu": counts["no_gpu"], "valid_before_uwin": counts["valid_before_uwin"],
               "uwin_days": len({r["uwin_start"][:10] for r in uwin_runs}), "uwin_first_start": uwin_runs[0]["uwin_start"] if uwin_runs else None,
               "val_start": fmt(locl[i_tr]), "test_start": fmt(locl[i_va])}
        for k, v in EXPECTED.items():
            checks.append({"check": k, "expected": v, "observed": obs[k], "ok": obs[k] == v})
        for h, v in EXPECTED_PERSIST_TEST.items():
            o = persist_pairs[h]["test"]["MAE"]
            checks.append({"check": f"persistence_MAE_test_h{h}", "expected": v, "observed": round(o, 6),
                           "ok": abs(o - v) <= PERSIST_TOL})
        n_fail = sum(not c["ok"] for c in checks)
        for c in checks:
            log(f"CHECK {'ok      ' if c['ok'] else 'MISMATCH'} {c['check']}: expected {c['expected']} | "
                f"observed {c['observed']}")
        log(f"reference checks: {len(checks) - n_fail} ok, {n_fail} mismatch")

    meta = {
        "script": "src/data/build_model_dataset.py", "version": VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "input": {"path": str(a.input), "sha256": sha256(a.input), "rows": n, "columns": df.width},
        "output": {"path": str(a.out), "rows": out.height, "columns": out.columns},
        "settings": {"horizons": list(HORIZONS), "input_len": INPUT_LEN, "L24_len": L24_LEN,
                     "L24_min_valid": L24_MIN_VALID, "split": "70/15/15 by minute index",
                     "covariates": COVARIATES, "time_axis": "America/New_York wall clock (file clock)",
                     "quantile_method": "numpy.quantile default (linear)"},
        "rules": __doc__,
        "maintenance_days": [d.isoformat() for d in mdays], "dst_gap_days": dst_days,
        "uwin_runs": uwin_runs, "counts": counts, "fill": {"util_temp_nulls_before": fill_before,
                                                           "covariate_nulls_outside_outage": cov_null_outside_outage},
        "split": {"i_tr": i_tr, "i_va": i_va, "val_start": fmt(locl[i_tr]), "test_start": fmt(locl[i_va]),
                  "valid_minutes": {sp: int((valid & (split == sp)).sum()) for sp in ("train", "val", "test")}},
        "thresholds": {str(h): v for h, v in thresholds.items()},
        "ramp_rates": {str(h): v for h, v in ramp_rates.items()},
        "sample_counts": {str(h): v for h, v in sample_counts.items()},
        "persistence_pairs": {str(h): v for h, v in persist_pairs.items()},
        "persistence_samples": {str(h): v for h, v in persist_samples.items()},
        "train_stats": train_stats, "checks": checks,
    }
    Path(a.meta).parent.mkdir(parents=True, exist_ok=True)
    with open(a.meta, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log(f"written {a.meta}")
    log(f"total {time.time() - T:.1f} s")
    if a.strict and checks and any(not c["ok"] for c in checks):
        sys.exit(3)


if __name__ == "__main__":
    main()

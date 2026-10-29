#!/usr/bin/env python
"""build_series_lowmem.py: low-memory, resumable version of build_series() (time-weighted, v3 rules).

Reads the stream Parquets in batches, computes per-minute partial sums per batch, writes them to
--work-dir, and adds them up at the end. Only per-stream quantities are needed for the weights, so
the result equals the one-shot aggregation (up to float summation order).

Rules (same as build_series.py v3):
  stream = (file, gpu_index); m0/m1 = first/last minute; interior = strictly between.
  full = median n_samples over the stream's interior minutes; fallback = global median of interior minutes.
  w = 1 (interior), min(n_samples / full, 1) (first/last minute).
  power_W = sum(power*w); util/temp = weighted means (nulls excluded); mem_* = weighted sums;
  n_gpu_equiv = sum(w); n_gpu_streams = distinct streams; n_edge_streams = non-interior stream-minutes.
  complete minute grid [t0, t1); empty minutes: counts 0, n_gpu_equiv 0.0, metrics null.

Streams without interior minutes need the global median, so their rows are set aside per batch and
processed at the end (after the global median is known from the merged n_samples histogram).
"""
import argparse, calendar, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import polars as pl

COLS = ["file", "gpu_index", "minute_epoch", "power_W", "util_gpu", "util_mem", "temp_gpu", "temp_mem",
        "mem_used", "mem_free", "n_samples"]
MEANS = [("util_gpu", "util_gpu_mean"), ("util_mem", "util_mem_mean"),
         ("temp_gpu", "temp_gpu_mean"), ("temp_mem", "temp_mem_mean")]
KEY = ["file", "gpu_index"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ep(d):
    return calendar.timegm(datetime.fromisoformat(d).timetuple())


def read_one(p):
    df = pl.read_parquet(p, columns=COLS)
    return df.with_columns([pl.col(c).cast(pl.Float64) for c in COLS[3:]] +
                           [pl.col("gpu_index").cast(pl.Int64), pl.col("minute_epoch").cast(pl.Int64),
                            pl.col("file").cast(pl.Utf8)])


def add_interior(df):
    return (df.with_columns(pl.col("minute_epoch").min().over(KEY).alias("_m0"),
                            pl.col("minute_epoch").max().over(KEY).alias("_m1"))
              .with_columns(((pl.col("minute_epoch") > pl.col("_m0")) &
                             (pl.col("minute_epoch") < pl.col("_m1"))).alias("_int")))


def partial(df, t0, t1):
    """df has _int and _full (non-null). Returns per-minute partial sums inside [t0, t1)."""
    df = (df.with_columns(pl.when(pl.col("_int")).then(1.0)
                            .otherwise(pl.min_horizontal(pl.col("n_samples") / pl.col("_full"), pl.lit(1.0)))
                            .alias("_w"))
            .filter((pl.col("minute_epoch") >= t0) & (pl.col("minute_epoch") < t1)))
    aggs = [(pl.col("power_W") * pl.col("_w")).sum().alias("s_power"),
            (pl.col("mem_used") * pl.col("_w")).sum().alias("s_mem_used"),
            (pl.col("mem_free") * pl.col("_w")).sum().alias("s_mem_free"),
            pl.col("_w").sum().alias("s_w"),
            pl.len().cast(pl.Int64).alias("s_streams"),
            (~pl.col("_int")).cast(pl.Int64).sum().alias("s_edge"),
            pl.col("power_W").is_not_null().cast(pl.Int64).sum().alias("s_nn_power"),
            pl.col("mem_used").is_not_null().cast(pl.Int64).sum().alias("s_nn_mem_used"),
            pl.col("mem_free").is_not_null().cast(pl.Int64).sum().alias("s_nn_mem_free")]
    for c, _ in MEANS:
        aggs += [(pl.col(c) * pl.col("_w")).sum().alias(f"n_{c}"),
                 pl.when(pl.col(c).is_not_null()).then(pl.col("_w")).otherwise(0.0).sum().alias(f"d_{c}")]
    return df.group_by("minute_epoch").agg(aggs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", required=True, help="local folder for partials (resumable; one per file set)")
    ap.add_argument("--win-start", required=True)
    ap.add_argument("--win-end", required=True)
    ap.add_argument("--files", nargs="*", default=None,
                    help="optional Parquet(s) with column 'name' (e.g. chunk_1.parquet); restricts the streams")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--readers", type=int, default=16)
    a = ap.parse_args()

    t0, t1 = ep(a.win_start), ep(a.win_end)
    os.makedirs(a.work_dir, exist_ok=True)
    T = time.time()

    log(f"listing {a.stream_dir} ...")
    paths = sorted(e.path for e in os.scandir(a.stream_dir) if e.name.endswith(".parquet"))
    log(f"stream parquets in folder: {len(paths):,}")
    if a.files:
        want = set()
        for f in a.files:
            want |= {Path(n).stem for n in pl.read_parquet(f, columns=["name"])["name"].to_list()}
        have = {Path(p).name[:-len(".parquet")]: p for p in paths}
        miss = want - have.keys()
        paths = sorted(have[s] for s in want if s in have)
        log(f"restricted to --files: {len(want):,} expected | {len(paths):,} found | missing {len(miss):,}")
        if miss:
            log("ABORT: missing streams for the requested files"); sys.exit(2)

    batches = [paths[i:i + a.batch] for i in range(0, len(paths), a.batch)]
    log(f"batches: {len(batches)} x up to {a.batch:,} files | work dir {a.work_dir}")

    for b, bp in enumerate(batches):
        part_f = f"{a.work_dir}/part_{b:04d}.parquet"
        hist_f = f"{a.work_dir}/hist_{b:04d}.parquet"
        noint_f = f"{a.work_dir}/noint_{b:04d}.parquet"
        if os.path.exists(part_f) and os.path.exists(hist_f) and os.path.exists(noint_f):
            log(f"batch {b + 1}/{len(batches)}: done earlier, skipped")
            continue
        tb = time.time()
        with ThreadPoolExecutor(a.readers) as ex:
            df = pl.concat(list(ex.map(read_one, bp)), how="vertical_relaxed")
        rows = df.height
        df = add_interior(df)
        df = df.with_columns(pl.col("n_samples").filter(pl.col("_int")).median().over(KEY).alias("_full"))
        hist = (df.filter(pl.col("_int")).group_by("n_samples").agg(pl.len().cast(pl.Int64).alias("cnt")))
        noint = df.filter(pl.col("_full").is_null()).select(COLS + ["_int"])
        part = partial(df.filter(pl.col("_full").is_not_null()), t0, t1)
        del df
        noint.write_parquet(noint_f + ".tmp"); os.replace(noint_f + ".tmp", noint_f)
        hist.write_parquet(hist_f + ".tmp"); os.replace(hist_f + ".tmp", hist_f)
        part.write_parquet(part_f + ".tmp"); os.replace(part_f + ".tmp", part_f)   # part last = batch complete
        log(f"batch {b + 1}/{len(batches)}: {len(bp):,} files | {rows:,} stream-minutes | "
            f"{noint.height:,} rows without interior | {time.time() - tb:.0f} s | total {(time.time() - T) / 60:.1f} min")

    # global fallback median from the merged histogram (polars median: mean of the two middle values)
    hist = (pl.concat([pl.read_parquet(f"{a.work_dir}/hist_{b:04d}.parquet") for b in range(len(batches))])
              .group_by("n_samples").agg(pl.col("cnt").sum()).sort("n_samples"))
    n = int(hist["cnt"].sum())
    cum = hist["cnt"].cum_sum().to_list(); vals = hist["n_samples"].to_list()
    def kth(k):  # 0-based
        for v, c in zip(vals, cum):
            if k < c:
                return v
    gmed = kth((n - 1) // 2) if n % 2 else (kth(n // 2 - 1) + kth(n // 2)) / 2
    log(f"global median n_samples over {n:,} interior stream-minutes: {gmed}")

    noint = pl.concat([pl.read_parquet(f"{a.work_dir}/noint_{b:04d}.parquet") for b in range(len(batches))],
                      how="vertical_relaxed")
    n_noint_streams = noint.select(KEY).unique().height
    log(f"streams without interior minutes: {n_noint_streams:,} ({noint.height:,} rows) -> fallback median")
    log("adding up the partials (incrementally, to keep RAM low) ...")
    s = None
    def acc(s, p):
        return p if s is None else pl.concat([s, p], how="vertical_relaxed").group_by("minute_epoch").agg(pl.all().sum())
    for b in range(len(batches)):
        s = acc(s, pl.read_parquet(f"{a.work_dir}/part_{b:04d}.parquet"))
    if noint.height:
        s = acc(s, partial(noint.with_columns(pl.lit(float(gmed)).alias("_full")), t0, t1))
    ser = s.select(
        "minute_epoch",
        pl.col("s_power").alias("power_W"),
        *[pl.when(pl.col(f"d_{c}") > 0).then(pl.col(f"n_{c}") / pl.col(f"d_{c}")).otherwise(None).alias(o)
          for c, o in MEANS],
        pl.col("s_mem_used").alias("mem_used_sum"),
        pl.col("s_mem_free").alias("mem_free_sum"),
        pl.col("s_w").alias("n_gpu_equiv"),
        pl.col("s_streams").alias("n_gpu_streams"),
        pl.col("s_edge").alias("n_edge_streams"))
    grid = pl.DataFrame({"minute_epoch": pl.int_range(t0, t1, 60, eager=True).cast(pl.Int64)})
    ser = (grid.join(ser, on="minute_epoch", how="left")
               .with_columns(pl.col("n_gpu_equiv").fill_null(0.0),
                             pl.col("n_gpu_streams").fill_null(0),
                             pl.col("n_edge_streams").fill_null(0),
                             pl.from_epoch("minute_epoch", time_unit="s").alias("minute_local"))
               .sort("minute_epoch"))
    ser.write_parquet(a.out)
    empty = ser.filter(pl.col("n_gpu_streams") == 0).height
    log(f"series rows {ser.height:,}; minutes with no stream {empty:,}; "
        f"power_W min {ser['power_W'].min()} / max {ser['power_W'].max()}")
    log(f"written {a.out} | total {(time.time() - T) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Build per-minute features from the Slurm log and attach them to the GPU power series.

All times are in the file clock (America/New_York wall clock stored as naive epoch seconds).
Slurm log times are true UTC and are converted with log_to_file_clock().

Added columns:
  cluster_gap      minute lies entirely inside a cluster-wide gap in the log (no job of any kind)
  outage           cluster_gap and no GPU telemetry in the minute; GPU files can keep recording
                   for up to a few minutes after jobs are killed at a maintenance start, and
                   telemetry takes precedence over the log
  maint_day        local calendar day of a long cluster-wide gap (>= --outage-min-hours);
                   the day is the date of (gap start + 2 h), so a gap starting just before
                   midnight counts for the next day
  active_jobs      jobs of any kind running at the end of the minute
  active_gpu_jobs  GPU jobs running at the end of the minute
  gpus_alloc       GPUs allocated to running GPU jobs at the end of the minute
  queued_jobs      jobs submitted but not yet started at the end of the minute
  queued_gpu_jobs  same, GPU jobs only
State variables are sampled at the minute end (t + 60 s), the moment the minute's power value
becomes available, so they carry no information from after the minute.

Empty minutes (n_gpu_streams == 0):
  - during an outage: power and all metrics stay null (cluster down; masked in modelling);
  - cluster up but no GPU job: power_W, mem_used_sum, mem_free_sum and n_gpu_equiv are set
    to 0 (true job-attributed value); utilization and temperature means stay null
    (undefined without GPUs).
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

GPU_RE = r"(?:^|,)100[12]=(\d+)"


def log_to_file_clock(col: pl.Expr) -> pl.Expr:
    return (pl.from_epoch(col, time_unit="s").dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York").dt.replace_time_zone(None).dt.epoch("s"))


def load_log(path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(path, infer_schema_length=0,
                    columns=["id_job", "tres_alloc", "time_submit", "time_start", "time_end"])
        .with_columns(
            pl.col("tres_alloc").str.extract(GPU_RE, 1).cast(pl.Int64).alias("n_gpu"),
            *[pl.col(c).cast(pl.Int64) for c in ("time_submit", "time_start", "time_end")],
        )
        .with_columns(
            log_to_file_clock(pl.col("time_submit")).alias("submit_fc"),
            log_to_file_clock(pl.col("time_start")).alias("start_fc"),
            log_to_file_clock(pl.col("time_end")).alias("end_fc"),
        )
    )


def count_at(t: np.ndarray, starts: np.ndarray, ends: np.ndarray, weights=None) -> np.ndarray:
    """Number (or weighted sum) of intervals with start <= t < end, for each instant t."""
    os_, oe = np.argsort(starts, kind="stable"), np.argsort(ends, kind="stable")
    s, e = starts[os_], ends[oe]
    i_s, i_e = np.searchsorted(s, t, side="right"), np.searchsorted(e, t, side="right")
    if weights is None:
        return i_s - i_e
    ws = np.concatenate([[0], np.cumsum(weights[os_])])
    we = np.concatenate([[0], np.cumsum(weights[oe])])
    return ws[i_s] - we[i_e]


def cluster_gaps(starts: np.ndarray, ends: np.ndarray):
    """Intervals (g0, g1) during which no job of any kind is running."""
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    cmax = np.maximum.accumulate(e)
    idx = np.nonzero(s[1:] > cmax[:-1])[0]
    return cmax[:-1][idx], s[1:][idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach Slurm-log features to the GPU power series.")
    ap.add_argument("--log", type=Path, required=True, help="slurm-log.csv")
    ap.add_argument("--series", type=Path, required=True, help="output of build_series.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outage-min-hours", type=float, default=6.0,
                    help="minimum cluster-wide gap length that marks a maintenance day")
    a = ap.parse_args()

    log = load_log(a.log)
    ran = log.filter(pl.col("end_fc") > pl.col("start_fc"))
    print(f"log rows: {log.height:,}; with end > start: {ran.height:,}")

    # Cluster-wide gaps and maintenance days (whole log)
    g0, g1 = cluster_gaps(ran["start_fc"].to_numpy(), ran["end_fc"].to_numpy())
    long_ = (g1 - g0) >= a.outage_min_hours * 3600
    maint_dates = (pl.DataFrame({"g": g0[long_] + 7200})
                   .select(pl.from_epoch("g").dt.date().alias("d"))["d"].unique().sort())
    print(f"cluster-wide gaps: {len(g0):,}; long (>= {a.outage_min_hours} h): {int(long_.sum())}")
    print("maintenance days in the whole log:", [str(d) for d in maint_dates])

    s = pl.read_parquet(a.series)
    m = s["minute_epoch"].to_numpy()
    t_end = m + 60

    # Outage: minute [m, m+60) entirely inside one gap (gaps are disjoint and sorted)
    k = np.searchsorted(g0, m, side="right") - 1
    outage = (k >= 0) & (g1[np.clip(k, 0, None)] >= m + 60)

    gpu = ran.filter(pl.col("n_gpu").is_not_null())
    queued_all = log.filter(pl.col("start_fc") >= pl.col("submit_fc"))
    queued_gpu = queued_all.filter(pl.col("n_gpu").is_not_null())

    feat = pl.DataFrame({
        "minute_epoch": m,
        "cluster_gap": outage,
        "active_jobs": count_at(t_end, ran["start_fc"].to_numpy(), ran["end_fc"].to_numpy()),
        "active_gpu_jobs": count_at(t_end, gpu["start_fc"].to_numpy(), gpu["end_fc"].to_numpy()),
        "gpus_alloc": count_at(t_end, gpu["start_fc"].to_numpy(), gpu["end_fc"].to_numpy(),
                               gpu["n_gpu"].to_numpy()),
        "queued_jobs": count_at(t_end, queued_all["submit_fc"].to_numpy(), queued_all["start_fc"].to_numpy()),
        "queued_gpu_jobs": count_at(t_end, queued_gpu["submit_fc"].to_numpy(), queued_gpu["start_fc"].to_numpy()),
    })

    no_gpu_up = (pl.col("n_gpu_streams") == 0) & ~pl.col("outage")
    out = (
        s.join(feat, on="minute_epoch", how="left")
        .with_columns((pl.col("cluster_gap") & (pl.col("n_gpu_streams") == 0)).alias("outage"))
        .with_columns(pl.col("minute_local").dt.date().is_in(maint_dates.implode()).alias("maint_day"))
        .with_columns([
            pl.when(no_gpu_up).then(pl.lit(0.0)).otherwise(pl.col(c)).alias(c)
            for c in ("power_W", "mem_used_sum", "mem_free_sum", "n_gpu_equiv")
        ])
    )

    conflict = out.filter(pl.col("cluster_gap") & (pl.col("n_gpu_streams") > 0)).height
    print(f"minutes: {out.height:,}; outage: {int(out['outage'].sum()):,}; "
          f"cluster up, no GPU job (set to 0): {out.filter(no_gpu_up).height:,}; "
          f"maint_day: {int(out['maint_day'].sum()):,}")
    print(f"null power_W after rules: {out['power_W'].null_count():,}; "
          f"cluster-gap minutes with GPU telemetry (kept, not outage): {conflict}")
    print(f"gpus_alloc max: {out['gpus_alloc'].max()} (cluster has 448 GPUs)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(f"written: {a.out} ({out.height:,} rows, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()

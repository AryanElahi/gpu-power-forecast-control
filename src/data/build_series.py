"""Build the aggregate GPU power series for a time window from MIT Supercloud per-job GPU files.

Time frame: GPU file timestamps are New York wall-clock values stored as epoch seconds
(empirical finding on 25 files; the Slurm log is true UTC). All minutes here use the file clock.

Per file (parallel, resumable):
  download in HTTP Range chunks (< 8 MiB) -> verify size -> clip to the window ->
  mean of each metric per (gpu_index, minute) -> write a small Parquet -> delete the raw CSV.
Then: sum power over all streams per minute on a complete minute grid.

Notes:
  * Each file is one node of one job; a stream is (file name, gpu_index).
  * Raw 100 ms samples are averaged per minute before summing across streams.
  * Edge minutes of a job are partial; we keep the mean of the available samples.
  * Minutes with no stream get n_gpu_streams = 0 and null power (decision pending).
  * The result is job-attributed GPU power, not facility power.
"""
from __future__ import annotations

import argparse
import calendar
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import polars as pl

BASE = "https://mit-supercloud-dataset.s3.amazonaws.com/datacenter-challenge/202201/gpu"
HEADERS = {"User-Agent": "Mozilla/5.0"}
CHUNK = 4 * 1024 * 1024  # below the 8 MiB VPN truncation limit

METRICS = {
    "power_draw_W": "power_W",
    "utilization_gpu_pct": "util_gpu",
    "utilization_memory_pct": "util_mem",
    "temperature_gpu": "temp_gpu",
    "temperature_memory": "temp_mem",
    "memory_used_MiB": "mem_used",
    "memory_free_MiB": "mem_free",
}


def day_to_epoch(day: str) -> int:
    """'YYYY-MM-DD' as a naive (file-clock) epoch, independent of the machine time zone."""
    return calendar.timegm(datetime.strptime(day, "%Y-%m-%d").timetuple())


def download(sub: str, name: str, size: int, dest: Path, retries: int = 5) -> None:
    url = f"{BASE}/{sub}/{name}"
    pos = dest.stat().st_size if dest.exists() else 0
    if pos > size:
        dest.unlink()
        pos = 0
    with open(dest, "ab") as fh:
        while pos < size:
            end = min(pos + CHUNK, size) - 1
            for attempt in range(retries):
                try:
                    req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={pos}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) != end - pos + 1:
                        raise IOError(f"short read: {len(data)} bytes")
                    break
                except Exception:
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 ** attempt)
            fh.write(data)
            pos = end + 1
    if dest.stat().st_size != size:
        raise IOError(f"size mismatch for {name}")


def aggregate(path: Path, t0: int, t1: int, name: str) -> pl.DataFrame:
    return (
        pl.scan_csv(path, infer_schema_length=10_000)
        .with_columns(pl.col("timestamp").cast(pl.Float64, strict=False))
        .filter((pl.col("timestamp") >= t0) & (pl.col("timestamp") < t1))
        .with_columns((pl.col("timestamp") // 60 * 60).cast(pl.Int64).alias("minute_epoch"))
        .group_by("gpu_index", "minute_epoch")
        .agg(
            *[pl.col(s).cast(pl.Float64, strict=False).mean().alias(d) for s, d in METRICS.items()],
            pl.len().cast(pl.Int64).alias("n_samples"),
        )
        .with_columns(pl.col("gpu_index").cast(pl.Int64), pl.lit(name).alias("file"))
        .sort("gpu_index", "minute_epoch")
        .collect(engine="streaming")
    )


def process_one(row: dict, t0: int, t1: int, stream_dir: Path, tmp_dir: Path) -> tuple[str, int]:
    out = stream_dir / (Path(row["name"]).stem + ".parquet")
    if out.exists():
        return "skipped", 0
    raw = tmp_dir / row["name"]
    download(row["sub"], row["name"], row["size"], raw)
    df = aggregate(raw, t0, t1, row["name"])
    tmp_out = stream_dir / (Path(row["name"]).stem + ".parquet.tmp")
    df.write_parquet(tmp_out)
    tmp_out.rename(out)  # the final name marks the file as done
    raw.unlink()
    return "done", row["size"]


def build_series(stream_dir: Path, t0: int, t1: int) -> pl.DataFrame:
    """Aggregate per-stream minute rows into one series on a complete minute grid.

    A stream is one (file, gpu_index) pair. Each stream-minute gets a weight equal to the
    fraction of the minute the stream actually covers:
      - interior minutes (strictly between the stream's first and last minute): weight 1,
        because the job ran for the whole minute; fewer samples only mean irregular sampling;
      - first and last minute: weight = min(n_samples / full, 1), where `full` is the median
        n_samples over that stream's interior minutes (falls back to the global median of
        interior minutes for streams without interior minutes).
    Power and memory are weighted sums (power_W = energy / 60 s); utilization and temperature
    are weighted means. This avoids counting a GPU twice when two jobs share it within one minute.
    """
    key = ["file", "gpu_index"]
    w = pl.col("_w")

    def wsum(c: str, name: str) -> pl.Expr:
        return (pl.col(c) * w).sum().alias(name)

    def wmean(c: str, name: str) -> pl.Expr:
        return ((pl.col(c) * w).sum() / pl.when(pl.col(c).is_not_null()).then(w).sum()).alias(name)

    agg = (
        pl.scan_parquet(str(stream_dir / "*.parquet"))
        .with_columns(
            pl.col("minute_epoch").min().over(key).alias("_m0"),
            pl.col("minute_epoch").max().over(key).alias("_m1"),
        )
        .with_columns(
            ((pl.col("minute_epoch") > pl.col("_m0")) & (pl.col("minute_epoch") < pl.col("_m1"))).alias("_interior")
        )
        .with_columns(
            pl.coalesce(
                pl.when(pl.col("_interior")).then(pl.col("n_samples")).median().over(key),
                pl.when(pl.col("_interior")).then(pl.col("n_samples")).median(),
            ).alias("_full")
        )
        .with_columns(
            pl.when(pl.col("_interior"))
            .then(pl.lit(1.0))
            .otherwise((pl.col("n_samples") / pl.col("_full")).clip(upper_bound=1.0))
            .alias("_w")
        )
        .group_by("minute_epoch")
        .agg(
            wsum("power_W", "power_W"),
            wmean("util_gpu", "util_gpu_mean"),
            wmean("util_mem", "util_mem_mean"),
            wmean("temp_gpu", "temp_gpu_mean"),
            wmean("temp_mem", "temp_mem_mean"),
            wsum("mem_used", "mem_used_sum"),
            wsum("mem_free", "mem_free_sum"),
            w.sum().alias("n_gpu_equiv"),
            pl.struct(key).n_unique().cast(pl.Int64).alias("n_gpu_streams"),
            (~pl.col("_interior")).sum().cast(pl.Int64).alias("n_edge_streams"),
        )
        .collect()
    )
    grid = pl.DataFrame({"minute_epoch": pl.int_range(t0, t1, 60, eager=True, dtype=pl.Int64)})
    return (
        grid.join(agg, on="minute_epoch", how="left")
        .with_columns(
            pl.col("n_gpu_streams").fill_null(0),
            pl.col("n_edge_streams").fill_null(0),
            pl.col("n_gpu_equiv").fill_null(0.0),
            pl.from_epoch("minute_epoch", time_unit="s").alias("minute_local"),
        )
        .sort("minute_epoch")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=Path, required=True, help="Parquet with columns sub, name, size")
    ap.add_argument("--stream-dir", type=Path, required=True)
    ap.add_argument("--tmp-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--win-start", required=True, help="YYYY-MM-DD, file clock, inclusive")
    ap.add_argument("--win-end", required=True, help="YYYY-MM-DD, file clock, exclusive")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-size-mib", type=float, default=None)
    a = ap.parse_args()

    t0, t1 = day_to_epoch(a.win_start), day_to_epoch(a.win_end)
    a.stream_dir.mkdir(parents=True, exist_ok=True)
    a.tmp_dir.mkdir(parents=True, exist_ok=True)

    files = pl.read_parquet(a.files).select("sub", "name", "size").sort("name")
    if a.max_size_mib is not None:
        files = files.filter(pl.col("size") < a.max_size_mib * 1024**2)
    if a.limit is not None:
        files = files.head(a.limit)
    rows = files.to_dicts()
    total_gib = sum(r["size"] for r in rows) / 1024**3
    print(f"files to handle: {len(rows):,} ({total_gib:.2f} GiB), workers: {a.workers}", flush=True)

    start, n_done, n_skip, bytes_done, failures = time.time(), 0, 0, 0, []
    todo_bytes = sum(r["size"] for r in rows
                     if not (a.stream_dir / (Path(r["name"]).stem + ".parquet")).exists())
    print(f"still to download: {todo_bytes / 1024**3:.2f} GiB", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_one, r, t0, t1, a.stream_dir, a.tmp_dir): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                status, nbytes = fut.result()
                n_done += status == "done"
                n_skip += status == "skipped"
                bytes_done += nbytes
            except Exception as e:  # keep going; record the failure
                failures.append({"name": r["name"], "error": repr(e)[:300]})
            if i % 10 == 0 or i == len(rows):
                dt = time.time() - start
                rate = bytes_done / max(dt, 1)
                pct = 100 * bytes_done / max(todo_bytes, 1)
                eta_min = (todo_bytes - bytes_done) / max(rate, 1) / 60
                print(f"  [{pct:5.1f}%] files {i:,}/{len(rows):,}  done={n_done:,} skipped={n_skip:,} "
                      f"failed={len(failures)}  {bytes_done / 1024**3:.2f}/{todo_bytes / 1024**3:.2f} GiB  "
                      f"{rate / 1024**2:.2f} MiB/s  elapsed {dt / 60:.1f} min  ETA {eta_min:.0f} min",
                      flush=True)

    if failures:
        pl.DataFrame(failures).write_csv(a.stream_dir / "_failures.csv")
        print(f"failures written to {a.stream_dir / '_failures.csv'}")

    series = build_series(a.stream_dir, t0, t1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    series.write_parquet(a.out)
    print(f"series rows: {series.height:,}; minutes with no stream: "
          f"{series.filter(pl.col('n_gpu_streams') == 0).height:,}; "
          f"power_W min/max: {series['power_W'].min()} / {series['power_W'].max()}")


if __name__ == "__main__":
    main()

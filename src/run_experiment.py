#!/usr/bin/env python
"""run_experiment.py: run one experiment from a YAML config (Week 3).

Usage (from the project root):
  python src/run_experiment.py --config configs/rnn_lstm.yaml
  python src/run_experiment.py --config configs/rnn_lstm.yaml --set seed=1 scaling=local

Writes experiments/<group>/<name>/ with config.yaml (resolved), predictions_val.parquet, history.csv, metrics.json,
model.pt, and a tracking record (experiments/runs/<group>/<name>.json + W&B run, see src/tracking.py).
Currently supported: family "rnn" (LSTM, GRU). The test split is never scored here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation import metrics as M  # noqa: E402
from tracking import log_run  # noqa: E402


def parse_value(v):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    ap.add_argument("--tracking-mode", default=None, help="online | offline | auto | disabled")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
    name = cfg.get("name") or f"{cfg['cell']}_{cfg['scaling']}{'_cov' if cfg.get('covariates') else ''}_s{cfg['seed']}"
    group = cfg.get("group", "week3_rnn")
    out = ROOT / "experiments" / group / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[{time.strftime('%H:%M:%S')}] run {name} | config {a.config} | overrides {a.set}", flush=True)

    if cfg["family"] != "rnn":
        raise SystemExit(f"family {cfg['family']} not supported by this runner yet")
    import torch
    from models import rnn as R

    md = pl.read_parquet(ROOT / cfg["dataset"])
    meta = json.loads((ROOT / cfg["dataset_meta"]).read_text())
    model, data, hist, info = R.train(md, cfg, log=lambda s: print(s, flush=True))
    pl.DataFrame(hist).write_csv(out / "history.csv")
    torch.save(model.state_dict(), out / "model.pt")

    va = data.origins("val")
    t0 = time.time()
    pv = R.predict(model, data, va)
    infer_s = time.time() - t0
    L24 = md["L24_kW"].to_numpy().astype(float)
    mod5 = md["mod5"].to_numpy().astype(int)
    ep = md["minute_epoch"].to_numpy()
    rows, metrics, parts = [], {}, []
    for j, h in enumerate(R.HORIZONS):
        t = data.idx[h]["val"]
        yhat = pv[np.searchsorted(va, t), j]
        y, p0 = data.P[t + h], data.P[t]
        ramp = md[f"ramp_h{h}"].to_numpy().astype(float)[t]
        r = M.evaluate(y, yhat, p0, l24=L24[t], ramp=ramp, thr=meta["thresholds"][str(h)]["rel_p95"],
                       mod5_target=mod5[t + h] if h == 1 else None, dm_lag=60)
        rows.append({"h": h, **r})
        metrics.update({f"h{h}/{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        parts.append(pl.DataFrame({"minute_epoch": ep[t], "h": np.full(len(t), h, np.int16), "y": y, "p0": p0,
                                   "yhat": yhat}))
    pl.concat(parts).write_parquet(out / "predictions_val.parquet")
    metrics.update({"n_params": info["n_params"], "train_s": info["train_s"], "infer_s_val": infer_s,
                    "infer_ms_per_sample": 1000 * infer_s / max(len(va), 1), "best_epoch": info["best_epoch"],
                    "epochs_run": info["epochs_run"]})
    (out / "metrics.json").write_text(json.dumps({"info": info, "val": rows, "metrics": metrics}, indent=1, default=str))
    rec, used = log_run(ROOT, name, group, {**cfg, **{f"data_{k}": v for k, v in info.items()}}, metrics,
                        files=[out / "config.yaml", out / "metrics.json"], tables={"val_metrics": pl.DataFrame(rows)},
                        mode=a.tracking_mode)
    print(f"[{time.strftime('%H:%M:%S')}] done {name} | params {info['n_params']:,} | train {info['train_s']:.0f} s | "
          f"best epoch {info['best_epoch']} | val MAE " +
          " / ".join(f"{r['MAE']:.4f}" for r in rows) + f" kW | tracking: {used} | {rec.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

"""tracking.py: experiment logging (Week 3).

Every run is written twice:
  1. a local JSON record experiments/runs/<group>/<name>.json (always; the source of truth for the tables), and
  2. a Weights & Biases run (project and entity from configs/tracking.yaml) when `mode` allows it.

mode: "online"  -> log to W&B, fail loudly if that is impossible
      "offline" -> W&B offline run under experiments/wandb (upload later with `wandb sync`)
      "auto"    -> try online, fall back to offline if the connection fails
      "disabled"-> JSON record only
"""
import json
import os
import time
from pathlib import Path


def _clean(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        if hasattr(v, "item"):
            return v.item()
        return str(v)


def load_tracking_config(root: Path) -> dict:
    import yaml
    p = Path(root) / "configs" / "tracking.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {"mode": "disabled"}


def log_run(root, name, group, config, metrics, files=(), tables=None, mode=None, tracking=None):
    """Write the JSON record and (optionally) a W&B run. Returns the path of the JSON record and the W&B mode used."""
    root = Path(root)
    tracking = tracking or load_tracking_config(root)
    mode = mode or tracking.get("mode", "auto")
    rec_dir = root / "experiments" / "runs" / group
    rec_dir.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "group": group, "logged": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": {k: _clean(v) for k, v in config.items()},
              "metrics": {k: _clean(v) for k, v in metrics.items()},
              "files": [str(f) for f in files]}
    used = "disabled"
    if mode != "disabled":
        try:
            import wandb
            os.environ.setdefault("WANDB_DIR", str(root / "experiments"))
            (root / "experiments" / "wandb").mkdir(parents=True, exist_ok=True)
            tries = ["online", "offline"] if mode == "auto" else [mode]
            for m in tries:
                try:
                    run = wandb.init(project=tracking.get("project"), entity=tracking.get("entity"), name=name,
                                     group=group, config=record["config"], mode=m, reinit="finish_previous",
                                     settings=wandb.Settings(init_timeout=60, silent=True))
                    run.summary.update(record["metrics"])
                    for tname, df in (tables or {}).items():
                        run.log({tname: wandb.Table(dataframe=df.to_pandas() if hasattr(df, "to_pandas") else df)})
                    run.finish()
                    used = m
                    record["wandb"] = {"mode": m, "url": run.url if m == "online" else None}
                    break
                except Exception as e:  # noqa: BLE001
                    record.setdefault("wandb_errors", []).append(f"{m}: {type(e).__name__}: {e}")
        except ImportError:
            record["wandb_errors"] = ["wandb not installed"]
    out = rec_dir / f"{name}.json"
    out.write_text(json.dumps(record, indent=1, default=str))
    return out, used

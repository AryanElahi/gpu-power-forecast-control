"""week6.py: Week 6 experiments (data fraction, qubit sweep) for the hybrid QNN, QRC and their classical references.

Data fraction (provisional design, assistant's decision):
  * A fraction f keeps the MOST RECENT contiguous block of train origins (the last ceil(f * N) of the sorted union
    of usable train origins; per horizon the usable origins at or after the same cutoff minute). This is the
    "short history" scenario of a new or changed cluster, and it keeps the temporal structure; validation is unchanged.
  * Networks (hybrid QNN, MLP control, PatchTST, univariate GRU) use the shared protocol with ONE change: an epoch has
    a fixed number of optimiser steps, ceil(N_full / batch), taken from repeated permutations of the available origins.
    At f = 100% this is exactly the usual epoch; for small f it keeps the optimisation budget and the early-stopping
    rule comparable (otherwise a 1% run would get 1% of the gradient steps).
  * LightGBM keeps the tuned parameters of notebook 08 and chooses the number of rounds by early stopping on
    validation (50 rounds patience, at most 3,000) at every fraction, including 100% (rerun), seeds 0-2.
  * QRC and its controls: reservoir states are cached once per minute; only the ridge readout is refitted on the
    fraction (penalty chosen on validation as in notebook 19).
Validation only. The test split is opened once in notebook 23.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import rnn as R

HORIZONS = R.HORIZONS


# ----------------------------------------------------------------------------------------------------------------
# fractions
# ----------------------------------------------------------------------------------------------------------------
def fraction_cutoff(data, frac):
    """Cutoff minute index: origins >= cutoff form the most recent fraction `frac` of the train origins."""
    tr = data.origins("train")
    k = max(1, int(math.ceil(frac * len(tr))))
    return int(tr[-k]) if frac < 1.0 else int(tr[0])


def frac_tag(frac):
    return f"f{int(round(frac * 100)):03d}"


# ----------------------------------------------------------------------------------------------------------------
# generic network trainer (fixed steps per epoch)
# ----------------------------------------------------------------------------------------------------------------
def _predict(model, data, t, device, batch=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(t), batch):
            outs.append(model(data.windows(t[i:i + batch]).to(device)).float().cpu().numpy())
    out = np.concatenate(outs) if outs else np.zeros((0, len(HORIZONS)))
    return data.P[t][:, None] + out * data.scale


def build_network(kind, cfg, data):
    """kind: hqnn / ctrl_mlp / ctrl_bottleneck (src/models/quantum.py) or patchtst / gru_uni (src/models/nf_models.py)."""
    if kind in ("hqnn", "ctrl_mlp", "ctrl_bottleneck"):
        from models import quantum as Q
        middle = {"hqnn": "quantum", "ctrl_mlp": "mlp", "ctrl_bottleneck": "bottleneck"}[kind]
        model, p = Q.build({**cfg, "middle": middle}, R.INPUT_LEN * data.n_in)
        return model, p
    from models import nf_models as NF
    model, n = NF.build({"patchtst": "patchtst", "gru_uni": "gru"}[kind], cfg)
    return model, {"n_params": n}


def train_network(kind, cfg, data, train_origins, log=print):
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    from models import nf_models as NF
    device = NF.pick_device(cfg.get("device", "cpu"))
    model, pinfo = build_network(kind, cfg, data)
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]),
                           weight_decay=float(cfg.get("weight_decay", 0.0)))
    tr = np.asarray(train_origins)
    va = data.origins("val")
    Ytr, Mtr = data.targets(tr, "train")
    bs = int(cfg["batch"])
    n_full = len(data.origins("train"))
    steps = int(math.ceil(n_full / bs))
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    t_start = time.time()
    for ep in range(1, int(cfg["epochs"]) + 1):
        t_ep = time.time()
        model.train()
        need = steps * bs
        idx = np.concatenate([rng.permutation(len(tr)) for _ in range(int(math.ceil(need / len(tr))))])
        tot, cnt = 0.0, 0.0
        for i in range(steps):
            b = idx[i * bs:(i + 1) * bs]
            if len(b) == 0:
                break
            x = data.windows(tr[b]).to(device)
            y, mk = Ytr[b].to(device), Mtr[b].to(device)
            out = model(x)
            loss = (((out - y) ** 2) * mk).sum() / mk.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            tot += loss.item() * float(mk.sum())
            cnt += float(mk.sum())
        pv = _predict(model, data, va, device)
        maes = []
        for j, h in enumerate(HORIZONS):
            sel = np.isin(va, data.idx[h]["val"])
            maes.append(float(np.mean(np.abs(pv[sel, j] - data.P[va[sel] + h]))))
        vm = float(np.mean(maes))
        hist.append({"epoch": ep, "train_loss": tot / max(cnt, 1.0), "val_MAE_mean": vm,
                     **{f"val_MAE_h{h}": m for h, m in zip(HORIZONS, maes)}, "epoch_s": time.time() - t_ep})
        log(f"  epoch {ep:>3} | loss {tot / max(cnt, 1):.5f} | val MAE " + " / ".join(f"{m:.4f}" for m in maes))
        if vm < best - 1e-6:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    info = {**pinfo, "train_s": time.time() - t_start, "epochs_run": len(hist),
            "best_epoch": int(np.argmin([h_["val_MAE_mean"] for h_ in hist]) + 1), "best_val_MAE_mean": best,
            "n_train_origins": int(len(tr)), "steps_per_epoch": steps, "device": str(device)}
    info["cap_hit"] = bool(info["best_epoch"] == int(cfg["epochs"]))
    return model, hist, info, device


# ----------------------------------------------------------------------------------------------------------------
# scoring and saving (same layout as all other runs)
# ----------------------------------------------------------------------------------------------------------------
def score_save(root, md, meta, data, group, run, cfg, info, preds, tracking_mode="auto", extra_files=None,
               variant_row=None):
    """preds: {h: yhat on data.idx[h]['val']}. Writes config.yaml, predictions_val.parquet, metrics.json (last)."""
    import polars as pl
    import yaml
    from evaluation import metrics as M
    from tracking import log_run
    from models.quantum import log_variant
    out = Path(root) / "experiments" / group / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    L24 = md["L24_kW"].to_numpy().astype(float)
    mod5 = md["mod5"].to_numpy().astype(int)
    ep = md["minute_epoch"].to_numpy()
    rows, metrics, parts = [], {}, []
    for h in HORIZONS:
        t = data.idx[h]["val"]
        y, p0 = data.P[t + h], data.P[t]
        ramp = md[f"ramp_h{h}"].to_numpy().astype(float)[t]
        r = M.evaluate(y, preds[h], p0, l24=L24[t], ramp=ramp, thr=meta["thresholds"][str(h)]["rel_p95"],
                       mod5_target=mod5[t + h] if h == 1 else None, dm_lag=60)
        rows.append({"h": h, **r})
        metrics.update({f"h{h}/{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        parts.append(pl.DataFrame({"minute_epoch": ep[t], "h": np.full(len(t), h, np.int16), "y": y, "p0": p0,
                                   "yhat": preds[h]}))
    pl.concat(parts).write_parquet(out / "predictions_val.parquet")
    metrics.update({k: v for k, v in info.items() if isinstance(v, (int, float, bool))})
    res = {"info": info, "val": rows, "metrics": metrics}
    log_run(root, run, group, {**cfg, **{f"info_{k}": v for k, v in info.items()}}, metrics,
            files=[out / "config.yaml"], tables={"val_metrics": pl.DataFrame(rows)}, mode=tracking_mode)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    log_variant(root, {"group": group, "run": run, **(variant_row or {}),
                       **{f"val_MAE_h{r_['h']}": round(r_["MAE"], 5) for r_ in rows},
                       **{f"val_skill_h{r_['h']}": round(r_["skill_MAE"], 4) for r_ in rows}})
    return res


def done(root, group, run):
    p = Path(root) / "experiments" / group / run / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_network_fraction(root, md, meta, data, kind, cfg, frac, seed, group, tracking_mode="auto", log=print,
                         tag=None):
    import yaml  # noqa: F401
    run = f"{tag or kind}_{frac_tag(frac)}_s{seed}"
    prev = done(root, group, run)
    if prev:
        log(f"[{time.strftime('%H:%M:%S')}] {run}: done earlier, skipped")
        return prev
    cut = fraction_cutoff(data, frac)
    tr = data.origins("train")
    tr = tr[tr >= cut]
    cfg = {**cfg, "kind": kind, "seed": seed, "fraction": frac, "cutoff_index": cut, "name": run, "group": group}
    log(f"[{time.strftime('%H:%M:%S')}] {run}: {len(tr):,} train origins")
    model, hist, info, device = train_network(kind, cfg, data, tr, log=lambda s: None)
    out = Path(root) / "experiments" / group / run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    import polars as pl
    pl.DataFrame(hist).write_csv(out / "history.csv")
    pv = _predict(model, data, data.origins("val"), device)
    va = data.origins("val")
    preds = {h: pv[np.searchsorted(va, data.idx[h]["val"]), j] for j, h in enumerate(HORIZONS)}
    res = score_save(root, md, meta, data, group, run, cfg, info, preds, tracking_mode,
                     variant_row={"stage": "data_fraction", "variant": tag or kind, "fraction": frac, "seed": seed,
                                  "n_params": info["n_params"], "best_epoch": info["best_epoch"]})
    log(f"[{time.strftime('%H:%M:%S')}] {run}: params {info['n_params']:,} | best epoch {info['best_epoch']}/"
        f"{info['epochs_run']}{' (CAP HIT)' if info['cap_hit'] else ''} | {info['train_s']:.0f} s | skill " +
        " / ".join(f"{r_['skill_MAE']:+.3f}" for r_ in res["val"]))
    return res


# ----------------------------------------------------------------------------------------------------------------
# LightGBM (tuned parameters of notebook 08, rounds by early stopping on validation)
# ----------------------------------------------------------------------------------------------------------------
def run_lightgbm_fraction(root, md, meta, data, frac, seed, group, tracking_mode="auto", log=print,
                          max_rounds=3000, patience=50):
    import lightgbm as lgb
    from models import tabular_features as TF
    run = f"lightgbm_{frac_tag(frac)}_s{seed}"
    prev = done(root, group, run)
    if prev:
        log(f"[{time.strftime('%H:%M:%S')}] {run}: done earlier, skipped")
        return prev
    clas = json.loads((Path(root) / "reports" / "classical" / "classical_summary.json").read_text())
    A = TF.arrays(md)
    cut = fraction_cutoff(data, frac)
    preds, rounds, t0 = {}, {}, time.time()
    for h in HORIZONS:
        tr = data.idx[h]["train"]
        tr = tr[tr >= cut]
        va = data.idx[h]["val"]
        tune = clas["lightgbm"][f"{h}|gbm"]
        Xtr, names = TF.features(A, tr, "gbm")
        Xva, _ = TF.features(A, va, "gbm")
        p = {"metric": "l1", "verbosity": -1, "num_threads": 0, "bagging_freq": 1, "seed": seed, "bagging_seed": seed,
             "feature_fraction_seed": seed, "deterministic": True, "force_row_wise": True, **tune["params"]}
        p["min_data_in_leaf"] = int(min(p.get("min_data_in_leaf", 20), max(5, len(tr) // 20)))
        dtr = lgb.Dataset(Xtr, data.P[tr + h] - data.P[tr], feature_name=names, params={"feature_pre_filter": False})
        dva = lgb.Dataset(Xva, data.P[va + h] - data.P[va], reference=dtr)
        b = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(patience, verbose=False)])
        rounds[h] = int(b.best_iteration or max_rounds)
        preds[h] = data.P[va] + b.predict(Xva, num_iteration=rounds[h])
    cfg = {"kind": "lightgbm", "seed": seed, "fraction": frac, "cutoff_index": cut, "name": run, "group": group,
           "params": "notebook 08 tuned (gbm arm); min_data_in_leaf capped at n_train/20",
           "rounds": "early stopping on validation", "max_rounds": max_rounds, "patience": patience}
    info = {"n_params": None, "train_s": time.time() - t0, **{f"rounds_h{h}": rounds[h] for h in HORIZONS},
            "n_train_origins": int(np.sum(data.idx[1]["train"] >= cut))}
    res = score_save(root, md, meta, data, group, run, cfg, info, preds, tracking_mode,
                     variant_row={"stage": "data_fraction", "variant": "lightgbm", "fraction": frac, "seed": seed})
    log(f"[{time.strftime('%H:%M:%S')}] {run}: rounds {rounds} | skill " +
        " / ".join(f"{r_['skill_MAE']:+.3f}" for r_ in res["val"]))
    return res


# ----------------------------------------------------------------------------------------------------------------
# QRC and controls (cached reservoir, readout on the fraction)
# ----------------------------------------------------------------------------------------------------------------
def run_qrc_fraction(root, md, meta, data, U, kind, scale, seed, frac, group, depth=4, tracking_mode="auto",
                     log=print):
    from models import qrc as QR
    n = U.shape[1]
    base = QR.variant_name(kind, n, depth)
    run = f"{base}_{frac_tag(frac)}_s{seed}"
    prev = done(root, group, run)
    if prev:
        log(f"[{time.strftime('%H:%M:%S')}] {run}: done earlier, skipped")
        return prev
    Phi = QR.reservoir_cached(Path(root) / "experiments" / "_cache_qrc", kind, U, depth, scale, seed)
    cut = fraction_cutoff(data, frac)
    info, preds, fits, _ = QR.run_variant(data, md, U, kind, scale, seed, depth=depth, Phi=Phi, min_origin=cut)
    cfg = {"kind": kind, "variant": base, "n_qubits": n, "depth": depth, "scale": scale, "seed": seed,
           "fraction": frac, "cutoff_index": cut, "name": run, "group": group}
    res = score_save(root, md, meta, data, group, run, cfg, info, preds, tracking_mode,
                     variant_row={"stage": "data_fraction" if frac < 1 else "qubit_sweep", "variant": base,
                                  "fraction": frac, "seed": seed, "n_qubits": n, "n_params": info["n_params"]})
    out = Path(root) / "experiments" / group / run
    np.savez(out / "readout.npz", **{f"{k}_h{h}": np.asarray(fits[h][k]) for h in HORIZONS
                                    for k in ("coef", "mean", "std", "intercept", "ridge")})
    log(f"[{time.strftime('%H:%M:%S')}] {run}: skill " + " / ".join(f"{r_['skill_MAE']:+.3f}" for r_ in res["val"]))
    return res

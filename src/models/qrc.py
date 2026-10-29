"""qrc.py: quantum reservoir computing (QRC) and its controls (Week 5).

Design (follows the QRC load-forecasting template, arXiv 2606.12806, adapted to our protocol):

1. Per-minute input (causal, independent of the forecast origin, so every reservoir state is computed ONCE per minute
   and cached): u(t) = tanh(x(t)), x(t) = [dP(t), dP(t - 1), ..., dP(t - n + 1)] / s, where dP is the 1-minute change
   of power (0 when either end is invalid) and s = std of valid 1-minute changes on train (the same s as the local
   scaling of the networks). tanh bounds the heavy-tailed changes.
2. Reservoir map phi(u) in R^F, F = 5n:
   "qrc"        fixed random brickwork circuit on n qubits: RY(a * u_i) on qubit i, then `depth` layers of fixed random
                Rot gates + CZ on even / odd neighbour pairs (alternating); readout <X_i>, <Y_i>, <Z_i>, <Z_i Z_i+1>,
                <X_i X_i+1> (ring). Exact statevector expectation values (no shots, no noise).
   "qrc_noent"  the same circuit without CZ gates (entanglement ablation; a product state).
   "rf"         classical random-feature reservoir of the same size: cos(a * W u + b), W ~ N(0, 1/n), b ~ U(0, 2 pi).
   "raw"        no reservoir: phi(u) = u (Elastic Net on the raw inputs with the same aggregation and readout).
3. Window aggregation over the 60 input minutes of origin t (exponential kernels, normalised weights):
   F_lam(t) = sum_k lam^k phi(t - k) / sum_k lam^k, k = 0..59, lam in LAMBDAS (lam = 0 is the last state).
4. Readout: ridge regression per horizon on standardised features, target (P(t + h) - P(t)) / s, penalty chosen on
   validation MAE (kW) from a fixed grid (equal budget for all reservoir types). Elastic Net (the template's readout)
   was tried first and did not converge on the collinear kernel aggregates (see fit_readout).

Only the readout is trained. The circuit is small enough to be simulated exactly on a laptop; no claim of
computational quantum advantage is made.
"""
import math
import time

import numpy as np

from models import rnn as R

HORIZONS = R.HORIZONS
LAMBDAS = (0.0, 0.5, 0.8, 0.95)
RIDGE_GRID = tuple(float(v) for v in np.logspace(-7, 1, 17))   # penalty / n_train


# ----------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------
def minute_inputs(data, md, n):
    """(n_minutes, n) array u(t) = tanh([dP(t), ..., dP(t - n + 1)] / s), zeros where undefined."""
    P = data.P
    valid = md["valid"].to_numpy()
    dP = np.zeros(len(P))
    ok = valid[1:] & valid[:-1]
    dP[1:] = np.where(ok, P[1:] - P[:-1], 0.0)
    x = np.zeros((len(P), n))
    for j in range(n):
        x[j:, j] = dP[:len(P) - j]
    return np.tanh(x / data.stats["std_dP_train"]).astype(np.float64)


# ----------------------------------------------------------------------------------------------------------------
# reservoirs
# ----------------------------------------------------------------------------------------------------------------
def pairs(n):
    return [(i, (i + 1) % n) for i in range(n)] if n > 2 else ([(0, 1)] if n == 2 else [])


def n_features(kind, n):
    return n if kind == "raw" else 3 * n + 2 * len(pairs(n))


def circuit_angles(n, depth, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(depth, n, 3))


def quantum_features(U, depth=4, scale=math.pi / 2, seed=0, entangle=True, batch=8192, log=None):
    """Exact expectation values of the fixed random circuit for every row of U (n_rows, n). Returns float32."""
    import pennylane as qml
    import torch
    n = U.shape[1]
    ang = circuit_angles(n, depth, seed)
    dev = qml.device("default.qubit", wires=n)
    prs = pairs(n)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circ(u):
        for i in range(n):
            qml.RY(scale * u[:, i], wires=i)
        for d in range(depth):
            for i in range(n):
                qml.Rot(*ang[d, i], wires=i)
            if entangle and n > 1:
                for i in range(d % 2, n - 1, 2):
                    qml.CZ(wires=[i, i + 1])
        obs = ([qml.expval(qml.PauliX(i)) for i in range(n)] + [qml.expval(qml.PauliY(i)) for i in range(n)] +
               [qml.expval(qml.PauliZ(i)) for i in range(n)] +
               [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in prs] +
               [qml.expval(qml.PauliX(a) @ qml.PauliX(b)) for a, b in prs])
        return obs

    out = np.empty((len(U), n_features("qrc", n)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(U), batch):
            ub = torch.as_tensor(U[i:i + batch], dtype=torch.float64)
            res = circ(ub)
            out[i:i + len(ub)] = torch.stack([r.reshape(-1) for r in res], dim=1).numpy()
            if log and (i // batch) % 10 == 0:
                log(f"    reservoir rows {i + len(ub):,}/{len(U):,} | {time.time() - t0:.0f} s")
    return out


def rf_features(U, n_out, scale=math.pi / 2, seed=0):
    rng = np.random.default_rng(10_000 + seed)
    n = U.shape[1]
    W = rng.normal(0.0, 1.0 / math.sqrt(n), size=(n, n_out))
    b = rng.uniform(0.0, 2 * np.pi, size=n_out)
    return np.cos(scale * (U @ W) + b).astype(np.float32)


def reservoir(kind, U, depth=4, scale=math.pi / 2, seed=0, log=None):
    n = U.shape[1]
    if kind == "qrc":
        return quantum_features(U, depth, scale, seed, entangle=True, log=log)
    if kind == "qrc_noent":
        return quantum_features(U, depth, scale, seed, entangle=False, log=log)
    if kind == "rf":
        return rf_features(U, n_features("qrc", n), scale, seed)
    if kind == "raw":
        return U.astype(np.float32)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------------------------
# aggregation and readout
# ----------------------------------------------------------------------------------------------------------------
def aggregate(Phi, t, lambdas=LAMBDAS, L=R.INPUT_LEN):
    """(len(t), F * len(lambdas)) window features for origins t (all t >= L - 1)."""
    t = np.asarray(t)
    F = Phi.shape[1]
    out = np.zeros((len(t), F * len(lambdas)), dtype=np.float32)
    for j, lam in enumerate(lambdas):
        w = np.array([lam ** k if lam > 0 else float(k == 0) for k in range(L)], dtype=np.float64)
        w /= w.sum()
        acc = np.zeros((len(t), F), dtype=np.float64)
        for k in range(L):
            if w[k] < 1e-8:
                continue
            acc += w[k] * Phi[t - k]
        out[:, j * F:(j + 1) * F] = acc
    return out


def targets(data, t, h):
    return (data.P[t + h] - data.P[t]) / data.scale


def fit_readout(Xtr, ytr, Xva, yva_kw, p0_va, scale, ridge_grid=RIDGE_GRID):
    """Ridge readout on standardised features, closed form for the whole penalty grid from one eigendecomposition of
    the Gram matrix. Penalty lam = r * n_train (r in ridge_grid), chosen by validation MAE in kW.
    (Elastic Net was the first choice; coordinate descent did not converge on the strongly collinear kernel
    aggregates, so the readout is ridge for every reservoir type: provisional decision, same budget for all.)"""
    mu = Xtr.mean(0, dtype=np.float64)
    sd = Xtr.std(0, dtype=np.float64)
    sd[sd < 1e-12] = 1.0
    Z = (Xtr - mu) / sd
    ym = float(ytr.mean())
    G = Z.T @ Z
    zy = Z.T @ (ytr - ym)
    e, V = np.linalg.eigh(G)
    e = np.clip(e, 0.0, None)
    vzy = V.T @ zy
    Zva = (Xva - mu) / sd
    n = len(ytr)
    grid, best = [], None
    for r in ridge_grid:
        w = V @ (vzy / (e + r * n))
        pred = p0_va + (Zva @ w + ym) * scale
        m = float(np.mean(np.abs(pred - yva_kw)))
        grid.append({"ridge": r, "val_MAE": m})
        if best is None or m < best["val_MAE"]:
            best = {"ridge": r, "val_MAE": m, "coef": w}
    best.update({"intercept": ym, "mean": mu, "std": sd, "grid": grid, "n_coef": int(len(best["coef"])),
                 "at_grid_edge": bool(best["ridge"] in (min(ridge_grid), max(ridge_grid)))})
    return best


def predict_readout(fit, X, p0, scale):
    return p0 + (((X - fit["mean"]) / fit["std"]) @ fit["coef"] + fit["intercept"]) * scale


def variant_name(kind, n, depth=4):
    return {"qrc": f"qrc_q{n}_d{depth}", "qrc_noent": f"qrc_noent_q{n}_d{depth}",
            "rf": f"ctrl_rf_q{n}_f{n_features('qrc', n)}", "raw": f"ctrl_raw_q{n}"}[kind]


def run_variant(data, md, U, kind, scale, seed, depth=4, log=print, Phi=None, min_origin=None):
    """Reservoir + aggregation + readout for one variant. Returns (result dict, predictions per horizon on val).
    min_origin: if given, the readout is fitted only on train origins >= min_origin (data-fraction experiment)."""
    t0 = time.time()
    if Phi is None:
        Phi = reservoir(kind, U, depth=depth, scale=scale, seed=seed, log=log)
    t_res = time.time() - t0
    tr_all = data.origins("train")
    va_all = data.origins("val")
    t1 = time.time()
    Xtr_all = aggregate(Phi, tr_all)
    Xva_all = aggregate(Phi, va_all)
    t_agg = time.time() - t1
    fits, preds = {}, {}
    t2 = time.time()
    for h in HORIZONS:
        ttr, tva = data.idx[h]["train"], data.idx[h]["val"]
        if min_origin is not None:
            ttr = ttr[ttr >= min_origin]
        Xtr = Xtr_all[np.searchsorted(tr_all, ttr)]
        Xva = Xva_all[np.searchsorted(va_all, tva)]
        fit = fit_readout(Xtr, targets(data, ttr, h), Xva, data.P[tva + h], data.P[tva], data.scale)
        fits[h] = fit
        preds[h] = predict_readout(fit, Xva, data.P[tva], data.scale)
    t_fit = time.time() - t2
    info = {"kind": kind, "scale": scale, "seed": seed, "depth": depth, "n_inputs": U.shape[1],
            "n_reservoir_features": int(Phi.shape[1]), "n_window_features": int(Phi.shape[1] * len(LAMBDAS)),
            "n_params": int(sum(f["n_coef"] + 1 for f in fits.values())),
            "n_train_origins_h1": int(np.sum(data.idx[1]["train"] >= (min_origin or 0))),
            "ridge_at_grid_edge": bool(any(f["at_grid_edge"] for f in fits.values())),
            "reservoir_s": t_res, "aggregate_s": t_agg, "readout_s": t_fit,
            **{f"ridge_h{h}": fits[h]["ridge"] for h in HORIZONS},
            **{f"val_MAE_h{h}": fits[h]["val_MAE"] for h in HORIZONS}}
    info["val_MAE_mean"] = float(np.mean([fits[h]["val_MAE"] for h in HORIZONS]))
    return info, preds, fits, Phi


def scale_tag(scale):
    return f"a{scale:.3f}".replace(".", "p")


def run_and_score(root, md, meta, data, U, kind, scale, seed, group, depth=4, tracking_mode="auto", log=print,
                  screen=False):
    """One QRC / control variant: reservoir -> aggregation -> readout -> validation metrics; same file layout as the
    networks (config.yaml, predictions_val.parquet, metrics.json written last; tracking record; variant log)."""
    import json
    from pathlib import Path
    import polars as pl
    import yaml
    from evaluation import metrics as M
    from tracking import log_run
    from models.quantum import log_variant

    root = Path(root)
    n = U.shape[1]
    base = variant_name(kind, n, depth)
    run = f"{base}_{scale_tag(scale)}_s{seed}" if kind != "raw" else f"{base}_s{seed}"
    out = root / "experiments" / group / run
    if (out / "metrics.json").exists():
        log(f"[{time.strftime('%H:%M:%S')}] {run}: done earlier, skipped")
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": kind, "variant": base, "n_qubits": n, "depth": depth, "scale": scale, "seed": seed,
           "lambdas": list(LAMBDAS), "ridge_grid": list(RIDGE_GRID), "group": group, "name": run, "screen": screen,
           "input": "tanh of the last n 1-minute changes / s, per minute", "readout": "ridge per horizon"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log(f"[{time.strftime('%H:%M:%S')}] {run}: reservoir + readout")
    info, preds, fits, _ = run_variant(data, md, U, kind, scale, seed, depth=depth, log=None)
    L24 = md["L24_kW"].to_numpy().astype(float)
    mod5 = md["mod5"].to_numpy().astype(int)
    ep = md["minute_epoch"].to_numpy()
    rows, metrics, parts = [], {}, []
    for h in HORIZONS:
        t = data.idx[h]["val"]
        yhat = preds[h]
        y, p0 = data.P[t + h], data.P[t]
        ramp = md[f"ramp_h{h}"].to_numpy().astype(float)[t]
        r = M.evaluate(y, yhat, p0, l24=L24[t], ramp=ramp, thr=meta["thresholds"][str(h)]["rel_p95"],
                       mod5_target=mod5[t + h] if h == 1 else None, dm_lag=60)
        rows.append({"h": h, **r})
        metrics.update({f"h{h}/{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        parts.append(pl.DataFrame({"minute_epoch": ep[t], "h": np.full(len(t), h, np.int16), "y": y, "p0": p0,
                                   "yhat": yhat}))
    pl.concat(parts).write_parquet(out / "predictions_val.parquet")
    np.savez(out / "readout.npz", **{f"{k}_h{h}": np.asarray(fits[h][k]) for h in HORIZONS
                                    for k in ("coef", "mean", "std", "intercept", "ridge")})
    metrics.update({k: v for k, v in info.items() if isinstance(v, (int, float, bool))})
    res = {"info": info, "val": rows, "metrics": metrics, "ridge_grid": {h: fits[h]["grid"] for h in HORIZONS}}
    log_run(root, run, group, {**cfg, **{f"info_{k}": v for k, v in info.items()}}, metrics,
            files=[out / "config.yaml"], tables={"val_metrics": pl.DataFrame(rows)}, mode=tracking_mode)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    log_variant(root, {"stage": "qrc_screen" if screen else "qrc_validation", "group": group, "run": run,
                       "variant": base, "kind": kind, "n_qubits": n, "depth": depth, "scale": round(scale, 4),
                       "seed": seed, "n_params": info["n_params"],
                       **{f"val_MAE_h{r_['h']}": round(r_["MAE"], 5) for r_ in rows},
                       **{f"val_skill_h{r_['h']}": round(r_["skill_MAE"], 4) for r_ in rows}})
    log(f"[{time.strftime('%H:%M:%S')}] {run}: features {info['n_window_features']} | readout params {info['n_params']}"
        f" | reservoir {info['reservoir_s']:.0f} s | val MAE " + " / ".join(f"{r_['MAE']:.4f}" for r_ in rows) +
        " kW | skill " + " / ".join(f"{r_['skill_MAE']:+.3f}" for r_ in rows))
    return res


def predict_split(fits, Phi, data, split):
    """Forecasts (kW) of a fitted readout on the usable origins of `split`: {h: yhat}."""
    out = {}
    for h in HORIZONS:
        t = data.idx[h][split]
        out[h] = predict_readout(fits[h], aggregate(Phi, t), data.P[t], data.scale)
    return out


def load_readout(path):
    """Readout saved by run_and_score (readout.npz) -> {h: fit dict}."""
    z = np.load(path)
    return {h: {k: z[f"{k}_h{h}"] for k in ("coef", "mean", "std", "intercept", "ridge")} for h in HORIZONS}


def reservoir_cached(cache_dir, kind, U, depth, scale, seed, log=None):
    """Reservoir states for every minute, cached as .npy (they depend only on the inputs, kind, scale and seed)."""
    from pathlib import Path
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = U.shape[1]
    f = cache_dir / f"{variant_name(kind, n, depth)}_{scale_tag(scale)}_s{seed}_rows{len(U)}.npy"
    if f.exists():
        return np.load(f)
    Phi = reservoir(kind, U, depth=depth, scale=scale, seed=seed, log=log)
    np.save(f, Phi)
    return Phi


def quantum_features_noisy(U, depth=4, scale=math.pi / 2, seed=0, entangle=True, p1=0.0, p2=0.0, p_ro=0.0,
                           shots=None, batch=512, rng_seed=0, log=None):
    """Reservoir states with noise (inference only). p1 after RY and Rot, p2 on both qubits after every CZ
    (density matrix, default.mixed); readout flip p_ro applied analytically ((1 - 2p) per measured qubit:
    single-qubit observables x (1 - 2p), two-qubit correlators x (1 - 2p)^2); shots: binomial sampling of each
    observable with `shots` shots (each observable measured separately: a simplification)."""
    import pennylane as qml
    import torch
    n = U.shape[1]
    ang = circuit_angles(n, depth, seed)
    prs = pairs(n)
    noisy = (p1 > 0) or (p2 > 0)
    dev = qml.device("default.mixed" if noisy else "default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circ(u):
        for i in range(n):
            qml.RY(scale * u[:, i], wires=i)
            if p1 > 0:
                qml.DepolarizingChannel(p1, wires=i)
        for d in range(depth):
            for i in range(n):
                qml.Rot(*ang[d, i], wires=i)
                if p1 > 0:
                    qml.DepolarizingChannel(p1, wires=i)
            if entangle and n > 1:
                for i in range(d % 2, n - 1, 2):
                    qml.CZ(wires=[i, i + 1])
                    if p2 > 0:
                        qml.DepolarizingChannel(p2, wires=i)
                        qml.DepolarizingChannel(p2, wires=i + 1)
        return ([qml.expval(qml.PauliX(i)) for i in range(n)] + [qml.expval(qml.PauliY(i)) for i in range(n)] +
                [qml.expval(qml.PauliZ(i)) for i in range(n)] +
                [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in prs] +
                [qml.expval(qml.PauliX(a) @ qml.PauliX(b)) for a, b in prs])

    F = n_features("qrc", n)
    out = np.empty((len(U), F), dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(U), batch):
            ub = torch.as_tensor(U[i:i + batch], dtype=torch.float64)
            res = circ(ub)
            out[i:i + len(ub)] = torch.stack([r.reshape(-1) for r in res], dim=1).numpy()
            if log and (i // batch) % 20 == 0:
                log(f"    noisy reservoir rows {i + len(ub):,}/{len(U):,} | {time.time() - t0:.0f} s")
    if p_ro > 0:
        f1 = 1.0 - 2.0 * p_ro
        out[:, :3 * n] *= f1
        out[:, 3 * n:] *= f1 ** 2
    if shots:
        rng = np.random.default_rng(rng_seed)
        pr = np.clip((1.0 + out) / 2.0, 0.0, 1.0)
        out = 2.0 * rng.binomial(int(shots), pr) / float(shots) - 1.0
    return out.astype(np.float32)

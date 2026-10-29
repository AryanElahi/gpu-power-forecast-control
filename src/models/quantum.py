"""quantum.py: hybrid quantum-classical forecasters and their classical controls (Week 5).

All models share the protocol of every other network in the benchmark (src/models/rnn.py, src/models/nf_models.py):
the same usable origins, 60-minute univariate input window, local scaling (P(t - k) - P(t)) / s, direct output of the
scaled changes for h = 1, 5, 15, masked MSE, Adam, early stopping on the mean validation MAE in kW.

Architecture ("dressed" hybrid network, Kaseb et al. style; called "hybrid QNN" in the paper):

    x (B, 60) --Linear--> z (B, n) --tanh, * a (trainable, init pi/2)--> angles in (-a, a)
              --> middle block --> m (B, n) in [-1, 1] --Linear--> (B, 3) scaled changes

Middle blocks (the ONLY part that differs between a hybrid model and its controls):
    "quantum"    : variational circuit on n qubits, L layers. Each layer: (optional) angle embedding of the angles
                   (data re-uploading if reupload=True, otherwise only before the first layer), then one
                   Rot(phi, theta, omega) per qubit, then (if entangle=True) a CNOT ring with range
                   r = (l mod (n - 1)) + 1, i.e. exactly qml.StronglyEntanglingLayers. Readout <Z_i> per qubit.
                   entangle=False is the entanglement ablation (same parameters, no CNOTs: a product state).
    "mlp"        : classical control, n -> k -> n with tanh, k chosen so that its parameter count is as close as
                   possible to the circuit's 3 n L (same bounded output range [-1, 1] as <Z>).
    "bottleneck" : classical control without any middle block (cos of the angles, which is exactly what the
                   circuit returns with no trainable gates and no entanglement); shows what the encoder/decoder alone
                   can do.

Simulation is exact (statevector, no shots, no noise) and classically tractable for n <= 8 qubits; no claim of
computational quantum advantage is made. Backends: "default.qubit" with backprop and native parameter broadcasting
(whole batch in one call) or "lightning.qubit" with adjoint differentiation (one circuit per sample). Both are exact;
which one is used is a speed choice made in notebook 17.
"""
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import rnn as R

HORIZONS = R.HORIZONS


# ----------------------------------------------------------------------------------------------------------------
# middle blocks
# ----------------------------------------------------------------------------------------------------------------
def circuit_param_count(n_qubits, n_layers):
    return 3 * n_qubits * n_layers


def matched_hidden(n_qubits, n_layers):
    """k for the n -> k -> n MLP whose parameter count k(2n + 1) + n is closest to the circuit's 3 n L."""
    target = circuit_param_count(n_qubits, n_layers)
    best = min(range(1, 4 * target + 2), key=lambda k: (abs(k * (2 * n_qubits + 1) + n_qubits - target), k))
    return best


def ring_range(layer, n_qubits):
    return (layer % (n_qubits - 1)) + 1 if n_qubits > 1 else 0


class QuantumBlock(nn.Module):
    """Variational circuit as a torch module: angles (B, n) -> <Z_i> (B, n), float32."""

    def __init__(self, n_qubits, n_layers, reupload=True, entangle=True, backend="default.qubit",
                 diff_method=None, init_scale=math.pi):
        super().__init__()
        import pennylane as qml
        self.n, self.L = int(n_qubits), int(n_layers)
        self.reupload, self.entangle, self.backend = bool(reupload), bool(entangle), backend
        self.weights = nn.Parameter(init_scale * (2 * torch.rand(self.L, self.n, 3) - 1))   # U(-pi, pi)
        if diff_method is None:
            diff_method = "backprop" if backend == "default.qubit" else "adjoint"
        self.diff_method = diff_method
        dev = qml.device(backend, wires=self.n)
        n, L, reup, ent = self.n, self.L, self.reupload, self.entangle

        def circuit(angles, w):
            for l in range(L):
                if l == 0 or reup:
                    qml.AngleEmbedding(angles, wires=range(n))
                for i in range(n):
                    qml.Rot(w[l, i, 0], w[l, i, 1], w[l, i, 2], wires=i)
                if ent and n > 1:
                    r = ring_range(l, n)
                    for i in range(n):
                        qml.CNOT(wires=[i, (i + r) % n])
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        self.qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)

    def forward(self, angles):
        if self.backend == "default.qubit":
            out = self.qnode(angles, self.weights)                      # broadcast over the batch
            return torch.stack(list(out), dim=-1).to(torch.float32)
        rows = [torch.stack(list(self.qnode(a, self.weights))) for a in angles]   # one circuit per sample
        return torch.stack(rows).to(torch.float32)


class MLPBlock(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, k), nn.Tanh(), nn.Linear(k, n), nn.Tanh())

    def forward(self, a):
        return self.net(a)


class CosBlock(nn.Module):
    """No trainable middle: cos(angle) = <Z> of an angle-embedded qubit without gates (bottleneck control)."""

    def forward(self, a):
        return torch.cos(a)


class HybridForecaster(nn.Module):
    def __init__(self, n_in, n_qubits=4, n_layers=2, middle="quantum", reupload=True, entangle=True,
                 backend="default.qubit", diff_method=None, input_scale_init=math.pi / 2, train_input_scale=True,
                 control_hidden=None, n_out=len(HORIZONS)):
        super().__init__()
        n = int(n_qubits)
        self.encoder = nn.Linear(n_in, n)
        scale = torch.full((n,), float(input_scale_init))
        self.in_scale = nn.Parameter(scale) if train_input_scale else nn.Parameter(scale, requires_grad=False)
        if middle == "quantum":
            self.middle = QuantumBlock(n, n_layers, reupload, entangle, backend, diff_method)
        elif middle == "mlp":
            self.middle = MLPBlock(n, int(control_hidden or matched_hidden(n, n_layers)))
        elif middle == "bottleneck":
            self.middle = CosBlock()
        else:
            raise ValueError(middle)
        self.middle_kind = middle
        self.decoder = nn.Linear(n, n_out)

    def angles(self, x):
        return self.in_scale * torch.tanh(self.encoder(x.reshape(x.shape[0], -1)))

    def forward(self, x):
        return self.decoder(self.middle(self.angles(x)))


def param_breakdown(model):
    cnt = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
    enc = cnt(model.encoder) + (model.in_scale.numel() if model.in_scale.requires_grad else 0)
    mid = cnt(model.middle)
    dec = cnt(model.decoder)
    return {"n_params": enc + mid + dec, "n_params_encoder": enc, "n_params_middle": mid, "n_params_decoder": dec}


def build(cfg, n_in):
    model = HybridForecaster(
        n_in, n_qubits=int(cfg.get("n_qubits", 4)), n_layers=int(cfg.get("n_layers", 2)),
        middle=cfg.get("middle", "quantum"), reupload=bool(cfg.get("reupload", True)),
        entangle=bool(cfg.get("entangle", True)), backend=cfg.get("backend", "default.qubit"),
        diff_method=cfg.get("diff_method"), input_scale_init=float(cfg.get("input_scale_init", math.pi / 2)),
        train_input_scale=bool(cfg.get("train_input_scale", True)), control_hidden=cfg.get("control_hidden"))
    return model, param_breakdown(model)


def variant_name(cfg):
    m = cfg.get("middle", "quantum")
    base = f"q{cfg.get('n_qubits', 4)}_l{cfg.get('n_layers', 2)}"
    if m == "quantum":
        tag = "hqnn" + ("" if cfg.get("entangle", True) else "_noent") + ("" if cfg.get("reupload", True) else "_noreup")
    else:
        tag = {"mlp": "ctrl_mlp", "bottleneck": "ctrl_bottleneck"}[m]
    return f"{tag}_{base}"


# ----------------------------------------------------------------------------------------------------------------
# training (same loop as nf_models.train; only the model differs). CPU only.
# ----------------------------------------------------------------------------------------------------------------
def predict(model, data, t, batch=4096):
    """Forecasts of P(t + h) in kW, shape (len(t), 3)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(t), batch):
            outs.append(model(data.windows(t[i:i + batch])).float().numpy())
    out = np.concatenate(outs) if outs else np.zeros((0, len(HORIZONS)))
    return data.P[t][:, None] + out * data.scale


def subsample(t, n, seed):
    if not n or n >= len(t):
        return t
    return np.sort(np.random.default_rng(seed).choice(t, int(n), replace=False))


def val_maes(model, data, va, split="val"):
    """MAE in kW per horizon on origins va, using the usable origins of `split` for each horizon."""
    pv = predict(model, data, va)
    maes = []
    for j, h in enumerate(HORIZONS):
        sel = np.isin(va, data.idx[h][split])
        maes.append(float(np.mean(np.abs(pv[sel, j] - data.P[va[sel] + h]))))
    return maes


def train(md, cfg, log=print, data=None, train_origins=None, val_origins=None):
    """Train one model with the shared protocol. Returns model, data, history, info.

    cfg keys: seed, lr, batch, epochs, patience, grad_clip, weight_decay, max_train_samples, max_val_samples,
    plus the model keys of build(). train_origins / val_origins override the split origins (used by the overfit check).
    """
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 0)) or torch.get_num_threads())
    data = data or R.Data(md, {"scaling": "local", "calendar": False, "covariates": False})
    n_in = R.INPUT_LEN * data.n_in
    model, pinfo = build(cfg, n_in)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]),
                           weight_decay=float(cfg.get("weight_decay", 0.0)))
    loss_fn = nn.MSELoss(reduction="none")
    tr = train_origins if train_origins is not None else subsample(data.origins("train"), cfg.get("max_train_samples"), seed)
    va = val_origins if val_origins is not None else subsample(data.origins("val"), cfg.get("max_val_samples"), seed + 1)
    tr_split = cfg.get("train_split", "train")
    Ytr, Mtr = data.targets(tr, tr_split)
    rng = np.random.default_rng(seed)
    bs = int(cfg["batch"])
    best, best_state, bad, hist = np.inf, None, 0, []
    t_start = time.time()
    for ep in range(1, int(cfg["epochs"]) + 1):
        t_ep = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        tot, cnt = 0.0, 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = data.windows(tr[b])
            y, mk = Ytr[b], Mtr[b]
            out = model(x)
            loss = (loss_fn(out, y) * mk).sum() / mk.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            tot += loss.item() * float(mk.sum())
            cnt += float(mk.sum())
        train_s = time.time() - t_ep
        maes = val_maes(model, data, va, cfg.get("val_split", "val"))
        vm = float(np.mean(maes))
        hist.append({"epoch": ep, "train_loss": tot / max(cnt, 1.0), "val_MAE_mean": vm,
                     **{f"val_MAE_h{h}": m for h, m in zip(HORIZONS, maes)},
                     "epoch_train_s": train_s, "epoch_s": time.time() - t_ep, "s": time.time() - t_start})
        log(f"  epoch {ep:>3} | train loss {tot / max(cnt, 1):.5f} | val MAE " +
            " / ".join(f"{m:.4f}" for m in maes) + f" kW | epoch {time.time() - t_ep:.0f} s")
        if vm < best - 1e-6:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    info = {"variant": variant_name(cfg), **pinfo, "train_s": time.time() - t_start, "epochs_run": len(hist),
            "best_epoch": int(np.argmin([h_["val_MAE_mean"] for h_ in hist]) + 1), "best_val_MAE_mean": best,
            "n_train_origins": int(len(tr)), "n_val_origins": int(len(va)), "device": "cpu",
            "backend": cfg.get("backend", "default.qubit") if cfg.get("middle", "quantum") == "quantum" else "torch",
            **data.stats}
    return model, data, hist, info


# ----------------------------------------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------------------------------------
def time_steps(cfg, data, n_batches=10, bs=512, n_pred=4096, seed=0):
    """Time forward+backward on n_batches random train batches (after 1 warm-up batch) and prediction."""
    torch.manual_seed(seed)
    model, pinfo = build(cfg, R.INPUT_LEN * data.n_in)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    tr, va = data.origins("train"), data.origins("val")
    rng = np.random.default_rng(seed)
    model.train()
    t0, n_done = None, 0
    for i in range(n_batches + 1):
        if i == 1:
            t0 = time.time()
        b = np.sort(rng.choice(tr, bs, replace=False))
        y, mk = data.targets(b, "train")
        out = model(data.windows(b))
        loss = (((out - y) ** 2) * mk).sum() / mk.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i >= 1:
            n_done += bs
    train_s = time.time() - t0
    t1 = time.time()
    predict(model, data, va[:n_pred], batch=min(4096, n_pred))
    pred_s = time.time() - t1
    ms_tr, ms_pr = 1000 * train_s / n_done, 1000 * pred_s / n_pred
    return {"variant": variant_name(cfg), "backend": cfg.get("backend", "default.qubit"), **pinfo,
            "ms_per_train_sample": ms_tr, "ms_per_pred_sample": ms_pr,
            "est_epoch_min_full": (ms_tr * len(tr) + ms_pr * len(va)) / 60000}


def grad_variance(cfg, data, n_init=20, bs=256, seed=0):
    """Variance over random initialisations of d loss / d theta for the middle-block parameters (barren-plateau
    diagnostic, cf. McClean et al. 2018; Cerezo et al. 2025). Fixed batch of real train samples. Returns the mean
    over parameters of the per-parameter variance, and the same for the encoder weights."""
    rng = np.random.default_rng(seed)
    b = np.sort(rng.choice(data.origins("train"), bs, replace=False))
    x = data.windows(b)
    y, mk = data.targets(b, "train")
    g_mid, g_enc = [], []
    for k in range(n_init):
        torch.manual_seed(1000 + k)
        model, _ = build(cfg, R.INPUT_LEN * data.n_in)
        out = model(x)
        loss = (((out - y) ** 2) * mk).sum() / mk.sum().clamp(min=1.0)
        model.zero_grad()
        loss.backward()
        mids = [p.grad.flatten() for p in model.middle.parameters() if p.grad is not None]
        g_mid.append(torch.cat(mids).numpy() if mids else np.zeros(0))
        g_enc.append(model.encoder.weight.grad.flatten().numpy())
    gm, ge = np.stack(g_mid), np.stack(g_enc)
    return {"variant": variant_name(cfg), "n_init": n_init,
            "var_grad_middle": float(gm.var(axis=0).mean()) if gm.size else float("nan"),
            "var_grad_encoder": float(ge.var(axis=0).mean())}


def log_variant(root, row, path="reports/week5/quantum_variants_log.csv"):
    """Append one tried variant (any stage) to the variant log, so that every quantum variant tried is recorded."""
    p = Path(root) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    new = not p.exists()
    if not new:
        with open(p) as fh:
            header = next(csv.reader(fh), [])
        keys = header + [k for k in row if k not in header]
        if keys != header:                       # rewrite with the wider header
            with open(p) as fh:
                old = list(csv.DictReader(fh))
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(old)
    else:
        keys = list(row)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------------------------------------------
# one validation run with the shared scoring and file layout (same layout as notebook 11: config.yaml, history.csv,
# model.pt, predictions_val.parquet, metrics.json written last = run complete; tracking record via src/tracking.py)
# ----------------------------------------------------------------------------------------------------------------
def run_and_score(root, md, meta, data, cfg, group, tracking_mode="auto", log=print, epoch_log=False):
    """Train, score on validation and save one run. epoch_log=True also prints one line per epoch."""
    import json
    import polars as pl
    import yaml
    from evaluation import metrics as M
    from tracking import log_run

    root = Path(root)
    run = f"{variant_name(cfg)}_s{int(cfg['seed'])}"
    out = root / "experiments" / group / run
    if (out / "metrics.json").exists():
        log(f"[{time.strftime('%H:%M:%S')}] {run}: done earlier, skipped")
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "name": run, "group": group, "variant": variant_name(cfg)}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log(f"[{time.strftime('%H:%M:%S')}] {run}: training")
    model, d, hist, info = train(md, cfg, log=log if epoch_log else (lambda s: None), data=data)
    pl.DataFrame(hist).write_csv(out / "history.csv")
    torch.save(model.state_dict(), out / "model.pt")
    va = d.origins("val")
    t0 = time.time()
    pv = predict(model, d, va)
    infer_s = time.time() - t0
    L24 = md["L24_kW"].to_numpy().astype(float)
    mod5 = md["mod5"].to_numpy().astype(int)
    ep = md["minute_epoch"].to_numpy()
    rows, metrics, parts = [], {}, []
    for j, h in enumerate(HORIZONS):
        t = d.idx[h]["val"]
        yhat = pv[np.searchsorted(va, t), j]
        y, p0 = d.P[t + h], d.P[t]
        ramp = md[f"ramp_h{h}"].to_numpy().astype(float)[t]
        r = M.evaluate(y, yhat, p0, l24=L24[t], ramp=ramp, thr=meta["thresholds"][str(h)]["rel_p95"],
                       mod5_target=mod5[t + h] if h == 1 else None, dm_lag=60)
        rows.append({"h": h, **r})
        metrics.update({f"h{h}/{k}": v for k, v in r.items() if isinstance(v, (int, float))})
        parts.append(pl.DataFrame({"minute_epoch": ep[t], "h": np.full(len(t), h, np.int16), "y": y, "p0": p0,
                                   "yhat": yhat}))
    pl.concat(parts).write_parquet(out / "predictions_val.parquet")
    info["cap_hit"] = bool(info["best_epoch"] == int(cfg["epochs"]))
    metrics.update({k: info[k] for k in ("n_params", "n_params_encoder", "n_params_middle", "n_params_decoder",
                                         "train_s", "best_epoch", "epochs_run", "cap_hit")})
    metrics.update({"infer_s_val": infer_s, "infer_ms_per_sample": 1000 * infer_s / max(len(va), 1),
                    "mean_epoch_train_s": float(np.mean([h_["epoch_train_s"] for h_ in hist]))})
    res = {"info": info, "val": rows, "metrics": metrics}
    log_run(root, run, group, {**cfg, **{f"data_{k}": v for k, v in info.items()}}, metrics,
            files=[out / "config.yaml"], tables={"val_metrics": pl.DataFrame(rows)}, mode=tracking_mode)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=str))   # written last = run complete
    log_variant(root, {"stage": "validation_run", "group": group, "run": run, "variant": info["variant"],
                       "n_qubits": cfg.get("n_qubits"), "n_layers": cfg.get("n_layers"), "middle": cfg.get("middle"),
                       "entangle": cfg.get("entangle", True), "reupload": cfg.get("reupload", True),
                       "seed": cfg["seed"], "n_params": info["n_params"], "n_params_middle": info["n_params_middle"],
                       "best_epoch": info["best_epoch"], "train_s": round(info["train_s"], 1),
                       **{f"val_MAE_h{r_['h']}": round(r_["MAE"], 5) for r_ in rows},
                       **{f"val_skill_h{r_['h']}": round(r_["skill_MAE"], 4) for r_ in rows}})
    log(f"[{time.strftime('%H:%M:%S')}] {run}: params {info['n_params']:,} (middle {info['n_params_middle']}) | "
        f"train {info['train_s']:.0f} s | best epoch {info['best_epoch']}/{info['epochs_run']}"
        f"{' (CAP HIT)' if info['cap_hit'] else ''} | val MAE " + " / ".join(f"{r_['MAE']:.4f}" for r_ in rows) +
        " kW | skill " + " / ".join(f"{r_['skill_MAE']:+.3f}" for r_ in rows))
    return res


# ----------------------------------------------------------------------------------------------------------------
# inference-only noise (Week 6): the trained circuit weights are evaluated on a noisy simulator
# ----------------------------------------------------------------------------------------------------------------
class NoisyQuantumBlock(nn.Module):
    """Evaluates a trained QuantumBlock with noise. p1: single-qubit depolarizing probability after every embedding
    rotation and every Rot gate; p2: single-qubit depolarizing probability applied to BOTH qubits after every CNOT
    (an approximation of two-qubit depolarizing noise); p_ro: symmetric readout flip, applied analytically
    (<Z> -> (1 - 2 p_ro) <Z>, exact for independent symmetric readout errors); shots: finite sampling (None = exact).
    Density-matrix simulation (default.mixed) when p1 or p2 > 0, otherwise default.qubit."""

    def __init__(self, block, p1=0.0, p2=0.0, p_ro=0.0, shots=None, seed=0):
        super().__init__()
        import pennylane as qml
        self.n, self.L, self.p_ro = block.n, block.L, float(p_ro)
        self.register_buffer("w", block.weights.detach().clone())
        n, L, reup, ent = block.n, block.L, block.reupload, block.entangle
        noisy = (p1 > 0) or (p2 > 0)
        dev = qml.device("default.mixed" if noisy else "default.qubit", wires=n, seed=seed if shots else None)

        def circuit(angles, w):
            for l in range(L):
                if l == 0 or reup:
                    qml.AngleEmbedding(angles, wires=range(n))
                    if p1 > 0:
                        for i in range(n):
                            qml.DepolarizingChannel(p1, wires=i)
                for i in range(n):
                    qml.Rot(w[l, i, 0], w[l, i, 1], w[l, i, 2], wires=i)
                    if p1 > 0:
                        qml.DepolarizingChannel(p1, wires=i)
                if ent and n > 1:
                    r = ring_range(l, n)
                    for i in range(n):
                        qml.CNOT(wires=[i, (i + r) % n])
                        if p2 > 0:
                            qml.DepolarizingChannel(p2, wires=i)
                            qml.DepolarizingChannel(p2, wires=(i + r) % n)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        qn = qml.QNode(circuit, dev, interface="torch", diff_method=None)
        self.qnode = qml.set_shots(qn, shots=shots) if shots else qn

    def forward(self, angles):
        with torch.no_grad():
            out = self.qnode(angles.to(torch.float64), self.w.to(torch.float64))
            z = torch.stack(list(out), dim=-1).to(torch.float32)
        return (1.0 - 2.0 * self.p_ro) * z


def noisy_copy(model, p1=0.0, p2=0.0, p_ro=0.0, shots=None, seed=0):
    """A copy of a trained HybridForecaster whose circuit runs with noise (encoder/decoder unchanged)."""
    import copy
    m = copy.deepcopy(model)
    m.middle = NoisyQuantumBlock(model.middle, p1, p2, p_ro, shots, seed)
    return m.eval()


def load_trained(run_dir, data):
    """Rebuild a saved hybrid / control model from experiments/<group>/<run>/ (config.yaml + model.pt)."""
    import yaml
    cfg = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    model, _ = build(cfg, R.INPUT_LEN * data.n_in)
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location="cpu"))
    return model.eval(), cfg

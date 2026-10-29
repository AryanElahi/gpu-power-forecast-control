"""rnn.py: LSTM / GRU direct multi-horizon forecasters on the modelling dataset (Week 3).

One network predicts the change P(t + h) - P(t) for all horizons (1, 5, 15) at once from the 60 minutes ending at
t. Training uses every train origin that is usable for at least one horizon; the loss is masked per horizon
(ok_h{h} and split_h{h} == "train"). Early stopping on the validation MAE in kW (mean over horizons).

Per-step input channels:
  power     scaling "minmax": (P - min) / (max - min), min/max of valid train minutes (protocol default)
            scaling "local" : (P(t - k) - P(t)) / s, s = std of valid 1-minute changes on train
  calendar  minute-mod-5 one-hot (5) + hour sin/cos (2)            (if cfg["calendar"])
  covariates 1-minute changes of the 12 covariates / train std     (if cfg["covariates"]; covariate arm)
Targets are (P(t + h) - P(t)) / scale, scale = (max - min) for "minmax" and s for "local".
"""
import time

import numpy as np
import torch
from torch import nn

HORIZONS = (1, 5, 15)
INPUT_LEN = 60
COVARIATES = ["util_gpu_mean", "util_mem_mean", "temp_gpu_mean", "temp_mem_mean", "mem_used_sum", "mem_free_sum",
              "n_gpu_equiv", "gpus_alloc", "active_gpu_jobs", "active_jobs", "queued_gpu_jobs", "queued_jobs"]


class RNNForecaster(nn.Module):
    def __init__(self, cell, n_in, hidden, layers, dropout, n_out=len(HORIZONS)):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class Data:
    """Holds the per-minute arrays and builds batches of windows on the fly."""

    def __init__(self, md, cfg):
        self.cfg = cfg
        n = md.height
        self.n = n
        self.P = md["P_kW"].fill_null(0.0).to_numpy().astype(np.float64)
        valid = md["valid"].to_numpy()
        split = md["split"].to_numpy()
        trv = valid & (split == "train")
        self.idx = {}
        for h in HORIZONS:
            ok = md[f"ok_h{h}"].to_numpy()
            sp = np.array(md[f"split_h{h}"].to_list(), dtype=object)
            self.idx[h] = {s: np.flatnonzero(ok & (sp == s)) for s in ("train", "val", "test")}
        pmin, pmax = float(self.P[trv].min()), float(self.P[trv].max())
        pair = valid[1:] & valid[:-1] & trv[1:]
        s = float(np.std(np.diff(self.P)[pair]))
        self.scaling = cfg["scaling"]
        if self.scaling == "minmax":
            self.pmin, self.scale = pmin, pmax - pmin
        elif self.scaling == "local":
            self.pmin, self.scale = 0.0, s
        else:
            raise ValueError(cfg["scaling"])
        chans = []
        if cfg.get("calendar", True):
            m5 = md["mod5"].to_numpy().astype(int)
            chans += [(m5 == k).astype(np.float32) for k in range(5)]
            chans += [md["hour_sin"].to_numpy().astype(np.float32), md["hour_cos"].to_numpy().astype(np.float32)]
        if cfg.get("covariates", False):
            for c in COVARIATES:
                d = md[f"d1_{c}"].fill_null(0.0).to_numpy().astype(np.float64)
                sd = float(np.std(d[trv])) or 1.0
                chans.append((d / sd).astype(np.float32))
        self.extra = torch.from_numpy(np.stack(chans, axis=1)) if chans else None
        self.n_in = 1 + (self.extra.shape[1] if chans else 0)
        self.Pt = torch.from_numpy(self.P)
        self.offsets = torch.arange(-(INPUT_LEN - 1), 1)
        self.stats = {"pmin_train": pmin, "pmax_train": pmax, "std_dP_train": s, "scale": self.scale,
                      "n_in": self.n_in}

    def origins(self, split):
        return np.unique(np.concatenate([self.idx[h][split] for h in HORIZONS]))

    def targets(self, t, split):
        """(N, 3) scaled targets and (N, 3) mask for origins t."""
        Y = np.zeros((len(t), len(HORIZONS)), dtype=np.float32)
        Mk = np.zeros((len(t), len(HORIZONS)), dtype=np.float32)
        for j, h in enumerate(HORIZONS):
            m = np.isin(t, self.idx[h][split])
            Mk[m, j] = 1.0
            tt = t[m]
            Y[m, j] = ((self.P[tt + h] - self.P[tt]) / self.scale).astype(np.float32)
        return torch.from_numpy(Y), torch.from_numpy(Mk)

    def windows(self, t):
        tt = torch.as_tensor(t, dtype=torch.long)
        w = tt[:, None] + self.offsets[None, :]                       # (B, 60)
        p = self.Pt[w]
        if self.scaling == "minmax":
            x = (p - self.pmin) / self.scale
        else:
            x = (p - self.Pt[tt][:, None]) / self.scale
        x = x.float().unsqueeze(-1)
        if self.extra is not None:
            x = torch.cat([x, self.extra[w]], dim=-1)
        return x


def predict(model, data, t, batch=4096):
    """Forecasts of P(t + h) in kW, shape (len(t), 3)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(t), batch):
            outs.append(model(data.windows(t[i:i + batch])).numpy())
    out = np.concatenate(outs) if outs else np.zeros((0, len(HORIZONS)))
    return data.P[t][:, None] + out * data.scale


def train(md, cfg, log=print):
    """Train one network. Returns model, data, history, info."""
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 0)) or torch.get_num_threads())
    data = Data(md, cfg)
    model = RNNForecaster(cfg["cell"], data.n_in, int(cfg["hidden"]), int(cfg["layers"]), float(cfg["dropout"]))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    loss_fn = {"mse": nn.MSELoss(reduction="none"), "l1": nn.L1Loss(reduction="none"),
               "huber": nn.HuberLoss(reduction="none", delta=1.0)}[cfg.get("loss", "mse")]
    tr = data.origins("train")
    if cfg.get("max_train_samples"):
        rng0 = np.random.default_rng(seed)
        tr = np.sort(rng0.choice(tr, min(len(tr), int(cfg["max_train_samples"])), replace=False))
    va = data.origins("val")
    Ytr, Mtr = data.targets(tr, "train")
    rng = np.random.default_rng(seed)
    bs = int(cfg["batch"])
    best, best_state, bad, hist = np.inf, None, 0, []
    t_start = time.time()
    for ep in range(1, int(cfg["epochs"]) + 1):
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
        pv = predict(model, data, va)
        maes = []
        for j, h in enumerate(HORIZONS):
            sel = np.isin(va, data.idx[h]["val"])
            maes.append(float(np.mean(np.abs(pv[sel, j] - data.P[va[sel] + h]))))
        vm = float(np.mean(maes))
        hist.append({"epoch": ep, "train_loss": tot / max(cnt, 1.0), "val_MAE_mean": vm,
                     **{f"val_MAE_h{h}": m for h, m in zip(HORIZONS, maes)}, "s": time.time() - t_start})
        log(f"  epoch {ep:>3} | train loss {tot / max(cnt, 1):.5f} | val MAE " +
            " / ".join(f"{m:.4f}" for m in maes) + f" kW | {time.time() - t_start:.0f} s")
        if vm < best - 1e-6:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    info = {"n_params": n_params, "train_s": time.time() - t_start, "epochs_run": len(hist),
            "best_epoch": int(np.argmin([h_["val_MAE_mean"] for h_ in hist]) + 1), "best_val_MAE_mean": best,
            "n_train_origins": int(len(tr)), **data.stats}
    return model, data, hist, info

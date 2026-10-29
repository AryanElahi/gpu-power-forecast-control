"""nf_models.py: neuralforecast architectures (N-HiTS, PatchTST, DLinear, NLinear) trained with OUR protocol (Week 4).

Why not the neuralforecast fit/predict API: our protocol needs exactly the same usable origins (ok_h{h}, split_h{h}),
the same input window (60 valid minutes), the same local scaling, the same masked loss on the three horizons and the
same early stopping as the Week 3 LSTM/GRU (src/models/rnn.py). We therefore take the neuralforecast model classes as
plain torch modules (the architecture) and drive them with the Week 3 data loader and training loop. Only the
architecture differs between the networks in the benchmark.

Input  : rnn.Data windows, univariate, local scaling: (P(t - k) - P(t)) / s, k = 59 .. 0, shape (B, 60, 1).
         (PatchTST, DLinear and NLinear in neuralforecast take no exogenous inputs, so all four run univariate;
          the Week 3 LSTM/GRU also had calendar channels: difference documented in the benchmark.)
Output : neuralforecast forecasts 15 steps; we keep steps 1, 5, 15 = (P(t + h) - P(t)) / s.
Loss   : masked MSE on the three horizons (as rnn.py); early stopping on mean validation MAE in kW.
"""
import time

import numpy as np
import torch
from torch import nn

from models import rnn as R

H_MAX = max(R.HORIZONS)
STEP_IDX = [h - 1 for h in R.HORIZONS]          # positions of h = 1, 5, 15 in the 15-step output


def make_nf(name, cfg):
    """Build a neuralforecast model object (used only as a torch module)."""
    seed = int(cfg.get("seed", 0))
    common = dict(h=H_MAX, input_size=R.INPUT_LEN, max_steps=1, random_seed=seed)
    if name == "nhits":
        from neuralforecast.models import NHITS
        return NHITS(**common, mlp_units=cfg.get("mlp_units", [[256, 256]] * 3),
                     n_pool_kernel_size=cfg.get("n_pool_kernel_size", [2, 2, 1]),
                     n_freq_downsample=cfg.get("n_freq_downsample", [4, 2, 1]),
                     dropout_prob_theta=float(cfg.get("dropout", 0.0)))
    if name == "patchtst":
        from neuralforecast.models import PatchTST
        return PatchTST(**common, patch_len=int(cfg.get("patch_len", 16)), stride=int(cfg.get("stride", 8)),
                        hidden_size=int(cfg.get("hidden", 64)), n_heads=int(cfg.get("n_heads", 4)),
                        encoder_layers=int(cfg.get("encoder_layers", 2)), linear_hidden_size=int(cfg.get("ff", 128)),
                        dropout=float(cfg.get("dropout", 0.1)), revin=bool(cfg.get("revin", True)))
    if name == "dlinear":
        from neuralforecast.models import DLinear
        return DLinear(**common, moving_avg_window=int(cfg.get("moving_avg_window", 25)))
    if name == "nlinear":
        from neuralforecast.models import NLinear
        return NLinear(**common)
    raise ValueError(name)


class NFNet(nn.Module):
    """Wraps a neuralforecast model: x (B, 60, 1) -> (B, 3) scaled changes for h = 1, 5, 15."""

    def __init__(self, nf_model):
        super().__init__()
        self.nf = nf_model
        self.layout = None          # "BL1" (B, L, 1) or "BL" (B, L); found on the first call

    def _call(self, y, layout):
        if layout == "BL":
            y = y[..., 0]
        wb = {"insample_y": y, "insample_mask": torch.ones_like(y),
              "futr_exog": None, "hist_exog": None, "stat_exog": None}
        return self.nf(wb)

    def forward(self, x):
        b = x.shape[0]
        y = x[..., :1]
        if self.layout is None:
            errs = []
            for lay in ("BL1", "BL"):
                try:
                    out = self._call(y, lay)
                    if out.reshape(b, -1).shape[1] == H_MAX:
                        self.layout = lay
                        break
                    errs.append(f"{lay}: output shape {tuple(out.shape)}")
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{lay}: {type(e).__name__}: {e}")
            if self.layout is None:
                raise RuntimeError("no input layout worked: " + " | ".join(errs))
        out = self._call(y, self.layout)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.reshape(b, -1)
        if out.shape[1] != H_MAX:
            raise RuntimeError(f"unexpected output shape {tuple(out.shape)} (expected {H_MAX} steps)")
        return out[:, STEP_IDX]


def build(name, cfg):
    if name in ("gru", "lstm"):            # univariate Week 3 RNN, used as a timing / architecture reference
        net = R.RNNForecaster(name, 1, int(cfg.get("rnn_hidden", 64)), int(cfg.get("rnn_layers", 2)),
                              float(cfg.get("rnn_dropout", 0.1)))
    else:
        net = NFNet(make_nf(name, cfg))
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return net, n_params


def pick_device(pref="cpu"):
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, data, t, device, batch=4096):
    """Forecasts of P(t + h) in kW, shape (len(t), 3)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(t), batch):
            outs.append(model(data.windows(t[i:i + batch]).to(device)).float().cpu().numpy())
    out = np.concatenate(outs) if outs else np.zeros((0, len(R.HORIZONS)))
    return data.P[t][:, None] + out * data.scale


def train(md, cfg, log=print, data=None):
    """Train one model with the Week 3 protocol. Returns model, data, history, info."""
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 0)) or torch.get_num_threads())
    dcfg = {"scaling": "local", "calendar": False, "covariates": False}
    data = data or R.Data(md, dcfg)
    device = pick_device(cfg.get("device", "cpu"))
    model, n_params = build(cfg["model"], cfg)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    loss_fn = nn.MSELoss(reduction="none")
    tr = data.origins("train")
    if cfg.get("max_train_samples"):
        rng0 = np.random.default_rng(seed)
        tr = np.sort(rng0.choice(tr, min(len(tr), int(cfg["max_train_samples"])), replace=False))
    va = data.origins("val")
    if cfg.get("max_val_samples"):
        rng1 = np.random.default_rng(seed + 1)
        va = np.sort(rng1.choice(va, min(len(va), int(cfg["max_val_samples"])), replace=False))
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
            x = data.windows(tr[b]).to(device)
            y, mk = Ytr[b].to(device), Mtr[b].to(device)
            out = model(x)
            loss = (loss_fn(out, y) * mk).sum() / mk.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            tot += loss.item() * float(mk.sum())
            cnt += float(mk.sum())
        pv = predict(model, data, va, device)
        maes = []
        for j, h in enumerate(R.HORIZONS):
            sel = np.isin(va, data.idx[h]["val"])
            maes.append(float(np.mean(np.abs(pv[sel, j] - data.P[va[sel] + h]))))
        vm = float(np.mean(maes))
        hist.append({"epoch": ep, "train_loss": tot / max(cnt, 1.0), "val_MAE_mean": vm,
                     **{f"val_MAE_h{h}": m for h, m in zip(R.HORIZONS, maes)}, "s": time.time() - t_start})
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
    info = {"model": cfg["model"], "n_params": n_params, "train_s": time.time() - t_start, "epochs_run": len(hist),
            "best_epoch": int(np.argmin([h_["val_MAE_mean"] for h_ in hist]) + 1), "best_val_MAE_mean": best,
            "n_train_origins": int(len(tr)), "device": str(device), **data.stats}
    return model, data, hist, info


def time_steps(name, cfg, data, device, n_batches=20, bs=512, seed=0):
    """Time forward+backward on n_batches random train batches and prediction on 4,096 val origins."""
    torch.manual_seed(seed)
    model, n_params = build(name, cfg)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = data.origins("train")
    va = data.origins("val")
    rng = np.random.default_rng(seed)
    model.train()
    t0, n_done = None, 0
    for i in range(n_batches + 2):                      # first 2 batches = warm-up, not timed
        if i == 2:
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
        b = np.sort(rng.choice(tr, bs, replace=False))
        y, mk = data.targets(b, "train")
        x = data.windows(b).to(device)
        out = model(x)
        loss = (((out - y.to(device)) ** 2) * mk.to(device)).sum() / mk.sum().clamp(min=1.0).to(device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i >= 2:
            n_done += bs
    if device.type == "mps":
        torch.mps.synchronize()
    train_s = time.time() - t0
    t1 = time.time()
    predict(model, data, va[:4096], device)
    pred_s = time.time() - t1
    return {"model": name, "device": str(device), "n_params": n_params,
            "ms_per_train_sample": 1000 * train_s / n_done, "ms_per_pred_sample": 1000 * pred_s / 4096,
            "est_epoch_min": train_s / n_done * len(tr) / 60 + pred_s / 4096 * len(va) / 60}

"""frozen.py: re-create every benchmark model's forecasts for any split from the frozen Week 3-4 settings (Week 4).

Used once to open the test split. Nothing is tuned or selected here:
  baselines (notebook 07)   persistence, minute-mod-5 corrections (static from train; rolling causal 7 days),
                            daily seasonal naive, ridge AR on changes refit on train with the alpha chosen on validation
  state space (notebook 08) AutoARIMA order/parameters and AutoETS parameters fitted on the train tail, run as fixed
                            filters from the same start minute (S0) to the end of the requested split
  LightGBM (notebook 08)    tuned parameters and boosting rounds from validation, refit on train, seeds 0, 1, 2
  networks (notebooks 09, 11) the saved checkpoints (train only, early stopped on validation), no retraining
Every function returns {h: {model: forecasts}} on the usable origins IDX[h][split] (ok_h and split_h == split),
or {h: {model: {seed: forecasts}}} for multi-seed models. `reproduce_check` compares the validation forecasts with
the ones stored by the original notebooks: the test split may only be opened after that check passes.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

HORIZONS = (1, 5, 15)
INPUT_LEN = 60
ROLL_WINDOW = 7 * 1440
SEEDS = (0, 1, 2)
AR_KIND = {"ar_lags": "lags", "ar_lags_cal": "cal", "ar_lags_cal_cov": "cal_cov"}


class Ctx:
    """Dataset, arrays and usable-origin indices shared by all predictors."""

    def __init__(self, root):
        import sys
        self.root = Path(root)
        sys.path.insert(0, str(self.root / "src"))
        from models import tabular_features as TF
        self.TF = TF
        self.md = pl.read_parquet(self.root / "data" / "processed" / "model_dataset.parquet")
        self.meta = json.loads((self.root / "data" / "processed" / "model_dataset_meta.json").read_text())
        self.A = TF.arrays(self.md)
        self.P = self.A["P"]
        self.n = len(self.P)
        self.VALID = self.md["valid"].to_numpy()
        self.MOD5 = self.A["MOD5"]
        self.L24 = self.A["L24"]
        self.EP = self.md["minute_epoch"].to_numpy()
        self.IDX = {}
        for h in HORIZONS:
            ok = self.md[f"ok_h{h}"].to_numpy()
            sp = np.array(self.md[f"split_h{h}"].to_list(), dtype=object)
            self.IDX[h] = {s: np.flatnonzero(ok & (sp == s)) for s in ("train", "val", "test")}
        self.base = json.loads((self.root / "reports" / "baselines" / "baselines_summary.json").read_text())
        self.clas = json.loads((self.root / "reports" / "classical" / "classical_summary.json").read_text())


# ------------------------------------------------------------------ notebook 07
def _ridge(X, y, a):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return mu, sd, ym, beta


def predict_baselines(c, split):
    P, MOD5, VALID, n = c.P, c.MOD5, c.VALID, c.n
    out = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        t = c.IDX[h][split]
        tr = c.IDX[h]["train"]
        out[h]["persistence"] = P[t].copy()
        dtr = P[tr + h] - P[tr]
        cs_ = np.array([dtr[MOD5[tr] == m].mean() for m in range(5)])
        out[h]["pers_mod5_static"] = P[t] + cs_[MOD5[t]]
        pair = np.zeros(n, dtype=bool)
        pair[:n - h] = VALID[:n - h] & VALID[h:]
        s = np.flatnonzero(pair)
        d = P[s + h] - P[s]
        corr = np.zeros(len(t))
        for m in range(5):
            sm, dm_ = s[MOD5[s] == m], d[MOD5[s] == m]
            avail = sm + h
            cum = np.concatenate([[0.0], np.cumsum(dm_)])
            hi = np.searchsorted(avail, t, side="right")
            lo = np.searchsorted(avail, t - ROLL_WINDOW, side="right")
            cnt = hi - lo
            mean = np.where(cnt > 0, (cum[hi] - cum[lo]) / np.maximum(cnt, 1), 0.0)
            sel = MOD5[t] == m
            corr[sel] = np.where(cnt[sel] >= 100, mean[sel], 0.0)
        out[h]["pers_mod5_rolling"] = P[t] + corr
        j = t + h - 1440
        ok = (j >= 0) & VALID[np.clip(j, 0, n - 1)]
        out[h]["seasonal_naive_1d"] = np.where(ok, P[np.clip(j, 0, n - 1)], P[t])
        for name, kind in AR_KIND.items():
            a = float(c.base["chosen_alpha"][f"{h}|{name}"])
            Xtr, _ = c.TF.features(c.A, tr, kind)
            X, _ = c.TF.features(c.A, t, kind)
            mu, sd, ym, beta = _ridge(Xtr, dtr, a)
            out[h][name] = P[t] + ym + ((X - mu) / sd) @ beta
    return out


# ------------------------------------------------------------------ notebook 08: state space
def predict_statespace(c, split):
    from models import statespace as SS
    start = c.clas["fit_segment"]["start"]
    S0 = int(np.flatnonzero(c.md["minute_local"].cast(pl.Utf8).str.slice(0, 19).to_numpy() == start)[0])
    last = int(max(c.IDX[h][split].max() + h for h in HORIZONS))
    y_run = np.where(c.VALID[S0:last + 1], c.P[S0:last + 1], np.nan)
    orig = {h: c.IDX[h][split] - S0 for h in HORIZONS}
    if min(o.min() for o in orig.values()) < 0:
        raise ValueError("split starts before the state-space start minute S0")
    allo = np.unique(np.concatenate(list(orig.values())))
    ar = c.clas["auto_arima"]
    order = tuple(int(v) for v in ar["order"])
    params = np.array(list(ar["params"].values()), dtype=float)   # values in SARIMAX order (keys mislabelled in nb 08)
    fa, _ = SS.arima_rolling(y_run, order, params, allo, HORIZONS)
    spec = dict(c.clas["auto_ets"])
    fe, _ = SS.ets_rolling(y_run, spec, allo, HORIZONS)
    out = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        pos = np.searchsorted(allo, orig[h])
        out[h]["auto_arima"] = fa[h][pos]
        out[h]["auto_ets"] = fe[h][pos]
    return out, {"S0": S0, "order": order, "params": params.tolist()}


# ------------------------------------------------------------------ notebook 08: LightGBM
def _lgb_params(p, seed):
    q = {"metric": "l1", "verbosity": -1, "num_threads": 0, "bagging_freq": 1, "seed": seed, "bagging_seed": seed,
         "feature_fraction_seed": seed, "deterministic": True, "force_row_wise": True}
    q.update(p)
    return q


def predict_lightgbm(c, split, seeds=SEEDS, log=print):
    import lightgbm as lgb
    out = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        tr, t = c.IDX[h]["train"], c.IDX[h][split]
        for arm, name in (("gbm", "lightgbm"), ("gbm_cov", "lightgbm_cov")):
            tune = c.clas["lightgbm"][f"{h}|{arm}"]
            Xtr, names = c.TF.features(c.A, tr, arm)
            X, _ = c.TF.features(c.A, t, arm)
            dtr = lgb.Dataset(Xtr, c.P[tr + h] - c.P[tr], feature_name=names, params={"feature_pre_filter": False},
                              free_raw_data=False)
            out[h][name] = {}
            for s in seeds:
                b = lgb.train(_lgb_params(tune["params"], s), dtr, num_boost_round=max(int(tune["rounds"]), 1))
                out[h][name][s] = c.P[t] + b.predict(X)
            log(f"  lightgbm h={h:>2} {name}: {len(seeds)} seeds, {tune['rounds']} rounds")
    return out


# ------------------------------------------------------------------ notebooks 09 and 11: saved networks
def predict_networks(c, split, log=print):
    import torch
    import yaml
    from models import rnn as R
    from models import nf_models as NF
    dev = torch.device("cpu")
    out = {h: {} for h in HORIZONS}
    groups = [("week3_rnn", "lstm", "lstm_local_s{s}"), ("week3_rnn", "gru", "gru_local_s{s}")] + \
             [("week4_nf", m, m + "_s{s}") for m in ("nhits", "patchtst", "dlinear", "nlinear", "gru_uni")]
    data_cache = {}
    for group, name, pat in groups:
        for s in SEEDS:
            d = c.root / "experiments" / group / pat.format(s=s)
            if not (d / "model.pt").exists():
                log(f"  missing {d}")
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            key = (cfg.get("scaling"), bool(cfg.get("calendar", True)), bool(cfg.get("covariates", False)))
            if key not in data_cache:
                data_cache[key] = R.Data(c.md, {"scaling": key[0], "calendar": key[1], "covariates": key[2]})
            data = data_cache[key]
            if group == "week3_rnn":
                net = R.RNNForecaster(cfg["cell"], data.n_in, int(cfg["hidden"]), int(cfg["layers"]), float(cfg["dropout"]))
            else:
                net, _ = NF.build(cfg["model"], cfg)
            net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
            net.eval()
            ta = data.origins(split)
            pv = NF.predict(net, data, ta, dev)
            for j, h in enumerate(HORIZONS):
                t = c.IDX[h][split]
                out[h].setdefault(name, {})[s] = pv[np.searchsorted(ta, t), j]
        log(f"  networks {name}: seeds {sorted(out[1].get(name, {}).keys())}")
    return out


# ------------------------------------------------------------------ reproduction check on validation
def stored_val(c):
    """Validation forecasts stored by the original notebooks: {h: {model: {seed: yhat}}} (single-seed -> seed 0)."""
    out = {h: {} for h in HORIZONS}
    for src in ("baselines", "classical"):
        d = pl.read_parquet(c.root / "reports" / src / "val_predictions.parquet")
        for (m, h), g in d.group_by("model", "h"):
            out[int(h)].setdefault(m, {})[0] = g.sort("minute_epoch")["yhat"].to_numpy()
    for group, name, pat in [("week3_rnn", "lstm", "lstm_local_s{s}"), ("week3_rnn", "gru", "gru_local_s{s}")] + \
            [("week4_nf", m, m + "_s{s}") for m in ("nhits", "patchtst", "dlinear", "nlinear", "gru_uni")]:
        for s in SEEDS:
            f = c.root / "experiments" / group / pat.format(s=s) / "predictions_val.parquet"
            if f.exists():
                d = pl.read_parquet(f)
                for h in HORIZONS:
                    out[h].setdefault(name, {})[s] = d.filter(pl.col("h") == h).sort("minute_epoch")["yhat"].to_numpy()
    return out


def reproduce_check(c, new, tol=1e-3):
    """Max |new - stored| per model, horizon and seed on validation; passes when every value is below tol (kW)."""
    old = stored_val(c)
    rows = []
    for h in HORIZONS:
        for m, by_seed in new[h].items():
            if not isinstance(by_seed, dict):
                by_seed = {0: by_seed}
            for s, v in by_seed.items():
                ref = old[h].get(m, {}).get(s)
                if ref is None:
                    rows.append({"h": h, "model": m, "seed": s, "n": len(v), "max_abs_diff": None, "ok": None})
                    continue
                dd = float(np.max(np.abs(np.asarray(v) - ref))) if len(ref) == len(v) else float("inf")
                rows.append({"h": h, "model": m, "seed": s, "n": len(v), "max_abs_diff": dd, "ok": dd < tol})
    tab = pl.DataFrame(rows)
    checked = tab.filter(pl.col("ok").is_not_null())
    return tab, bool(checked.height > 0 and checked["ok"].all())

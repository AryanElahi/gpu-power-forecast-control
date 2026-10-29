"""statespace.py: rolling multi-step forecasts of ARIMA and ETS models with fixed parameters (Week 3).

Order/model selection and parameter estimation are done once by statsforecast (AutoARIMA, AutoETS) on a
training segment. The fitted model is then run forward over a longer series (training tail + validation) as a
causal filter: at each origin t the state is updated with P(t) and forecasts for t + h are read off the state.
Missing minutes (NaN) are skipped: the state is propagated without an update. Nothing after t is used.

ARIMA: statsforecast coefficients are mapped to a statsmodels SARIMAX state-space model (no constant; same sign
convention as R: y_t = sum ar_i y_{t-i} + e_t + sum ma_j e_{t-j}); the Kalman filter gives x_{t|t}, and the h-step
forecast is Z T^h x_{t|t}.
ETS: additive-error, non-seasonal models (ANN, AAN, AAdN) with the R/statsforecast state equations
  yhat_t = l_{t-1} + phi b_{t-1};  e_t = y_t - yhat_t
  l_t = l_{t-1} + phi b_{t-1} + alpha e_t;  b_t = phi b_{t-1} + beta e_t
  h-step forecast from t: l_t + (phi + ... + phi^h) b_t
"""
import numpy as np


# ------------------------------------------------------------------ ARIMA
def arima_spec(model_):
    """(order, params, names) for SARIMAX from a statsforecast AutoARIMA `model_` dict."""
    p, q, P, Q, m, d, D = model_["arma"]
    if P or Q or D:
        raise ValueError(f"seasonal terms not supported here: arma={model_['arma']}")
    coef = model_["coef"]
    extra = [k for k in coef if not (k.startswith("ar") or k.startswith("ma"))]
    if extra:
        raise ValueError(f"unsupported coefficients (mean/drift/xreg): {extra}")
    ar = [coef[f"ar{i}"] for i in range(1, p + 1)]
    ma = [coef[f"ma{j}"] for j in range(1, q + 1)]
    names = [f"ar.L{i}" for i in range(1, p + 1)] + [f"ma.L{j}" for j in range(1, q + 1)] + ["sigma2"]
    return (int(p), int(d), int(q)), np.array(ar + ma + [model_["sigma2"]], dtype=float), names


def arima_rolling(y, order, params, origins, horizons):
    """Forecasts {h: array over origins} of y[t + h] made at t (after observing y[t]). y may contain NaN."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    mod = SARIMAX(y, order=order, trend="n", enforce_stationarity=False, enforce_invertibility=False)
    res = mod.filter(params)
    fr = res.filter_results
    Z = np.asarray(fr.design)[:, :, 0]
    T = np.asarray(fr.transition)[:, :, 0]
    X = np.asarray(fr.filtered_state)[:, origins]          # x_{t|t}
    out = {}
    for h in horizons:
        out[h] = (Z @ np.linalg.matrix_power(T, h) @ X).ravel()
    one_step = np.asarray(res.forecasts).ravel()              # E[y_t | y_{<t}] for every t
    return out, one_step


# ------------------------------------------------------------------ ETS
def ets_spec(model_):
    """dict(alpha, beta, phi, l0, b0, trend, damped) from a statsforecast AutoETS `model_` dict."""
    comp = model_["components"]
    if comp[0] != "A" or comp[2] != "N":
        raise ValueError(f"only additive-error non-seasonal ETS supported, got {comp}")
    par = np.asarray(model_["par"], dtype=float)
    trend = comp[1] != "N"
    damped = len(comp) > 3 and comp[3] == "D"
    return {"alpha": par[0], "beta": par[1] if trend else 0.0, "phi": (par[3] if damped else 1.0) if trend else 0.0,
            "l0": par[4], "b0": par[5] if trend else 0.0, "trend": trend, "damped": damped, "components": comp}


def ets_rolling(y, spec, origins, horizons, l_init=None, b_init=None):
    """Forecasts {h: array over origins} of y[t + h] made at t; NaN in y = no update. Also returns one-step fits."""
    a, b, phi = spec["alpha"], spec["beta"], spec["phi"]
    l = spec["l0"] if l_init is None else l_init
    bb = spec["b0"] if b_init is None else b_init
    n = len(y)
    L, B, fit = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        yh = l + phi * bb
        fit[t] = yh
        if np.isfinite(y[t]):
            e = y[t] - yh
            l, bb = yh + a * e, phi * bb + b * e
        else:
            l, bb = yh, phi * bb
        L[t], B[t] = l, bb
    out = {}
    for h in horizons:
        damp = sum(phi ** k for k in range(1, h + 1))
        out[h] = L[origins] + damp * B[origins]
    return out, fit

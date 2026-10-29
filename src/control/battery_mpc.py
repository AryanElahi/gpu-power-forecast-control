"""battery_mpc.py: battery model and ramp-limiting MPC on the measured GPU power (Sprint 4, Week 7, RQ4).

Setting (1-minute steps, dt = 1/60 h). The data center draws P(t) (kW, measured at minute t). A battery behind the
meter charges with pc(t) >= 0 or discharges with pd(t) >= 0, so the grid sees

    g(t) = P(t) - pd(t) + pc(t)                         (kW)
    SoC(t+1) = SoC(t) + (eta * pc(t) - pd(t) / eta) * dt / E      (E in kWh, one-way efficiency eta)

with 0 <= pc, pd <= Pmax and SoC_min <= SoC <= SoC_max. Goal: keep the grid ramp |g(t) - g(t-1)| below a limit
R (kW/min).

MPC at every minute t (horizon N = 15 minutes, matching the longest forecast horizon). P(t) is measured, P(t+k) for
k = 1..N-1 comes from a forecast trajectory Phat. Decision variables pc, pd (N), SoC (N+1), slacks s (N) >= 0:

    minimise   W_VIOL * sum s_k  +  W_SOC * sum (SoC_k - SOC_REF)^2  +  W_USE * sum (pc_k + pd_k)
    subject to |g_k - g_{k-1}| <= R + s_k  (g_{-1} = grid power applied at t-1), battery dynamics and limits.

Only the first move (pc_0, pd_0) is applied (receding horizon); it acts on the measured P(t), so the realised grid
power is exact. The slack makes the problem always feasible (pc = pd = 0 is feasible for any SoC in bounds); the L1
penalty keeps violations sparse. Forecast quality enters only through the planned trajectory: a forecast that sees
a ramp coming lets the controller pre-position the SoC and spread the change over several minutes.

Forecast trajectories (scenarios):
  oracle      true P(t+k)
  model       piecewise-linear interpolation through P(t), yhat(t+1), yhat(t+5), yhat(t+15) of a stored forecast
  persistence flat P(t)
  myopic      horizon N = 1 (no look-ahead at all; "no forecast")
"""
import numpy as np

DT = 1.0 / 60.0
W_VIOL, W_SOC, W_USE = 1000.0, 1.0, 0.01
SOC_MIN, SOC_MAX, SOC_REF = 0.1, 0.9, 0.5
ETA = 0.95


class MPC:
    """Parametrised cvxpy problem (compiled once, solved every minute)."""

    def __init__(self, E_kWh, Pmax_kW, R_kW_min, N=15, eta=ETA, solver="CLARABEL"):
        import cvxpy as cp
        self.N, self.E, self.Pmax, self.R, self.eta, self.solver = N, E_kWh, Pmax_kW, R_kW_min, eta, solver
        self.phat = cp.Parameter(N)
        self.soc0 = cp.Parameter()
        self.gprev = cp.Parameter()
        pc, pd = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
        soc, s = cp.Variable(N + 1), cp.Variable(N, nonneg=True)
        g = self.phat - pd + pc
        dg = cp.hstack([g[0] - self.gprev] + ([g[1:] - g[:-1]] if N > 1 else []))
        cons = [soc[0] == self.soc0,
                soc[1:] == soc[:-1] + (eta * pc - pd / eta) * DT / E_kWh,
                soc[1:] >= SOC_MIN, soc[1:] <= SOC_MAX, pc <= Pmax_kW, pd <= Pmax_kW,
                dg <= R_kW_min + s, -dg <= R_kW_min + s]
        obj = W_VIOL * cp.sum(s) + W_SOC * cp.sum_squares(soc[1:] - SOC_REF) + W_USE * cp.sum(pc + pd)
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.pc, self.pd = pc, pd
        assert self.prob.is_dpp()

    def step(self, phat, soc0, gprev):
        self.phat.value = np.asarray(phat, float)[: self.N]
        self.soc0.value = float(soc0)
        self.gprev.value = float(gprev)
        try:
            self.prob.solve(solver=self.solver, warm_start=True)
            ok = self.prob.status in ("optimal", "optimal_inaccurate")
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            return 0.0, 0.0, False
        return max(float(self.pc.value[0]), 0.0), max(float(self.pd.value[0]), 0.0), True


def segments(valid, min_len=60):
    """[start, end) runs of consecutive valid minutes with length >= min_len."""
    v = np.asarray(valid, bool).astype(int)
    d = np.diff(np.concatenate([[0], v, [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def trajectory_model(P_t, y1, y5, y15, N=15):
    """Piecewise-linear trajectory k = 0..N-1 through (0, P(t)), (1, y1), (5, y5), (15, y15)."""
    return np.interp(np.arange(N), [0, 1, 5, 15], [P_t, y1, y5, y15])


def simulate(P, segs, scenario, mpc, forecasts=None, log_every=0, log=print):
    """Closed loop over the given segments. P: power array (kW) over the minute grid.
    forecasts: dict minute -> (y1, y5, y15) for scenario 'model' (missing -> flat persistence fallback).
    Returns per-minute arrays over the segments: minute, P, g, soc, pc, pd, ok, and the number of fallbacks."""
    N = mpc.N
    out = {k: [] for k in ("minute", "P", "g", "soc", "pc", "pd", "ok", "first")}
    fallback = 0
    done = 0
    for a, b in segs:
        soc, gprev = SOC_REF, P[a]
        for t in range(a, b):
            Pt = P[t]
            if scenario == "oracle":
                idx = np.minimum(np.arange(t, t + N), b - 1)
                ph = P[idx]
            elif scenario == "model":
                f = forecasts.get(t) if forecasts is not None else None
                if f is None:
                    ph = np.full(N, Pt); fallback += 1
                else:
                    ph = trajectory_model(Pt, *f, N=N)
            else:                                   # persistence or myopic (N = 1)
                ph = np.full(N, Pt)
            pc, pd, ok = mpc.step(ph, soc, gprev)
            g = Pt - pd + pc
            soc = float(np.clip(soc + (mpc.eta * pc - pd / mpc.eta) * DT / mpc.E, 0.0, 1.0))
            for k, v in (("minute", t), ("P", Pt), ("g", g), ("soc", soc), ("pc", pc), ("pd", pd), ("ok", ok), ("first", t == a)):
                out[k].append(v)
            gprev = g
            done += 1
            if log_every and done % log_every == 0:
                log(f"    {scenario}: {done:,} minutes")
    res = {k: np.asarray(v) for k, v in out.items()}
    res["fallback"] = fallback
    return res


def ramp_metrics(res, R, key="g"):
    """Ramp statistics of the grid power (key 'g') or of the load itself (key 'P'), within segments only."""
    x = res[key]
    d = np.abs(np.diff(x))
    d = d[~res["first"][1:]]                     # no ramp across a segment boundary
    viol = d > R + 1e-6
    out = {"n_steps": int(len(d)), "max_ramp": float(d.max()), "p999_ramp": float(np.quantile(d, 0.999)),
           "p99_ramp": float(np.quantile(d, 0.99)), "viol_steps": int(viol.sum()), "viol_share": float(viol.mean()),
           "viol_excess_kW": float(np.clip(d - R, 0, None).sum())}
    if key == "g":
        thr = (res["pc"] + res["pd"]).sum() * DT
        out.update({"throughput_kWh": float(thr), "soc_min": float(res["soc"].min()), "soc_max": float(res["soc"].max()),
                    "solver_fail": int((~res["ok"]).sum())})
    return out

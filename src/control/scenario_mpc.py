"""scenario_mpc.py: uncertainty-aware MPC with conformal bounds (Sprint 4, Week 7 extension; notebook 27).

Kept separate from battery_mpc.py so that file stays byte-identical to the version recorded in the notebook-26
freeze record. Same battery model, constraints and objective as battery_mpc.MPC.
"""
import numpy as np

from .battery_mpc import DT, ETA, SOC_MAX, SOC_MIN, SOC_REF, W_SOC, W_USE, W_VIOL, trajectory_model


class ScenarioMPC:
    """Scenario MPC: S forecast trajectories (e.g. lower bound, point forecast, upper bound of a conformal interval)
    share ONE first move (non-anticipativity: P(t) is measured, so the first move must suit every scenario); later
    moves may differ per scenario. Objective = weighted mean over scenarios of the single-scenario MPC objective.
    With S = 1 it is identical to MPC."""

    def __init__(self, E_kWh, Pmax_kW, R_kW_min, N=15, S=3, weights=None, eta=ETA, solver="CLARABEL"):
        import cvxpy as cp
        self.N, self.S, self.E, self.Pmax, self.R, self.eta, self.solver = N, S, E_kWh, Pmax_kW, R_kW_min, eta, solver
        w = np.full(S, 1.0 / S) if weights is None else np.asarray(weights, float) / np.sum(weights)
        self.phat = cp.Parameter((S, N))
        self.soc0, self.gprev = cp.Parameter(), cp.Parameter()
        cons, obj = [], 0
        self.pc, self.pd = [], []
        for s in range(S):
            pc, pd = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
            soc, sl = cp.Variable(N + 1), cp.Variable(N, nonneg=True)
            g = self.phat[s] - pd + pc
            dg = cp.hstack([g[0] - self.gprev] + ([g[1:] - g[:-1]] if N > 1 else []))
            cons += [soc[0] == self.soc0, soc[1:] == soc[:-1] + (eta * pc - pd / eta) * DT / E_kWh,
                     soc[1:] >= SOC_MIN, soc[1:] <= SOC_MAX, pc <= Pmax_kW, pd <= Pmax_kW,
                     dg <= R_kW_min + sl, -dg <= R_kW_min + sl]
            obj = obj + w[s] * (W_VIOL * cp.sum(sl) + W_SOC * cp.sum_squares(soc[1:] - SOC_REF) + W_USE * cp.sum(pc + pd))
            if s > 0:
                cons += [pc[0] == self.pc[0][0], pd[0] == self.pd[0][0]]
            self.pc.append(pc); self.pd.append(pd)
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dpp()

    def step(self, phat, soc0, gprev):
        self.phat.value = np.asarray(phat, float)[:, : self.N]
        self.soc0.value, self.gprev.value = float(soc0), float(gprev)
        try:
            self.prob.solve(solver=self.solver, warm_start=True)
            ok = self.prob.status in ("optimal", "optimal_inaccurate")
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            return 0.0, 0.0, False
        return max(float(self.pc[0].value[0]), 0.0), max(float(self.pd[0].value[0]), 0.0), True


def trajectory_band(P_t, lo, mid, hi, N=15):
    """(3, N) trajectories through the lower bound, point forecast and upper bound at h = 1, 5, 15.
    lo, mid, hi: sequences of length 3 (h = 1, 5, 15)."""
    return np.stack([trajectory_model(P_t, *v, N=N) for v in (lo, mid, hi)])


def simulate_band(P, segs, mpc, bands, log_every=0, log=print):
    """Closed loop with ScenarioMPC. bands: dict minute -> (lo[3], mid[3], hi[3]); missing -> all scenarios flat."""
    N, S = mpc.N, mpc.S
    out = {k: [] for k in ("minute", "P", "g", "soc", "pc", "pd", "ok", "first")}
    fallback, done = 0, 0
    for a, b in segs:
        soc, gprev = SOC_REF, P[a]
        for t in range(a, b):
            Pt = P[t]
            f = bands.get(t)
            if f is None:
                ph = np.full((S, N), Pt); fallback += 1
            else:
                ph = trajectory_band(Pt, *f, N=N)
            pc, pd, ok = mpc.step(ph, soc, gprev)
            g = Pt - pd + pc
            soc = float(np.clip(soc + (mpc.eta * pc - pd / mpc.eta) * DT / mpc.E, 0.0, 1.0))
            for k, v in (("minute", t), ("P", Pt), ("g", g), ("soc", soc), ("pc", pc), ("pd", pd), ("ok", ok), ("first", t == a)):
                out[k].append(v)
            gprev = g
            done += 1
            if log_every and done % log_every == 0:
                log(f"    band: {done:,} minutes")
    res = {k: np.asarray(v) for k, v in out.items()}
    res["fallback"] = fallback
    return res

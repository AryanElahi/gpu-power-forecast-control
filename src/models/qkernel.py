"""qkernel.py: quantum kernel ridge regression and its controls (Week 6, supplementary item of the plan).

Design (provisional, assistant's decision; validation only):

1. Input: the same 60-minute window as the networks with local scaling, x(t) = [(P(t - k) - P(t)) / s], k = 1..59
   (k = 0 is always 0 and is dropped). PCA fitted on the training subsample only, n components (n = qubits),
   whitened with the training standard deviations: z(t) in R^n.
2. Kernels on z (all deterministic, no random parameters):
   "qk"        fidelity kernel k(z, z') = |<psi(z)|psi(z')>|^2 of an IQP-type feature map (Havlicek et al. 2019 style):
               L = 2 layers of [H on every qubit, RZ(2 a z_i), IsingZZ(2 a^2 z_i z_j) on ring pairs]. Exact
               statevectors (classically simulable; no advantage claim).
   "qk_noent"  the same map without the IsingZZ gates (product state; the kernel then has a classical closed form).
   "rbf"       classical Gaussian kernel exp(-gamma ||z - z'||^2) with gamma = a^2 / 2 (matched bandwidth grid).
3. Kernel ridge per horizon on the target (P(t + h) - P(t)) / s, alpha = (K + lam n I)^-1 y. One eigendecomposition
   of K gives all penalties; lam = r * n with r in RIDGE_GRID (same grid as the QRC readout). The bandwidth a and the
   penalty are chosen on validation MAE in kW (mean over horizons for a; per horizon for lam).
4. Training subsample: the most recent block of train origins (as in the data-fraction experiment of notebook 21),
   restricted to origins usable for all three horizons; optionally thinned (every k-th origin) to bound memory.
"""
import math
import time

import numpy as np

from models import rnn as R

HORIZONS = R.HORIZONS
RIDGE_GRID = tuple(float(v) for v in np.logspace(-7, 1, 17))
SCALES = (0.01, 0.03, 0.1, 0.3, 1.0)   # near-linear end included (lesson from the QRC scale screen)
N_LAYERS = 2


# ----------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------
def windows(data, t):
    """(len(t), 59) local-scaled windows (k = 1..59)."""
    x = data.windows(t)[:, :, 0].numpy().astype(np.float64)       # (N, 60), last column (k = 0) is 0
    return x[:, :-1]


class PCA:
    def __init__(self, n):
        self.n = n

    def fit(self, X):
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[: self.n].T
        Z = (X - self.mu) @ self.V
        self.sd = Z.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        var = S ** 2
        self.explained = float(var[: self.n].sum() / var.sum())
        return self

    def transform(self, X):
        return ((X - self.mu) @ self.V) / self.sd


# ----------------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------------
def pairs(n):
    return [(i, (i + 1) % n) for i in range(n)] if n > 2 else ([(0, 1)] if n == 2 else [])


def states(Z, scale, entangle=True, n_layers=N_LAYERS, batch=4096):
    """Exact statevectors (len(Z), 2^n), complex128, of the IQP-type feature map."""
    import pennylane as qml
    n = Z.shape[1]
    dev = qml.device("default.qubit", wires=n)
    prs = pairs(n)

    @qml.qnode(dev, interface="numpy", diff_method=None)
    def circ(z):
        for _ in range(n_layers):
            for i in range(n):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * scale * z[:, i], wires=i)
            if entangle:
                for a, b in prs:
                    qml.IsingZZ(2.0 * scale ** 2 * z[:, a] * z[:, b], wires=[a, b])
        return qml.state()

    out = np.empty((len(Z), 2 ** n), dtype=np.complex128)
    for i in range(0, len(Z), batch):
        zb = np.asarray(Z[i:i + batch], dtype=np.float64)
        out[i:i + len(zb)] = np.asarray(circ(zb)).reshape(len(zb), -1)
    return out


class Kernel:
    """Feature representation for one (kind, scale); gram(A, B) gives the kernel matrix block."""

    def __init__(self, kind, scale):
        self.kind, self.scale = kind, scale

    def embed(self, Z):
        if self.kind == "rbf":
            return Z.astype(np.float64)
        return states(Z, self.scale, entangle=(self.kind == "qk"))

    def gram(self, A, B):
        if self.kind == "rbf":
            g = self.scale ** 2 / 2.0
            d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
            return np.exp(-g * np.clip(d2, 0.0, None))
        return np.abs(A.conj() @ B.T) ** 2


def offdiag_stats(K):
    m = ~np.eye(len(K), dtype=bool)
    v = K[m]
    return {"mean_offdiag": float(v.mean()), "var_offdiag": float(v.var())}


# ----------------------------------------------------------------------------------------------------------------
# kernel ridge
# ----------------------------------------------------------------------------------------------------------------
def fit_all(K, Y, ridge_grid=RIDGE_GRID):
    """K (n, n) train Gram, Y (n, H) targets. Returns dual coefficients (len(grid), n, H) and the target means."""
    n = len(K)
    ym = Y.mean(0)
    e, V = np.linalg.eigh((K + K.T) / 2.0)
    e = np.clip(e, 0.0, None)
    vty = V.T @ (Y - ym)
    A = np.stack([V @ (vty / (e + r * n)[:, None]) for r in ridge_grid])
    return A, ym, float(e.min()), float(e.max())


def predict_all(kern, Etr, Eq, A, ym, chunk=4096):
    """Predictions (len(grid), len(Eq), H) of the scaled targets."""
    G, n, H = A.shape
    Aflat = A.transpose(1, 0, 2).reshape(n, G * H)
    out = np.empty((len(Eq), G * H))
    for i in range(0, len(Eq), chunk):
        out[i:i + chunk] = kern.gram(Eq[i:i + chunk], Etr) @ Aflat
    return out.reshape(len(Eq), G, H).transpose(1, 0, 2) + ym[None, None, :]


def train_origins(data, frac, thin=1):
    """Most recent block of train origins (fraction frac), usable for every horizon, every `thin`-th origin."""
    from models import week6 as W
    cut = W.fraction_cutoff(data, frac)
    common = data.idx[HORIZONS[0]]["train"]
    for h in HORIZONS[1:]:
        common = np.intersect1d(common, data.idx[h]["train"])
    t = common[common >= cut]
    return t[::-1][::thin][::-1] if thin > 1 else t


def run_config(data, kind, n, frac, thin, scales=SCALES, log=print):
    """Screen the bandwidth grid for one (kind, n qubits, training subsample). Returns a list of results (one per
    scale): info, predictions per horizon on validation, chosen penalty per horizon."""
    ttr = train_origins(data, frac, thin)
    va = data.origins("val")
    Xtr, Xva = windows(data, ttr), windows(data, va)
    pca = PCA(n).fit(Xtr)
    Ztr, Zva = pca.transform(Xtr), pca.transform(Xva)
    Y = np.stack([(data.P[ttr + h] - data.P[ttr]) / data.scale for h in HORIZONS], 1)
    res = []
    for a in scales:
        t0 = time.time()
        kern = Kernel(kind, a)
        Etr, Eva = kern.embed(Ztr), kern.embed(Zva)
        K = kern.gram(Etr, Etr)
        st = offdiag_stats(K)
        A, ym, emin, emax = fit_all(K, Y)
        Pva = predict_all(kern, Etr, Eva, A, ym)                     # (G, len(va), H)
        preds, ridge, maes, edge = {}, {}, {}, False
        for j, h in enumerate(HORIZONS):
            sel = np.searchsorted(va, data.idx[h]["val"])
            tva = data.idx[h]["val"]
            y = data.P[tva + h]
            cand = data.P[tva][None, :] + Pva[:, sel, j] * data.scale
            m = np.abs(cand - y[None, :]).mean(1)
            g = int(np.argmin(m))
            preds[h], ridge[h], maes[h] = cand[g], RIDGE_GRID[g], float(m[g])
            edge |= g in (0, len(RIDGE_GRID) - 1)
        info = {"kind": kind, "n_qubits": n, "scale": a, "fraction": frac, "thin": thin, "n_train": int(len(ttr)),
                "n_params": int(len(ttr) * len(HORIZONS)), "pca_explained": pca.explained, **st,
                "gram_eig_min": emin, "gram_eig_max": emax, "ridge_at_grid_edge": bool(edge),
                **{f"ridge_h{h}": ridge[h] for h in HORIZONS}, **{f"val_MAE_h{h}": maes[h] for h in HORIZONS},
                "val_MAE_mean": float(np.mean(list(maes.values()))), "fit_s": time.time() - t0}
        res.append((info, preds))
        if log:
            log(f"    {kind:8s} q{n} {frac:.2f}/{thin} a={a:<4} | n_train {len(ttr):,} | offdiag mean "
                f"{st['mean_offdiag']:.3f} var {st['var_offdiag']:.2e} | val MAE " +
                " / ".join(f"{maes[h]:.4f}" for h in HORIZONS) + f" | {time.time() - t0:.0f} s")
    return res


def concentration(data, kinds, qubits, scale, frac=0.01, n_pts=500, seed=0):
    """Variance of off-diagonal kernel values on n_pts training points, per kind and qubit count (diagnostic)."""
    rows = []
    rng = np.random.default_rng(seed)
    ttr = train_origins(data, frac)
    pick = np.sort(rng.choice(ttr, min(n_pts, len(ttr)), replace=False))
    X = windows(data, pick)
    for n in qubits:
        Z = PCA(n).fit(X).transform(X)
        for k in kinds:
            kern = Kernel(k, scale)
            E = kern.embed(Z)
            rows.append({"kind": k, "n_qubits": n, "scale": scale, **offdiag_stats(kern.gram(E, E))})
    return rows


def run_name(kind, n, frac, thin, scale):
    from models import week6 as W
    t = f"{W.frac_tag(frac)}" + (f"t{thin}" if thin > 1 else "")
    return f"{kind}_q{n}_{t}_a{scale:.3f}".replace(".", "p")

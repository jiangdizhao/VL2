import numpy as np
import scipy.linalg as la
from numpy.linalg import slogdet
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any

# ----------------------------
# Small linear algebra helpers
# ----------------------------

def invs(A: np.ndarray) -> np.ndarray:
    """Safe inverse. If nearly singular, add a tiny diagonal ridge."""
    A = np.asarray(A)
    if A.ndim == 1 or (A.ndim == 2 and np.allclose(A, np.diag(np.diag(A)))):
        d = np.diag(A) if A.ndim == 2 else A
        eps = 1e-12
        return np.diag(1.0 / (d + eps))
    try:
        return la.inv(A)
    except la.LinAlgError:
        eps = 1e-12
        n = A.shape[0]
        return la.inv(A + eps * np.eye(n))

def logdet(A: np.ndarray) -> float:
    """Robust log|det(A)| using slogdet; fallback to SVD for safety."""
    A = np.asarray(A)
    sign, ldet = slogdet(A)
    if sign <= 0:
        s = la.svd(A, compute_uv=False)
        s = s[(s > 1e-16) & (s < 1e16)]
        if s.size == 0:
            return float(-1e16)
        return float(np.sum(np.log(s)))
    return float(ldet)

# ----------------------------
# Model containers
# ----------------------------

@dataclass
class Model:
    g: Optional[Callable[[np.ndarray, Dict[str, Any], Dict[str, Any]], np.ndarray]] = None  # y = g(p, M, U)
    IS: Optional[Callable[[np.ndarray, Dict[str, Any], Dict[str, Any]], np.ndarray]] = None # integrator (unused here)
    pE: Optional[np.ndarray] = None  # prior mean (parameters)
    pC: Optional[np.ndarray] = None  # prior covariance (parameters)
    hE: Optional[np.ndarray] = None  # prior mean (hyperparameters, log-precisions)
    hC: Optional[np.ndarray] = None  # prior covariance (hyperparameters)
    X:  Optional[np.ndarray] = None  # design matrix for GLM demos

    # Filled internally
    ipC: Optional[np.ndarray] = None
    ihC: Optional[np.ndarray] = None
    Ny:  Optional[int] = None
    Np:  Optional[int] = None
    Nh:  Optional[int] = None
    fun: Optional[Callable[[np.ndarray, Dict[str, Any], Dict[str, Any]], np.ndarray]] = None

@dataclass
class DataBundle:
    y: np.ndarray
    Q: Optional[List[np.ndarray]] = None  # precision components

# ----------------------------
# Variational Laplace (general)
# ----------------------------

class VariationalLaplace:
    def __init__(self, M: Model, U: Optional[Dict[str, Any]], Y: DataBundle, max_it: int = 100):
        self.M = M
        self.U = U if U is not None else {}
        self.Y = Y
        self.max_it = max_it
        self._init_structs()

    def _init_structs(self):
        M, Y = self.M, self.Y
        if Y.Q is None:
            Y.Q = [np.eye(len(Y.y))]
        # Priors -> precisions
        M.ipC = invs(M.pC)
        M.ihC = invs(M.hC if np.ndim(M.hC) == 2 else np.array([[M.hC]]))
        # Shapes
        M.Ny = len(Y.y)
        M.Np = M.pC.shape[0]
        M.Nh = len(Y.Q)
        # Choose generative function
        M.fun = M.IS if M.IS is not None else M.g
        if M.fun is None:
            raise ValueError("Please provide M.g or M.IS")
        # Coerce hyperparam shapes
        if np.ndim(M.hE) == 0:
            M.hE = np.array([float(M.hE)])
        if np.ndim(M.hC) == 0:
            M.hC = np.array([[float(M.hC)]])

    # ----- Finite-difference Jacobian -----
    def compute_g_gradient(self, p: np.ndarray) -> np.ndarray:
        M, U = self.M, self.U
        Ny, Np = M.Ny, M.Np
        J = np.zeros((Ny, Np))
        dx = np.exp(-8.0)  # same as MATLAB
        p0 = p.copy()
        g0 = M.fun(p0, M.__dict__, U)
        for i in range(Np):
            p_ = p0.copy()
            p_[i] += dx
            g1 = M.fun(p_, M.__dict__, U)
            J[:, i] = (g1 - g0) / dx
        return J

    # ----- Build data precision from hyperparameters -----
    def compute_data_precision(self, h: np.ndarray, Q: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        iS = np.zeros_like(Q[0])
        P = []
        for i in range(len(Q)):
            Pi = Q[i] * (np.exp(-32.0) + np.exp(h[i]))
            P.append(Pi)
            iS = iS + Pi
        return iS, P

    # ----- Parameter gradient & curvature -----
    def compute_F_gradient(self, p, y, J, iS) -> np.ndarray:
        M, U = self.M, self.U
        ep = p - M.pE
        ey = y - self.M.fun(p, M.__dict__, U)
        dFdp = J.T @ iS @ ey - M.ipC @ ep
        return dFdp

    def compute_F_curvature(self, J, iS) -> Tuple[np.ndarray, np.ndarray]:
        M = self.M
        dFdpp = -(J.T @ iS @ J) - M.ipC  # negative definite approx
        Pp = -dFdpp                      # posterior precision of p
        Cp = invs(Pp)
        return dFdpp, Cp

    # ----- Ozaki/LM-style parameter step -----
    def update_step(self, dFdpp, dFdp, logv) -> np.ndarray:
        n = dFdp.shape[0]
        # scale by average curvature magnitude
        t = np.exp(logv - logdet(dFdpp) / n)
        E = la.expm(dFdpp * t) - np.eye(n)
        dp = E @ la.inv(dFdpp) @ dFdp
        return dp

    # ----- Hyperparameter update (simple ReML-like per-component step) -----
    def update_h(self, h, p, J, Cp, P, iS, y) -> Tuple[np.ndarray, np.ndarray, bool]:
        M, U = self.M, self.U
        Nh = M.Nh
        r = y - M.fun(p, M.__dict__, U)
        S = invs(iS)
        S_pred = J @ Cp @ J.T  # obs covariance from param uncertainty
        eh = h - M.hE
        ihC = M.ihC

        Ch = np.zeros((Nh, Nh))
        has_converged = True
        for i in range(Nh):
            # Gradient approx
            tr_Qi_S = float(np.trace(P[i] @ S))
            rr_Qi = float(r.T @ P[i] @ r)
            tr_Qi_Spred = float(np.trace(P[i] @ S_pred))
            g_i = 0.5 * tr_Qi_S - 0.5 * (rr_Qi + tr_Qi_Spred) - float((ihC @ eh.reshape(-1,1))[i])

            # Diagonal Hessian approx
            H_ii = -0.5 * float(np.trace((P[i] @ S) @ (P[i] @ S))) - float(ihC[i, i])

            step = - g_i / (H_ii if H_ii != 0 else -1.0)
            old = h[i]
            h[i] = h[i] + step
            if abs(h[i] - old) > 1e-6:
                has_converged = False
            Ch[i, i] = -1.0 / (H_ii if H_ii != 0 else -1.0)

        return h, Ch, has_converged

    # ----- Free energy (ELBO) -----
    def calc_free_energy(self, p, h, y, iS, Cp, Ch) -> float:
        M, U = self.M, self.U
        ey = y - M.fun(p, M.__dict__, U)
        ep = p - M.pE
        eh = h - M.hE

        ipC = M.ipC
        ihC = M.ihC

        S = invs(iS)
        L1 = float(ey.T @ iS @ ey) + logdet(S)                # data term
        L2 = float(ep.T @ ipC @ ep) + logdet(self.M.pC)       # param prior
        L3 = float(eh.T @ ihC @ eh) + logdet(self.M.hC)       # hyper prior
        L4 = logdet(Cp) + logdet(Ch if Ch.size else np.eye(len(h)))  # post entropies

        F = -0.5 * L1 - 0.5 * L2 - 0.5 * L3 + 0.5 * L4
        return float(F)

    # ----- Main loop -----
    def fit(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, list]]:
        M, Y = self.M, self.Y
        y = Y.y.reshape(-1,)

        # Init
        p = M.pE.copy().reshape(-1,)
        h = M.hE.copy().reshape(-1,)
        logv = -4.0
        C = {"p": p.copy(), "h": h.copy(), "F": -np.inf, "Cp": np.eye(M.Np)}
        trace = {"F": [], "logv": [], "h": [], "step_norm": []}
        criterion = [False, False, False]

        for it in range(self.max_it):
            # Jacobian
            J = self.compute_g_gradient(p)

            # Hyperparameter refinement
            for _ in range(8):
                iS, P = self.compute_data_precision(h, Y.Q)
                _, Cp = self.compute_F_curvature(J, iS)
                h, Ch, has_converged = self.update_h(h, p, J, Cp, P, iS, y)
                if has_converged:
                    break

            # Parameter gradient/curvature and ELBO
            iS, P = self.compute_data_precision(h, Y.Q)
            dFdpp, Cp = self.compute_F_curvature(J, iS)
            dFdp = self.compute_F_gradient(p, y, J, iS)
            F = self.calc_free_energy(p, h, y, iS, Cp, Ch)

            # Accept / reject
            if F > C["F"]:
                C["p"], C["h"], C["F"], C["Cp"] = p.copy(), h.copy(), float(F), Cp.copy()
                logv = min(logv + 0.5, 4.0)         # be bolder
            else:
                p, h, Cp = C["p"].copy(), C["h"].copy(), C["Cp"].copy()
                logv = min(logv - 2.0, -4.0)        # be conservative
                # recompute around accepted state
                J = self.compute_g_gradient(p)
                iS, P = self.compute_data_precision(h, Y.Q)
                dFdpp, Cp = self.compute_F_curvature(J, iS)
                dFdp = self.compute_F_gradient(p, y, J, iS)
                F = C["F"]

            # Trace
            trace["F"].append(float(F))
            trace["logv"].append(float(logv))
            trace["h"].append(h.copy())

            # Parameter step
            dp = self.update_step(dFdpp, dFdp, logv)
            p = p + dp
            trace["step_norm"].append(float(np.linalg.norm(dp)))

            # Convergence heuristic
            dF_pred = float(dFdp.T @ dp)
            criterion = [ (dF_pred < 1e-1) ] + criterion[:2]
            if all(criterion):
                break

        Ep, Cp = C["p"], C["Cp"]
        Eh, F = C["h"], C["F"]
        Ch = Ch if 'Ch' in locals() else 1e-6*np.eye(M.Nh)
        return Ep, Cp, Eh, Ch, F, trace

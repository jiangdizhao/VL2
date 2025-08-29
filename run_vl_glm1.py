import numpy as np
import matplotlib.pyplot as plt
from variational_laplace_py import Model, DataBundle, VariationalLaplace

def run_glm1(seed: int = 42, N: int = 120):
    rng = np.random.default_rng(seed)

    # True GLM: y = X p + e
    x = np.linspace(0, 1, N)
    X = np.column_stack([np.ones_like(x), x])
    p_true = np.array([1.0, 5.0])
    h_true = np.array([4.0])  # log-precision (tau = exp(h))

    tau = float(np.exp(h_true[0]))
    y = X @ p_true + rng.normal(0.0, 1.0/np.sqrt(tau), size=N)

    # Build model/prior (mirrors run_VL_GLM1.m)
    M = Model()
    M.X = X
    M.g = lambda P, Mdict, U: Mdict["X"] @ P
    M.pE = np.zeros(2)             # prior mean (parameters)
    M.pC = np.eye(2) * 1.0         # prior covariance (parameters)
    M.hE = np.array([4.0])         # prior mean (hyperparameters)
    M.hC = np.array([[1.0]])       # prior covariance (hyperparameters)

    Y = DataBundle(y=y, Q=[np.eye(N)])  # single precision component

    # Fit via Variational Laplace
    vl = VariationalLaplace(M, U={}, Y=Y, max_it=64)
    Ep, Cp, Eh, Ch, F, trace = vl.fit()

    print("Posterior Ep (parameters):", Ep, "| True:", p_true)
    print("Posterior Eh (log-prec):  ", Eh, "| True:", h_true)
    print("Free Energy (ELBO):       ", F)

    # Plots like the MATLAB example
    plt.figure()
    plt.plot(np.array(trace["h"])[:,0])
    plt.xlabel("Iteration"); plt.ylabel("h (log-precision)"); plt.title("Hyperparameter trace")

    plt.figure()
    plt.plot(trace["F"])
    plt.xlabel("Iteration"); plt.ylabel("F (ELBO)"); plt.title("Free energy trace")

    plt.figure()
    plt.plot(y, linestyle='--', label='data')
    plt.plot(X @ Ep, label='fit')
    plt.xlabel("t / sample index"); plt.ylabel("y"); plt.title("GLM fit")
    plt.legend()

    plt.figure()
    mu = Ep
    sd = np.sqrt(np.diag(Cp))
    xpos = np.arange(len(mu))
    plt.errorbar(xpos, mu, yerr=sd, fmt='o', capsize=4, label='posterior')
    plt.plot(xpos, M.pE, marker='x', linestyle='None', label='prior mean')
    plt.xticks(xpos, ['intercept','slope'])
    plt.ylabel("value"); plt.title("Posterior parameters (mean ± 1 sd)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_glm1()

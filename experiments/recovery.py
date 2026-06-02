"""Recovering the initial data from a trained network. With y_j = 2 t W_j and
g_j = -b_j - |y_j|^2/(4t), the recovered initial data approaches the target
g(y) = |y| as eps -> 0. Writes figures/initial_data_recovery.pdf."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C1, C2, C3, C4,
    setup_style, figures_dir,
    lse_1d, weights_1d, mse_grad_1d, adam_step,
)

import numpy as np
from scipy.optimize import minimize

plt = setup_style()
OUT = figures_dir()

# -- Training helper -----------------------------------------------------------

def train_lbfgs(x_tr, y_tr, N, eps, n_adam=4000, lr=0.01, seed=0):
    """Adam warm-start then L-BFGS-B."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(N) * 0.4
    b = rng.standard_normal(N) * 0.2
    mW = vW = mb = vb = 0.0
    for t in range(1, n_adam + 1):
        loss, dW, db = mse_grad_1d(x_tr, y_tr, W, b, eps)
        upW, mW, vW = adam_step(dW, mW, vW, t, lr)
        upb, mb, vb = adam_step(db, mb, vb, t, lr)
        W -= upW;  b -= upb

    def obj(p):
        W_, b_ = p[:N], p[N:]
        l, dW_, db_ = mse_grad_1d(x_tr, y_tr, W_, b_, eps)
        return float(l), np.concatenate([dW_, db_])

    res = minimize(obj, np.concatenate([W, b]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 10000, "ftol": 1e-24, "gtol": 1e-14})
    return res.x[:N], res.x[N:], float(res.fun)

# -- Ground truth: Hopf-Lax (eps=0) with g*(y) = |y| -------------------------

print("Experiment A: initial-data recovery at multiple eps")

t_pde = 1.0
N_true = 8
y_true = np.linspace(-2.0, 2.0, N_true)
g_true = np.abs(y_true)

x_all  = np.linspace(-2.4, 2.4, 400)
costs  = g_true[None, :] + (x_all[:, None] - y_true[None, :]) ** 2 / (4 * t_pde)
u_0    = costs.min(axis=1)
f_star = x_all ** 2 / (4 * t_pde) - u_0

# -- Train at three eps values -------------------------------------------------

eps_list = [0.5, 0.1, 0.04]
N = 10

results = {}
for eps in eps_list:
    best_loss, best_W, best_b = np.inf, None, None
    for seed in range(8):
        W, b, loss = train_lbfgs(x_all, f_star, N, eps, n_adam=5000,
                                  lr=0.008, seed=seed)
        if loss < best_loss:
            best_loss, best_W, best_b = loss, W, b

    y_rec = 2 * t_pde * best_W
    g_rec = -best_b - y_rec ** 2 / (4 * t_pde)
    dev   = np.abs(g_rec - np.abs(y_rec))

    pi_tr  = weights_1d(x_all, best_W, best_b, eps)
    w_mean = pi_tr.mean(axis=0)
    dev_wavg = float((w_mean * dev).sum() / w_mean.sum())

    f_fit = lse_1d(x_all, best_W, best_b, eps)
    fit_err = np.max(np.abs(f_fit - f_star))

    print(f"  eps={eps:.2f}  MSE={best_loss:.2e}"
          f"  max_fit_err={fit_err:.4f}"
          f"  wtd_curve_dev={dev_wavg:.4f}")

    results[eps] = dict(W=best_W, b=best_b, y_rec=y_rec, g_rec=g_rec,
                        f_fit=f_fit, dev=dev, w_mean=w_mean,
                        dev_wavg=dev_wavg, mse=best_loss)

# -- Figure --------------------------------------------------------------------

y_curve = np.linspace(-2.5, 2.5, 300)
g_curve = np.abs(y_curve)

fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6))

for col, eps in enumerate(eps_list):
    r = results[eps]

    ax = axes[0, col]
    ax.plot(x_all, f_star, "-",  color=C1, lw=1.8, label=r"$f^*$ (Hopf-Lax)")
    ax.plot(x_all, r["f_fit"], "--", color=C2, lw=1.2,
            label=fr"trained ($\varepsilon={eps}$)")
    ax.set_title(fr"$\varepsilon = {eps}$")
    ax.set_xlabel(r"$x$");  ax.set_ylabel(r"$f(x)$")
    ax.legend(loc="upper left", handlelength=0.8, fontsize=6.5,
              borderpad=0.3, labelspacing=0.2)
    ax.grid(True, alpha=0.2, lw=0.4)

    ax = axes[1, col]
    ax.plot(y_curve, g_curve, "-", color="0.65", lw=1.2,
            label=r"$g^*(y)=|y|$")
    w = r["w_mean"]
    sizes = 20 + 180 * w / (w.max() + 1e-10)
    ax.scatter(r["y_rec"], r["g_rec"], c=r["dev"],
               cmap="RdYlGn_r", vmin=0, vmax=0.5,
               s=sizes, zorder=5, label=r"$(y_j,\,g_j)$ recovered")
    ax.set_xlabel(r"$y_j$");  ax.set_ylabel(r"$g_j$")
    ax.set_title(fr"Recovery ($\bar\delta = {r['dev_wavg']:.3f}$)")
    ax.legend(loc="upper right", handlelength=0.8, fontsize=6.5,
              borderpad=0.3, labelspacing=0.2)
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.set_xlim(-2.8, 2.8);  ax.set_ylim(-0.1, 2.6)

fig.tight_layout(pad=0.5)
fig.subplots_adjust(hspace=0.38)
fig.savefig(os.path.join(OUT, "initial_data_recovery.pdf"))
fig.savefig(os.path.join(OUT, "initial_data_recovery.png"), dpi=150)
plt.close(fig)
print("  Saved initial_data_recovery.pdf")

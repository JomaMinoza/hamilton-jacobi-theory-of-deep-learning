"""Robustness bound on CIFAR-10. An LSE network (Adam, PCA-64 inputs, N=128) is
checked against ||grad^2_x f||_2 <= ||W||_{2,inf}^2 / eps for eps in
{0.1, 0.3, 1, 3, 10}. Writes figures/hessian_cifar10.pdf."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C1, C4,
    setup_style, figures_dir,
    load_cifar10, hessian_norm_nd, adam_step,
)

import numpy as np

plt = setup_style(large=True)
OUT = figures_dir()

np.random.seed(7)

# -- LSE network helpers -------------------------------------------------------

def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def lse_fwd(X, W, b, eps):
    logits = X @ W.T + b[None, :]
    m = logits.max(axis=1, keepdims=True)
    return eps * (np.log(np.exp((logits - m) / eps).sum(axis=1)) + m[:, 0] / eps)

def lse_pi(X, W, b, eps):
    """Softmax weights pi_j(x). Returns (n, N)."""
    logits = X @ W.T + b[None, :]
    return _softmax(logits / eps)

def net_fwd(X, W, b, w2, eps):
    return lse_fwd(X, W, b, eps) * w2

def mse_grad(X, y, W, b, w2, eps):
    n    = len(y)
    h    = lse_fwd(X, W, b, eps)
    pi   = lse_pi(X, W, b, eps)
    f    = h * w2
    r    = f - y
    loss = np.mean(r ** 2)
    dW   = (2 / n) * (r * w2)[:, None, None] * pi[:, :, None] * X[:, None, :]
    dW   = dW.sum(axis=0)
    db   = (2 / n) * ((r * w2)[:, None] * pi).sum(axis=0)
    dw2  = (2 / n) * np.sum(r * h)
    return loss, dW, db, dw2

def train(X_tr, y_tr, N, eps, n_iter=5000, lr=0.005, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    d   = X_tr.shape[1]
    W   = rng.standard_normal((N, d)) * (0.1 / d ** 0.5)
    b   = np.zeros(N)
    w2  = 1.0
    mW = vW = mb = vb = mw2 = vw2 = 0.0
    n = len(X_tr)
    for step in range(1, n_iter + 1):
        idx = rng.choice(n, batch, replace=False)
        loss, dW, db, dw2_ = mse_grad(X_tr[idx], y_tr[idx], W, b, w2, eps)
        upW,  mW,  vW  = adam_step(dW,   mW,  vW,  step, lr)
        upb,  mb,  vb  = adam_step(db,   mb,  vb,  step, lr)
        upw2, mw2, vw2 = adam_step(dw2_, mw2, vw2, step, lr)
        W -= upW;  b -= upb;  w2 -= upw2
    return W, b, w2

# -- Main ----------------------------------------------------------------------

print("Experiment F: Hessian bound on CIFAR-10 (Corollary 8.2)")
print("Loading CIFAR-10 ...")
X_tr, y_tr, X_te, y_te = load_cifar10()
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

print("PCA projection to d=64 ...")
X_mean = X_tr.mean(axis=0)
Xc = X_tr - X_mean
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Vt64   = Vt[:64]
X_tr_p = (X_tr - X_mean) @ Vt64.T
X_te_p = (X_te - X_mean) @ Vt64.T
print(f"  Projected shape: {X_tr_p.shape}")

N_neurons = 128
N_test    = 300
N_ITER    = 5000
LR        = 0.005

eps_vals = np.array([0.1, 0.3, 1.0, 3.0, 10.0])

rng = np.random.default_rng(0)
test_idx   = rng.choice(len(X_te_p), N_test, replace=False)
X_test_sub = X_te_p[test_idx]

measured_max  = []
theory_bounds = []
fit_rmse      = []

print(f"\nTraining LSE network (N={N_neurons}, d=64) for each eps ...")
for eps in eps_vals:
    W, b, w2 = train(X_tr_p, y_tr, N_neurons, eps, n_iter=N_ITER, lr=LR, seed=0)

    f_te = net_fwd(X_te_p[:2000], W, b, w2, eps)
    rmse = np.sqrt(np.mean((f_te - y_te[:2000]) ** 2))
    fit_rmse.append(rmse)

    hsn     = hessian_norm_nd(X_test_sub, W, b, eps)
    max_hsn = hsn.max()
    measured_max.append(max_hsn)

    bound = (W ** 2).sum(axis=1).max() / eps
    theory_bounds.append(bound)

    status = "OK" if max_hsn <= bound + 1e-8 else "VIOLATED"
    print(f"  eps={eps:.1f}  RMSE={rmse:.3f}  "
          f"max||H||={max_hsn:.4f}  bound={bound:.4f}  {status}")

measured_max  = np.array(measured_max)
theory_bounds = np.array(theory_bounds)

assert np.all(measured_max <= theory_bounds + 1e-8), \
    "Bound violated -- check implementation"
print("\nBound holds for all eps values on CIFAR-10 test set.")

# -- Figure --------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3.2, 2.5))

ax.loglog(eps_vals, measured_max,  "o-",  color=C1,
          label=r"$\|\nabla^2 f\|_2$ (CIFAR-10)")
ax.loglog(eps_vals, theory_bounds, "--s", color=C4, markersize=3,
          label=r"$\|W\|_{2,\infty}^2/\varepsilon$ (bound)")

ax.set_xlabel(r"Viscosity $\varepsilon$")
ax.set_ylabel(r"Spectral norm")
ax.set_title(r"Hessian bound: CIFAR-10")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
          handlelength=0.8, fontsize=7.5, borderpad=0.3,
          labelspacing=0.2, columnspacing=0.8)
ax.grid(True, which="both", alpha=0.25, lw=0.5)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.26)
fig.savefig(os.path.join(OUT, "hessian_cifar10.pdf"))
fig.savefig(os.path.join(OUT, "hessian_cifar10.png"), dpi=150)
plt.close(fig)
print("Saved hessian_cifar10.pdf")

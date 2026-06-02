"""Robustness bound for small 1-D networks trained on f(x) = |x| (N=30), swept
over six values of eps. Writes figures/robustness_sgd.pdf."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C1, C4,
    setup_style, figures_dir,
    lse_1d, mse_grad_1d, adam_step, hessian_norm_1d,
)

import numpy as np

plt = setup_style(large=True)
OUT = figures_dir()

np.random.seed(1)

# -- Training helper -----------------------------------------------------------

def train(x_tr, y_tr, N, eps, n_iter=3000, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(N) * 0.4
    b = rng.standard_normal(N) * 0.2
    mW = vW = mb = vb = 0.0
    for t in range(1, n_iter + 1):
        loss, dW, db = mse_grad_1d(x_tr, y_tr, W, b, eps)
        upW, mW, vW = adam_step(dW, mW, vW, t, lr)
        upb, mb, vb = adam_step(db, mb, vb, t, lr)
        W -= upW;  b -= upb
    return W, b

# -- Experiment ----------------------------------------------------------------

print("Experiment C: robustness bound for trained networks")

N_rob   = 30
N_TRAIN = 500
N_TEST  = 200
N_ITER  = 4000
LR      = 0.01

rng = np.random.default_rng(99)
x_tr = rng.uniform(-2.0, 2.0, N_TRAIN)
x_te = np.linspace(-1.8, 1.8, N_TEST)
y_tr = np.abs(x_tr)
y_te = np.abs(x_te)

eps_vals = np.array([0.08, 0.12, 0.2, 0.35, 0.6, 1.0])

measured_max  = []
theory_bounds = []
fit_rmse      = []

for eps in eps_vals:
    best_rmse = np.inf
    best_W, best_b = None, None
    for seed in range(3):
        W, b = train(x_tr, y_tr, N_rob, eps, N_ITER, LR, seed=seed)
        f_te = lse_1d(x_te, W, b, eps)
        rmse = np.sqrt(np.mean((f_te - y_te) ** 2))
        if rmse < best_rmse:
            best_rmse, best_W, best_b = rmse, W, b

    W, b = best_W, best_b
    fit_rmse.append(best_rmse)

    hsn = hessian_norm_1d(x_te, W, b, eps)
    max_hsn = hsn.max()
    measured_max.append(max_hsn)

    M2 = np.max(W ** 2)
    bound = M2 / eps
    theory_bounds.append(bound)

    print(f"  eps={eps:.2f}  fit RMSE={best_rmse:.4f}"
          f"  max_Hess={max_hsn:.4f}  bound={bound:.4f}"
          f"  {'OK' if max_hsn <= bound + 1e-10 else 'VIOLATED'}")

measured_max  = np.array(measured_max)
theory_bounds = np.array(theory_bounds)

assert np.all(measured_max <= theory_bounds + 1e-10), \
    "Bound violated -- check implementation"
print("  Bound verified: ||nabla^2 f||_2 <= ||W||_{2,inf}^2/eps for all eps")

# -- Figure --------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3.2, 2.5))

ax.loglog(eps_vals, measured_max,  "o-",  color=C1,
          label=r"$\|\nabla^2 f\|_2$ (measured)")
ax.loglog(eps_vals, theory_bounds, "--s", color=C4, markersize=3,
          label=r"$\|W\|_{2,\infty}^2/\varepsilon$ (bound)")

eps_ref = np.array([eps_vals[0], eps_vals[-1]], dtype=float)
ax.loglog(eps_ref,
          measured_max[0] * (eps_ref / eps_vals[0]) ** (-1),
          ":", color="0.5", lw=1.0, label=r"$\varepsilon^{-1}$ slope")

ax.set_xlabel(r"Viscosity $\varepsilon$")
ax.set_ylabel(r"Spectral norm")
ax.set_title(r"Hessian bound (trained networks)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
          handlelength=0.8, fontsize=6.5, borderpad=0.3,
          labelspacing=0.2, columnspacing=0.8)
ax.grid(True, which="both", alpha=0.25, lw=0.5)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.28)
fig.savefig(os.path.join(OUT, "robustness_sgd.pdf"))
fig.savefig(os.path.join(OUT, "robustness_sgd.png"), dpi=150)
plt.close(fig)
print("  Saved robustness_sgd.pdf")

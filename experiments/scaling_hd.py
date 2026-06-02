"""Extends the scaling law to d=4. Reads the d=1,2 runs from exp_B_results.csv,
trains d=4, and regenerates the combined figure. Run scaling.py first."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C1, C2, C3,
    setup_style, figures_dir,
    lse_nd, mse_grad_nd, adam_step, log_slope,
)

import numpy as np
import csv

plt = setup_style(large=True)
OUT = figures_dir()

np.random.seed(0)

CSV = os.path.join(os.path.dirname(__file__), "exp_B_results.csv")

# -- Training helper -----------------------------------------------------------

def train_nd(x_tr, y_tr, N, eps, n_iter, lr, seed):
    rng = np.random.default_rng(seed)
    d = x_tr.shape[1]
    W = rng.standard_normal((N, d)) * (0.5 / np.sqrt(d))
    b = rng.standard_normal(N) * 0.3
    mW = np.zeros_like(W);  vW = np.zeros_like(W)
    mb = np.zeros_like(b);  vb = np.zeros_like(b)
    for t in range(1, n_iter + 1):
        loss, dW, db = mse_grad_nd(x_tr, y_tr, W, b, eps)
        upW, mW, vW = adam_step(dW, mW, vW, t, lr)
        upb, mb, vb = adam_step(db, mb, vb, t, lr)
        W -= upW;  b -= upb
    return W, b

# -- Load existing d=1, d=2 results from CSV -----------------------------------

print("Loading d=1, d=2 from exp_B_results.csv ...")
results = {}
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = int(row["d"])
        N = int(row["N"])
        rmse = float(row["rmse"])
        if d not in results:
            results[d] = ([], [])
        results[d][0].append(N)
        results[d][1].append(rmse)

for d in results:
    print(f"  d={d}: {len(results[d][0])} points loaded")

# -- Train d=4 -----------------------------------------------------------------

print("Experiment B (d=4): scaling law from Adam-trained networks")

N_TRAIN = 4000
N_TEST  = 800
N_ITER  = 3000
LR      = 0.006

d = 4
N_list = [50, 100, 200, 500]

rng_data = np.random.default_rng(42 + d)
x_tr = rng_data.uniform(-2.0, 2.0, (N_TRAIN, d))
x_te = rng_data.uniform(-2.0, 2.0, (N_TEST,  d))
y_tr = np.linalg.norm(x_tr, axis=1)
y_te = np.linalg.norm(x_te, axis=1)

rmse_list = []
for N in N_list:
    eps = N ** (-1.0 / d)
    best_rmse = np.inf
    for seed in range(5):
        W, b  = train_nd(x_tr, y_tr, N, eps, N_ITER, LR, seed=seed + d * 100)
        f_te  = lse_nd(x_te, W, b, eps)
        rmse  = np.sqrt(np.mean((f_te - y_te) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse

    rmse_list.append(best_rmse)
    print(f"  d={d}  N={N:4d}  eps={eps:.4f}  RMSE={best_rmse:.4f}")

results[d] = (N_list, rmse_list)

# -- Append d=4 results to CSV -------------------------------------------------

with open(CSV, "a", newline="") as f:
    writer = csv.writer(f)
    for N, rmse in zip(N_list, rmse_list):
        eps = N ** (-1.0 / d)
        writer.writerow([d, N, f"{eps:.4f}", f"{rmse:.4f}"])
print("  Appended d=4 results to exp_B_results.csv")

# -- Figure (all three dimensions) ---------------------------------------------

fig, ax = plt.subplots(figsize=(3.2, 2.5))

for d, col, mk in [(1, C1, "o"), (2, C2, "s"), (4, C3, "^")]:
    Ns, Es = results[d]
    alpha = -log_slope(Ns, Es)
    ax.loglog(Ns, Es, mk + "-", color=col,
              label=fr"$d={d}$,  $\hat\alpha={alpha:.2f}$")
    Nref = np.array([Ns[1], Ns[-1]], dtype=float)
    ax.loglog(Nref, Es[1] * (Nref / Ns[1]) ** (-1 / d),
              "--", color=col, alpha=0.45, lw=1.0)

ax.set_xlabel(r"Network width $N$")
ax.set_ylabel(r"Test RMSE")
ax.set_title(r"Scaling law (trained networks)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
          handlelength=0.8, fontsize=6.5, borderpad=0.3,
          labelspacing=0.2, columnspacing=0.8)
ax.grid(True, which="both", alpha=0.25, lw=0.5)

ax.text(0.62, 0.62, r"$N^{-1}$",    transform=ax.transAxes, fontsize=7, color=C1)
ax.text(0.62, 0.48, r"$N^{-1/2}$",  transform=ax.transAxes, fontsize=7, color=C2)
ax.text(0.62, 0.34, r"$N^{-1/4}$",  transform=ax.transAxes, fontsize=7, color=C3)

fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.25)
fig.savefig(os.path.join(OUT, "scaling_law_sgd.pdf"))
fig.savefig(os.path.join(OUT, "scaling_law_sgd.png"), dpi=150)
plt.close(fig)
print("  Saved scaling_law_sgd.pdf (d=1,2,4)")

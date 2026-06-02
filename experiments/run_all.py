"""Runs the core checks and writes their outputs to figures/: the LSE = Hopf-Cole
identity, the N^{-1/d} quadrature rate, the scaling law, and the robustness
bound."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C1, C2, C3, C4,
    setup_style, figures_dir,
    log_slope,
)

import numpy as np

plt = setup_style(large=True)

np.random.seed(42)

OUT = figures_dir()

# -- helpers -------------------------------------------------------------------

def lse_stable(logits, eps):
    """eps * log sum_j exp(logits_j / eps), stable."""
    m = logits.max(axis=-1, keepdims=True)
    return eps * (np.log(np.exp((logits - m) / eps).sum(axis=-1))
                  + m.squeeze(-1) / eps)

def hopf_cole_1d(x, y, g, eps, t=1.0):
    """Hopf-Cole solution u(x,t) for 1-D. Returns (N_x,)."""
    exp = (-g[None, :] - (x[:, None] - y[None, :])**2 / (4*t)) / eps
    m   = exp.max(axis=1, keepdims=True)
    return -eps * (np.log(np.exp(exp - m).sum(axis=1)) + m[:, 0])

def lse_1d(x, y, g, eps, t=1.0):
    """LSE(W x + b) for 1-D neurons. Equals |x|^2/(4t) - u_HC(x,t)."""
    W = y / (2*t);  b = -g - y**2 / (4*t)
    return lse_stable(x[:, None] * W[None, :] + b[None, :], eps)

def lse_nd(x, y, g, eps, t=1.0):
    """LSE(W x + b) for n-D neurons. x:(N_x,d), y:(N_y,d), g:(N_y,)."""
    W = y / (2*t);  b = -g - (y**2).sum(-1) / (4*t)
    return lse_stable(x @ W.T + b[None, :], eps)

def norm_lse_nd(x, y, g, eps, t=1.0):
    """Normalized LSE: eps*log(mean_j exp(logit_j/eps)) = lse_nd - eps*log(N)."""
    N = len(y)
    return lse_nd(x, y, g, eps, t) - eps * np.log(N)

# -----------------------------------------------------------------------------
# Experiment 1  Verification table (Theorem 4.1)
# -----------------------------------------------------------------------------

print("Experiment 1: Verification (Theorem 4.1)")

t  = 1.0
y1 = np.array([-1.5, -0.5,  0.5,  1.5])
g1 = np.array([ 1.0,  0.3,  0.3,  1.0])
x1 = np.linspace(-2.5, 2.5, 500)

lines = ["eps       max| LSE + u_HC - |x|^2/(4t) |",
         "-" * 46]
for eps in [1.0, 0.5, 0.2, 0.1, 0.05]:
    lv  = lse_1d(x1, y1, g1, eps, t)
    uhc = hopf_cole_1d(x1, y1, g1, eps, t)
    err = np.max(np.abs(lv + uhc - x1**2 / (4*t)))
    lines.append(f"{eps:.2f}      {err:.2e}")

with open(os.path.join(OUT, "verification_table.txt"), "w") as f:
    f.write("\n".join(lines))
for r in lines:
    print("  " + r)

# -----------------------------------------------------------------------------
# Experiment 2  Quadrature convergence rate  O(N^{-1/d})
# -----------------------------------------------------------------------------

print("\nExperiment 2: Quadrature rate O(N^{-1/d})")

eps_q = 1.0;  t_q = 1.0;  L_lip = 1.0

def grid_nd(d, n_per_dim, lo=-2.0, hi=2.0):
    axes  = [np.linspace(lo, hi, n_per_dim)] * d
    grids = np.meshgrid(*axes, indexing="ij")
    y     = np.stack([g.ravel() for g in grids], axis=1)
    g_arr = L_lip * np.sqrt((y**2).sum(-1))
    return y, g_arr

def run_qrate(d, N_per_dim_list, n_ref, n_eval=400, mc_ref=None):
    rng  = np.random.default_rng(100 + d)
    x_ev = rng.uniform(-1.5, 1.5, (n_eval, d))
    if mc_ref is not None:
        rng_mc = np.random.default_rng(777 + d)
        y_r    = rng_mc.uniform(-2.0, 2.0, (mc_ref, d))
        g_r    = L_lip * np.sqrt((y_r**2).sum(-1))
    else:
        y_r, g_r = grid_nd(d, n_ref)
    f_ref = norm_lse_nd(x_ev, y_r, g_r, eps_q, t_q)
    errs = []
    for Npd in N_per_dim_list:
        y_c, g_c = grid_nd(d, Npd)
        f_c      = norm_lse_nd(x_ev, y_c, g_c, eps_q, t_q)
        errs.append(np.max(np.abs(f_c - f_ref)))
        N = Npd**d
        print(f"  d={d} N={N:5d}  err={errs[-1]:.3e}")
    return errs

N_pd1  = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
N_pd2  = [3, 5, 8, 12, 18, 28, 40, 60]
N_pd4  = [2, 3, 4, 5, 6, 7, 9, 12, 15]

errs1 = run_qrate(1, N_pd1, n_ref=16384)
errs2 = run_qrate(2, N_pd2, n_ref=120)
errs4 = run_qrate(4, N_pd4, n_ref=13, mc_ref=100000)

N_tot1 = [n    for n in N_pd1]
N_tot2 = [n**2 for n in N_pd2]
N_tot4 = [n**4 for n in N_pd4]

s1 = log_slope(N_tot1, errs1)
s2 = log_slope(N_tot2, errs2)
s4 = log_slope(N_tot4, errs4)
print(f"\n  Fitted slopes: d=1 {s1:.2f} (theory -1.00)")
print(f"                 d=2 {s2:.2f} (theory -0.50)")
print(f"                 d=4 {s4:.2f} (theory -0.25)")

fig, ax = plt.subplots(figsize=(3.0, 2.3))
ax.loglog(N_tot1, errs1, "o-", color=C1, label=r"$d=1$")
ax.loglog(N_tot2, errs2, "s-", color=C2, label=r"$d=2$")
ax.loglog(N_tot4, errs4, "^-", color=C3, label=r"$d=4$")

for d, Nts, Es, col in [(1,N_tot1,errs1,C1),(2,N_tot2,errs2,C2),(4,N_tot4,errs4,C3)]:
    Nref = np.array([Nts[1], Nts[-1]], dtype=float)
    ax.loglog(Nref, Es[1]*(Nref/Nts[1])**(-1/d), "--", color=col, alpha=0.5, lw=1.0)

ax.set_xlabel(r"Neurons $N$")
ax.set_ylabel(r"$\ell^\infty$ error")
ax.set_title(r"Rate $O(N^{-1/d})$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
          handlelength=0.8, fontsize=6.5, borderpad=0.3,
          labelspacing=0.2, columnspacing=1.0)
ax.grid(True, which="both", alpha=0.25, lw=0.5)
fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.25)
fig.savefig(os.path.join(OUT, "gen_rate.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "gen_rate.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved gen_rate.pdf")

# -----------------------------------------------------------------------------
# Experiment 3  Scaling law  alpha = 1/d_eff
# -----------------------------------------------------------------------------

print("\nExperiment 3: Scaling law alpha = 1/d_eff")

def scaling_curve(d, N_per_dim_list, n_eval=400):
    rng  = np.random.default_rng(200 + d)
    x_ev = rng.uniform(-1.5, 1.5, (n_eval, d))
    x_sq = (x_ev**2).sum(-1)
    losses = []
    for Npd in N_per_dim_list:
        N   = Npd**d
        eps = N**(-1.0/d)
        target = x_sq / 12.0 + (d/2)*eps*np.log(4*np.pi*eps/3) - d*eps*np.log(4)
        y_c, _ = grid_nd(d, Npd)
        g_c    = 0.5 * (y_c**2).sum(-1)
        f_c    = norm_lse_nd(x_ev, y_c, g_c, eps, t_q)
        losses.append(np.mean((f_c - target)**2)**0.5)
        print(f"  d_eff={d} N={N:5d}  eps={eps:.4f}  loss={losses[-1]:.4e}")
    return losses

L1 = scaling_curve(1, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
L2 = scaling_curve(2, [3, 5,  8, 12, 18,  28, 40, 60])
L4 = scaling_curve(4, [7, 9, 11, 13, 15])

N_sl1 = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
N_sl2 = [n**2 for n in [3, 5,  8, 12, 18,  28, 40, 60]]
N_sl4 = [n**4 for n in [7, 9, 11, 13, 15]]

a1 = log_slope(N_sl1, L1)
a2 = log_slope(N_sl2, L2)
a4 = log_slope(N_sl4, L4)
print(f"\n  Fitted alpha: d=1 {-a1:.2f} (theory 1.00)")
print(f"               d=2 {-a2:.2f} (theory 0.50)")
print(f"               d=4 {-a4:.2f} (theory 0.25)")

fig, ax = plt.subplots(figsize=(3.0, 2.3))
ax.loglog(N_sl1, L1, "o-", color=C1,
          label=f"$d=1$\n$\\hat\\alpha={-a1:.2f}$")
ax.loglog(N_sl2, L2, "s-", color=C2,
          label=f"$d=2$\n$\\hat\\alpha={-a2:.2f}$")
ax.loglog(N_sl4, L4, "^-", color=C3,
          label=f"$d=4$\n$\\hat\\alpha={-a4:.2f}$")

for d, Nts, Ls, col in [(1,N_sl1,L1,C1),(2,N_sl2,L2,C2),(4,N_sl4,L4,C3)]:
    if len(Nts) >= 2:
        Nref = np.array([Nts[2], Nts[-1]], dtype=float)
        ax.loglog(Nref, Ls[2]*(Nref/Nts[2])**(-1/d),
                  "--", color=col, alpha=0.5, lw=1.0)

ax.set_xlabel(r"Width $N$")
ax.set_ylabel(r"RMSE loss")
ax.set_title(r"Scaling law $N^{-\alpha}$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=3,
          handlelength=0.8, fontsize=6.5, borderpad=0.3,
          labelspacing=0.15, columnspacing=1.0)
ax.grid(True, which="both", alpha=0.25, lw=0.5)
fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.30)
fig.savefig(os.path.join(OUT, "scaling_law.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "scaling_law.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved scaling_law.pdf")

# -----------------------------------------------------------------------------
# Experiment 4  Robustness  (Corollary 8.2)
# -----------------------------------------------------------------------------

print("\nExperiment 4: Robustness vs epsilon (Corollary 8.2)")

rng   = np.random.default_rng(3)
N_rob = 40;  D_rob = 8
W_rob = rng.standard_normal((N_rob, D_rob))
W_rob /= np.linalg.norm(W_rob, axis=1, keepdims=True)
b_rob = np.zeros(N_rob)
x0    = rng.standard_normal(D_rob);  x0 /= np.linalg.norm(x0)
M     = np.max(np.linalg.norm(W_rob, axis=1))

eps_vals   = np.logspace(-1.3, 1, 60)
h_measured = []
h_bound    = []
for eps in eps_vals:
    logits = x0 @ W_rob.T + b_rob
    pi     = np.exp((logits - logits.max()) / eps);  pi /= pi.sum()
    Wp     = W_rob * pi[:, None]
    H      = (Wp.T @ W_rob - np.outer(W_rob.T @ pi, W_rob.T @ pi)) / eps
    h_measured.append(np.linalg.svd(H, compute_uv=False)[0])
    h_bound.append(M**2 / eps)

fig, ax = plt.subplots(figsize=(3.0, 2.3))
ax.loglog(eps_vals, h_measured, "-",  color=C1,
          label=r"$\|\nabla^2_x f\|_2$ (measured)")
ax.loglog(eps_vals, h_bound,    "--", color=C4,
          label=r"$\|W\|_{2,\infty}^2/\varepsilon$ (bound)")
ax.set_xlabel(r"Viscosity $\varepsilon$")
ax.set_ylabel(r"Spectral norm")
ax.set_title(r"Hessian bound")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
          handlelength=0.8, fontsize=6.5, borderpad=0.3,
          labelspacing=0.2, columnspacing=0.8)
ax.grid(True, which="both", alpha=0.25, lw=0.5)
fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.25)
fig.savefig(os.path.join(OUT, "robustness.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "robustness.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved robustness.pdf")

print(f"\nDone. Figures in: {OUT}")

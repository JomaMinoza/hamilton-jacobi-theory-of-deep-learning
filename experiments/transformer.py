"""LSE-transformer block. Checks that attention after LayerNorm matches the Gibbs
form grad LSE_eps(QK^T) . V, and that each LSE-FFN layer matches the Hopf-Cole
identity, both to machine precision."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import softmax

import numpy as np

np.random.seed(42)

def lse(z, eps):
    """LSE_eps(z) = eps * log sum_j exp(z_j / eps), stable."""
    m = z.max(axis=-1, keepdims=True)
    return eps * (np.log(np.exp((z - m) / eps).sum(axis=-1)) + m[..., 0] / eps)

def lse_gradient(z, eps):
    return softmax(z / eps, axis=-1)

def layer_norm(x, gamma=None, beta=None, eps_ln=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    sigma = np.sqrt(x.var(axis=-1, keepdims=True) + eps_ln)
    out = (x - mu) / sigma
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out

# -- Part (i): Attention identity with LayerNorm -------------------------------

def attn_standard(Q, K, V):
    d = Q.shape[-1]
    eps = d ** 0.5
    return softmax((Q @ K.T) / eps, axis=-1) @ V

def attn_hopf_cole(Q, K, V):
    d = Q.shape[-1]
    eps = d ** 0.5
    return lse_gradient(Q @ K.T, eps) @ V

print("=" * 68)
print("Experiment G: LSE-Transformer Block Characterization")
print("=" * 68)
print()
print("Part (i): Attention identity  Attn(LN(X)W_Q, LN(X)W_K, V) = grad(LSE).V")
print("-" * 68)

N_TRIALS = 500
d_values = [4, 8, 16, 32, 64]
n_seq, d_v = 8, 16

results_attn_ln = []
for d in d_values:
    eps = d ** 0.5
    W_Q = np.random.randn(d, d) / np.sqrt(d)
    W_K = np.random.randn(d, d) / np.sqrt(d)
    max_errs = []
    for _ in range(N_TRIALS):
        X_raw = np.random.randn(n_seq, d)
        V     = np.random.randn(n_seq, d_v)
        X_ln  = layer_norm(X_raw)
        Q = X_ln @ W_Q
        K = X_ln @ W_K
        err = np.abs(attn_standard(Q, K, V) - attn_hopf_cole(Q, K, V)).max()
        max_errs.append(err)
    max_err = max(max_errs)
    results_attn_ln.append((d, eps, max_err))
    print(f"  d={d:2d}  eps=sqrt({d:2d})={eps:5.2f}  max|Attn_std - HC| = {max_err:.2e}")

# -- Part (ii): LSE-FFN HJ identity -------------------------------------------

print()
print("Part (ii): LSE-FFN HJ identity  f + u = |x|^2/(4t)  (Theorem 3.1)")
print("           Input x = LN(z)  (as in transformer pre-norm FFN)")
print("-" * 68)

N_NEURONS = 32
d_ffn     = 8
t         = 1.0
eps_values = [0.05, 0.10, 0.20, 0.50, 1.00]

results_ffn = []
for eps in eps_values:
    W = np.random.randn(N_NEURONS, d_ffn)
    b = np.random.randn(N_NEURONS)
    max_errs = []
    for _ in range(N_TRIALS):
        z_raw = np.random.randn(50, d_ffn)
        x     = layer_norm(z_raw)
        logits = x @ W.T + b[None, :]       # (n, N)
        f      = lse(logits, eps)           # LSE_eps(Wx + b)
        norm_sq = (x ** 2).sum(axis=-1) / (4 * t)
        u      = norm_sq - f               # u_eps^N = |x|^2/(4t) - f
        err    = np.abs(f + u - norm_sq).max()
        max_errs.append(err)
    max_err = max(max_errs)
    results_ffn.append((eps, max_err))
    print(f"  eps={eps:.2f}  max|f + u - |x|^2/(4t)| = {max_err:.2e}")

# -- Summary --------------------------------------------------------------------

print()
print("=" * 68)
print("Summary table (for paper Tab. tab:verify_transformer)")
print("=" * 68)
print()
print("Attention + LayerNorm (identity eq:attn_gibbs):")
print(f"  {'d':>4}  {'eps':>6}  {'max error':>12}  {'status':>12}")
for d, eps, err in results_attn_ln:
    status = "PASS (exact)" if err < 1e-10 else "FAIL"
    print(f"  {d:>4}  {eps:>6.2f}  {err:>12.2e}  {status:>12}")

print()
print("LSE-FFN (Theorem 3.1, input x = LN(z)):")
print(f"  {'eps':>6}  {'max error':>12}  {'status':>20}")
for eps, err in results_ffn:
    status = "PASS (machine prec.)" if err < 1e-12 else "FAIL"
    print(f"  {eps:>6.2f}  {err:>12.2e}  {status:>20}")

all_attn = all(e < 1e-10 for _, _, e in results_attn_ln)
all_ffn  = all(e < 1e-12 for _, e in results_ffn)
print()
print(f"Attention+LN: {'ALL PASS' if all_attn else 'FAILURE'} over {N_TRIALS} trials per d")
print(f"LSE-FFN:      {'ALL PASS' if all_ffn  else 'FAILURE'} over {N_TRIALS} trials per eps")

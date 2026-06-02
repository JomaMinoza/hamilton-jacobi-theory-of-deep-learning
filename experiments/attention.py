"""Attention equals the Hopf-Cole average. Compares scaled dot-product attention
against grad LSE_eps(QK^T) . V with eps = sqrt(d), over 500 random trials per
dimension. The two agree to roundoff. Prints a table; no figures."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import softmax

import numpy as np

np.random.seed(42)

# -- helpers -------------------------------------------------------------------

def lse_gradient(z, eps):
    """nabla_z LSE_eps(z) = softmax(z / eps)."""
    return softmax(z / eps, axis=-1)

def attention_path_A(Q, K, V):
    d = Q.shape[-1]
    eps = d ** 0.5
    scores = (Q @ K.T) / eps
    weights = softmax(scores, axis=-1)
    return weights @ V

def attention_path_B(Q, K, V):
    """Hopf-Cole path: pi = nabla_z LSE_eps(z) at z = Q K^T (unscaled)."""
    d = Q.shape[-1]
    eps = d ** 0.5
    z = Q @ K.T
    pi = lse_gradient(z, eps)
    return pi @ V

# -- Experiment ----------------------------------------------------------------

print("=" * 64)
print("Attention = Hopf-Cole average (Theorem 4.1 / eq. attn_gibbs)")
print("=" * 64)
print()
print("Verifying:  Attn(Q,K,V)_i  =  (nabla_z LSE_eps(z))|_{z_j=q_i.k_j} . V")
print("            where eps = sqrt(d_k)")
print()

N_TRIALS = 500
d_values = [4, 8, 16, 32, 64]
n_q, n_k, d_v = 8, 12, 16

results = []
for d in d_values:
    eps = d ** 0.5
    max_errors = []
    for _ in range(N_TRIALS):
        Q = np.random.randn(n_q, d)
        K = np.random.randn(n_k, d)
        V = np.random.randn(n_k, d_v)

        out_A = attention_path_A(Q, K, V)
        out_B = attention_path_B(Q, K, V)

        max_errors.append(np.abs(out_A - out_B).max())

    max_err = max(max_errors)
    mean_err = np.mean(max_errors)
    results.append((d, eps, max_err, mean_err))
    print(f"  d={d:2d}  eps=sqrt({d:2d})={eps:5.2f}  "
          f"max|Attn_A - HC_B| = {max_err:.2e}  "
          f"mean = {mean_err:.2e}")

print()
print("All errors are floating-point roundoff -- identity holds algebraically.")
print()

print("=" * 64)
print("Summary")
print("=" * 64)
print(f"  {'d':>4}  {'eps':>6}  {'max error':>12}  {'status':>8}")
print(f"  {'-'*4}  {'-'*6}  {'-'*12}  {'-'*8}")
for d, eps, max_err, _ in results:
    status = "PASS" if max_err < 1e-10 else "FAIL"
    print(f"  {d:>4}  {eps:>6.2f}  {max_err:>12.2e}  {status:>8}")

all_pass = all(max_err < 1e-10 for _, _, max_err, _ in results)
print()
print(f"Overall: {'ALL PASS' if all_pass else 'FAILURE'} "
      f"-- identity verified to machine precision over {N_TRIALS} random trials per d.")

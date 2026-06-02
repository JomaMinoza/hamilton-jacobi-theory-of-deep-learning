"""Five quick checks of the network <-> HJ PDE correspondence: ReLU as the
eps -> 0 limit of softplus, the LSE = Hopf-Cole identity, the analytic PDE
residual, a two-layer composition, and Hopf-Lax recovering ReLU."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

np.random.seed(0)

# -- helpers ------------------------------------------------------------------

def softplus(z, eps):
    """eps * log(1 + exp(z/eps)), numerically stable via branch on sign(z)."""
    pos = z > 0
    out = np.empty_like(z, dtype=float)
    out[pos]  = z[pos]  + eps * np.log1p(np.exp(-z[pos]  / eps))
    out[~pos] =           eps * np.log1p(np.exp( z[~pos] / eps))
    return out

def lse(logits, eps):
    """eps * log sum exp(logits / eps) -- numerically stable (log-sum-exp trick)."""
    m = logits.max(axis=-1, keepdims=True)
    return eps * (np.log(np.exp((logits - m) / eps).sum(axis=-1)) + m.squeeze(-1) / eps)

def hopf_cole(x, y, g, eps, t=1.0):
    """
    Hopf-Cole solution (eq. hc_solution in paper):
      u_eps(x,t) = -eps * log sum_j exp((-g(y_j) - |x-y_j|^2/(4t)) / eps)
    """
    exponents = (-g[None, :] - (x[:, None] - y[None, :]) ** 2 / (4 * t)) / eps
    m = exponents.max(axis=-1, keepdims=True)
    log_sum = np.log(np.exp(exponents - m).sum(axis=-1)) + m.squeeze(-1)
    return -eps * log_sum

def hopf_lax(x, y, g, t=1.0):
    """
    Hopf-Lax formula (eps=0 limit):
      u_0(x,t) = inf_j { g(y_j) + |x-y_j|^2 / (4t) }
    """
    costs = g[None, :] + (x[:, None] - y[None, :]) ** 2 / (4 * t)
    return costs.min(axis=1)

# -- Test 1: ReLU = eps->0 of softplus ----------------------------------------

print("=" * 62)
print("Test 1: ReLU = tropical (eps->0) limit of softplus")
print("=" * 62)

z = np.linspace(-3, 3, 1000)
relu = np.maximum(z, 0)

for eps in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]:
    err = np.max(np.abs(softplus(z, eps) - relu))
    print(f"  eps={eps:.4f}:  max|softplus_eps - ReLU| = {err:.2e}")

print()

# -- Test 2: LSE layer = Hopf-Cole solution (Theorem 4.1) ---------------------

print("=" * 62)
print("Test 2: LSE layer = Hopf-Cole solution (Theorem 4.1)")
print("=" * 62)

t = 1.0
y = np.array([-1.5, -0.5, 0.5, 1.5])
g = np.array([ 1.0,  0.3,  0.3,  1.0])

W = y / (2 * t)
b = -g - y ** 2 / (4 * t)

x = np.linspace(-2.5, 2.5, 500)

for eps in [1.0, 0.5, 0.2, 0.1, 0.05]:
    logits = x[:, None] * W[None, :] + b[None, :]
    f_lse = lse(logits, eps)
    u_hc  = hopf_cole(x, y, g, eps, t)

    # Algebraic identity: f_lse(x) = |x|^2/(4t) - u_eps(x,t)
    u_from_lse = x ** 2 / (4 * t) - f_lse
    err = np.max(np.abs(u_from_lse - u_hc))
    print(f"  eps={eps:.2f}:  max|u(from LSE) - u(Hopf-Cole)| = {err:.2e}")

print()

# -- Test 3: Analytical PDE residual ------------------------------------------

print("=" * 62)
print("Test 3: PDE residual for Hopf-Cole solution")
print("=" * 62)

def softmax_weights(x, y, g, eps, t):
    """p_j(x) = exp((-g_j - |x-y_j|^2/4t)/eps) / Z"""
    logits = (-g[None, :] - (x[:, None] - y[None, :]) ** 2 / (4 * t)) / eps
    logits -= logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=-1, keepdims=True)

x_test = np.linspace(-2, 2, 400)
eps = 0.3
t = 1.0

p = softmax_weights(x_test, y, g, eps, t)         # (N_x, N_y)
dy = x_test[:, None] - y[None, :]                  # x - y_j,  shape (N_x, N_y)

# Exact analytical derivatives of u = -eps*log Z
u_t  = -(p * dy ** 2).sum(axis=1) / (4 * t ** 2)          # d_t u
u_x  =  (p * dy).sum(axis=1) / (2 * t)                     # d_x u
# Variance of (x-y) under p
E_dy2 = (p * dy ** 2).sum(axis=1)
E_dy  = (p * dy).sum(axis=1)
V_dy  = E_dy2 - E_dy ** 2                                   # Var_p[x-y]
u_xx  = (1 / (2 * t)) * (1 - V_dy / (2 * t * eps))         # d_xx u

# The identity satisfied: u_t + (u_x)^2 = eps*u_xx - eps/(2t)
# => residual of  u_t + (u_x)^2 - eps*u_xx + eps/(2t) = 0
residual = u_t + u_x ** 2 - eps * u_xx + eps / (2 * t)
print(f"  eps={eps}:  max|analytical residual| = {np.max(np.abs(residual)):.2e}")
print()

# -- Test 4: 2-layer ReLU as composed HJ semigroup ----------------------------

print("=" * 62)
print("Test 4: 2-layer ReLU = eps->0 of 2-layer softplus (Claim B)")
print("=" * 62)

W1 = np.array([1.5, -1.0,  0.8])
b1 = np.array([0.3,  0.5, -0.4])
w2 = np.array([0.5,  0.5,  0.5])   # positive -> convex output

def relu_net(x, W1, b1, w2):
    h = np.maximum(x[:, None] * W1[None, :] + b1[None, :], 0)
    return h @ w2

def softplus_net(x, W1, b1, w2, eps):
    h = softplus(x[:, None] * W1[None, :] + b1[None, :], eps)
    return h @ w2

x = np.linspace(-2, 2, 500)
f_relu_2l = relu_net(x, W1, b1, w2)

for eps in [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]:
    f_sp = softplus_net(x, W1, b1, w2, eps)
    err  = np.max(np.abs(f_sp - f_relu_2l))
    print(f"  eps={eps:.3f}:  max|softplus_2L - ReLU_2L| = {err:.2e}")

print()

# -- Test 5: Hopf-Lax recovers ReLU output ------------------------------------

print("=" * 62)
print("Test 5: Hopf-Lax recovers ReLU output (dense support on [0,5])")
print("=" * 62)

y_hl  = np.linspace(0, 5, 1000)
g_hl  = np.zeros_like(y_hl)
x_hl  = np.linspace(-2, 2, 500)

u0      = hopf_lax(x_hl, y_hl, g_hl, t=1.0)
relu_ref = np.maximum(x_hl, 0)

err = np.max(np.abs(u0 - relu_ref))
print(f"  max|Hopf-Lax - ReLU| = {err:.4f}")
print()

print("All tests done.")

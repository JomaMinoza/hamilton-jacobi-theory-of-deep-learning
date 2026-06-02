"""Shared helpers for the experiment scripts: LSE and softmax, the Gibbs
attribution weights pi_j, the Hopf-Cole solution, MSE gradients and Adam, the
Hessian/robustness bound, entropy, and MNIST/CIFAR-10 loading."""

import os
import gzip
import struct
import urllib.request
import numpy as np

# -- Colour scheme ------------------------------------------------------------

C1 = "#0077BB"    # blue
C2 = "#EE7733"    # orange
C3 = "#009988"    # teal
C4 = "#CC3311"    # red

C_PARTICLE = "#1565C0"   # eps -> 0, Hopf-Lax (particle)
C_CRITICAL = "#2E7D32"   # eps*, optimal temperature
C_WAVE     = "#B71C1C"   # eps -> inf, heat equation (wave)

# -- Matplotlib style ----------------------------------------------------------

def setup_style(large=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = 10 if large else 9
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":          fs,
        "axes.labelsize":     fs,
        "axes.titlesize":     fs,
        "axes.linewidth":     0.8,
        "legend.fontsize":    fs - 2,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.7",
        "xtick.labelsize":    fs - 1,
        "ytick.labelsize":    fs - 1,
        "lines.linewidth":    1.4,
        "lines.markersize":   4.0 if not large else 4.5,
        "figure.dpi":         200,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })
    return plt

# -- Output directory ----------------------------------------------------------

def figures_dir():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(path, exist_ok=True)
    return path

# -- LSE / softmax core --------------------------------------------------------

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def lse_1d(x, W, b, eps):
    """LSE_eps(W*x + b) for 1-D input. Returns (n,)."""
    logits = x[:, None] * W[None, :] + b[None, :]
    m = logits.max(axis=1, keepdims=True)
    return eps * (np.log(np.exp((logits - m) / eps).sum(axis=1)) + m[:, 0] / eps)

def lse_nd(x, W, b, eps):
    """LSE_eps(W x + b) for N-D input. Returns (n,)."""
    logits = x @ W.T + b[None, :]
    m = logits.max(axis=1, keepdims=True)
    return eps * (np.log(np.exp((logits - m) / eps).sum(axis=1)) + m[:, 0] / eps)

def weights_1d(x, W, b, eps):
    """Gibbs weights pi_j(x), 1-D. Returns (n, N)."""
    logits = x[:, None] * W[None, :] + b[None, :]
    m = logits.max(axis=1, keepdims=True)
    e = np.exp((logits - m) / eps)
    return e / e.sum(axis=1, keepdims=True)

def weights_nd(x, W, b, eps):
    """Gibbs weights pi_j(x), N-D. Returns (n, N)."""
    logits = x @ W.T + b[None, :]
    m = logits.max(axis=1, keepdims=True)
    e = np.exp((logits - m) / eps)
    return e / e.sum(axis=1, keepdims=True)

# -- Loss and gradients --------------------------------------------------------

def mse_grad_1d(x, y, W, b, eps):
    """MSE loss and (dW, db) for 1-D LSE. Returns (loss, dW, db)."""
    n   = len(x)
    f   = lse_1d(x, W, b, eps)
    pi  = weights_1d(x, W, b, eps)
    r   = f - y
    dW  = (2 / n) * (r[:, None] * pi * x[:, None]).sum(axis=0)
    db  = (2 / n) * (r[:, None] * pi).sum(axis=0)
    return np.mean(r ** 2), dW, db

def mse_grad_nd(x, y, W, b, eps):
    """MSE loss and (dW, db) for N-D LSE. Returns (loss, dW, db)."""
    n   = len(x)
    f   = lse_nd(x, W, b, eps)
    pi  = weights_nd(x, W, b, eps)
    r   = f - y
    dW  = (2 / n) * (r[:, None] * pi).T @ x
    db  = (2 / n) * (r[:, None] * pi).sum(axis=0)
    return np.mean(r ** 2), dW, db

# -- Adam ----------------------------------------------------------------------

def adam_step(g, m, v, t, lr, b1=0.9, b2=0.999, ea=1e-8):
    """One Adam update. Returns (update, new_m, new_v)."""
    m  = b1 * m + (1 - b1) * g
    v  = b2 * v + (1 - b2) * g ** 2
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return lr * mh / (np.sqrt(vh) + ea), m, v

# -- Hessian analysis ----------------------------------------------------------

def hessian_norm_1d(x, W, b, eps):
    """Var_{pi}(W) / eps - exact Hessian spectral norm, 1-D. Returns (n,)."""
    pi   = weights_1d(x, W, b, eps)
    EW   = (pi * W[None, :]).sum(axis=1)
    EW2  = (pi * W[None, :] ** 2).sum(axis=1)
    return (EW2 - EW ** 2) / eps

def hessian_norm_nd(X, W, b, eps):
    """Hessian spectral norm, N-D. Returns (n,)."""
    pi    = weights_nd(X, W, b, eps)
    norms = np.empty(len(X))
    for i in range(len(X)):
        p    = pi[i]
        cov  = (W * p[:, None]).T @ W - np.outer((p[:, None] * W).sum(0),
                                                   (p[:, None] * W).sum(0))
        norms[i] = np.linalg.norm(cov / eps, ord=2)
    return norms

def log_slope(Ns, Es):
    """Fit log-log slope to (N, error) pairs."""
    lN = np.log(np.array(Ns, dtype=float))
    lE = np.log(np.clip(Es, 1e-12, None))
    return np.polyfit(lN, lE, 1)[0]

def entropy(p):
    """Shannon entropy of each row of p. Returns (n,)."""
    p = np.clip(p, 1e-15, None)
    return -(p * np.log(p)).sum(axis=1)

# -- MNIST loading -------------------------------------------------------------

_MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def _mnist_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data")

def _download_mnist():
    d = _mnist_dir()
    os.makedirs(d, exist_ok=True)
    for key, fname in _MNIST_FILES.items():
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(_MNIST_URL + fname, p)

def _parse_images(path):
    with gzip.open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, r * c).astype(np.float32) / 255.0

def _parse_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8).astype(np.int32)

def load_mnist():
    """Return (X_tr, y_tr, X_te, y_te), downloading if needed."""
    _download_mnist()
    d = _mnist_dir()
    X_tr = _parse_images(os.path.join(d, _MNIST_FILES["train_images"]))
    y_tr = _parse_labels(os.path.join(d, _MNIST_FILES["train_labels"]))
    X_te = _parse_images(os.path.join(d, _MNIST_FILES["test_images"]))
    y_te = _parse_labels(os.path.join(d, _MNIST_FILES["test_labels"]))
    return X_tr, y_tr, X_te, y_te

# -- CIFAR-10 loading ----------------------------------------------------------

_CIFAR_URL  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR_FILE = "cifar-10-python.tar.gz"

def _cifar_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_data")

def _download_cifar10():
    import tarfile
    d    = _cifar_dir()
    os.makedirs(d, exist_ok=True)
    tar  = os.path.join(d, _CIFAR_FILE)
    extr = os.path.join(d, "cifar-10-batches-py")
    if not os.path.exists(tar):
        print("  Downloading CIFAR-10 (~170 MB) ...")
        urllib.request.urlretrieve(_CIFAR_URL, tar)
    if not os.path.exists(extr):
        print("  Extracting ...")
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(d)
    return extr

def _load_batch(path):
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].astype(np.float32) / 255.0, np.array(d[b"labels"], dtype=np.int32)

def load_cifar10():
    """Return (X_tr, y_tr, X_te, y_te), downloading if needed."""
    extr = _download_cifar10()
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = _load_batch(os.path.join(extr, f"data_batch_{i}"))
        Xs.append(X); ys.append(y)
    X_tr = np.concatenate(Xs); y_tr = np.concatenate(ys)
    X_te, y_te = _load_batch(os.path.join(extr, "test_batch"))
    return X_tr, y_tr, X_te, y_te

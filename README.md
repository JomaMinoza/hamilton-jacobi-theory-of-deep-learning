# The Hamilton-Jacobi Theory of Deep Learning

Code for the paper:

> **The Hamilton-Jacobi Theory of Deep Learning**
> Jose Marie Antonio Minoza, Erika Fille T. Legara, Christopher P. Monterola
> [arXiv](https://arxiv.org/abs/2605.28983) | [Project Page](https://jomaminoza.github.io/hamilton-jacobi-theory-of-deep-learning/)

## Overview

A trained log-sum-exp (LSE) layer is, exactly, the Hopf-Cole solution of a viscous
Hamilton-Jacobi equation. The weights encode the initial data, the architecture
encodes the Hamiltonian, and a forward pass evaluates that PDE solution at the
input. A single parameter `eps` is at once the softmax temperature, the PDE
viscosity, and the convex-regularization strength; sending `eps -> 0` is the
tropical (max-plus) limit, where the layer becomes the Hopf-Lax formula.

```
  NN  (f^N_eps, eps>0)  --- eps->0 --->  Tropical NN  (f^N_0)
        |  exact                               |  exact
        v                                       v
  Viscous HJ  (u_eps)   --- eps->0 --->  Inviscid HJ / Hopf-Lax  (u_0)
```

These scripts are numerical checks of that correspondence and of its consequences
(generalization rate, robustness bound, attribution weights and their
bifurcations). Everything is plain NumPy/Matplotlib; there is no model to train at
scale and no package to install.

## Installation

```bash
git clone https://github.com/JomaMinoza/hamilton-jacobi-theory-of-deep-learning.git
cd hamilton-jacobi-theory-of-deep-learning
pip install -r requirements.txt
```

MNIST and CIFAR-10 download automatically on first use. `umap-learn` and
`scikit-learn` are optional (used by `phase_mnist_dense.py`, which falls back to
PCA without them).

## Quick start

```bash
# Exact LSE = Hopf-Cole identity and the tropical limit (prints a table)
python experiments/verify.py

# Attention = Gibbs/Hopf-Cole average, to machine precision
python experiments/attention.py

# Core figures (identity, quadrature rate, scaling law, robustness)
python experiments/run_all.py
```

Scripts run from any directory and write figures to `figures/`.

## Experiments

| Script | What it shows | Output |
|---|---|---|
| `verify.py` | LSE = Hopf-Cole identity, tropical limit, PDE residual, semigroup | printed table |
| `attention.py` | Attention = Gibbs average `grad LSE_eps(QK^T).V` | printed table |
| `transformer.py` | LSE-transformer block (attention + LSE-FFN) identities | printed table |
| `recovery.py` | Initial-data recovery as `eps -> 0` | `initial_data_recovery.pdf` |
| `scaling.py` | Scaling law at `d=1,2`, optimal `eps = N^{-1/d}` | `scaling_law_adam.pdf`, `exp_B_results.csv` |
| `scaling_hd.py` | Extends the scaling law to `d=4` (run `scaling.py` first) | `scaling_law_sgd.pdf` |
| `robustness.py` | Hessian bound on a 1-D target | `robustness_sgd.pdf` |
| `hessian_mnist.py` | Hessian bound on MNIST (PCA-50, N=128) | `hessian_mnist.pdf` |
| `hessian_cifar10.py` | Hessian bound on CIFAR-10 (PCA-64, N=128) | `hessian_cifar10.pdf` |
| `phase_diagram.py` | Particle-to-wave phase diagram, synthetic 2-cluster | `phase_diagram.pdf`, `.gif` |
| `phase_mnist.py` | Phase diagram on MNIST 3 vs 7 | `mnist_phase_diagram.pdf`, `.gif` |
| `phase_mnist_dense.py` | Phase diagram on all-class MNIST (N=2000) | `mnist_dense_diagram.pdf`, `.gif` |
| `bifurcation.py` | Fold bifurcations of the attribution entropy `H(pi)` | `bifurcation_diagram.pdf`, `.gif` |
| `run_all.py` | Runs the four core checks above | `figures/` |

`utils.py` holds the shared math (LSE, Gibbs weights, Hopf-Cole, gradients, the
Hessian bound) and the dataset loaders.

## Structure

```
.
  utils.py            # shared math and data loading
  experiments/        # one script per check / figure
  docs/               # project page (GitHub Pages)
  requirements.txt
```

## Citation

```bibtex
@article{minoza2026hjdl,
  title   = {The Hamilton--Jacobi Theory of Deep Learning},
  author  = {Mi{\~n}oza, Jose Marie Antonio and Legara, Erika Fille T. and Monterola, Christopher P.},
  journal = {arXiv preprint arXiv:2605.28983},
  year    = {2026},
}
```

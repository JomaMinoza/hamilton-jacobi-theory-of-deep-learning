"""The same phase diagram as phase_diagram.py, but with real MNIST support points
(two digits, projected to 2-D by PCA). The Hopf-Cole formula is unchanged. Saves
the figure and a GIF to figures/."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C_PARTICLE, C_CRITICAL, C_WAVE,
    setup_style, figures_dir, entropy,
)

import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

plt = setup_style()
OUT = figures_dir()

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

# -- Load MNIST two-class subset -----------------------------------------------

def load_mnist_two_classes(class_a=3, class_b=7, n_per_class=60, seed=0):
    from utils import load_mnist
    rng = np.random.default_rng(seed)
    X_tr, y_tr, _, _ = load_mnist()

    idx_a = np.where(y_tr == class_a)[0]
    idx_b = np.where(y_tr == class_b)[0]
    idx_a = rng.choice(idx_a, n_per_class, replace=False)
    idx_b = rng.choice(idx_b, n_per_class, replace=False)

    imgs_a = X_tr[idx_a]
    imgs_b = X_tr[idx_b]
    imgs_all = np.vstack([imgs_a, imgs_b])

    labels = np.array([0.0]*n_per_class + [1.0]*n_per_class)

    mu  = imgs_all.mean(axis=0)
    Xc  = imgs_all - mu
    cov = Xc.T @ Xc / (len(imgs_all) - 1)
    vals, vecs = np.linalg.eigh(cov)
    top2 = vecs[:, -2:][:, ::-1]
    proj = Xc @ top2

    raw_imgs = np.vstack([imgs_a, imgs_b]).reshape(-1, 28, 28)
    return proj, labels, raw_imgs, class_a, class_b, mu, top2

# -- Load data -----------------------------------------------------------------
CLASS_A, CLASS_B = 3, 7
N_PER_CLASS = 60
print(f"Loading MNIST digits {CLASS_A} vs {CLASS_B}  ({N_PER_CLASS} each)...")
y_pts, g, raw_imgs, CA, CB, pca_mu, pca_vecs = \
    load_mnist_two_classes(CLASS_A, CLASS_B, n_per_class=N_PER_CLASS)

N = len(y_pts)
t = 1.0
eps_star = N ** (-1.0 / 2.0)

print(f"N={N}, eps*={eps_star:.3f}")
y = y_pts

# -- Exact Hopf-Cole formulas --------------------------------------------------

def hc_weights(x, eps):
    log_w = -(np.sum((x - y)**2, axis=1) / (4.0*t) + g) / eps
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()

def hj_gradient(x, eps):
    pi = hc_weights(x, eps)
    return (x - pi @ y) / (2.0*t)

def soft_predict(x, eps):
    return hc_weights(x, eps) @ g

def _ent(x, eps):
    return entropy(hc_weights(x, eps)[None, :])[0]

# -- Spatial grid --------------------------------------------------------------
MARGIN = 2.0
x1_min = y[:, 0].min() - MARGIN;  x1_max = y[:, 0].max() + MARGIN
x2_min = y[:, 1].min() - MARGIN;  x2_max = y[:, 1].max() + MARGIN

res  = 70
xv   = np.linspace(x1_min, x1_max, res)
yv   = np.linspace(x2_min, x2_max, res)
XX, YY = np.meshgrid(xv, yv)
grid   = np.stack([XX.ravel(), YY.ravel()], axis=1)
step   = 8

ENT_MAX = np.log(N)
G_star  = np.array([hj_gradient(pt, eps_star) for pt in grid])
GNORM_MAX = np.percentile(np.linalg.norm(G_star, axis=1), 98) * 1.3

COLOR_A = '#00a8d6'
COLOR_B = '#e85d26'

# -- Scatter helper ------------------------------------------------------------

def scatter_digits(ax, zoom=0.18, with_thumbnails=True):
    ax.scatter(y[:N_PER_CLASS, 0], y[:N_PER_CLASS, 1],
               c=COLOR_A, marker='o', s=30, zorder=4,
               edgecolors='#333333', linewidths=0.3, label=f'digit {CA}')
    ax.scatter(y[N_PER_CLASS:, 0], y[N_PER_CLASS:, 1],
               c=COLOR_B, marker='^', s=30, zorder=4,
               edgecolors='#333333', linewidths=0.3, label=f'digit {CB}')
    if with_thumbnails:
        for i in range(N):
            img = raw_imgs[i]
            color_img = np.zeros((*img.shape, 4))
            c = np.array(matplotlib.colors.to_rgba(COLOR_A if i < N_PER_CLASS else COLOR_B))
            color_img[..., :3] = c[:3]
            color_img[..., 3]  = img * 0.55 + 0.08
            oi = OffsetImage(color_img, zoom=zoom)
            ab = AnnotationBbox(oi, (y[i, 0], y[i, 1]),
                                frameon=False, zorder=3)
            ax.add_artist(ab)

# ==============================================================================
# STATIC SUMMARY FIGURE
# ==============================================================================

eps_range = np.logspace(-2, np.log10(eps_star * 8), 300)
query_pts = [
    (y[N_PER_CLASS // 2],
     f'digit {CA} centre', COLOR_A),
    ((y[:N_PER_CLASS].mean(0) + y[N_PER_CLASS:].mean(0)) / 2,
     'midpoint', '#E65100'),
    (y[N_PER_CLASS + N_PER_CLASS // 2],
     f'digit {CB} centre', COLOR_B),
]

print("Building static figure...")
fig_s = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 3, figure=fig_s, hspace=0.48, wspace=0.38)

# (a) Phase curves
ax_a = fig_s.add_subplot(gs[0, :2])
for xq, label, color in query_pts:
    H_vals = [_ent(xq, eps) for eps in eps_range]
    ax_a.semilogx(eps_range, H_vals, label=label, color=color, lw=2)
ax_a.axvline(eps_star, color=C_CRITICAL, ls='--', lw=2,
             label=r'$\varepsilon^* \approx %.3f$' % eps_star)
ax_a.axhline(ENT_MAX, color='#666666', ls=':', lw=1.2, label=r'$\log N$')
ax_a.set_xlabel(r'Viscosity $\varepsilon$')
ax_a.set_ylabel(r'Entropy $H(\pi)$')
ax_a.set_title(r'Phase diagram: MNIST %d vs %d' % (CA, CB))
ax_a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
ax_a.set_ylim([-0.05, ENT_MAX + 0.35])
ax_a.axvspan(eps_range[0], eps_star * 0.3, alpha=0.10, color=C_PARTICLE)
ax_a.axvspan(eps_star * 3, eps_range[-1],  alpha=0.10, color=C_WAVE)

# (b) PCA scatter with thumbnails
ax_b = fig_s.add_subplot(gs[0, 2])
scatter_digits(ax_b, zoom=0.20, with_thumbnails=True)
ax_b.set_title(f'PCA support: {CA} vs {CB}')
ax_b.set_xlabel('PC 1'); ax_b.set_ylabel('PC 2')
ax_b.legend(fontsize=7, loc='upper right')
ax_b.set_xlim([x1_min, x1_max]); ax_b.set_ylim([x2_min, x2_max])

# (c) Entropy heatmaps at cold / critical / hot
temps = [
    (eps_star * 0.1,  r'Cold  $\varepsilon=%.3f$' % (eps_star * 0.1)),
    (eps_star,        r'Critical  $\varepsilon^*\approx%.3f$' % eps_star),
    (eps_star * 6.0,  r'Hot  $\varepsilon=%.2f$' % (eps_star * 6.0)),
]
for col, (eps_v, title) in enumerate(temps):
    ax = fig_s.add_subplot(gs[1, col])
    H_map = np.array([_ent(pt, eps_v) for pt in grid]).reshape(res, res)
    im = ax.contourf(XX, YY, H_map, levels=25, cmap='plasma', vmin=0, vmax=ENT_MAX)
    scatter_digits(ax, zoom=0.14, with_thumbnails=(col == 1))
    fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$H(\pi)$')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_xlim([x1_min, x1_max]); ax.set_ylim([x2_min, x2_max])
    if col == 0:
        ax.legend(fontsize=7, loc='upper right')

fig_s.suptitle(
    r'LSE Network = Hopf-Cole  (MNIST %d vs %d,  $N=%d$,  $\varepsilon^*\approx%.3f$)' % (
        CA, CB, N, eps_star),
    fontsize=11)

fig_s.savefig(os.path.join(OUT, 'mnist_phase_diagram.pdf'), bbox_inches='tight')
fig_s.savefig(os.path.join(OUT, 'mnist_phase_diagram.png'), dpi=150, bbox_inches='tight')
print("Saved mnist_phase_diagram.pdf/.png")
plt.close(fig_s)

# ==============================================================================
# ANIMATION FRAMES
# ==============================================================================
anim_dir = os.path.join(OUT, 'mnist_phase_animation')
os.makedirs(anim_dir, exist_ok=True)

eps_frames = np.logspace(-2, np.log10(eps_star * 8), 60)

xq_attr = (y[:N_PER_CLASS].mean(0) + y[N_PER_CLASS:].mean(0)) / 2.0

print(f"\nGenerating {len(eps_frames)} animation frames -> {anim_dir}")
frame_paths = []

for i, eps in enumerate(eps_frames):
    log_frac = i / (len(eps_frames) - 1)
    if log_frac < 0.35:
        regime, rc = 'particle (eps -> 0)', C_PARTICLE
    elif log_frac > 0.65:
        regime, rc = 'wave (eps -> inf)', C_WAVE
    else:
        regime, rc = 'critical (optimal eps*)', C_CRITICAL

    H_map  = np.array([_ent(pt, eps) for pt in grid]).reshape(res, res)
    P_map  = np.array([soft_predict(pt, eps) for pt in grid]).reshape(res, res)
    xq_sub = grid[::step]
    Gq     = np.array([hj_gradient(pt, eps) for pt in xq_sub])
    nrm_q  = np.linalg.norm(Gq, axis=1, keepdims=True).clip(min=1e-8)
    pi_attr = hc_weights(xq_attr, eps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(r'$\varepsilon=%.4f$   $\varepsilon^*\approx%.3f$   %s' % (eps, eps_star, regime),
                 fontsize=10, color=rc)
    cb_kw = dict(fraction=0.046, pad=0.04)

    ax = axes[0]
    im = ax.contourf(XX, YY, H_map, levels=25, cmap='plasma', vmin=0, vmax=ENT_MAX)
    scatter_digits(ax, zoom=0.12, with_thumbnails=False)
    fig.colorbar(im, ax=ax, **cb_kw).set_label(r'$H(\pi)$')
    ax.set_title(r'Entropy $H(\pi)$')
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_xlim([x1_min, x1_max]); ax.set_ylim([x2_min, x2_max])

    ax = axes[1]
    im2 = ax.contourf(XX, YY, P_map, levels=25, cmap='RdBu_r', vmin=0, vmax=1)
    scatter_digits(ax, zoom=0.12, with_thumbnails=False)
    ax.quiver(xq_sub[:, 0], xq_sub[:, 1],
              Gq[:, 0]/nrm_q[:, 0], Gq[:, 1]/nrm_q[:, 0],
              color='#333333', alpha=0.40, scale=38, width=0.0025)
    fig.colorbar(im2, ax=ax, **cb_kw).set_label(r'$\mathbb{E}_\pi[g_j]$')
    ax.set_title(r'Prediction + characteristics')
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_xlim([x1_min, x1_max]); ax.set_ylim([x2_min, x2_max])

    ax = axes[2]
    bar_colors = [COLOR_A]*N_PER_CLASS + [COLOR_B]*N_PER_CLASS
    ax.bar(range(N), pi_attr, color=bar_colors, alpha=0.85, edgecolor='none')
    ax.axvline(N_PER_CLASS - 0.5, color='#666666', ls=':', lw=1)
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('Neuron $j$')
    ax.set_ylabel(r'$\pi_j$')
    ax.set_title(r'Attribution at midpoint')

    plt.tight_layout()
    out_path = os.path.join(anim_dir, f'frame_{i:03d}.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    frame_paths.append(out_path)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  {i+1}/{len(eps_frames)}  eps={eps:.4f}")

print(f"All {len(eps_frames)} frames saved.")

# ==============================================================================
# ASSEMBLE GIF
# ==============================================================================
gif_path = os.path.join(OUT, 'mnist_phase_transition.gif')

if HAS_IMAGEIO:
    from PIL import Image as PILImage
    print(f"\nAssembling GIF -> {gif_path}")
    raw = [PILImage.open(p).convert('RGB') for p in frame_paths]
    W_px, H_px = raw[0].size
    raw = [im.resize((W_px, H_px), PILImage.LANCZOS) for im in raw]
    loop_frames = raw + raw[::-1]
    loop_frames[0].save(
        gif_path, save_all=True, append_images=loop_frames[1:],
        duration=90, loop=0, optimize=False,
    )
    print(f"Saved {gif_path}  ({len(loop_frames)} frames)")
else:
    print("\nimageio not found.  Install: pip install imageio pillow")

print("\nDone.")

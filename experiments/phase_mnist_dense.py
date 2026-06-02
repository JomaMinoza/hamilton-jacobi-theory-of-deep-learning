"""All-class MNIST phase diagram. Uses 2000 support points (200 per digit)
projected to 2-D with UMAP, falling back to PCA. Initial data is flat, so class
identity comes from summing the attribution weights within each class. Saves the
figure and a GIF to figures/."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C_PARTICLE, C_CRITICAL, C_WAVE,
    setup_style, figures_dir, load_mnist,
)

import numpy as np
import warnings
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap

plt = setup_style()
OUT = figures_dir()
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

# -- Parameters ----------------------------------------------------------------
N_PER_CLASS = 200
N_CLASSES   = 10
SEED        = 42
t           = 1.0
D_PROJ      = 2

print(f"Loading MNIST  ({N_PER_CLASS} per class x 10 = {N_PER_CLASS*N_CLASSES} total)...")
X_all, y_all, _, _ = load_mnist()

rng = np.random.default_rng(SEED)
idx_list = []
for c in range(N_CLASSES):
    idx_c = np.where(y_all == c)[0]
    idx_list.append(rng.choice(idx_c, N_PER_CLASS, replace=False))
idx_sel = np.concatenate(idx_list)
X_sel   = X_all[idx_sel]
labels  = np.repeat(np.arange(N_CLASSES), N_PER_CLASS).astype(np.int32)
N       = len(X_sel)

# -- Dimensionality reduction: UMAP -> PCA fallback ---------------------------
try:
    import umap as umap_lib
    print("Fitting UMAP...")
    reducer = umap_lib.UMAP(n_components=D_PROJ, n_neighbors=20,
                             min_dist=0.1, random_state=SEED, verbose=False)
    proj = reducer.fit_transform(X_sel).astype(np.float64)
    PROJ_NAME = "UMAP"
except ImportError:
    print("umap-learn not found -- falling back to PCA.")
    mu  = X_sel.mean(axis=0)
    Xc  = X_sel - mu
    try:
        from sklearn.utils.extmath import randomized_svd
        _, _, Vt = randomized_svd(Xc, n_components=D_PROJ, random_state=SEED)
        proj = (Xc @ Vt.T).astype(np.float64)
    except ImportError:
        cov  = Xc.T @ Xc / (N - 1)
        vals, vecs = np.linalg.eigh(cov)
        top  = vecs[:, -D_PROJ:][:, ::-1]
        proj = (Xc @ top).astype(np.float64)
    PROJ_NAME = "PCA"

print(f"{PROJ_NAME} done. "
      f"Range: x1=[{proj[:,0].min():.2f},{proj[:,0].max():.2f}]  "
      f"x2=[{proj[:,1].min():.2f},{proj[:,1].max():.2f}]")

y = proj
eps_star = N ** (-1.0 / D_PROJ)
print(f"N={N}, eps*={eps_star:.4f}")

# -- Colour scheme -------------------------------------------------------------
TAB10       = matplotlib.colormaps.get_cmap('tab10').resampled(10)
CLASS_COLORS = [TAB10(c) for c in range(10)]

# -- Spatial grid --------------------------------------------------------------
MARGIN = 1.5 if PROJ_NAME == "UMAP" else 2.5
x1_min = y[:,0].min() - MARGIN;  x1_max = y[:,0].max() + MARGIN
x2_min = y[:,1].min() - MARGIN;  x2_max = y[:,1].max() + MARGIN

res  = 72
xv   = np.linspace(x1_min, x1_max, res)
yv2  = np.linspace(x2_min, x2_max, res)
XX, YY = np.meshgrid(xv, yv2)
grid   = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float64)
n_grid = len(grid)
step   = 7

y_sq    = np.sum(y**2, axis=1)
grid_sq = np.sum(grid**2, axis=1)

# -- Vectorised Hopf-Cole core -------------------------------------------------
def compute_pi_all(eps):
    xy    = grid @ y.T
    sq_d  = grid_sq[:, None] + y_sq[None, :] - 2.0 * xy
    log_w = -sq_d / (4.0 * t * eps)
    log_w -= log_w.max(axis=1, keepdims=True)
    w     = np.exp(log_w)
    w    /= w.sum(axis=1, keepdims=True)
    return w

def compute_class_probs(pi):
    cp = np.zeros((n_grid, N_CLASSES))
    for c in range(N_CLASSES):
        cp[:, c] = pi[:, labels == c].sum(axis=1)
    return cp

def compute_grad(eps, pi=None):
    if pi is None:
        pi = compute_pi_all(eps)
    centroid = pi @ y
    return (grid - centroid) / (2.0 * t)

def entropy_vec(p):
    p = np.clip(p, 1e-15, None)
    return -np.sum(p * np.log(p), axis=1)

# -- Fixed colour ranges --------------------------------------------------------
ENT_CLASS_MAX = np.log(N_CLASSES)

pi_star   = compute_pi_all(eps_star)
grad_star = compute_grad(eps_star, pi_star)
GNORM_MAX = np.percentile(np.linalg.norm(grad_star, axis=1), 98) * 1.3

bounds = np.arange(-0.5, 10.5, 1.0)
cmap10 = ListedColormap([CLASS_COLORS[c] for c in range(10)])
norm10 = BoundaryNorm(bounds, cmap10.N)

def scatter_support(ax, s=8, alpha=0.55):
    for c in range(N_CLASSES):
        mask = labels == c
        ax.scatter(y[mask, 0], y[mask, 1],
                   c=[CLASS_COLORS[c]], s=s, alpha=alpha,
                   edgecolors='none', zorder=4, label=str(c))

# ==============================================================================
# STATIC SUMMARY FIGURE
# ==============================================================================
print("Building static figure...")

eps_vals = [0.001, eps_star, 4.0]
maps = []
for ev in eps_vals:
    pi  = compute_pi_all(ev)
    cp  = compute_class_probs(pi)
    maps.append({
        'eps':      ev,
        'pi':       pi,
        'cp':       cp,
        'H_class':  entropy_vec(cp).reshape(res, res),
        'pred':     np.argmax(cp, axis=1).reshape(res, res).astype(float),
        'conf':     cp.max(axis=1).reshape(res, res),
        'gnorm':    np.linalg.norm(compute_grad(ev, pi), axis=1).reshape(res, res),
    })

fig_s = plt.figure(figsize=(16, 10))
gs    = gridspec.GridSpec(2, 4, figure=fig_s, hspace=0.50, wspace=0.38)

# Row 0: phase curves + projection scatter
ax_a = fig_s.add_subplot(gs[0, :2])
eps_range = np.logspace(-3, 1.0, 250)
query_pts_idx = [
    N_PER_CLASS * 0 + N_PER_CLASS // 2,
    N_PER_CLASS * 3 + N_PER_CLASS // 2,
    N_PER_CLASS * 7 + N_PER_CLASS // 2,
    N_PER_CLASS * 9 + N_PER_CLASS // 2,
]
qp_labels = ['digit 0', 'digit 3', 'digit 7', 'digit 9']
qp_colors = [CLASS_COLORS[0], CLASS_COLORS[3], CLASS_COLORS[7], CLASS_COLORS[9]]

for qi, ql, qc in zip(query_pts_idx, qp_labels, qp_colors):
    xq = y[qi]
    Hc_vals = []
    for ev in eps_range:
        xy_q = np.sum((xq - y)**2, axis=1)
        lw   = -xy_q / (4.0 * t * ev)
        lw  -= lw.max()
        w    = np.exp(lw); w /= w.sum()
        cp   = np.array([w[labels == c].sum() for c in range(N_CLASSES)])
        Hc_vals.append(entropy_vec(cp[None, :])[0])
    ax_a.semilogx(eps_range, Hc_vals, label=ql, color=qc, lw=2)

ax_a.axvline(eps_star, color=C_CRITICAL, ls='--', lw=2,
             label=r'$\varepsilon^* \approx %.3f$' % eps_star)
ax_a.axhline(ENT_CLASS_MAX, color='#888888', ls=':', lw=1.2, label=r'$\log 10$')
ax_a.set_xlabel(r'Viscosity $\varepsilon$')
ax_a.set_ylabel(r'Class entropy $H(p_c)$')
ax_a.set_title(r'Phase diagram: MNIST 0--9')
ax_a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
ax_a.set_ylim([-0.05, ENT_CLASS_MAX + 0.25])

ax_b = fig_s.add_subplot(gs[0, 2:])
scatter_support(ax_b, s=10, alpha=0.65)
ax_b.legend(title='digit', fontsize=7, ncol=2, loc='upper right', markerscale=1.5)
ax_b.set_title(f'{PROJ_NAME} support (N={N})')
ax_b.set_xlabel('dim 1'); ax_b.set_ylabel('dim 2')
ax_b.set_xlim([x1_min, x1_max]); ax_b.set_ylim([x2_min, x2_max])

# Row 1: class prediction maps at cold / critical / hot + class entropy at eps*
titles = [
    r'Cold  $\varepsilon=%.4f$' % eps_vals[0],
    r'Critical  $\varepsilon^*=%.4f$' % eps_vals[1],
    r'Hot  $\varepsilon=%.3f$' % eps_vals[2],
]
for col, (m, title) in enumerate(zip(maps, titles)):
    ax = fig_s.add_subplot(gs[1, col])
    im = ax.pcolormesh(XX, YY, m['pred'], cmap=cmap10, norm=norm10,
                       shading='auto', alpha=0.85)
    scatter_support(ax, s=5, alpha=0.45)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')
    ax.set_xlim([x1_min, x1_max]); ax.set_ylim([x2_min, x2_max])
    if col == 2:
        cbar = fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(range(10))

ax_ent = fig_s.add_subplot(gs[1, 3])
im_ent = ax_ent.contourf(XX, YY, maps[1]['H_class'], levels=25, cmap='plasma',
                          vmin=0, vmax=ENT_CLASS_MAX)
scatter_support(ax_ent, s=5, alpha=0.45)
fig_s.colorbar(im_ent, ax=ax_ent, fraction=0.046, pad=0.04, label=r'$H(p_c)$')
ax_ent.set_title(r'Class entropy at $\varepsilon^*$')
ax_ent.set_xlabel('dim 1'); ax_ent.set_ylabel('dim 2')
ax_ent.set_xlim([x1_min, x1_max]); ax_ent.set_ylim([x2_min, x2_max])

fig_s.suptitle(
    r'LSE Network = Hopf-Cole  (MNIST 0--9,  %s$_2$,  $N=%d$,  $\varepsilon^*\approx%.4f$)' % (
        PROJ_NAME, N, eps_star),
    fontsize=11)

fig_s.savefig(os.path.join(OUT, 'mnist_dense_diagram.pdf'), bbox_inches='tight')
fig_s.savefig(os.path.join(OUT, 'mnist_dense_diagram.png'), dpi=150, bbox_inches='tight')
print("Saved mnist_dense_diagram.pdf/.png")
plt.close(fig_s)

# ==============================================================================
# ANIMATION FRAMES
# ==============================================================================
anim_dir = os.path.join(OUT, 'mnist_dense_animation')
os.makedirs(anim_dir, exist_ok=True)

eps_frames = np.logspace(-3, 1.0, 60)
xq_attr   = y.mean(axis=0)

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

    pi       = compute_pi_all(eps)
    cp       = compute_class_probs(pi)
    H_class  = entropy_vec(cp).reshape(res, res)
    pred_map = np.argmax(cp, axis=1).reshape(res, res).astype(float)
    gnorm_map = np.linalg.norm(compute_grad(eps, pi), axis=1).reshape(res, res)

    pi_attr  = pi[np.argmin(np.sum((grid - xq_attr)**2, axis=1))]
    cp_attr  = np.array([pi_attr[labels == c].sum() for c in range(N_CLASSES)])
    H_class_attr = entropy_vec(cp_attr[None, :])[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(r'$\varepsilon=%.5f$   $\varepsilon^*\approx%.4f$   %s' % (eps, eps_star, regime),
                 fontsize=10, color=rc)
    cb_kw = dict(fraction=0.046, pad=0.04)

    ax = axes[0]
    ax.pcolormesh(XX, YY, pred_map, cmap=cmap10, norm=norm10, shading='auto', alpha=0.85)
    scatter_support(ax, s=5, alpha=0.40)
    ax.set_title(r'Class prediction $\arg\max_c p_c$')
    ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')
    ax.set_xlim([x1_min, x1_max]); ax.set_ylim([x2_min, x2_max])

    ax = axes[1]
    im2 = ax.contourf(XX, YY, H_class, levels=25, cmap='plasma',
                      vmin=0, vmax=ENT_CLASS_MAX)
    scatter_support(ax, s=5, alpha=0.40)
    fig.colorbar(im2, ax=ax, **cb_kw).set_label(r'$H(p_c)$')
    ax.set_title(r'Class entropy $H(p_c)$')
    ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')
    ax.set_xlim([x1_min, x1_max]); ax.set_ylim([x2_min, x2_max])

    ax = axes[2]
    ax.bar(range(N_CLASSES), cp_attr,
           color=[CLASS_COLORS[c] for c in range(N_CLASSES)],
           alpha=0.90, edgecolor='none')
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels([str(c) for c in range(N_CLASSES)])
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('Digit class $c$')
    ax.set_ylabel(r'$p_c$')
    ax.set_title(r'Class attribution at centroid')

    plt.tight_layout()
    out_path = os.path.join(anim_dir, f'frame_{i:03d}.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    frame_paths.append(out_path)

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  {i+1}/{len(eps_frames)}  eps={eps:.5f}")

print(f"All {len(eps_frames)} frames saved.")

# ==============================================================================
# ASSEMBLE GIF
# ==============================================================================
gif_path = os.path.join(OUT, 'mnist_dense_transition.gif')

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

"""Phase diagram of the LSE network read as a Hopf-Cole solution, on a synthetic
two-cluster dataset. Sweeps eps from the particle (Hopf-Lax) regime to the wave
(heat) regime and saves the figure and a transition GIF to figures/."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C_PARTICLE, C_CRITICAL, C_WAVE,
    setup_style, figures_dir, entropy,
)

import numpy as np
import matplotlib.gridspec as gridspec

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

np.random.seed(42)

# -- HJ / network parameters ---------------------------------------------------
d    = 2
N    = 16
t    = 1.0

y = np.vstack([
    np.random.randn(N // 2, d) * 0.4 + np.array([-1.0, -1.0]),
    np.random.randn(N // 2, d) * 0.4 + np.array([ 1.0,  1.0]),
])
g = np.array([0.0] * (N // 2) + [1.0] * (N // 2))

eps_star = N ** (-1.0 / d)

# -- Exact Hopf-Cole formulas (Theorem 3.1) ------------------------------------

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

# -- Spatial grid --------------------------------------------------------------
res  = 80
xv   = np.linspace(-2.5, 2.5, res)
yv   = np.linspace(-2.5, 2.5, res)
XX, YY = np.meshgrid(xv, yv)
grid   = np.stack([XX.ravel(), YY.ravel()], axis=1)
step   = 10

ENT_MAX   = np.log(N)
G_star    = np.array([hj_gradient(pt, eps_star) for pt in grid])
GNORM_MAX = np.percentile(np.linalg.norm(G_star, axis=1), 98) * 1.3

CLUSTER1_COLOR = '#00a8d6'
CLUSTER2_COLOR = '#e85d26'

# -- Helpers -------------------------------------------------------------------

def scatter_clusters(ax):
    ax.scatter(y[:N//2, 0], y[:N//2, 1],
               c=CLUSTER1_COLOR, marker='o', s=45, zorder=5,
               edgecolors='#333333', linewidths=0.3, label=r'$g_j=0$')
    ax.scatter(y[N//2:, 0], y[N//2:, 1],
               c=CLUSTER2_COLOR, marker='^', s=45, zorder=5,
               edgecolors='#333333', linewidths=0.3, label=r'$g_j=1$')

def _ent(x, eps):
    return entropy(hc_weights(x, eps)[None, :])[0]

def make_frame(eps, anim_dir, idx, eps_frames):
    """Render one 3-panel animation frame and save to anim_dir."""
    H_map = np.array([_ent(pt, eps) for pt in grid]).reshape(res, res)
    P_map = np.array([soft_predict(pt, eps) for pt in grid]).reshape(res, res)

    xq_sub = grid[::step]
    Gq     = np.array([hj_gradient(pt, eps) for pt in xq_sub])
    nrm_q  = np.linalg.norm(Gq, axis=1, keepdims=True).clip(min=1e-8)

    log_frac = (np.log10(eps) + 2) / (np.log10(eps_frames[-1]) + 2)
    if log_frac < 0.35:
        regime, rc = 'particle (eps -> 0)', C_PARTICLE
    elif log_frac > 0.65:
        regime, rc = 'wave (eps -> inf)', C_WAVE
    else:
        regime, rc = 'critical (optimal eps*)', C_CRITICAL

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(r'$\varepsilon=%.4f$   $\varepsilon^*\approx%.2f$   %s' % (eps, eps_star, regime),
                 fontsize=10, color=rc)
    cb_kw = dict(fraction=0.046, pad=0.04)

    ax = axes[0]
    im = ax.contourf(XX, YY, H_map, levels=25, cmap='plasma', vmin=0, vmax=ENT_MAX)
    scatter_clusters(ax)
    fig.colorbar(im, ax=ax, **cb_kw).set_label(r'$H(\pi)$')
    ax.set_title(r'Entropy $H(\pi)$')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    ax = axes[1]
    im2 = ax.contourf(XX, YY, P_map, levels=25, cmap='RdBu_r', vmin=0, vmax=1)
    scatter_clusters(ax)
    ax.quiver(xq_sub[:, 0], xq_sub[:, 1],
              Gq[:, 0]/nrm_q[:, 0], Gq[:, 1]/nrm_q[:, 0],
              color='#333333', alpha=0.45, scale=38, width=0.0025)
    fig.colorbar(im2, ax=ax, **cb_kw).set_label(r'$\mathbb{E}_\pi[g_j]$')
    ax.set_title(r'Prediction + characteristics')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    ax = axes[2]
    pi_attr = hc_weights(np.array([0.3, -0.2]), eps)
    bar_colors = [CLUSTER1_COLOR]*(N//2) + [CLUSTER2_COLOR]*(N//2)
    ax.bar(range(N), pi_attr, color=bar_colors, alpha=0.85, edgecolor='none')
    ax.axvline(N//2 - 0.5, color='#666666', ls=':', lw=1)
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('Neuron $j$')
    ax.set_ylabel(r'$\pi_j$')
    ax.set_title(r'Attribution at query')

    plt.tight_layout()
    out_path = os.path.join(anim_dir, f'frame_{idx:03d}.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    return out_path


# ==============================================================================
# STATIC SUMMARY FIGURE
# ==============================================================================

eps_range = np.logspace(-2, 1.3, 300)
query_pts = [
    (np.array([-1.0, -1.0]), r'$x\in$ cluster 1 $(g=0)$', C_PARTICLE),
    (np.array([ 0.0,  0.0]), r'$x$ at boundary',            '#E65100'),
    (np.array([ 1.0,  1.0]), r'$x\in$ cluster 2 $(g=1)$', C_WAVE),
]

fig_s = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 3, figure=fig_s, hspace=0.45, wspace=0.38)

# (a) Phase curves
ax_a = fig_s.add_subplot(gs[0, :2])
for xq, label, color in query_pts:
    H_vals = [_ent(xq, eps) for eps in eps_range]
    ax_a.semilogx(eps_range, H_vals, label=label, color=color, lw=2)
ax_a.axvline(eps_star, color=C_CRITICAL, ls='--', lw=2,
             label=r'$\varepsilon^* \approx %.2f$' % eps_star)
ax_a.axhline(ENT_MAX, color='#666666', ls=':', lw=1.2, label=r'$\log N$')
ax_a.set_xlabel(r'Viscosity $\varepsilon$')
ax_a.set_ylabel(r'Entropy $H(\pi)$')
ax_a.set_title(r'Phase diagram: particle vs. wave')
ax_a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=3)
ax_a.set_ylim([-0.05, ENT_MAX + 0.35])
ax_a.axvspan(1e-2, 0.08, alpha=0.10, color=C_PARTICLE)
ax_a.axvspan(2.5,  20.0, alpha=0.10, color=C_WAVE)

# (b) Attribution at query point
ax_c = fig_s.add_subplot(gs[0, 2])
xq_attr = np.array([0.3, -0.2])
for eps_v, label, color in [(0.05, r'$\varepsilon=0.05$', C_PARTICLE),
                              (eps_star, r'$\varepsilon^*$', C_CRITICAL),
                              (3.0, r'$\varepsilon=3.0$', C_WAVE)]:
    pi = hc_weights(xq_attr, eps_v)
    ax_c.plot(range(N), pi, 'o-', label=label, color=color, alpha=0.9, ms=4, lw=1.5)
ax_c.axvline(N//2 - 0.5, color='#666666', ls=':', lw=1)
ax_c.set_xlabel('Neuron $j$'); ax_c.set_ylabel(r'$\pi_j$')
ax_c.set_title('Attribution at query')
ax_c.legend(fontsize=8)

# (c) Entropy heatmaps at cold / critical / hot
temps = [
    (0.05,     r'Cold  $\varepsilon=0.05$'),
    (eps_star, r'Critical  $\varepsilon^*\approx%.2f$' % eps_star),
    (4.0,      r'Hot  $\varepsilon=4.0$'),
]
for col, (eps_v, title) in enumerate(temps):
    ax = fig_s.add_subplot(gs[1, col])
    H_map = np.array([_ent(pt, eps_v) for pt in grid]).reshape(res, res)
    im = ax.contourf(XX, YY, H_map, levels=25, cmap='plasma', vmin=0, vmax=ENT_MAX)
    scatter_clusters(ax)
    fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$H(\pi)$')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    if col == 0:
        ax.legend(fontsize=7, loc='upper right')

fig_s.suptitle(r'LSE Network = Hopf-Cole Solution  ($N=%d$, $d=%d$, $\varepsilon^*\approx%.2f$)' % (N, d, eps_star),
               fontsize=11)

fig_s.savefig(os.path.join(OUT, 'phase_diagram.pdf'), bbox_inches='tight')
fig_s.savefig(os.path.join(OUT, 'phase_diagram.png'), dpi=150, bbox_inches='tight')
print("Saved phase_diagram.pdf/.png")
plt.close(fig_s)

# ==============================================================================
# ANIMATION FRAMES
# ==============================================================================
anim_dir   = os.path.join(OUT, 'phase_animation')
os.makedirs(anim_dir, exist_ok=True)

eps_frames = np.logspace(-2, 1.0, 60)
frame_paths = []

print(f"\nGenerating {len(eps_frames)} animation frames -> {anim_dir}")
for i, eps in enumerate(eps_frames):
    path = make_frame(eps, anim_dir, i, eps_frames)
    frame_paths.append(path)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  {i+1}/{len(eps_frames)}  eps={eps:.4f}")

print(f"All {len(eps_frames)} frames saved.")

# ==============================================================================
# ASSEMBLE GIF
# ==============================================================================
gif_path = os.path.join(OUT, 'phase_transition.gif')

if HAS_IMAGEIO:
    from PIL import Image as PILImage
    print(f"\nAssembling GIF -> {gif_path}")
    raw = [PILImage.open(p).convert('RGB') for p in frame_paths]
    W_px, H_px = raw[0].size
    raw = [im.resize((W_px, H_px), PILImage.LANCZOS) for im in raw]
    frames_loop = raw + raw[::-1]
    frames_loop[0].save(
        gif_path, save_all=True, append_images=frames_loop[1:],
        duration=80, loop=0, optimize=False,
    )
    print(f"Saved {gif_path}  ({len(frames_loop)} frames)")
else:
    print("\nimageio not found.  Install: pip install imageio pillow")

print("\nDone.")

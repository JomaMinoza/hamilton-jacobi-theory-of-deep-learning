"""Bifurcations of the attribution-entropy landscape. Follows the critical points
of H(pi(x;eps)) as the viscosity eps varies, via the closed-form gradient
grad_x H = Cov_pi(y, -log pi) / (2 t eps), and records where minima, saddles and
maxima are created or annihilated. Writes the bifurcation diagram and a GIF of the
fold cascade to figures/."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    C_PARTICLE, C_CRITICAL, C_WAVE,
    setup_style, figures_dir,
)

import numpy as np
import warnings
import matplotlib.gridspec as gridspec
from scipy.optimize import fsolve

plt = setup_style()
OUT = figures_dir()
warnings.filterwarnings('ignore')

CLUSTER1_COLOR = '#00a8d6'
CLUSTER2_COLOR = '#e85d26'
MIN_COLOR      = '#1565C0'
MAX_COLOR      = '#B71C1C'
SAD_COLOR      = '#F9A825'

# -- HJ parameters -------------------------------------------------------------
np.random.seed(42)
d    = 2
N    = 16
t    = 1.0
y = np.vstack([
    np.random.randn(N//2, d) * 0.4 + np.array([-1.0, -1.0]),
    np.random.randn(N//2, d) * 0.4 + np.array([ 1.0,  1.0]),
])
g = np.array([0.0]*(N//2) + [1.0]*(N//2))
eps_star = N**(-1.0/d)

# -- Core Hopf-Cole functions --------------------------------------------------

def hc_weights(x, eps):
    log_w = -(np.sum((x - y)**2, axis=1) / (4.0*t) + g) / eps
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()

def _entropy_at(x, eps):
    """H(pi(x;eps)) = -sum_j pi_j log pi_j  in [0, log N]."""
    pi = hc_weights(x, eps)
    return float(-np.sum(pi * np.log(pi + 1e-300)))

def grad_H(x, eps):
    """grad_x H = (1/(2t eps)) Cov_pi(y, -log pi)."""
    pi         = hc_weights(x, eps)
    neg_log_pi = -np.log(pi + 1e-300)
    centroid   = pi @ y
    H_val      = float(pi @ neg_log_pi)
    E_y_nlp    = (pi * neg_log_pi) @ y
    return (E_y_nlp - centroid * H_val) / (2.0 * t * eps)

def hessian_H_numerical(x, eps, h=2e-4):
    """Second-order finite-difference Hessian of H."""
    Hmat = np.zeros((2, 2))
    for i in range(2):
        for j in range(i, 2):
            ei = np.zeros(2); ei[i] = h
            ej = np.zeros(2); ej[j] = h
            val = (_entropy_at(x+ei+ej, eps)
                   - _entropy_at(x+ei-ej, eps)
                   - _entropy_at(x-ei+ej, eps)
                   + _entropy_at(x-ei-ej, eps)) / (4.0*h*h)
            Hmat[i, j] = val
            Hmat[j, i] = val
    return Hmat

def classify_critical(Hmat):
    eigs = np.linalg.eigvalsh(Hmat)
    if eigs[0] > 1e-6:
        return 'min'
    elif eigs[1] < -1e-6:
        return 'max'
    else:
        return 'saddle'

# -- Grid for background landscape ---------------------------------------------
res  = 80
xv   = np.linspace(-3.0, 3.0, res)
yv   = np.linspace(-3.0, 3.0, res)
XX, YY = np.meshgrid(xv, yv)
grid   = np.stack([XX.ravel(), YY.ravel()], axis=1)

# -- Find critical points at a given eps ---------------------------------------
DEDUP_TOL = 0.12

_rng_fixed  = np.random.default_rng(0)
_rand_seeds = np.column_stack([
    _rng_fixed.uniform(-2.5, 2.5, 150),
    _rng_fixed.uniform(-2.5, 2.5, 150),
])
_ALL_SEEDS = np.vstack([y, _rand_seeds])

def find_critical_points(eps):
    """Multi-start fsolve on grad_H; deduplicates and classifies results."""
    found = []
    for x0 in _ALL_SEEDS:
        xc, info, ier, _ = fsolve(
            lambda x: grad_H(x, eps), x0, full_output=True, xtol=1e-12)
        res_norm = np.linalg.norm(info['fvec'])
        if res_norm > 1e-6:
            continue
        if not (-2.9 < xc[0] < 2.9 and -2.9 < xc[1] < 2.9):
            continue
        if any(np.linalg.norm(xc - u['pos']) < DEDUP_TOL for u in found):
            continue
        Hmat = hessian_H_numerical(xc, eps)
        eigs = np.linalg.eigvalsh(Hmat)
        found.append({
            'pos':     xc.copy(),
            'type':    classify_critical(Hmat),
            'H':       _entropy_at(xc, eps),
            'eigs':    eigs,
            'min_eig': float(np.min(np.abs(eigs))),
        })
    return found

# -- Scan eps range -------------------------------------------------------------
eps_scan = np.logspace(-2, 0.8, 120)
print(f"Scanning {len(eps_scan)} eps values for critical points of H...")

all_cpts = []
for i, eps in enumerate(eps_scan):
    cpts = find_critical_points(eps)
    all_cpts.append(cpts)
    if (i+1) % 20 == 0 or i == 0:
        n  = len(cpts)
        ns = sum(1 for c in cpts if c['type']=='saddle')
        nm = sum(1 for c in cpts if c['type'] in ('min','max'))
        print(f"  eps={eps:.4f}  total={n}  saddles={ns}  min/max={nm}")

# -- Count series --------------------------------------------------------------
count_total  = [len(c)                                   for c in all_cpts]
count_saddle = [sum(1 for p in c if p['type']=='saddle') for c in all_cpts]
count_min    = [sum(1 for p in c if p['type']=='min')    for c in all_cpts]
count_max    = [sum(1 for p in c if p['type']=='max')    for c in all_cpts]

min_eig_saddle = []
for c in all_cpts:
    saddles = [p['min_eig'] for p in c if p['type']=='saddle']
    min_eig_saddle.append(min(saddles) if saddles else np.nan)

# ==============================================================================
# STATIC SUMMARY FIGURE
# ==============================================================================
print("\nBuilding static figure...")

eps_show  = [0.02, eps_star, 3.0]
H_maps    = [np.array([_entropy_at(pt, ev) for pt in grid]).reshape(res, res)
             for ev in eps_show]
cpts_show = [find_critical_points(ev) for ev in eps_show]
H_max_global = np.log(N)

def _closest_idx(ev):
    return int(np.argmin(np.abs(eps_scan - ev)))

MARKER = {'min': 'v', 'max': '^', 'saddle': 'D'}
MCOLOR = {'min': MIN_COLOR, 'max': MAX_COLOR, 'saddle': SAD_COLOR}
MSIZE  = {'min': 70, 'max': 70, 'saddle': 60}

fig_s = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig_s, hspace=0.52, wspace=0.38)

panel_titles = [
    r'Cold  $\varepsilon=0.02$',
    r'Critical  $\varepsilon^*=%.2f$' % eps_star,
    r'Hot  $\varepsilon=3.0$',
]
for col, (ev, hm, cps, ptitle) in enumerate(
        zip(eps_show, H_maps, cpts_show, panel_titles)):
    ax = fig_s.add_subplot(gs[0, col])
    im = ax.contourf(XX, YY, hm, levels=30,
                     cmap='magma_r', vmin=0, vmax=H_max_global)
    fig_s.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$H(\pi;\varepsilon)$')
    ax.scatter(y[:N//2,0], y[:N//2,1], c=CLUSTER1_COLOR, s=25,
               edgecolors='#333333', linewidths=0.4, zorder=5)
    ax.scatter(y[N//2:,0], y[N//2:,1], c=CLUSTER2_COLOR, marker='^', s=25,
               edgecolors='#333333', linewidths=0.4, zorder=5)
    for cp in cps:
        ax.scatter(*cp['pos'], marker=MARKER[cp['type']],
                   color=MCOLOR[cp['type']], s=MSIZE[cp['type']],
                   edgecolors='black', linewidths=0.7, zorder=6)
    ax.set_xlim([-3, 3]); ax.set_ylim([-3, 3])
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    idx = _closest_idx(ev)
    ax.set_title(
        ptitle + f'\n#min={count_min[idx]}  #saddle={count_saddle[idx]}  #max={count_max[idx]}',
        fontsize=9)

from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], marker='v', color='w', markerfacecolor=MIN_COLOR,
           markeredgecolor='k', markersize=9, label='minimum of $H$'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=SAD_COLOR,
           markeredgecolor='k', markersize=8, label='saddle of $H$'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor=MAX_COLOR,
           markeredgecolor='k', markersize=9, label='maximum of $H$'),
]
fig_s.axes[2].legend(handles=handles, fontsize=9, loc='upper left')

ax_b = fig_s.add_subplot(gs[1, :2])
ax_b.semilogx(eps_scan, count_saddle, color=SAD_COLOR, lw=2.0, label='saddles')
ax_b.semilogx(eps_scan, count_min,    color=MIN_COLOR, lw=2.0, label='minima')
ax_b.semilogx(eps_scan, count_max,    color=MAX_COLOR, lw=2.0, label='maxima')
ax_b.semilogx(eps_scan, count_total,  color='#555555', lw=1.5, ls='--',
              label='total', alpha=0.7)
ax_b.axvline(eps_star, color=C_CRITICAL, ls='--', lw=1.5, label=r'$\varepsilon^*$')
ax_b.set_xlabel(r'Viscosity $\varepsilon$')
ax_b.set_ylabel('Number of critical points')
ax_b.set_title(r'Critical point count vs $\varepsilon$')
ax_b.legend(fontsize=9, ncol=3)
ax_b.set_ylim(bottom=0)

ax_c = fig_s.add_subplot(gs[1, 2])
ax_c.semilogx(eps_scan, min_eig_saddle, color=SAD_COLOR, lw=2)
ax_c.axvline(eps_star, color=C_CRITICAL, ls='--', lw=1.5, label=r'$\varepsilon^*$')
ax_c.set_xlabel(r'Viscosity $\varepsilon$')
ax_c.set_ylabel(r'$\min|\lambda|$ at saddle')
ax_c.set_title(r'Hessian eigenvalue at saddle')
ax_c.legend(fontsize=9)

ax_d = fig_s.add_subplot(gs[2, :])
for ev, cps in zip(eps_scan, all_cpts):
    for cp in cps:
        ax_d.plot(ev, cp['H'], '.', color=MCOLOR[cp['type']], ms=3.5, alpha=0.6)
ax_d.axvline(eps_star, color=C_CRITICAL, ls='--', lw=1.5, label=r'$\varepsilon^*$')
ax_d.axhline(np.log(N), color='#888888', ls=':', lw=1.2, label=r'$\log N$')
ax_d.set_xscale('log')
ax_d.set_xlabel(r'Viscosity $\varepsilon$')
ax_d.set_ylabel(r'$H(\pi;\varepsilon)$ at critical point')
ax_d.set_title(r'Bifurcation diagram: $H$ at each critical point')
ax_d.legend(fontsize=9)
ax_d.set_ylim([-0.1, H_max_global + 0.2])

for ax in [ax_b, ax_c, ax_d]:
    ylims = ax.get_ylim()
    ax.text(0.013, ylims[0] + 0.92*(ylims[1]-ylims[0]), 'PARTICLE',
            fontsize=8, color=C_PARTICLE, va='top')
    ax.text(1.5,   ylims[0] + 0.92*(ylims[1]-ylims[0]), 'WAVE',
            fontsize=8, color=C_WAVE, va='top')

fig_s.suptitle(
    r'Bifurcation Analysis: Attribution-Entropy Landscape  '
    r'($N=%d$, $d=%d$, $\varepsilon^*\approx%.2f$)' % (N, d, eps_star),
    fontsize=11)

fig_s.savefig(os.path.join(OUT, 'bifurcation_diagram.pdf'), bbox_inches='tight')
fig_s.savefig(os.path.join(OUT, 'bifurcation_diagram.png'), dpi=150, bbox_inches='tight')
print("Saved bifurcation_diagram.pdf/.png")
plt.close(fig_s)

# -- Animation ------------------------------------------------------------------
anim_dir = os.path.join(OUT, 'bifurcation_animation')
os.makedirs(anim_dir, exist_ok=True)

eps_frames = np.logspace(-2, 0.8, 60)
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

    H_map = np.array([_entropy_at(pt, eps) for pt in grid]).reshape(res, res)
    cps   = find_critical_points(eps)

    ns = sum(1 for p in cps if p['type']=='saddle')
    nm = sum(1 for p in cps if p['type']=='min')
    nM = sum(1 for p in cps if p['type']=='max')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        r'$\varepsilon=%.4f$   $\varepsilon^*\approx%.2f$   %s'
        % (eps, eps_star, regime)
        + f'   |  #min={nm}  #saddle={ns}  #max={nM}',
        fontsize=10, color=rc)

    ax = axes[0]
    im = ax.contourf(XX, YY, H_map, levels=30,
                     cmap='magma_r', vmin=0, vmax=H_max_global)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$H(\pi;\varepsilon)$')
    ax.scatter(y[:N//2,0], y[:N//2,1], c=CLUSTER1_COLOR, s=22,
               edgecolors='#333333', linewidths=0.4, zorder=5)
    ax.scatter(y[N//2:,0], y[N//2:,1], c=CLUSTER2_COLOR, marker='^', s=22,
               edgecolors='#333333', linewidths=0.4, zorder=5)
    for cp in cps:
        ax.scatter(*cp['pos'], marker=MARKER[cp['type']],
                   color=MCOLOR[cp['type']], s=MSIZE[cp['type']],
                   edgecolors='black', linewidths=0.8, zorder=6)
    ax.set_xlim([-3,3]); ax.set_ylim([-3,3])
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(r'Entropy $H(\pi)$ + critical points')

    gH_map = np.array([np.linalg.norm(grad_H(pt, eps)) for pt in grid]).reshape(res, res)
    gH_max = np.percentile(gH_map, 97)
    ax = axes[1]
    im2 = ax.contourf(XX, YY, gH_map, levels=30, cmap='YlOrRd',
                      vmin=0, vmax=max(gH_max, 1e-8))
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label=r'$|\nabla H|$')
    for cp in cps:
        ax.scatter(*cp['pos'], marker=MARKER[cp['type']],
                   color=MCOLOR[cp['type']], s=MSIZE[cp['type']],
                   edgecolors='black', linewidths=0.8, zorder=6)
    ax.set_xlim([-3,3]); ax.set_ylim([-3,3])
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(r'$|\nabla_x H|$ + critical points')

    ax = axes[2]
    idx_so_far = _closest_idx(eps) + 1
    for ev, clist in zip(eps_scan[:idx_so_far], all_cpts[:idx_so_far]):
        for cp in clist:
            ax.plot(ev, cp['H'], '.', color=MCOLOR[cp['type']], ms=3, alpha=0.55)
    if cps:
        for cp in cps:
            ax.plot(eps, cp['H'], marker=MARKER[cp['type']],
                    color=MCOLOR[cp['type']], ms=9,
                    markeredgecolor='black', markeredgewidth=0.8)
    ax.axhline(np.log(N), color='#888888', ls=':', lw=1.2)
    ax.axvline(eps_star, color=C_CRITICAL, ls='--', lw=1.3)
    ax.set_xscale('log')
    ax.set_xlim([eps_scan[0]*0.9, eps_scan[-1]*1.1])
    ax.set_ylim([-0.1, H_max_global + 0.2])
    ax.set_xlabel(r'Viscosity $\varepsilon$')
    ax.set_ylabel(r'$H$ at critical point')
    ax.set_title('Bifurcation diagram')
    ax.axvline(eps, color=rc, ls='-', lw=2, alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(anim_dir, f'frame_{i:03d}.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    frame_paths.append(out_path)

    if (i+1) % 10 == 0 or i == 0:
        print(f"  {i+1}/{len(eps_frames)}  eps={eps:.4f}  "
              f"#cpts={len(cps)} (min={nm} sad={ns} max={nM})")

print(f"All {len(eps_frames)} frames saved.")

# -- Assemble GIF ---------------------------------------------------------------
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

gif_path = os.path.join(OUT, 'bifurcation_transition.gif')

if HAS_IMAGEIO:
    from PIL import Image as PILImage
    print(f"\nAssembling GIF -> {gif_path}")
    raw = [PILImage.open(p).convert('RGB') for p in frame_paths]
    W_px, H_px = raw[0].size
    raw = [im.resize((W_px, H_px), PILImage.LANCZOS) for im in raw]
    loop_frames = raw + raw[::-1]
    loop_frames[0].save(
        gif_path, save_all=True, append_images=loop_frames[1:],
        duration=100, loop=0, optimize=False,
    )
    print(f"Saved {gif_path}  ({len(loop_frames)} frames)")
else:
    print("\nimageio not found.  Install: pip install imageio pillow")

print("\nDone.")

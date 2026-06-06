// builder.js -- SVG architecture builder with pointer-based drag-and-drop.
// Renders Input -> [blocks] -> LSE readout as an SVG diagram. Drag blocks to
// reorder, drag a palette chip in to insert, click the small x to remove. An
// invalid connection (ResNet with no layer before it) is drawn red and blocks Play.

const ARCH_LABEL = { dense: 'Dense', resnet: 'ResNet', attn: 'Self-Attention' };
const ARCH_MAX = 10;   // max hidden blocks (Input + 10 + LSE readout = 12 nodes)
let archDrag = null, archIndic = -1;
const SVGNS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }
function archNodes() { return ['__in'].concat(arch, ['__out']); }

function archGeom(W) {
  // tall, narrow nodes with vertical labels, centered horizontally
  const nodes = archNodes(), N = nodes.length, pad = 8, gap = 34;
  let nodeW = 46, total = N * nodeW + (N - 1) * gap;
  if (total + 2 * pad > W) { nodeW = Math.max(28, (W - 2 * pad - (N - 1) * gap) / N); total = N * nodeW + (N - 1) * gap; }
  const startX = Math.max(pad, (W - total) / 2);
  const xs = []; let x = startX; for (let i = 0; i < N; i++) { xs.push(x); x += nodeW + gap; }
  return { nodes, xs, nodeW, g: gap, h: 132, ny: 14, nh: 104 };
}
function archBlockCenters(G) { const c = []; for (let l = 0; l < arch.length; l++) c.push(G.xs[l + 1] + G.nodeW / 2); return c; }
function archInsertIndex(localX, G) { let idx = 0; for (const cx of archBlockCenters(G)) if (localX > cx) idx++; return Math.max(0, Math.min(arch.length, idx)); }

function archRenderSVG() {
  const svg = document.getElementById('arch_svg'); if (!svg) return;
  const W = Math.max(svg.clientWidth || (svg.parentElement ? svg.parentElement.clientWidth - 12 : 600), 320);
  const invalid = archValid(arch), G = archGeom(W);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + G.h);
  svg.setAttribute('height', G.h);

  const defs = svgEl('defs', {});
  [['ah2', '#9ca3af'], ['ah2b', '#dc2626']].forEach(p => {
    const mk = svgEl('marker', { id: p[0], markerWidth: 8, markerHeight: 8, refX: 6, refY: 3, orient: 'auto', markerUnits: 'strokeWidth' });
    mk.appendChild(svgEl('path', { d: 'M0,0 L6,3 L0,6 Z', fill: p[1] })); defs.appendChild(mk);
  });
  svg.appendChild(defs);

  for (let i = 0; i < G.nodes.length - 1; i++) {
    const x1 = G.xs[i] + G.nodeW, x2 = G.xs[i + 1], y = G.ny + G.nh / 2, bad = (invalid !== -1 && i === invalid);
    svg.appendChild(svgEl('line', { x1: x1, y1: y, x2: x2 - 7, y2: y, stroke: bad ? '#dc2626' : '#9ca3af', 'stroke-width': bad ? 2 : 1.4, 'marker-end': 'url(#' + (bad ? 'ah2b' : 'ah2') + ')' }));
    if (bad) { const t = svgEl('text', { x: (x1 + x2) / 2, y: y - 6, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: '#dc2626' }); t.textContent = 'X'; svg.appendChild(t); }
  }

  if (archDrag && archIndic >= 0) {
    const ix = (G.xs[archIndic] + G.nodeW + G.xs[archIndic + 1]) / 2;
    svg.appendChild(svgEl('line', { x1: ix, y1: G.ny - 6, x2: ix, y2: G.ny + G.nh + 6, stroke: '#2563eb', 'stroke-width': 2.5 }));
  }

  for (let i = 0; i < G.nodes.length; i++) {
    const n = G.nodes[i], x = G.xs[i], fixed = (n === '__in' || n === '__out'), isRes = (n === 'resnet'), isAttn = (n === 'attn');
    const grp = svgEl('g', { class: 'arch-node' }); if (!fixed) grp.setAttribute('data-idx', i - 1);
    const stroke = fixed ? '#9ca3af' : (isAttn ? '#0d9488' : isRes ? '#7c3aed' : '#2563eb');
    const fill = fixed ? '#ffffff' : (isAttn ? '#f0fdfa' : isRes ? '#f5f3ff' : '#eff6ff');
    grp.setAttribute('opacity', (archDrag && archDrag.kind === 'move' && archDrag.idx === i - 1) ? 0.4 : 1);
    grp.appendChild(svgEl('rect', { x: x, y: G.ny, width: G.nodeW, height: G.nh, rx: 7, fill: fill, stroke: stroke, 'stroke-width': 1.5, style: fixed ? 'cursor:default' : 'cursor:grab' }));
    const label = fixed ? (n === '__in' ? 'Input (x)' : 'LSE readout') : ARCH_LABEL[n];
    const cx = x + G.nodeW / 2, cy = G.ny + G.nh / 2;
    const txt = svgEl('text', { x: cx, y: cy, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-size': 11, 'font-weight': 600, fill: stroke, 'pointer-events': 'none', transform: 'rotate(-90 ' + cx + ' ' + cy + ')' }); txt.textContent = label; grp.appendChild(txt);
    if (!fixed) {
      const xg = svgEl('g', { class: 'arch-x', 'data-idx': i - 1, style: 'cursor:pointer' });
      xg.appendChild(svgEl('circle', { cx: x + G.nodeW - 9, cy: G.ny + 9, r: 7, fill: '#fff', stroke: '#d1d5db' }));
      const xt = svgEl('text', { x: x + G.nodeW - 9, y: G.ny + 12.5, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: '#6b7280', 'pointer-events': 'none' }); xt.textContent = 'x'; xg.appendChild(xt);
      grp.appendChild(xg);
    }
    svg.appendChild(grp);
  }

  const msg = document.getElementById('arch_msg'), pl = document.getElementById('tr_play');
  if (invalid !== -1) { if (msg) { msg.textContent = 'Cannot connect: a ResNet or Self-Attention block needs a layer before it (it acts on the H hidden units, not the scalar input). Remove or reorder it to train.'; msg.className = 'arch-msg bad'; } if (pl) { pl.disabled = true; pl.classList.add('disabled'); } }
  else { if (msg) { msg.textContent = arch.length === 0 ? 'Single LSE layer (convex). Drag a block in to go deeper and fit non-convex targets.' : (arch.length + ' hidden block' + (arch.length > 1 ? 's' : '') + ' + LSE readout. ' + (arch.length >= ARCH_MAX ? 'Maximum ' + ARCH_MAX + ' blocks reached. ' : '') + 'Press Play to train this network.'); msg.className = 'arch-msg'; } if (pl) { pl.disabled = false; pl.classList.remove('disabled'); } }
}

function setupBuilder() {
  const svg = document.getElementById('arch_svg'); if (!svg) return;
  let archMoved = false;
  const dropIndex = e => archInsertIndex(e.clientX - svg.getBoundingClientRect().left, archGeom(Math.max(svg.clientWidth, 320)));
  const overSvg = e => { const r = svg.getBoundingClientRect(); return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom; };

  document.querySelectorAll('.arch-chip.palette').forEach(p => {
    p.addEventListener('pointerdown', e => { archDrag = { kind: 'new', type: p.dataset.type }; archIndic = arch.length; archMoved = false; e.preventDefault(); });
  });

  svg.addEventListener('pointerdown', e => {
    const xg = e.target.closest && e.target.closest('.arch-x');
    if (xg) { arch.splice(+xg.getAttribute('data-idx'), 1); archRenderSVG(); trRebuild(); return; }
    const ng = e.target.closest && e.target.closest('.arch-node[data-idx]');
    if (ng) { archDrag = { kind: 'move', idx: +ng.getAttribute('data-idx') }; archIndic = archDrag.idx; archMoved = false; e.preventDefault(); archRenderSVG(); }
  });

  document.addEventListener('pointermove', e => {
    if (!archDrag) return;
    archMoved = true; archIndic = dropIndex(e); archRenderSVG();
  });

  document.addEventListener('pointerup', e => {
    if (!archDrag) return;
    if (archDrag.kind === 'new') {
      if (arch.length >= ARCH_MAX) { /* at the cap: ignore the add */ }
      else if (archMoved && overSvg(e)) arch.splice(dropIndex(e), 0, archDrag.type);   // dragged into the diagram
      else arch.push(archDrag.type);                                              // a plain click appends once
    } else if (archMoved && overSvg(e)) {                                         // reorder only on a real drag
      const idx = dropIndex(e), b = arch.splice(archDrag.idx, 1)[0];
      let t = idx; if (archDrag.idx < idx) t--; arch.splice(Math.max(0, Math.min(arch.length, t)), 0, b);
    }
    archDrag = null; archIndic = -1; archMoved = false; archRenderSVG(); trRebuild();
  });

  const clr = document.getElementById('arch_clear'); if (clr) clr.addEventListener('click', () => { arch.length = 0; archRenderSVG(); trRebuild(); });
  archRenderSVG();
}

// bifurcation.js -- Interactive: fold bifurcations of the attribution landscape.

// 1-D support points (irregular spacing so basins merge at different eps).
const BIFY = [-2.6, -1.5, -0.4, 0.7, 1.7, 2.7];
const BIF_NB = BIFY.length;
const BIF_HMAX = Math.log(BIF_NB);
const BIF_LO = -4, BIF_HI = 4;
let bifEps = 0.5;

// Attribution entropy along the slice: pi_j(x) ~ exp(-(x-y_j)^2 / (2 eps)).
function bifEntropy(x, eps) {
  const raw = BIFY.map(y => -((x - y) * (x - y)) / 2 / eps);
  const m = Math.max(...raw);
  const ex = raw.map(r => Math.exp(r - m));
  const Z = ex.reduce((p, c) => p + c, 0);
  let H = 0;
  for (const e of ex) { const p = e / Z; if (p > 1e-12) H -= p * Math.log(p); }
  return H;
}

function bifCriticals(eps) {
  const N = 600, crit = [];
  const xs = [], hs = [];
  for (let i = 0; i < N; i++) { const x = BIF_LO + (BIF_HI - BIF_LO) * i / (N - 1); xs.push(x); hs.push(bifEntropy(x, eps)); }
  for (let i = 1; i < N - 1; i++) {
    if (hs[i] > hs[i-1] && hs[i] > hs[i+1]) crit.push({ x: xs[i], H: hs[i], type: 'max' });
    else if (hs[i] < hs[i-1] && hs[i] < hs[i+1]) crit.push({ x: xs[i], H: hs[i], type: 'min' });
  }
  return crit;
}

function triDown(ctx, x, y, s) { ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y - s); ctx.lineTo(x, y + s); ctx.closePath(); ctx.fill(); }
function triUp(ctx, x, y, s)   { ctx.beginPath(); ctx.moveTo(x - s, y + s); ctx.lineTo(x + s, y + s); ctx.lineTo(x, y - s); ctx.closePath(); ctx.fill(); }

function drawBifLandscape() {
  const c = CV['c_bif_land']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 18, r: 16, b: 30, l: 40 };
  const yMax = BIF_HMAX * 1.12;
  const toX = v => pad.l + (v - BIF_LO) / (BIF_HI - BIF_LO) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v / yMax) * (H - pad.t - pad.b);

  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 1; ctx.fillStyle = '#9ca3af';
  ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  [0, 0.5, 1, 1.5].filter(v => v <= yMax).forEach(yv => {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(yv)); ctx.lineTo(W - pad.r, toY(yv)); ctx.stroke();
    ctx.fillText(yv.toFixed(1), pad.l - 4, toY(yv) + 3);
  });
  ctx.textAlign = 'center';
  [-3,-2,-1,0,1,2,3].forEach(xv => ctx.fillText(xv, toX(xv), H - pad.b + 12));

  const Npts = 400;
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.5; ctx.beginPath();
  for (let i = 0; i < Npts; i++) {
    const x = BIF_LO + (BIF_HI - BIF_LO) * i / (Npts - 1);
    const cx = toX(x), cy = toY(bifEntropy(x, bifEps));
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  const crit = bifCriticals(bifEps);
  crit.forEach(p => {
    if (p.type === 'min') { ctx.fillStyle = '#2563eb'; triDown(ctx, toX(p.x), toY(p.H), 5); }
    else { ctx.fillStyle = '#dc2626'; triUp(ctx, toX(p.x), toY(p.H), 5); }
  });
  const cnt = document.getElementById('bif_count');
  if (cnt) cnt.textContent = crit.length;

  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('slice coordinate x', pad.l + (W - pad.l - pad.r) / 2, H - 4);
  ctx.save(); ctx.translate(10, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('H(x; eps)', 0, 0); ctx.restore();
}

function drawBifDiagram() {
  const c = CV['c_bif_diag']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 18, r: 16, b: 30, l: 40 };
  const lo = Math.log(0.02), hi = Math.log(3.0);
  const yMax = BIF_HMAX * 1.12;
  const toX = le => pad.l + (le - lo) / (hi - lo) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v / yMax) * (H - pad.t - pad.b);

  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 1; ctx.fillStyle = '#9ca3af';
  ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  [0, 0.5, 1, 1.5].filter(v => v <= yMax).forEach(yv => {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(yv)); ctx.lineTo(W - pad.r, toY(yv)); ctx.stroke();
    ctx.fillText(yv.toFixed(1), pad.l - 4, toY(yv) + 3);
  });
  ctx.textAlign = 'center';
  [0.02, 0.1, 0.5, 3.0].forEach(ev => ctx.fillText(ev, toX(Math.log(ev)), H - pad.b + 12));

  const M = 180;
  for (let i = 0; i < M; i++) {
    const le = lo + (hi - lo) * i / (M - 1);
    const crit = bifCriticals(Math.exp(le));
    crit.forEach(p => {
      ctx.fillStyle = p.type === 'min' ? 'rgba(37,99,235,0.55)' : 'rgba(220,38,38,0.55)';
      ctx.beginPath(); ctx.arc(toX(le), toY(p.H), 1.4, 0, 2 * Math.PI); ctx.fill();
    });
  }

  ctx.strokeStyle = '#111827'; ctx.lineWidth = 1.2; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(toX(Math.log(bifEps)), pad.t); ctx.lineTo(toX(Math.log(bifEps)), H - pad.b); ctx.stroke();
  ctx.setLineDash([]);
  bifCriticals(bifEps).forEach(p => {
    ctx.fillStyle = p.type === 'min' ? '#2563eb' : '#dc2626';
    ctx.beginPath(); ctx.arc(toX(Math.log(bifEps)), toY(p.H), 3.2, 0, 2 * Math.PI); ctx.fill();
  });

  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('viscosity eps (log scale)', pad.l + (W - pad.l - pad.r) / 2, H - 4);
  ctx.save(); ctx.translate(10, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('critical H-value', 0, 0); ctx.restore();
}

(function(){
  const sl = document.getElementById('sl_bifeps');
  if (sl) sl.addEventListener('input', function() {
    bifEps = parseFloat(this.value);
    document.getElementById('v_bifeps').textContent = bifEps.toFixed(2);
    drawBifLandscape(); drawBifDiagram();
  });
})();

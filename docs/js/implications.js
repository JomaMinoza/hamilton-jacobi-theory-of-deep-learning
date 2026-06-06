// implications.js -- Interactive: hallucination (OOD extrapolation) and double descent.

// --- Hallucination: LSE network from support points (Hopf-Cole reparam) ---
const HT = 1.0;
const HY = [-1.6, -1.0, -0.4, 0.4, 1.0, 1.6];
const HG = HY.map(y => 0.7 * Math.cos(1.3 * y) + 0.3);
const HW = HY.map(y => y / (2 * HT));
const HB = HY.map((y, i) => -HG[i] - y * y / (4 * HT));
let hEps = 0.3;

function hLSE(x, eps) {
  const z = HW.map((w, i) => w * x + HB[i]);
  const m = Math.max(...z);
  return m + eps * Math.log(z.reduce((s, zi) => s + Math.exp((zi - m) / eps), 0));
}
function hInDist(x, eps) {
  const r = Math.sqrt(2 * eps * HT);
  return HY.some(y => Math.abs(x - y) <= r);
}
function hEntropy(x, eps) {
  const z = HW.map((w, i) => w * x + HB[i]);
  const m = Math.max(...z);
  const e = z.map(zi => Math.exp((zi - m) / eps));
  const Z = e.reduce((a, b) => a + b, 0);
  let Hs = 0;
  for (const ei of e) { const p = ei / Z; if (p > 1e-12) Hs -= p * Math.log(p); }
  return Hs;
}

function drawHalluc() {
  const c = CV['c_halluc']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 16, r: 16, b: 30, l: 42 };
  const x0 = -4, x1 = 4, N = 360;
  const xs = [], fs = [];
  for (let i = 0; i < N; i++) { const x = x0 + (x1 - x0) * i / (N - 1); xs.push(x); fs.push(hLSE(x, hEps)); }
  let ymin = Math.min(...fs), ymax = Math.max(...fs);
  const padY = 0.12 * ((ymax - ymin) || 1); ymin -= padY; ymax += padY;
  const toX = v => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v - ymin) / (ymax - ymin) * (H - pad.t - pad.b);

  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  const r = Math.sqrt(2 * hEps * HT);
  ctx.fillStyle = '#eff6ff';
  HY.forEach(y => {
    const xa = Math.max(x0, y - r), xb = Math.min(x1, y + r);
    if (xb > xa) ctx.fillRect(toX(xa), pad.t, toX(xb) - toX(xa), H - pad.t - pad.b);
  });
  ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  [-4, -2, 0, 2, 4].forEach(xv => ctx.fillText(xv, toX(xv), H - pad.b + 12));
  for (let i = 1; i < N; i++) {
    const id = hInDist((xs[i - 1] + xs[i]) / 2, hEps);
    ctx.strokeStyle = id ? '#2563eb' : '#dc2626'; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(toX(xs[i - 1]), toY(fs[i - 1])); ctx.lineTo(toX(xs[i]), toY(fs[i])); ctx.stroke();
  }
  const logN = Math.log(HY.length);
  const toYe = h => H - pad.b - (h / logN) * (H - pad.t - pad.b);
  ctx.strokeStyle = '#0d9488'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const cx = toX(xs[i]), cy = toYe(hEntropy(xs[i], hEps));
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#111827';
  HY.forEach(y => { ctx.beginPath(); ctx.arc(toX(y), H - pad.b, 3, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('input x', pad.l + (W - pad.l - pad.r) / 2, H - 2);
  ctx.save(); ctx.translate(11, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('output', 0, 0); ctx.restore();
}

(function(){
  const sl = document.getElementById('sl_halluc');
  if (sl) sl.addEventListener('input', function() {
    hEps = parseFloat(this.value);
    document.getElementById('v_halluc').textContent = hEps.toFixed(2);
    drawHalluc();
  });
})();

// --- Double descent: curvature = Gibbs variance / eps ---
const ST = 1.0;
const SY = [-3, -2, -1, 0, 1, 2, 3];
const SW = SY.map(y => y / (2 * ST));
const SB = SY.map(y => -y * y / (4 * ST));
let sEps = 0.3;

function sCurv(x, eps) {
  const z = SW.map((w, i) => w * x + SB[i]);
  const m = Math.max(...z);
  const e = z.map(zi => Math.exp((zi - m) / eps));
  const Z = e.reduce((a, b) => a + b, 0);
  let EW = 0, EW2 = 0;
  for (let i = 0; i < SW.length; i++) { const p = e[i] / Z; EW += p * SW[i]; EW2 += p * SW[i] * SW[i]; }
  return (EW2 - EW * EW) / eps;
}

function drawShock() {
  const c = CV['c_shock']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 18, r: 16, b: 30, l: 44 };
  const x0 = -3.5, x1 = 3.5, N = 400;
  const xs = [], ks = [];
  for (let i = 0; i < N; i++) { const x = x0 + (x1 - x0) * i / (N - 1); xs.push(x); ks.push(sCurv(x, sEps)); }
  const ymax = Math.max(...ks, 1e-6) * 1.12;
  const toX = v => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  for (let k = 0; k <= 3; k++) { const yv = ymax * k / 3, y = toY(yv); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(yv.toFixed(1), pad.l - 4, y + 3); }
  ctx.textAlign = 'center';
  [-3, -2, -1, 0, 1, 2, 3].forEach(xv => ctx.fillText(xv, toX(xv), H - pad.b + 12));
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 0; i < N; i++) { const cx = toX(xs[i]), cy = toY(ks[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke();
  ctx.fillStyle = '#cbd5e1';
  SY.forEach(y => { ctx.beginPath(); ctx.arc(toX(y), H - pad.b, 2.5, 0, 2 * Math.PI); ctx.fill(); });
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('input x', pad.l + (W - pad.l - pad.r) / 2, H - 2);
  ctx.save(); ctx.translate(11, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('curvature', 0, 0); ctx.restore();
}

(function(){
  const sl = document.getElementById('sl_shock');
  if (sl) sl.addEventListener('input', function() {
    sEps = parseFloat(this.value);
    document.getElementById('v_shock').textContent = sEps.toFixed(2);
    drawShock();
  });
})();

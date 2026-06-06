// particle-wave.js -- Interactive: particle <-> wave attribution transition.

// Fixed neuron scores (smooth two-bump profile, deterministic).
const SCORES = [0.2, 0.6, 1.1, 1.7, 2.2, 2.0, 1.3, 0.9, 1.4, 1.8, 1.1, 0.5];
const NEU = SCORES.length;
let eps2 = 0.5;

function softmax(scores, e) {
  const m = Math.max(...scores);
  const ex = scores.map(s => Math.exp((s - m) / e));
  const Z = ex.reduce((p, c) => p + c, 0);
  return ex.map(v => v / Z);
}
function entropy(p) { return -p.reduce((s, v) => s + (v > 1e-12 ? v * Math.log(v) : 0), 0); }

function drawWeights() {
  const c = CV['c_weights']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 16, r: 14, b: 30, l: 40 };
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  const pi = softmax(SCORES, eps2);
  const maxP = Math.max(...pi, 0.15);
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const bw = plotW / NEU * 0.7, gap = plotW / NEU;

  ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#9ca3af';
  ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const yv = maxP * k / 4;
    const y = pad.t + plotH - (yv / maxP) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(yv.toFixed(2), pad.l - 4, y + 3);
  }
  for (let j = 0; j < NEU; j++) {
    const x = pad.l + j * gap + (gap - bw) / 2;
    const h = (pi[j] / maxP) * plotH;
    ctx.fillStyle = '#2563eb';
    ctx.fillRect(x, pad.t + plotH - h, bw, h);
  }
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('neuron index j', pad.l + plotW / 2, H - 4);
  ctx.save(); ctx.translate(10, pad.t + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('pi_j', 0, 0); ctx.restore();
}

function drawEntropy() {
  const c = CV['c_entropy']; if (!c) return;
  const { ctx, W, H } = c;
  const padE = { t: 18, r: 16, b: 30, l: 40 };
  const lo = Math.log(0.03), hi = Math.log(5.0);
  const Hmax = Math.log(NEU);
  const toX = le => padE.l + (le - lo) / (hi - lo) * (W - padE.l - padE.r);
  const toY = hv => H - padE.b - (hv / (Hmax * 1.05)) * (H - padE.t - padE.b);

  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#9ca3af';
  ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  [0, 0.5, 1, 1.5, 2, 2.5].filter(v => v <= Hmax * 1.05).forEach(hv => {
    ctx.beginPath(); ctx.moveTo(padE.l, toY(hv)); ctx.lineTo(W - padE.r, toY(hv)); ctx.stroke();
    ctx.fillText(hv.toFixed(1), padE.l - 4, toY(hv) + 3);
  });
  ctx.textAlign = 'center';
  [0.05, 0.2, 1.0, 5.0].forEach(ev => ctx.fillText(ev, toX(Math.log(ev)), H - padE.b + 12));
  ctx.strokeStyle = '#d1d5db'; ctx.setLineDash([5,4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padE.l, toY(Hmax)); ctx.lineTo(W - padE.r, toY(Hmax)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#9ca3af'; ctx.textAlign = 'left';
  ctx.fillText('log N (max entropy)', padE.l + 6, toY(Hmax) - 4);

  const Npts = 200;
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2.5; ctx.beginPath();
  for (let i = 0; i < Npts; i++) {
    const le = lo + (hi - lo) * i / (Npts - 1);
    const hv = entropy(softmax(SCORES, Math.exp(le)));
    const cx = toX(le), cy = toY(hv);
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  const epsStar = Math.pow(NEU, -1 / 2);
  ctx.strokeStyle = '#f9a825'; ctx.setLineDash([4,3]); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(toX(Math.log(epsStar)), padE.t); ctx.lineTo(toX(Math.log(epsStar)), H - padE.b); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#b45309'; ctx.textAlign = 'center'; ctx.font = '10px Inter, sans-serif';
  ctx.fillText('eps*', toX(Math.log(epsStar)), padE.t + 10);

  const curH = entropy(softmax(SCORES, eps2));
  ctx.fillStyle = '#dc2626';
  ctx.beginPath(); ctx.arc(toX(Math.log(eps2)), toY(curH), 4.5, 0, 2 * Math.PI); ctx.fill();

  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('viscosity eps (log scale)', padE.l + (W - padE.l - padE.r) / 2, H - 4);
}

(function(){
  const sl = document.getElementById('sl_eps2');
  if (sl) sl.addEventListener('input', function() {
    eps2 = parseFloat(this.value);
    document.getElementById('v_eps2').textContent = eps2.toFixed(2);
    drawWeights(); drawEntropy();
  });
})();

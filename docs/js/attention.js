// attention.js -- Interactive: real distilgpt2 attention (uses window.ATTN_DATA).

let attnSent = 0, attnLayer = 0, attnHead = 0;
let HEAT_GEO = null, attnReady = false;

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[68, 1, 84], [33, 144, 141], [253, 231, 37]];
  const i = t < 0.5 ? 0 : 1, f = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}
function attnTrunc(s) { return s.length > 6 ? s.slice(0, 6) : s; }

function pickMostParticle() {
  const ent = window.ATTN_DATA.sentences[attnSent].entropy;
  let best = 2, bl = 0, bh = 0;
  for (let l = 0; l < ent.length; l++)
    for (let h = 0; h < ent[l].length; h++)
      if (ent[l][h] < best) { best = ent[l][h]; bl = l; bh = h; }
  attnLayer = bl; attnHead = bh;
}

function drawAttnHeat() {
  const c = CV['c_attn_heat']; if (!c || !window.ATTN_DATA) return;
  const { ctx, W, H } = c; const D = window.ATTN_DATA;
  const ent = D.sentences[attnSent].entropy, L = D.layers, Hn = D.heads;
  const pad = { t: 22, r: 8, b: 12, l: 46 };
  const cw = (W - pad.l - pad.r) / Hn, ch = (H - pad.t - pad.b) / L;
  HEAT_GEO = { pad, cw, ch, L, Hn };
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  for (let l = 0; l < L; l++) for (let h = 0; h < Hn; h++) {
    ctx.fillStyle = viridis(ent[l][h]);
    ctx.fillRect(pad.l + h * cw, pad.t + l * ch, cw - 1, ch - 1);
  }
  ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 2;
  ctx.strokeRect(pad.l + attnHead * cw, pad.t + attnLayer * ch, cw - 1, ch - 1);
  ctx.fillStyle = '#6b7280'; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('head', pad.l + (W - pad.l - pad.r) / 2, 11);
  ctx.textAlign = 'right';
  for (let l = 0; l < L; l++) ctx.fillText('L' + l, pad.l - 4, pad.t + l * ch + ch / 2 + 3);
}

function drawAttnMat() {
  const c = CV['c_attn_mat']; if (!c || !window.ATTN_DATA) return;
  const { ctx, W, H } = c; const s = window.ATTN_DATA.sentences[attnSent];
  const A = s.attn[attnLayer][attnHead], toks = s.tokens, T = toks.length;
  const pad = { t: 8, r: 8, b: 50, l: 52 };
  const cw = (W - pad.l - pad.r) / T, ch = (H - pad.t - pad.b) / T;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  for (let q = 0; q < T; q++) for (let k = 0; k < T; k++) {
    const t = Math.min(1, A[q][k]);
    ctx.fillStyle = `rgb(${Math.round(255 - 218 * t)},${Math.round(255 - 156 * t)},${Math.round(255 - 20 * t)})`;
    ctx.fillRect(pad.l + k * cw, pad.t + q * ch, cw - 0.4, ch - 0.4);
  }
  ctx.fillStyle = '#374151'; ctx.font = Math.max(6, Math.min(10, ch * 0.85)) + 'px Inter, sans-serif';
  ctx.textAlign = 'right';
  for (let q = 0; q < T; q++) ctx.fillText(attnTrunc(toks[q]), pad.l - 3, pad.t + q * ch + ch / 2 + 3);
  for (let k = 0; k < T; k++) {
    ctx.save(); ctx.translate(pad.l + k * cw + cw / 2, H - pad.b + 5);
    ctx.rotate(-Math.PI / 4); ctx.textAlign = 'right'; ctx.fillText(attnTrunc(toks[k]), 0, 0); ctx.restore();
  }
  const ent = s.entropy[attnLayer][attnHead];
  const label = ent < 0.33 ? 'particle (focused / retrieval)' : ent > 0.66 ? 'wave (diffuse / mixing)' : 'mixed';
  const tEl = document.getElementById('attn-head-title');
  const nEl = document.getElementById('attn-head-note');
  if (tEl) tEl.innerHTML = 'Attention: layer ' + attnLayer + ', head ' + attnHead;
  if (nEl) nEl.innerHTML = 'H/log k = ' + ent.toFixed(3) + ', ' + label + ' (row = query, col = key)';
}

function drawAttnDepth() {
  const c = CV['c_attn_depth']; if (!c || !window.ATTN_DATA) return;
  const { ctx, W, H } = c; const D = window.ATTN_DATA, L = D.layers, Hn = D.heads;
  const m = D.sentences[attnSent].entropy;   // per (layer, head) for the selected sentence
  const pad = { t: 14, r: 14, b: 30, l: 44 };
  const toX = l => L > 1 ? pad.l + l / (L - 1) * (W - pad.l - pad.r) : pad.l + (W - pad.l - pad.r) / 2;
  const toY = v => H - pad.b - v * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  [0, 0.25, 0.5, 0.75, 1].forEach(v => {
    const y = toY(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), pad.l - 4, y + 3);
  });
  ctx.textAlign = 'center';
  for (let l = 0; l < L; l++) ctx.fillText('L' + l, toX(l), H - pad.b + 12);
  const mean = [], lo = [], hi = [];
  for (let l = 0; l < L; l++) {
    let s = 0, mn = 1, mx = 0;
    for (let h = 0; h < Hn; h++) { const v = m[l][h]; s += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    mean.push(s / Hn); lo.push(mn); hi.push(mx);
  }
  ctx.fillStyle = 'rgba(37,99,235,0.12)'; ctx.beginPath();
  for (let l = 0; l < L; l++) { const x = toX(l), y = toY(hi[l]); if (l === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  for (let l = L - 1; l >= 0; l--) ctx.lineTo(toX(l), toY(lo[l]));
  ctx.closePath(); ctx.fill();
  const jit = Math.min(16, (W - pad.l - pad.r) / Math.max(L, 1) * 0.55);
  ctx.fillStyle = 'rgba(148,163,184,0.7)';
  for (let l = 0; l < L; l++) for (let h = 0; h < Hn; h++) {
    const jx = Hn > 1 ? (h / (Hn - 1) - 0.5) * jit : 0;
    ctx.beginPath(); ctx.arc(toX(l) + jx, toY(m[l][h]), 2, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.4; ctx.beginPath();
  for (let l = 0; l < L; l++) { const x = toX(l), y = toY(mean[l]); if (l === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  ctx.stroke();
  ctx.fillStyle = '#2563eb';
  for (let l = 0; l < L; l++) { ctx.beginPath(); ctx.arc(toX(l), toY(mean[l]), 3.5, 0, 2 * Math.PI); ctx.fill(); }
  const hjx = Hn > 1 ? (attnHead / (Hn - 1) - 0.5) * jit : 0;
  ctx.fillStyle = '#dc2626';
  ctx.beginPath(); ctx.arc(toX(attnLayer) + hjx, toY(m[attnLayer][attnHead]), 4, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('layer (depth = HJ time)', pad.l + (W - pad.l - pad.r) / 2, H - 2);
  ctx.save(); ctx.translate(11, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('entropy H/log k', 0, 0); ctx.restore();
}

function setupAttn() {
  if (attnReady || !window.ATTN_DATA) return;
  const D = window.ATTN_DATA, sel = document.getElementById('attn-sentence');
  if (!sel) return;
  D.sentences.forEach((s, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = (i + 1) + '. ' + s.text;
    sel.appendChild(o);
  });
  sel.addEventListener('change', function() {
    attnSent = parseInt(this.value); pickMostParticle(); drawAttnHeat(); drawAttnMat(); drawAttnDepth();
  });
  const heat = document.getElementById('c_attn_heat');
  heat.addEventListener('click', function(e) {
    if (!HEAT_GEO) return;
    const h = Math.floor((e.offsetX - HEAT_GEO.pad.l) / HEAT_GEO.cw);
    const l = Math.floor((e.offsetY - HEAT_GEO.pad.t) / HEAT_GEO.ch);
    if (h >= 0 && h < HEAT_GEO.Hn && l >= 0 && l < HEAT_GEO.L) {
      attnHead = h; attnLayer = l; drawAttnHeat(); drawAttnMat(); drawAttnDepth();
    }
  });
  pickMostParticle();
  attnReady = true;
}

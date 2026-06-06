// core.js -- shared math, canvas state, and the generic line-chart helper.
// Loaded first; every other module depends on CV, initCanvas, drawChart.

// ---- Shared math -------------------------------------------------
const sigmoid = x => 1 / (1 + Math.exp(-x));
const lse2 = (a, b, eps) => {
  const m = Math.max(a, b);
  return m + eps * Math.log(Math.exp((a - m) / eps) + Math.exp((b - m) / eps));
};

// ---- Canvas state ------------------------------------------------
const CV = {};
function initCanvas(id, logicalH) {
  const el = document.getElementById(id);
  if (!el) return;
  const dpr = window.devicePixelRatio || 1;
  const logicalW = Math.max((el.parentElement ? el.parentElement.clientWidth - 24 : 0), 280);
  el.width  = Math.round(logicalW * dpr);
  el.height = Math.round(logicalH * dpr);
  el.style.height = logicalH + 'px';
  const ctx = el.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  CV[id] = { ctx, W: logicalW, H: logicalH };
}

// ---- Generic line chart ------------------------------------------
function drawChart(ctx, W, H, xRange, yRange, series, opts) {
  const pad = opts.pad || { t:18, r:20, b:28, l:38 };
  const N = 400;
  const toX = v => pad.l + (v - xRange[0]) / (xRange[1] - xRange[0]) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v - yRange[0]) / (yRange[1] - yRange[0]) * (H - pad.t - pad.b);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#f0f0f0'; ctx.lineWidth = 1;
  if (opts.yTicks) opts.yTicks.forEach(y => {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(y)); ctx.lineTo(W - pad.r, toY(y)); ctx.stroke();
  });

  ctx.strokeStyle = '#d1d5db'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, toY(yRange[0])); ctx.lineTo(W - pad.r, toY(yRange[0])); ctx.stroke();
  const x0 = (xRange[0] <= 0 && xRange[1] >= 0) ? 0 : xRange[0];
  ctx.beginPath(); ctx.moveTo(toX(x0), pad.t); ctx.lineTo(toX(x0), H - pad.b); ctx.stroke();

  ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  if (opts.xTicks) opts.xTicks.forEach(x => ctx.fillText(x, toX(x), H - pad.b + 12));
  ctx.textAlign = 'right';
  if (opts.yTicks) opts.yTicks.forEach(y => ctx.fillText(y, pad.l - 4, toY(y) + 3));

  if (opts.xlabel) {
    ctx.textAlign = 'center'; ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif';
    ctx.fillText(opts.xlabel, (pad.l + W - pad.r) / 2, H - 4);
  }
  if (opts.ylabel) {
    ctx.save(); ctx.translate(10, (pad.t + H - pad.b) / 2);
    ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
    ctx.fillText(opts.ylabel, 0, 0); ctx.restore();
  }

  series.forEach(s => {
    const xs = Array.from({length: N}, (_, i) => xRange[0] + (xRange[1] - xRange[0]) * i / (N - 1));
    const ys = xs.map(s.fn);
    ctx.save();
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2.2;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    xs.forEach((x, i) => { const cx = toX(x), cy = toY(ys[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); });
    ctx.stroke();
    ctx.restore();
    if (s.label) {
      const lx = pad.l + 8, ly = pad.t + (series.indexOf(s)) * 16 + 12;
      ctx.save();
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
      ctx.fillStyle = '#374151'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(s.label, lx + 22, ly + 3);
      ctx.restore();
    }
  });
  if (opts.marker) {
    ctx.save();
    ctx.fillStyle = opts.marker.color || '#dc2626';
    const mx = toX(opts.marker.x), my = toY(opts.marker.y);
    ctx.beginPath(); ctx.arc(mx, my, 4, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }
}

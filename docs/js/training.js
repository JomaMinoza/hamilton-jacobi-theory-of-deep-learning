// training.js -- Interactive: live in-browser training of a deep LSE network (Adam).
// L = 1 is a single convex LSE layer (the exact Hamilton-Jacobi reading: output is
// an HJ solution, weights are its initial data). L >= 2 stacks softplus feature
// layers before the LSE readout; each layer is convex but the composition is not,
// which is how depth represents non-convex targets (the paper's multilayer result).

const TR_n = 64, TR_DOM = [-2, 2], TR_T = 1.0, TR_H = 12, TR_NL = 16;
// Architecture as an ordered list of hidden blocks ('dense' = Dense+Softplus,
// 'resnet' = residual). Input (scalar) and an LSE readout are implicit endpoints.
// trL (effective depth) is derived = arch.length + 1.
let arch = [];
let trL = 1, trEps = 0.2, trLr = 0.05, trTarget = 'quad';
// A ResNet block adds its input to its output, so it needs input width = H; the
// first block sees the scalar input (width 1), so it cannot be a ResNet. Returns
// the index of the first invalid block, or -1 if the stack is valid.
function archValid(a) {
  for (let l = 0; l < a.length; l++) { const din = (l === 0) ? 1 : TR_H; if ((a[l] === 'resnet' || a[l] === 'attn') && din !== TR_H) return l; }
  return -1;
}
let trPlaying = false, trRAF = null, trStep = 0, trLastLoss = 0, trDinOut = 1;
const trXs = Array.from({ length: TR_n }, (_, i) => TR_DOM[0] + (TR_DOM[1] - TR_DOM[0]) * i / (TR_n - 1));

// parameters (feature layers + LSE readout) and their Adam moments
let featW = [], featB = [], fmW = [], fvW = [], fmB = [], fvB = [];
let outW, outB, mOutW, vOutW, mOutB, vOutB;

function trTgt(x) {
  switch (trTarget) {
    case 'softabs': return Math.sqrt(x * x + 0.09);
    case 'abs':     return Math.abs(x);
    case 'sine':    return 0.8 * Math.sin(3 * x);
    case 'well':    return 0.22 * (x * x - 1) * (x * x - 1) - 0.3;
    default:        return 0.5 * x * x;            // 'quad'
  }
}
let trYs = trXs.map(trTgt);

function trSoftplus(z, eps) { const t = z / eps; return t > 0 ? z + eps * Math.log1p(Math.exp(-t)) : eps * Math.log1p(Math.exp(t)); }
function trSigm(z, eps) { const t = z / eps; return t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t)); }

function trInit() {
  const H = TR_H, N = TR_NL;
  trL = arch.length + 1;
  featW = []; featB = []; fmW = []; fvW = []; fmB = []; fvB = [];
  let din = 1;
  for (let l = 0; l < arch.length; l++) {
    const dout = H;
    const Wm = new Float64Array(din * dout), Bm = new Float64Array(dout);
    const scale = (din === 1) ? 2.0 : 1.0 / Math.sqrt(din);
    for (let i = 0; i < dout; i++) {
      for (let k = 0; k < din; k++) Wm[i * din + k] = (Math.random() - 0.5) * 2 * scale;
      Bm[i] = (Math.random() - 0.5) * 0.4;
    }
    featW.push(Wm); featB.push(Bm);
    fmW.push(new Float64Array(din * dout)); fvW.push(new Float64Array(din * dout));
    fmB.push(new Float64Array(dout)); fvB.push(new Float64Array(dout));
    din = dout;
  }
  trDinOut = din;
  outW = new Float64Array(N * trDinOut); outB = new Float64Array(N);
  for (let j = 0; j < N; j++) {
    for (let k = 0; k < trDinOut; k++) outW[j * trDinOut + k] = (trDinOut === 1) ? ((j / (N - 1) - 0.5) * 4) : ((Math.random() - 0.5) * 2 / Math.sqrt(trDinOut));
    outB[j] = (Math.random() - 0.5) * 0.2;
  }
  mOutW = new Float64Array(N * trDinOut); vOutW = new Float64Array(N * trDinOut);
  mOutB = new Float64Array(N); vOutB = new Float64Array(N);
  trStep = 0; trLastLoss = 0; trYs = trXs.map(trTgt);
}

function trForward(x) {
  const H = TR_H, N = TR_NL, dinOut = trDinOut;
  let a = [x]; const acts = [a], gs = [], attnAs = [];
  let din = 1;
  for (let l = 0; l < arch.length; l++) {
    const dout = H, Wm = featW[l], Bm = featB[l];
    const g = new Float64Array(dout), na = new Float64Array(dout);
    for (let i = 0; i < dout; i++) { let s = Bm[i]; for (let k = 0; k < din; k++) s += Wm[i * din + k] * a[k]; g[i] = s; }
    if (arch[l] === 'attn') {
      // single-head self-attention over the H units: queries=keys=values=g, a
      // learned linear projection of the hidden vector. na = softmax(g g^T/sqrt H) g
      // (the Gibbs average, attention = grad LSE_eps) plus a residual.
      const r = Math.sqrt(dout), A = new Float64Array(dout * dout);
      for (let i = 0; i < dout; i++) {
        let mxs = -1e30; for (let j = 0; j < dout; j++) { const s = g[i] * g[j] / r; if (s > mxs) mxs = s; }
        let se = 0; for (let j = 0; j < dout; j++) { const e = Math.exp(g[i] * g[j] / r - mxs); A[i * dout + j] = e; se += e; }
        let o = 0; for (let j = 0; j < dout; j++) { A[i * dout + j] /= se; o += A[i * dout + j] * g[j]; }
        na[i] = o + a[i];                                   // residual (din == dout == H)
      }
      attnAs.push(A);
    } else {
      for (let i = 0; i < dout; i++) na[i] = trSoftplus(g[i], trEps);
      if (arch[l] === 'resnet' && din === dout) { for (let i = 0; i < dout; i++) na[i] += a[i]; }   // residual (din == dout == H)
      attnAs.push(null);
    }
    gs.push(g); a = na; acts.push(a); din = dout;
  }
  const aLast = a, z = new Float64Array(N); let m = -1e30;
  for (let j = 0; j < N; j++) { let s = outB[j]; for (let k = 0; k < dinOut; k++) s += outW[j * dinOut + k] * aLast[k]; z[j] = s; if (s > m) m = s; }
  let se = 0; const ez = new Float64Array(N);
  for (let j = 0; j < N; j++) { ez[j] = Math.exp((z[j] - m) / trEps); se += ez[j]; }
  const f = m + trEps * Math.log(se), pi = new Float64Array(N);
  for (let j = 0; j < N; j++) pi[j] = ez[j] / se;
  return { f, acts, gs, aLast, pi, attnAs };
}
function trFx(x) { return trForward(x).f; }

function trAdam(p, g, m, v, t) {
  const b1 = 0.9, b2 = 0.999, ea = 1e-8;
  for (let i = 0; i < p.length; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    const mh = m[i] / (1 - Math.pow(b1, t)), vh = v[i] / (1 - Math.pow(b2, t));
    p[i] -= trLr * mh / (Math.sqrt(vh) + ea);
  }
}

function trDoSteps(K) {
  const L = trL, H = TR_H, N = TR_NL, dinOut = trDinOut;
  for (let k = 0; k < K; k++) {
    const gOutW = new Float64Array(N * dinOut), gOutB = new Float64Array(N);
    const gFW = featW.map(w => new Float64Array(w.length)), gFB = featB.map(b => new Float64Array(b.length));
    let loss = 0;
    for (let s = 0; s < TR_n; s++) {
      const x = trXs[s], fw = trForward(x);
      const r = fw.f - trYs[s]; loss += r * r; const dLdf = (2 / TR_n) * r;
      const aLast = fw.aLast, pi = fw.pi, daLast = new Float64Array(dinOut);
      for (let j = 0; j < N; j++) {
        const dz = pi[j] * dLdf;
        for (let kk = 0; kk < dinOut; kk++) { gOutW[j * dinOut + kk] += dz * aLast[kk]; daLast[kk] += dz * outW[j * dinOut + kk]; }
        gOutB[j] += dz;
      }
      let da = daLast;
      for (let l = arch.length - 1; l >= 0; l--) {
        const din = (l === 0) ? 1 : H, dout = H, Wm = featW[l], g = fw.gs[l], uprev = fw.acts[l];
        const du = new Float64Array(din), dg = new Float64Array(dout);
        if (arch[l] === 'attn') {
          // backward through na_i = sum_j A_ij g_j (+ residual). A_ij = softmax_j(g_i g_j / r).
          const A = fw.attnAs[l], r = Math.sqrt(dout);
          for (let i = 0; i < dout; i++) {
            const di = da[i];
            let rowdot = 0; for (let m = 0; m < dout; m++) rowdot += A[i * dout + m] * (di * g[m]);
            for (let j = 0; j < dout; j++) {
              const Aij = A[i * dout + j];
              dg[j] += di * Aij;                                  // explicit value path
              const ds = Aij * (di * g[j] - rowdot);             // softmax jacobian on s_ij
              dg[i] += ds * g[j] / r; dg[j] += ds * g[i] / r;     // s_ij = g_i g_j / r
            }
          }
        } else {
          for (let i = 0; i < dout; i++) dg[i] = da[i] * trSigm(g[i], trEps);
        }
        for (let i = 0; i < dout; i++) {
          for (let kk = 0; kk < din; kk++) { gFW[l][i * din + kk] += dg[i] * uprev[kk]; du[kk] += dg[i] * Wm[i * din + kk]; }
          gFB[l][i] += dg[i];
        }
        if (arch[l] === 'resnet' || arch[l] === 'attn') { for (let kk = 0; kk < din; kk++) du[kk] += da[kk]; }   // identity path
        da = du;
      }
    }
    trLastLoss = loss / TR_n; trStep++;
    trAdam(outW, gOutW, mOutW, vOutW, trStep); trAdam(outB, gOutB, mOutB, vOutB, trStep);
    for (let l = 0; l < arch.length; l++) { trAdam(featW[l], gFW[l], fmW[l], fvW[l], trStep); trAdam(featB[l], gFB[l], fmB[l], fvB[l], trStep); }
  }
}

function trDrawMsg(ctx, W, H, l1, l2) {
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#9ca3af'; ctx.textAlign = 'center';
  ctx.font = '13px Inter, sans-serif'; ctx.fillText(l1, W / 2, H / 2 - 8);
  ctx.font = '11px Inter, sans-serif'; ctx.fillText(l2, W / 2, H / 2 + 12);
}

// Left panel: network output vs target (all depths).
function drawTrainFit() {
  const c = CV['c_train_fit']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 14, r: 12, b: 28, l: 40 };
  const x0 = TR_DOM[0], x1 = TR_DOM[1], Np = 160;
  const gx = [], gf = [];
  for (let i = 0; i < Np; i++) { const x = x0 + (x1 - x0) * i / (Np - 1); gx.push(x); gf.push(trFx(x)); }
  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < Np; i++) { const t = trTgt(gx[i]); ymin = Math.min(ymin, gf[i], t); ymax = Math.max(ymax, gf[i], t); }
  if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
  const padY = 0.12 * ((ymax - ymin) || 1); ymin -= padY; ymax += padY;
  const toX = v => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v - ymin) / (ymax - ymin) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  [-2, -1, 0, 1, 2].forEach(xv => ctx.fillText(xv, toX(xv), H - pad.b + 12));
  ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath();
  for (let i = 0; i < Np; i++) { const cx = toX(gx[i]), cy = toY(trTgt(gx[i])); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 0; i < Np; i++) { const cx = toX(gx[i]), cy = toY(gf[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke();
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('input x', pad.l + (W - pad.l - pad.r) / 2, H - 2);
}

function trLegendre(vals, xs, y) { let best = -1e30; for (let i = 0; i < xs.length; i++) { const v = xs[i] * y - vals[i]; if (v > best) best = v; } return best; }

// Right panel: HJ initial data g(y) recovered as the Legendre transform (L = 1 only).
function drawTrainInit() {
  const c = CV['c_train_init']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 14, r: 12, b: 28, l: 44 };
  const x0 = TR_DOM[0], x1 = TR_DOM[1], Nx = 121;
  const xs = [], ff = [], tt = [];
  for (let i = 0; i < Nx; i++) { const x = x0 + (x1 - x0) * i / (Nx - 1); xs.push(x); ff.push(trFx(x)); tt.push(trTgt(x)); }
  const ymin = -2.2, ymax = 2.2, Ny = 121;
  const gn = [], gt = [], yy = [];
  for (let i = 0; i < Ny; i++) { const y = ymin + (ymax - ymin) * i / (Ny - 1); yy.push(y); gn.push(trLegendre(ff, xs, y)); gt.push(trLegendre(tt, xs, y)); }
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < Ny; i++) { vmin = Math.min(vmin, gn[i], gt[i]); vmax = Math.max(vmax, gn[i], gt[i]); }
  const vp = 0.12 * ((vmax - vmin) || 1); vmin -= vp; vmax += vp;
  const toX = v => pad.l + (v - ymin) / (ymax - ymin) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v - vmin) / (vmax - vmin) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  [-2, -1, 0, 1, 2].forEach(yv => ctx.fillText(yv, toX(yv), H - pad.b + 12));
  ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath();
  for (let i = 0; i < Ny; i++) { const cx = toX(yy[i]), cy = toY(gt[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 0; i < Ny; i++) { const cx = toX(yy[i]), cy = toY(gn[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke();
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('y  (gradient variable)', pad.l + (W - pad.l - pad.r) / 2, H - 2);
  ctx.save(); ctx.translate(11, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('initial data g(y)', 0, 0); ctx.restore();
}

// Third panel: the trained equation evolving in time (L = 1 only).
let trHjS = 1.0;
function trHjU0(x, s) { let m = 1e30; for (let j = 0; j < TR_NL; j++) { const yj = 2 * TR_T * outW[j], gj = -outB[j] - TR_T * outW[j] * outW[j]; const v = gj + (x - yj) * (x - yj) / (4 * s); if (v < m) m = v; } return m; }
function trHjUe(x, s) { let m = 1e30; const c = []; for (let j = 0; j < TR_NL; j++) { const yj = 2 * TR_T * outW[j], gj = -outB[j] - TR_T * outW[j] * outW[j]; const v = gj + (x - yj) * (x - yj) / (4 * s); c.push(v); if (v < m) m = v; } let s2 = 0; for (const v of c) s2 += Math.exp(-(v - m) / trEps); return m - trEps * Math.log(s2); }
function drawTrainHJ() {
  const c = CV['c_train_hj']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 16, r: 14, b: 30, l: 42 };
  const x0 = -4, x1 = 4, Np = 161;
  const xs = new Float64Array(Np);
  for (let i = 0; i < Np; i++) xs[i] = x0 + (x1 - x0) * i / (Np - 1);
  let u0arr = null; const ue = new Float64Array(Np); const sy = [], sg = [];
  if (trL === 1) {
    // exact discrete solution, both directions, with support-point atoms
    u0arr = new Float64Array(Np);
    for (let i = 0; i < Np; i++) { u0arr[i] = trHjU0(xs[i], trHjS); ue[i] = trHjUe(xs[i], trHjS); }
    for (let j = 0; j < TR_NL; j++) { const yj = 2 * TR_T * outW[j]; if (yj >= x0 && yj <= x1) { sy.push(yj); sg.push(-outB[j] - TR_T * outW[j] * outW[j]); } }
  } else {
    // deep: forward Cole-Hopf flow (well-posed for s >= t) of the output's HJ
    // snapshot u(x,t) = |x|^2/(4t) - f(x). Reduces to the exact solution at s = t.
    const snap = new Float64Array(Np);
    for (let i = 0; i < Np; i++) snap[i] = xs[i] * xs[i] / (4 * TR_T) - trFx(xs[i]);
    const s = Math.max(trHjS, TR_T), tau = s - TR_T, dx = (x1 - x0) / (Np - 1);
    if (tau <= 1e-6) { for (let i = 0; i < Np; i++) ue[i] = snap[i]; }
    else {
      const w0 = new Float64Array(Np); for (let q = 0; q < Np; q++) w0[q] = Math.exp(-snap[q] / trEps);
      const cf = 1 / Math.sqrt(4 * Math.PI * trEps * tau), denom = 4 * trEps * tau, konst = 0.5 * trEps * Math.log(s / TR_T);
      for (let i = 0; i < Np; i++) {
        let acc = 0; const xi = xs[i];
        for (let q = 0; q < Np; q++) { const dz = xi - xs[q]; acc += cf * Math.exp(-dz * dz / denom) * w0[q] * dx; }
        ue[i] = -trEps * Math.log(Math.max(acc, 1e-300)) - konst;
      }
    }
  }
  let ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < Np; i++) { ymin = Math.min(ymin, ue[i]); ymax = Math.max(ymax, ue[i]); if (u0arr) { ymin = Math.min(ymin, u0arr[i]); ymax = Math.max(ymax, u0arr[i]); } }
  for (let i = 0; i < sg.length; i++) { ymin = Math.min(ymin, sg[i]); ymax = Math.max(ymax, sg[i]); }
  if (!isFinite(ymin)) { ymin = -1; ymax = 1; }
  const py = 0.12 * ((ymax - ymin) || 1); ymin -= py; ymax += py;
  const toX = v => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v - ymin) / (ymax - ymin) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  [-4, -2, 0, 2, 4].forEach(xv => ctx.fillText(xv, toX(xv), H - pad.b + 12));
  if (u0arr) {
    ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath();
    for (let i = 0; i < Np; i++) { const cx = toX(xs[i]), cy = toY(u0arr[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.4; ctx.beginPath();
  for (let i = 0; i < Np; i++) { const cx = toX(xs[i]), cy = toY(ue[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke();
  ctx.fillStyle = '#0d9488';
  for (let i = 0; i < sy.length; i++) { ctx.beginPath(); ctx.arc(toX(sy[i]), toY(sg[i]), 3, 0, 2 * Math.PI); ctx.fill(); }
  if (trL > 1) { ctx.fillStyle = '#9ca3af'; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'right'; ctx.fillText('forward HJ flow (s >= t)', W - pad.r, pad.t + 2); }
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('x', pad.l + (W - pad.l - pad.r) / 2, H - 2);
  ctx.save(); ctx.translate(11, pad.t + (H - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('u(x, s)', 0, 0); ctx.restore();
}

// 3D surface: the trained network output over (input x, viscosity eps). Works at
// any depth (just evaluates the net at a range of eps). Isometric, shaded, painted
// back-to-front. Reuses viridis() from attention.js for the height colormap.
function drawTrainSurface() {
  const c = CV['c_train_surf']; if (!c) return;
  const { ctx, W, H } = c;
  const Nx = 26, NE = 16, xmin = -2, xmax = 2, le0 = Math.log(0.03), le1 = Math.log(2.0);
  const saved = trEps;
  const Z = []; let zmin = Infinity, zmax = -Infinity;
  for (let j = 0; j < NE; j++) {
    trEps = Math.exp(le0 + (le1 - le0) * j / (NE - 1));
    const row = new Float64Array(Nx);
    for (let i = 0; i < Nx; i++) { const x = xmin + (xmax - xmin) * i / (Nx - 1); const z = trFx(x); row[i] = z; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
    Z.push(row);
  }
  trEps = saved;
  const pad = { l: 8, r: 10, t: 12, b: 18 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const spanX = plotW * 0.66, dpx = plotW * 0.30, dpy = -plotH * 0.42, spanZ = plotH * 0.46;
  const ox = pad.l + 4, oy = H - pad.b - 2;
  const nz = v => (v - zmin) / ((zmax - zmin) || 1);
  const proj = (i, j, z) => { const ax = i / (Nx - 1), ay = j / (NE - 1); return [ox + ax * spanX + ay * dpx, oy + ay * dpy - nz(z) * spanZ]; };
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  for (let j = NE - 1; j >= 1; j--) {
    for (let i = 0; i < Nx - 1; i++) {
      const a = proj(i, j, Z[j][i]), b = proj(i + 1, j, Z[j][i + 1]);
      const cc = proj(i + 1, j - 1, Z[j - 1][i + 1]), d = proj(i, j - 1, Z[j - 1][i]);
      const t = nz((Z[j][i] + Z[j][i + 1] + Z[j - 1][i] + Z[j - 1][i + 1]) / 4);
      ctx.fillStyle = viridis(t);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(cc[0], cc[1]); ctx.lineTo(d[0], d[1]); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 0.5; ctx.stroke();
    }
  }
  ctx.fillStyle = '#6b7280'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('x', ox + spanX * 0.5, H - 5);
  ctx.fillText('eps: small -> large', ox + spanX + dpx * 0.5, oy + dpy * 0.55 + 4);
}

// --- Consequences of the trained network (L = 1): hallucination + bounds ---
function trZ(x) { const z = new Float64Array(TR_NL); for (let j = 0; j < TR_NL; j++) z[j] = outW[j] * x + outB[j]; return z; } // L=1 (din=1)
// Last-layer logits for ANY depth: z_j = W_j . h(x) + b_j, with h the features.
function trLastLogits(x) {
  const fw = trForward(x), din = trDinOut, z = new Float64Array(TR_NL);
  for (let j = 0; j < TR_NL; j++) { let s = outB[j]; for (let k = 0; k < din; k++) s += outW[j * din + k] * fw.aLast[k]; z[j] = s; }
  return z;
}
function trDom(x) {
  const z = trLastLogits(x); let i1 = 0;
  for (let j = 1; j < TR_NL; j++) if (z[j] > z[i1]) i1 = j;
  let z2 = -1e30; for (let j = 0; j < TR_NL; j++) if (j !== i1 && z[j] > z2) z2 = z[j];
  return { zmax: z[i1], gap: z[i1] - z2 };
}
function trHessNorm1(x, eps) {
  const z = trZ(x); const m = Math.max(...z); const e = z.map(v => Math.exp((v - m) / eps));
  const Z = e.reduce((a, b) => a + b, 0); let EW = 0, EW2 = 0;
  for (let j = 0; j < TR_NL; j++) { const p = e[j] / Z; EW += p * outW[j]; EW2 += p * outW[j] * outW[j]; }
  return (EW2 - EW * EW) / eps;
}

function drawHallucMap() {
  const c = CV['c_halluc_map']; if (!c) return;
  const { ctx, W, H } = c;
  // General (any-depth) criterion from the paper's Remark: the last-layer energy
  // gap Delta(x)/eps relative to log N. Delta/eps >> log N: one neuron dominates
  // (hallucination-prone); << log N: many neurons (stochastic parrot).
  const pad = { t: 14, r: 12, b: 28, l: 40 };
  const x0 = -4, x1 = 4, le0 = Math.log(0.03), le1 = Math.log(2.0), Nx = 76, NE = 46;
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b, cw = plotW / Nx, ch = plotH / NE;
  const logN = Math.log(TR_NL), saved = trEps;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  for (let j = 0; j < NE; j++) {
    const eps = Math.exp(le0 + (le1 - le0) * j / (NE - 1)); trEps = eps;
    const yy = pad.t + plotH - (j + 1) * ch;
    for (let i = 0; i < Nx; i++) {
      const x = x0 + (x1 - x0) * i / (Nx - 1);
      const rho = trDom(x).gap / (eps * logN);
      const t = rho / (1 + rho);
      let col;
      if (t < 0.5) { const u = t / 0.5; col = `rgb(${Math.round(147 + 72 * u)},${Math.round(197 + 29 * u)},253)`; }
      else { const u = (t - 0.5) / 0.5; col = `rgb(252,${Math.round(165 - 127 * u)},${Math.round(165 - 127 * u)})`; }
      ctx.fillStyle = col; ctx.fillRect(pad.l + i * cw, yy, cw + 0.6, ch + 0.6);
    }
  }
  trEps = saved;
  const yline = pad.t + plotH - ((Math.log(trEps) - le0) / (le1 - le0)) * plotH;
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, yline); ctx.lineTo(W - pad.r, yline); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  [-4, -2, 0, 2, 4].forEach(xv => ctx.fillText(xv, pad.l + (xv - x0) / (x1 - x0) * plotW, H - pad.b + 12));
  ctx.textAlign = 'right';
  [0.05, 0.2, 1.0].forEach(ev => { const yy2 = pad.t + plotH - ((Math.log(ev) - le0) / (le1 - le0)) * plotH; ctx.fillText(ev, pad.l - 4, yy2 + 3); });
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('input x', pad.l + plotW / 2, H - 2);
  ctx.save(); ctx.translate(11, pad.t + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('viscosity eps', 0, 0); ctx.restore();
}

function drawHallucBound() {
  const c = CV['c_halluc_bound']; if (!c) return;
  const { ctx, W, H } = c;
  const pad = { t: 14, r: 12, b: 28, l: 44 }, x0 = -4, x1 = 4, Np = 200;
  const xs = [], dev = [], bnd = []; let ymax = 1e-9;
  for (let i = 0; i < Np; i++) {
    const x = x0 + (x1 - x0) * i / (Np - 1); const d = trDom(x);
    const dv = Math.abs(trFx(x) - d.zmax), b = trEps * Math.log(1 + (TR_NL - 1) * Math.exp(-d.gap / trEps));
    xs.push(x); dev.push(dv); bnd.push(b); ymax = Math.max(ymax, dv, b);
  }
  ymax *= 1.15;
  const toX = v => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const toY = v => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  for (let k = 0; k <= 3; k++) { const yv = ymax * k / 3, y = toY(yv); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(yv.toFixed(2), pad.l - 4, y + 3); }
  ctx.textAlign = 'center'; [-4, -2, 0, 2, 4].forEach(xv => ctx.fillText(xv, toX(xv), H - pad.b + 12));
  ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]); ctx.beginPath();
  for (let i = 0; i < Np; i++) { const cx = toX(xs[i]), cy = toY(bnd[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.2; ctx.beginPath();
  for (let i = 0; i < Np; i++) { const cx = toX(xs[i]), cy = toY(dev[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke();
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('input x', pad.l + (W - pad.l - pad.r) / 2, H - 2);
}

function drawRobustBound() {
  const c = CV['c_robust']; if (!c) return;
  const { ctx, W, H } = c;
  // Measured input curvature |f''(x)| (numeric), any depth. The closed-form bound
  // ||W||^2/eps is exact only for L = 1; for L > 1 only the measured value is shown.
  const pad = { t: 14, r: 14, b: 30, l: 46 }, le0 = Math.log(0.03), le1 = Math.log(3.0), NEv = 44;
  const xg = []; for (let i = 0; i <= 80; i++) xg.push(-3.6 + 7.2 * i / 80);
  const hh = 0.03, saved = trEps;
  const eps = [], meas = [];
  for (let j = 0; j < NEv; j++) {
    const e = Math.exp(le0 + (le1 - le0) * j / (NEv - 1)); trEps = e;
    let mx = 0; for (const x of xg) { const fxx = (trFx(x + hh) - 2 * trFx(x) + trFx(x - hh)) / (hh * hh); mx = Math.max(mx, Math.abs(fxx)); }
    eps.push(e); meas.push(mx);
  }
  trEps = saved;
  let bnd = null;
  if (trL === 1) { let maxW2 = 0; for (let j = 0; j < TR_NL; j++) maxW2 = Math.max(maxW2, outW[j] * outW[j]); bnd = eps.map(e => maxW2 / e); }
  let lymin = 1e9, lymax = -1e9;
  for (let j = 0; j < NEv; j++) { const lm = Math.log10(Math.max(meas[j], 1e-6)); lymin = Math.min(lymin, lm); lymax = Math.max(lymax, lm); if (bnd) { lymin = Math.min(lymin, Math.log10(bnd[j])); lymax = Math.max(lymax, Math.log10(bnd[j])); } }
  lymin = Math.floor(lymin); lymax = Math.ceil(lymax); if (lymax <= lymin) lymax = lymin + 1;
  const toX = le => pad.l + (le - le0) / (le1 - le0) * (W - pad.l - pad.r);
  const toY = ly => H - pad.b - (ly - lymin) / (lymax - lymin) * (H - pad.t - pad.b);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#9ca3af'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'right';
  for (let k = lymin; k <= lymax; k++) { const y = toY(k); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText('1e' + k, pad.l - 4, y + 3); }
  ctx.textAlign = 'center'; [0.05, 0.2, 1.0, 3.0].forEach(ev => ctx.fillText(ev, toX(Math.log(ev)), H - pad.b + 12));
  if (bnd) {
    ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]); ctx.beginPath();
    for (let j = 0; j < NEv; j++) { const cx = toX(Math.log(eps[j])), cy = toY(Math.log10(bnd[j])); if (j === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.2; ctx.beginPath();
  for (let j = 0; j < NEv; j++) { const cx = toX(Math.log(eps[j])), cy = toY(Math.log10(Math.max(meas[j], 1e-6))); if (j === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); }
  ctx.stroke();
  ctx.strokeStyle = '#111827'; ctx.lineWidth = 1.1; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(toX(Math.log(trEps)), pad.t); ctx.lineTo(toX(Math.log(trEps)), H - pad.b); ctx.stroke(); ctx.setLineDash([]);
  if (trL > 1) { ctx.fillStyle = '#9ca3af'; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'right'; ctx.fillText('measured only (||W||^2/eps bound is L=1)', W - pad.r, pad.t + 2); }
  ctx.fillStyle = '#6b7280'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('viscosity eps (log)', pad.l + (W - pad.l - pad.r) / 2, H - 2);
}

function trRedraw() {
  drawTrainFit(); drawTrainInit(); drawTrainHJ(); drawTrainSurface();
  drawHallucMap(); drawHallucBound(); drawRobustBound();
  const st = document.getElementById('tr_status');
  if (st) st.textContent = 'L=' + trL + '   step ' + trStep + (trStep ? '   loss ' + trLastLoss.toExponential(2) : '');
}

function trLoop() { if (!trPlaying) return; trDoSteps(trL > 1 ? 8 : 4); trRedraw(); trRAF = requestAnimationFrame(trLoop); }

// Run-state helpers at module scope so the architecture builder can drive them.
let trPlayBtn = null;
function trPauseLoop() { trPlaying = false; if (trRAF) { cancelAnimationFrame(trRAF); trRAF = null; } if (trPlayBtn) { trPlayBtn.textContent = 'Play'; trPlayBtn.classList.remove('active'); } }
function trStartLoop() { if (archValid(arch) !== -1) return; if (trRAF) { cancelAnimationFrame(trRAF); trRAF = null; } trPlaying = true; if (trPlayBtn) { trPlayBtn.textContent = 'Pause'; trPlayBtn.classList.add('active'); } trLoop(); }
function trRebuild() { trPauseLoop(); if (archValid(arch) !== -1) return; trInit(); trRedraw(); }   // builder calls this on stack change; skips invalid stacks

function setupTrain() {
  const pl = document.getElementById('tr_play'); if (!pl) return;
  trPlayBtn = pl; trInit();
  pl.addEventListener('click', () => { if (trPlaying) trPauseLoop(); else trStartLoop(); });
  document.getElementById('tr_step').addEventListener('click', () => { trPauseLoop(); if (archValid(arch) === -1) { trDoSteps(20); trRedraw(); } });
  document.getElementById('tr_reset').addEventListener('click', () => { trPauseLoop(); trInit(); trRedraw(); });
  document.getElementById('sl_tr_eps').addEventListener('input', function() {
    trEps = parseFloat(this.value); document.getElementById('v_tr_eps').textContent = trEps.toFixed(2); if (!trPlaying) trRedraw();
  });
  document.getElementById('sl_tr_lr').addEventListener('input', function() {
    trLr = parseFloat(this.value); document.getElementById('v_tr_lr').textContent = trLr.toFixed(3);
  });
  document.getElementById('tr_target').addEventListener('change', function() { trPauseLoop(); trTarget = this.value; trInit(); trRedraw(); });
  const hjs = document.getElementById('sl_hj_s');
  if (hjs) hjs.addEventListener('input', function() { trHjS = parseFloat(this.value); document.getElementById('v_hj_s').textContent = trHjS.toFixed(2); if (!trPlaying) drawTrainHJ(); });
  trRedraw();
}

// main.js -- orchestration: init all canvases, draw everything, wire load/resize.
// Loaded last; depends on every other module.

function initAll() {
  initCanvas('c_lse', 220);
  initCanvas('c_grad', 220);
  initCanvas('c_weights', 230);
  initCanvas('c_entropy', 230);
  initCanvas('c_bif_land', 230);
  initCanvas('c_bif_diag', 230);
  initCanvas('c_halluc', 230);
  initCanvas('c_shock', 230);
  initCanvas('c_attn_heat', 240);
  initCanvas('c_attn_mat', 240);
  initCanvas('c_attn_depth', 250);
  initCanvas('c_train_fit', 240);
  initCanvas('c_train_init', 240);
  initCanvas('c_train_hj', 240);
  initCanvas('c_train_surf', 240);
  initCanvas('c_halluc_map', 230);
  initCanvas('c_halluc_bound', 230);
  initCanvas('c_robust', 230);
}
function drawAll() {
  redrawDeform(); drawWeights(); drawEntropy();
  drawBifLandscape(); drawBifDiagram();
  drawHalluc(); drawShock();
  if (window.ATTN_DATA) { drawAttnHeat(); drawAttnMat(); drawAttnDepth(); }
  drawTrainFit(); drawTrainInit(); drawTrainHJ(); drawTrainSurface();
  drawHallucMap(); drawHallucBound(); drawRobustBound();
  if (typeof archRenderSVG === 'function') archRenderSVG();
}
window.addEventListener('load', () => { initAll(); setupAttn(); setupTrain(); setupBuilder(); drawAll(); });
window.addEventListener('resize', () => { initAll(); drawAll(); });

// ---- BibTeX copy ------------------------------------------------
function copyBibtex() {
  const text = document.getElementById('bibtex').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
  });
}

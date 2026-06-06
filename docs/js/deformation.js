// deformation.js -- Interactive: the deformation parameter epsilon.

let eps = 1.0;
function redrawDeform() {
  const padS = {t:18, r:16, b:30, l:40};
  const a = CV['c_lse'];
  if (a) drawChart(a.ctx, a.W, a.H, [-4, 4], [-0.2, 4.2], [
    { fn: v => Math.max(v, 0),     color: '#dc2626', dash: [5,4], width: 1.5, label: 'max(a,0)' },
    { fn: v => lse2(v, 0, eps),    color: '#2563eb', width: 2.5,  label: 'LSE_eps(a,0)' },
  ], { xTicks: [-3,-2,-1,0,1,2,3], yTicks: [0,1,2,3,4], xlabel: 'net input a', ylabel: 'output', pad: padS });

  const b = CV['c_grad'];
  if (b) drawChart(b.ctx, b.W, b.H, [-4, 4], [-0.05, 1.1], [
    { fn: v => v < 0 ? 0 : 1,       color: '#d1d5db', dash: [5,4], width: 1.5, label: 'step (eps->0)' },
    { fn: v => sigmoid(v / eps),     color: '#059669', width: 2.5, label: 'sigma(a/eps)' },
  ], { xTicks: [-3,-2,-1,0,1,2,3], yTicks: [0,0.25,0.5,0.75,1], xlabel: 'a', ylabel: 'weight', pad: padS });
}
(function(){
  const sl = document.getElementById('sl_eps');
  if (sl) sl.addEventListener('input', function() {
    eps = parseFloat(this.value);
    document.getElementById('v_eps').textContent = eps.toFixed(2);
    redrawDeform();
  });
})();

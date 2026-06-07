// nablaC: the LANDSCAPE view.
//
// Gradient descent is a marble rolling downhill. Here is the hill. We can't draw
// the full loss surface (it lives in hundreds of dimensions), so we take a
// 2-D slice: pick the two parameters the cost cares about most right now,
// and sweep C across them while every other weight is held fixed. The contours
// are that slice of C; the bright trail is the path descent has actually taken;
// the arrow is −∇C, the direction the marble is about to roll.

const ACCENT = '#46d6ff';
const INK = '#e9e3d4';

export function createLandscapeView(canvas, eqEl) {
  let trainer = null;
  let pi = 0, pj = 1;          // indices of the two visualised params
  let center = [0, 0];         // window center in param space
  let range = 2.5;             // half-width of the window
  const RES = 48;
  let surface = null;          // Float32Array RES*RES of C
  let lo = 0, hi = 1;
  const path = [];             // history of [θi, θj]
  let lastCompute = 0;

  function params() { return trainer.net.parameters(); }

  function pickParams() {
    // the two parameters with the largest |gradient| (distinct)
    const ps = params();
    let order = ps.map((p, i) => [i, Math.abs(p.grad)]).sort((a, b) => b[1] - a[1]);
    pi = order[0]?.[0] ?? 0;
    pj = order[1]?.[0] ?? 1;
  }

  function computeSurface() {
    const ps = params();
    center = [ps[pi].data, ps[pj].data];
    const oi = ps[pi].data, oj = ps[pj].data;
    surface = new Float32Array(RES * RES);
    lo = Infinity; hi = -Infinity;
    for (let b = 0; b < RES; b++) {
      const vj = center[1] + range * (1 - (2 * b) / (RES - 1));
      for (let a = 0; a < RES; a++) {
        const vi = center[0] - range + (2 * range * a) / (RES - 1);
        ps[pi].data = vi; ps[pj].data = vj;
        const c = trainer.lossNumeric();
        surface[b * RES + a] = c;
        if (c < lo) lo = c;
        if (c > hi) hi = c;
      }
    }
    ps[pi].data = oi; ps[pj].data = oj; // restore
    lastCompute = performance.now();
  }

  function build(t) {
    trainer = t;
    pickParams();
    path.length = 0;
    recordPoint();
    computeSurface();
    draw();
  }

  function recordPoint() {
    const ps = params();
    path.push([ps[pi].data, ps[pj].data]);
    if (path.length > 600) path.shift();
  }

  function toPx(vi, vj, w, h) {
    const x = ((vi - (center[0] - range)) / (2 * range)) * w;
    const y = ((center[1] + range - vj) / (2 * range)) * h;
    return [x, y];
  }

  function fit() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }

  function shade(t) {
    // t=0 high cost (dark), t=1 low cost (toward bone, basin tinted cool)
    const r = 10 + t * 150;
    const g = 12 + t * 150;
    const b = 16 + t * 140;
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  }

  function draw() {
    const { ctx, w, h } = fit();
    ctx.clearRect(0, 0, w, h);
    if (!surface) return;

    const cw = w / RES, ch = h / RES;
    const span = Math.max(1e-6, hi - lo);
    for (let b = 0; b < RES; b++) {
      for (let a = 0; a < RES; a++) {
        const c = surface[b * RES + a];
        const t = 1 - (c - lo) / span; // low cost → bright
        ctx.fillStyle = shade(t * t);
        ctx.fillRect(Math.floor(a * cw), Math.floor(b * ch), Math.ceil(cw) + 1, Math.ceil(ch) + 1);
      }
    }

    // contour lines (iso-cost) via simple level thresholding
    ctx.strokeStyle = 'rgba(233,227,212,0.10)';
    ctx.lineWidth = 1;
    const levels = 8;
    for (let L = 1; L < levels; L++) {
      const lvl = lo + (span * L) / levels;
      ctx.beginPath();
      for (let b = 0; b < RES - 1; b++) {
        for (let a = 0; a < RES - 1; a++) {
          const v = surface[b * RES + a];
          const vr = surface[b * RES + a + 1];
          const vd = surface[(b + 1) * RES + a];
          if ((v < lvl) !== (vr < lvl)) { ctx.moveTo((a + 0.5) * cw, (b + 0.5) * ch); ctx.lineTo((a + 1.5) * cw, (b + 0.5) * ch); }
          if ((v < lvl) !== (vd < lvl)) { ctx.moveTo((a + 0.5) * cw, (b + 0.5) * ch); ctx.lineTo((a + 0.5) * cw, (b + 1.5) * ch); }
        }
      }
      ctx.stroke();
    }

    // descent path
    if (path.length > 1) {
      ctx.beginPath();
      path.forEach(([vi, vj], k) => {
        const [x, y] = toPx(vi, vj, w, h);
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.strokeStyle = 'rgba(70,214,255,0.55)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // current marble + −∇C arrow
    const ps = params();
    const [cx, cy] = toPx(ps[pi].data, ps[pj].data, w, h);
    const gi = ps[pi].grad, gj = ps[pj].grad;
    const gl = Math.hypot(gi, gj) || 1;
    const arrow = 46;
    const ax = cx - (gi / gl) * arrow;
    const ay = cy + (gj / gl) * arrow; // canvas y is flipped vs param axis
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ax, ay);
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = ACCENT; ctx.fill();
    ctx.strokeStyle = '#05070a'; ctx.lineWidth = 1.4; ctx.stroke();

    renderEq();
  }

  function renderEq() {
    if (!eqEl || !trainer) return;
    const ps = params();
    eqEl.innerHTML =
      `2-D slice of the loss surface · axes = the two weights with largest |∇C| · ` +
      `θ<sub>i</sub> = <b>${ps[pi].data.toFixed(3)}</b> (∂C/∂θ<sub>i</sub> = ${ps[pi].grad.toFixed(3)}) · ` +
      `θ<sub>j</sub> = <b>${ps[pj].data.toFixed(3)}</b> (∂C/∂θ<sub>j</sub> = ${ps[pj].grad.toFixed(3)}) · ` +
      `<span style="color:${ACCENT}">arrow = −∇C</span>`;
  }

  function frame(running) {
    if (!trainer || !surface) return;
    if (running) {
      recordPoint();
      const ps = params();
      // recenter + recompute when the marble drifts near the window edge, or periodically
      const drifted = Math.abs(ps[pi].data - center[0]) > range * 0.8 || Math.abs(ps[pj].data - center[1]) > range * 0.8;
      if (drifted || performance.now() - lastCompute > 600) computeSurface();
    }
    draw();
  }

  return { build, frame };
}

// nablaC: the side panels, the input space (decision boundary) and the cost
// curve. Canvas, because both want per-pixel fills.
//
// INPUT SPACE: the 2-D plane the network sees. Background tinted by the net's
// current belief P(class B | x,y); the 0.5 contour is the decision boundary,
// and you watch it bend itself around the data as ∇C does its work.
//
// COST: C over training steps, the descent itself. The whole project is the
// story of this line going down.

const COOL = [44, 64, 92];  // class A region (dim)
const WARM = [96, 74, 46];  // class B region (dim)
const DOT_A = '#8fb4e6';
const DOT_B = '#e6b97f';
const INK = '#f1ead6';
const GRAD = '#46d6ff';

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

export function createBoundaryView(canvas, lim = 1.4) {
  let data = [];
  const off = document.createElement('canvas');

  const toPx = (x, y, w, h) => [((x + lim) / (2 * lim)) * w, ((lim - y) / (2 * lim)) * h];

  function setData(d) { data = d; }

  function draw(grid, res) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    // paint the belief field into a small offscreen buffer, then scale it up
    if (grid && res) {
      off.width = res; off.height = res;
      const octx = off.getContext('2d');
      const img = octx.createImageData(res, res);
      for (let i = 0; i < res * res; i++) {
        const p = grid[i];
        const r = COOL[0] + (WARM[0] - COOL[0]) * p;
        const g = COOL[1] + (WARM[1] - COOL[1]) * p;
        const b = COOL[2] + (WARM[2] - COOL[2]) * p;
        // brighten near the 0.5 contour, the boundary itself
        const edge = Math.max(0, 1 - Math.abs(p - 0.5) * 12);
        img.data[i * 4] = r + edge * 120;
        img.data[i * 4 + 1] = g + edge * 116;
        img.data[i * 4 + 2] = b + edge * 100;
        img.data[i * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#0c0e12';
      ctx.fillRect(0, 0, w, h);
    }

    // data points
    for (const pt of data) {
      const [px, py] = toPx(pt.x, pt.y, w, h);
      ctx.beginPath();
      ctx.arc(px, py, 3.4, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? DOT_A : DOT_B;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(8,10,14,0.85)';
      ctx.stroke();
    }
  }

  return { setData, draw };
}

export function createLossView(canvas) {
  function draw(history, gradHistory) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!history || history.length < 2) return;

    const pad = 4;
    const n = history.length;
    let max = 0;
    for (const v of history) if (v > max) max = v;
    max = Math.max(max, 0.05);

    const xAt = (i) => pad + (i / (n - 1)) * (w - pad * 2);
    const yAt = (v) => h - pad - (v / max) * (h - pad * 2);

    // area under C
    ctx.beginPath();
    ctx.moveTo(xAt(0), h - pad);
    for (let i = 0; i < n; i++) ctx.lineTo(xAt(i), yAt(history[i]));
    ctx.lineTo(xAt(n - 1), h - pad);
    ctx.closePath();
    ctx.fillStyle = 'rgba(241,234,214,0.06)';
    ctx.fill();

    // ‖∇C‖ (cyan), the descent's "urge", normalised to its own max
    if (gradHistory && gradHistory.length === n) {
      let gmax = 0;
      for (const v of gradHistory) if (v > gmax) gmax = v;
      gmax = Math.max(gmax, 1e-6);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const yy = h - pad - (gradHistory[i] / gmax) * (h - pad * 2);
        i ? ctx.lineTo(xAt(i), yy) : ctx.moveTo(xAt(i), yy);
      }
      ctx.strokeStyle = GRAD;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // C (bone)
    ctx.beginPath();
    for (let i = 0; i < n; i++) i ? ctx.lineTo(xAt(i), yAt(history[i])) : ctx.moveTo(xAt(i), yAt(history[i]));
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
  return { draw };
}

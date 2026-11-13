// What the network believes about the plane, drawn like a geological sheet:
// a flat wash of colour for the group it would answer at each spot (paler
// where it is unsure), a thin line where its answer changes, and the
// training dots on top with a ring round each one it currently gets wrong.

import { forward } from './net.js';
import { contour } from './contour.js';

export const EXTENT = 1.22; // the plane shown is [−EXTENT, EXTENT]²

export const GROUPS = [
  { deep: '#b06a1f', wash: [232, 196, 132] },
  { deep: '#2f6f55', wash: [160, 199, 168] },
  { deep: '#6b4687', wash: [196, 176, 214] },
];
const PAPER = [244, 239, 228];

export function makeTerritory(canvas) {
  const ctx = canvas.getContext('2d');
  const G = 80;
  const small = document.createElement('canvas');
  small.width = small.height = G;
  const sctx = small.getContext('2d');
  const img = sctx.createImageData(G, G);
  const field = [0, 1, 2].map(() => new Float32Array(G * G));
  let S = 0;
  let dpr = 1;

  function resize(css, ratio) {
    dpr = ratio;
    S = Math.round(css * ratio);
    canvas.width = canvas.height = S;
  }

  const toPx = (x) => ((x + EXTENT) / (2 * EXTENT)) * S;
  const toPy = (y) => ((EXTENT - y) / (2 * EXTENT)) * S;

  // Returns how many training dots the network gets wrong at θ.
  function draw(net, th, data) {
    if (!S) return 0;
    const px = img.data;
    for (let r = 0; r < G; r++) {
      const y = EXTENT - ((r + 0.5) / G) * 2 * EXTENT;
      for (let c = 0; c < G; c++) {
        const x = -EXTENT + ((c + 0.5) / G) * 2 * EXTENT;
        const a = forward(net, th, x, y);
        const k = r * G + c;
        let best = 0;
        for (let j = 1; j < 3; j++) if (a[j] > a[best]) best = j;
        let w0 = a[0] ** 4;
        let w1 = a[1] ** 4;
        let w2 = a[2] ** 4;
        const ws = w0 + w1 + w2 || 1;
        w0 /= ws; w1 /= ws; w2 /= ws;
        const others = (a[0] + a[1] + a[2] - a[best]) / 2;
        const sure = Math.min(1, Math.max(0, (a[best] - others) * 1.35)) ** 0.8;
        for (let q = 0; q < 3; q++) {
          const wash = w0 * GROUPS[0].wash[q] + w1 * GROUPS[1].wash[q] + w2 * GROUPS[2].wash[q];
          px[k * 4 + q] = PAPER[q] + (wash - PAPER[q]) * (0.18 + 0.82 * sure);
        }
        px[k * 4 + 3] = 255;
        for (let j = 0; j < 3; j++) {
          const o1 = a[(j + 1) % 3];
          const o2 = a[(j + 2) % 3];
          field[j][k] = a[j] - (o1 > o2 ? o1 : o2);
        }
      }
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(small, 0, 0, S, S);

    // Where the answer changes.
    const g2p = (g) => ((g + 0.5) / G) * S;
    ctx.strokeStyle = 'rgba(48, 38, 28, 0.55)';
    ctx.lineWidth = 0.9 * dpr;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let j = 0; j < 3; j++) {
      for (const { pts, closed } of contour(field[j], G, G, 0)) {
        ctx.moveTo(g2p(pts[0]), g2p(pts[1]));
        for (let i = 2; i < pts.length; i += 2) ctx.lineTo(g2p(pts[i]), g2p(pts[i + 1]));
        if (closed) ctx.closePath();
      }
    }
    ctx.stroke();

    // The dots.
    let wrong = 0;
    const rad = 3.3 * dpr;
    const rings = [];
    for (let i = 0; i < data.n; i++) {
      const a = forward(net, th, data.xs[i], data.ys[i]);
      let best = 0;
      for (let j = 1; j < 3; j++) if (a[j] > a[best]) best = j;
      const x = toPx(data.xs[i]);
      const y = toPy(data.ys[i]);
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, Math.PI * 2);
      ctx.fillStyle = GROUPS[data.labels[i]].deep;
      ctx.fill();
      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = 'rgba(250, 246, 238, 0.9)';
      ctx.stroke();
      if (best !== data.labels[i]) {
        wrong++;
        rings.push([x, y]);
      }
    }
    ctx.strokeStyle = '#211b16';
    ctx.lineWidth = 1.3 * dpr;
    ctx.beginPath();
    for (const [x, y] of rings) {
      ctx.moveTo(x + rad * 2.1, y);
      ctx.arc(x, y, rad * 2.1, 0, Math.PI * 2);
    }
    ctx.stroke();
    return wrong;
  }

  return { resize, draw };
}

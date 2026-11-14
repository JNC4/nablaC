// What the network believes about the plane, drawn like a geological sheet:
// a flat wash of colour for the group it would answer at each spot (paler
// where it is unsure), a thin line where its answer changes, and the
// training dots on top with a ring round each one it currently gets wrong.

import { forward } from './net.js';
import { contour } from './contour.js';

export const EXTENT = 1.22; // the plane shown is [−EXTENT, EXTENT]²

export const GROUPS = [
  { deep: '#b06a1f', wash: [232, 196, 132], name: 'ochre' },
  { deep: '#2f6f55', wash: [160, 199, 168], name: 'green' },
  { deep: '#6b4687', wash: [196, 176, 214], name: 'violet' },
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
  const base = document.createElement('canvas');
  const bctx = base.getContext('2d');
  let S = 0;
  let dpr = 1;
  let css = 0;
  let wrongMask = new Uint8Array(0);

  function resize(size, ratio) {
    dpr = ratio;
    css = size;
    S = Math.round(size * ratio);
    canvas.width = canvas.height = S;
    base.width = base.height = S;
  }

  const toPx = (x) => ((x + EXTENT) / (2 * EXTENT)) * S;
  const toPy = (y) => ((EXTENT - y) / (2 * EXTENT)) * S;

  // CSS pixels within the canvas → a point of the plane.
  const toPlane = (cx, cy) => [(cx / css) * 2 * EXTENT - EXTENT, EXTENT - (cy / css) * 2 * EXTENT];

  // The network's opinion over the whole plane, for θ. Returns how many
  // training dots it gets wrong.
  function render(net, th, data) {
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
        w0 /= ws;
        w1 /= ws;
        w2 /= ws;
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
    bctx.imageSmoothingEnabled = true;
    bctx.imageSmoothingQuality = 'high';
    bctx.drawImage(small, 0, 0, S, S);

    // Where the answer changes.
    const g2p = (g) => ((g + 0.5) / G) * S;
    bctx.strokeStyle = 'rgba(48, 38, 28, 0.55)';
    bctx.lineWidth = 0.9 * dpr;
    bctx.lineJoin = 'round';
    bctx.beginPath();
    for (let j = 0; j < 3; j++) {
      for (const { pts, closed } of contour(field[j], G, G, 0)) {
        bctx.moveTo(g2p(pts[0]), g2p(pts[1]));
        for (let i = 2; i < pts.length; i += 2) bctx.lineTo(g2p(pts[i]), g2p(pts[i + 1]));
        if (closed) bctx.closePath();
      }
    }
    bctx.stroke();

    if (wrongMask.length < data.n) wrongMask = new Uint8Array(data.n * 2);
    let wrong = 0;
    for (let i = 0; i < data.n; i++) {
      const a = forward(net, th, data.xs[i], data.ys[i]);
      let best = 0;
      for (let j = 1; j < 3; j++) if (a[j] > a[best]) best = j;
      wrongMask[i] = best !== data.labels[i] ? 1 : 0;
      wrong += wrongMask[i];
    }
    return wrong;
  }

  // The last render, with the dots and the probe on top. `probe` is a point
  // of the plane or null; `brush` is { x, y, radius } in plane units or null.
  function compose(data, probe, brush) {
    if (!S) return;
    ctx.drawImage(base, 0, 0);
    const rad = 3.3 * dpr;
    for (let i = 0; i < data.n; i++) {
      ctx.beginPath();
      ctx.arc(toPx(data.xs[i]), toPy(data.ys[i]), rad, 0, Math.PI * 2);
      ctx.fillStyle = GROUPS[data.labels[i]].deep;
      ctx.fill();
      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = 'rgba(250, 246, 238, 0.9)';
      ctx.stroke();
    }
    ctx.strokeStyle = '#211b16';
    ctx.lineWidth = 1.3 * dpr;
    ctx.beginPath();
    for (let i = 0; i < data.n; i++) {
      if (!wrongMask[i]) continue;
      const x = toPx(data.xs[i]);
      const y = toPy(data.ys[i]);
      ctx.moveTo(x + rad * 2.1, y);
      ctx.arc(x, y, rad * 2.1, 0, Math.PI * 2);
    }
    ctx.stroke();

    if (probe) {
      // A fine red crosshair: the point being fed through the network.
      const x = toPx(probe[0]);
      const y = toPy(probe[1]);
      const R = 9 * dpr;
      ctx.strokeStyle = 'rgba(251, 247, 238, 0.85)';
      ctx.lineWidth = 3.4 * dpr;
      cross(ctx, x, y, R);
      ctx.strokeStyle = '#c8302a';
      ctx.lineWidth = 1.4 * dpr;
      cross(ctx, x, y, R);
    }
    if (brush) {
      ctx.beginPath();
      ctx.arc(toPx(brush.x), toPy(brush.y), (brush.radius / (2 * EXTENT)) * S, 0, Math.PI * 2);
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.strokeStyle = 'rgba(33, 27, 22, 0.7)';
      ctx.lineWidth = 1 * dpr;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  return { resize, render, compose, toPlane, isWrong: (i) => wrongMask[i] === 1 };
}

function cross(ctx, x, y, R) {
  ctx.beginPath();
  ctx.arc(x, y, R, 0, Math.PI * 2);
  ctx.moveTo(x - R * 1.6, y);
  ctx.lineTo(x - R * 0.45, y);
  ctx.moveTo(x + R * 0.45, y);
  ctx.lineTo(x + R * 1.6, y);
  ctx.moveTo(x, y - R * 1.6);
  ctx.lineTo(x, y - R * 0.45);
  ctx.moveTo(x, y + R * 0.45);
  ctx.lineTo(x, y + R * 1.6);
  ctx.stroke();
}

// Surveying the cost after the walk.
//
// C is a function of every weight at once (hundreds of numbers), so it has
// no map. What can be drawn is a flat slice through it, chosen after the
// fact to hold the path descent actually took (in the spirit of Li et al.,
// "Visualizing the Loss Landscape of Neural Nets", NeurIPS 2018, who use the
// principal directions of the path): two unit directions d₁, d₂ through the
// final weights θ_end, and
//
//   C(θ_end + α d₁ + β d₂)
//
// on a grid of (α, β). Each grid cell is a real, complete network; its
// height is that network's real cost on the training set. The network never
// sees this map. It only ever knew the slope where it stood.

import { cost } from './net.js';

const SNAP_EVERY = 3;

// Everything the map needs from a finished run: the plane, the route drawn
// in it, the share of the route that lies in the plane, and the window.
// `all` holds θ_t for every step as rows; `start` and `end` are θ_0 and
// θ_steps in full precision.
export function planeOfRun({ net, start, all, end, steps }) {
  const P = net.nParams;
  const row = (t, p) => (t === 0 ? start[p] : all[t * P + p]);
  const offsets = [];
  for (let t = 0; t <= steps; t += SNAP_EVERY) {
    const s = new Float64Array(P);
    for (let p = 0; p < P; p++) s[p] = row(t, p) - end[p];
    offsets.push(s);
  }
  const { dirs, inPlane } = routePlane(offsets, offsets[0]);
  const path = new Float32Array((steps + 1) * 2);
  const pts = [];
  for (let t = 0; t <= steps; t++) {
    let a = 0;
    let b = 0;
    for (let p = 0; p < P; p++) {
      const v = row(t, p) - end[p];
      a += v * dirs[0][p];
      b += v * dirs[1][p];
    }
    path[t * 2] = a;
    path[t * 2 + 1] = b;
    pts.push([a, b]);
  }
  return { dirs, path, inPlane, win: frame(pts, 0.42) };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// The plane used for the map. The first direction is the straight line from
// where the walk ended back to where it began, so both ends of the route sit
// exactly on the surveyed ground. The second is the direction in which the
// route strays furthest from that line (the top principal direction of what
// is left of each step once its share along the first is removed). Returns
// the two unit directions and the share of the route's squared length that
// lies in the plane.
export function routePlane(steps, start) {
  const P = start.length;
  const d1 = start.slice();
  const n0 = Math.sqrt(dot(d1, d1));
  for (let p = 0; p < P; p++) d1[p] /= n0;
  let total = 0;
  let along = 0;
  const rest = steps.map((s) => {
    const k = dot(s, d1);
    total += dot(s, s);
    along += k * k;
    const r = new Float64Array(P);
    for (let p = 0; p < P; p++) r[p] = s[p] - k * d1[p];
    return r;
  });
  const { dirs, explained } = principalPlane(rest, 1);
  const d2 = dirs[0];
  const pr = dot(d2, d1);
  for (let p = 0; p < P; p++) d2[p] -= pr * d1[p];
  const n2 = Math.sqrt(dot(d2, d2));
  for (let p = 0; p < P; p++) d2[p] /= n2;
  const restTotal = total - along;
  return { dirs: [d1, d2], inPlane: (along + explained[0] * restTotal) / total };
}

// Top principal directions of the rows of `steps`, found by power iteration
// on the small Gram matrix G = M Mᵀ, then lifted back to parameter space as
// d = Mᵀu / ‖Mᵀu‖. Returns the directions and the share of the rows' total
// squared length that each one accounts for. (This uncentred form is the one
// Li et al. use; routePlane above is what the piece draws.)
export function principalPlane(steps, count = 2) {
  const k = steps.length;
  const P = steps[0].length;
  const G = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      const v = dot(steps[i], steps[j]);
      G[i * k + j] = v;
      G[j * k + i] = v;
    }
  }
  let trace = 0;
  for (let i = 0; i < k; i++) trace += G[i * k + i];

  const eig = [];
  for (let c = 0; c < count; c++) {
    let u = new Float64Array(k).map((_, i) => Math.sin(1 + i * (c + 1.37)));
    let lambda = 0;
    for (let it = 0; it < 500; it++) {
      const w = new Float64Array(k);
      for (let i = 0; i < k; i++) {
        let s = 0;
        for (let j = 0; j < k; j++) s += G[i * k + j] * u[j];
        w[i] = s;
      }
      for (const e of eig) {
        const p = dot(w, e.u);
        for (let i = 0; i < k; i++) w[i] -= p * e.u[i];
      }
      const n = Math.sqrt(dot(w, w));
      if (n === 0) break;
      for (let i = 0; i < k; i++) w[i] /= n;
      const delta = Math.abs(n - lambda);
      lambda = n;
      u = w;
      if (delta < 1e-12 * n) break;
    }
    eig.push({ u, lambda });
  }

  const dirs = eig.map(({ u }) => {
    const d = new Float64Array(P);
    for (let i = 0; i < k; i++) {
      const s = steps[i];
      for (let p = 0; p < P; p++) d[p] += u[i] * s[p];
    }
    const n = Math.sqrt(dot(d, d));
    for (let p = 0; p < P; p++) d[p] /= n;
    return d;
  });
  return { dirs, explained: eig.map((e) => e.lambda / trace) };
}

// A square window around the projected path, equal scale on both axes so
// that downhill on the map points the way downhill points in the plane.
export function frame(path, margin = 0.3) {
  let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
  for (const [a, b] of path) {
    if (a < a0) a0 = a;
    if (a > a1) a1 = a;
    if (b < b0) b0 = b;
    if (b > b1) b1 = b;
  }
  const half = (Math.max(a1 - a0, b1 - b0) / 2) * (1 + margin);
  const ca = (a0 + a1) / 2;
  const cb = (b0 + b1) / 2;
  return { a0: ca - half, a1: ca + half, b0: cb - half, b1: cb + half };
}

// C over rows r0..r1−1 of an n×n grid of the window. Row 0 is the top
// (largest β). Rows can be split between workers.
export function heights(net, data, end, dirs, win, n, r0 = 0, r1 = n) {
  const th = new Float64Array(end.length);
  const out = new Float32Array((r1 - r0) * n);
  for (let r = r0; r < r1; r++) {
    const b = win.b1 - ((win.b1 - win.b0) * r) / (n - 1);
    for (let c = 0; c < n; c++) {
      const a = win.a0 + ((win.a1 - win.a0) * c) / (n - 1);
      for (let p = 0; p < th.length; p++) th[p] = end[p] + a * dirs[0][p] + b * dirs[1][p];
      out[(r - r0) * n + c] = cost(net, th, data);
    }
  }
  return out;
}

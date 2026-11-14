// Training sets: points in the plane, each with one of three colours.
// Two-dimensional so the plane can be drawn and every point the network has
// an opinion about is a pixel on screen. Each preset is deterministic, and
// the reader can paint more points or rub them out.

import { rng } from './net.js';

export const MAX_POINTS = 360;

function gauss(r) {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

// [[x, y, colour], ...] → the typed form the maths reads.
export function pack(pts) {
  const n = pts.length;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const labels = new Uint8Array(n);
  pts.forEach((p, i) => {
    xs[i] = p[0];
    ys[i] = p[1];
    labels[i] = p[2];
  });
  return { n, xs, ys, labels };
}

// Three arms wound around the centre, each a third of a turn apart.
function spiral(r) {
  const pts = [];
  const per = 50;
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < per; i++) {
      const t = 0.12 + (0.88 * i) / (per - 1);
      const ang = t * 0.62 * Math.PI * 2 + (c * Math.PI * 2) / 3;
      const rad = t * 0.92;
      pts.push([rad * Math.cos(ang) + gauss(r) * 0.05, rad * Math.sin(ang) + gauss(r) * 0.05, c]);
    }
  }
  return pts;
}

// A disc inside two rings.
function rings(r) {
  const pts = [];
  const bands = [[0, 0.3], [0.48, 0.66], [0.84, 1.0]];
  bands.forEach(([r0, r1], c) => {
    const count = [36, 52, 62][c];
    for (let i = 0; i < count; i++) {
      const rad = Math.sqrt(r0 * r0 + (r1 * r1 - r0 * r0) * r());
      const ang = r() * Math.PI * 2;
      pts.push([rad * Math.cos(ang), rad * Math.sin(ang), c]);
    }
  });
  return pts;
}

// Five clusters for three colours: two colours come in two pieces each, on
// opposite sides, so no straight cut separates them.
function islands(r) {
  const centres = [
    [[-0.55, 0.5], [0.55, -0.55]],
    [[0.5, 0.5]],
    [[-0.55, -0.5], [0.02, 0.0]],
  ];
  const pts = [];
  centres.forEach((cs, c) => {
    for (let i = 0; i < 50; i++) {
      const [cx, cy] = cs[i % cs.length];
      pts.push([cx + gauss(r) * 0.15, cy + gauss(r) * 0.15, c]);
    }
  });
  return pts;
}

export const PRESETS = {
  spiral: () => spiral(rng(11)),
  rings: () => rings(rng(12)),
  islands: () => islands(rng(13)),
};

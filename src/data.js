// Training sets: points in the plane, each with one of three labels.
// Two-dimensional so the plane can be drawn and every point the network
// has an opinion about is a pixel on screen. Deterministic per seed.

import { rng } from './net.js';

function gauss(r) {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

function pack(pts) {
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
function arms(perClass, r, turn, noise) {
  const pts = [];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < perClass; i++) {
      const t = 0.12 + (0.88 * i) / (perClass - 1);
      const ang = t * turn * Math.PI * 2 + (c * Math.PI * 2) / 3;
      const rad = t * 0.92;
      pts.push([rad * Math.cos(ang) + gauss(r) * noise, rad * Math.sin(ang) + gauss(r) * noise, c]);
    }
  }
  return pts;
}

export const TASKS = {
  arms: (seed) => pack(arms(50, rng(seed), 0.62, 0.05)),
};

export function makeData(task, seed = 11) {
  return (TASKS[task] || TASKS.arms)(seed);
}

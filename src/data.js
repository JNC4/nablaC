// nablaC: toy 2D datasets.
//
// Each task is a set of points in the unit-ish square [-1,1]² with an integer
// class label. Two-dimensional on purpose: the input space is the screen, so the
// decision boundary the network carves can be drawn directly. All deterministic
// given a seed.

import { makeRng } from './nn.js';

function gaussian(rng) {
  // Box–Muller.
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Two interleaving half-moons. The canonical "needs a curved boundary" task.
function twoMoons(n, rng, noise = 0.12) {
  const pts = [];
  const per = Math.floor(n / 2);
  for (let i = 0; i < per; i++) {
    const t = Math.PI * (i / per);
    pts.push({ x: Math.cos(t), y: Math.sin(t), label: 0 });
    pts.push({ x: 1 - Math.cos(t), y: 0.5 - Math.sin(t), label: 1 });
  }
  return pts.map((p) => ({
    x: (p.x - 0.5) * 1.1 + gaussian(rng) * noise,
    y: (p.y - 0.25) * 1.6 + gaussian(rng) * noise,
    label: p.label,
  }));
}

// Two intertwined spiral arms, the hardest task here.
function spirals(n, rng, noise = 0.06) {
  const pts = [];
  const per = Math.floor(n / 2);
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < per; i++) {
      const r = i / per;
      const t = 1.5 * r * Math.PI * 2 + c * Math.PI;
      pts.push({
        x: r * Math.cos(t) + gaussian(rng) * noise,
        y: r * Math.sin(t) + gaussian(rng) * noise,
        label: c,
      });
    }
  }
  return pts;
}

// A blob inside a ring: radial separation.
function circle(n, rng, noise = 0.08) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const inner = i % 2 === 0;
    const r = inner ? rng() * 0.4 : 0.6 + rng() * 0.35;
    const t = rng() * Math.PI * 2;
    pts.push({
      x: r * Math.cos(t) + gaussian(rng) * noise,
      y: r * Math.sin(t) + gaussian(rng) * noise,
      label: inner ? 0 : 1,
    });
  }
  return pts;
}

// Classic XOR quadrants: not linearly separable, needs a hidden layer.
function xor(n, rng, noise = 0.18) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = rng() * 2 - 1;
    const y = rng() * 2 - 1;
    pts.push({
      x: x + gaussian(rng) * noise,
      y: y + gaussian(rng) * noise,
      label: x * y > 0 ? 0 : 1,
    });
  }
  return pts;
}

export const DATASETS = {
  moons: { name: 'two moons', fn: twoMoons },
  spirals: { name: 'spirals', fn: spirals },
  circle: { name: 'circle', fn: circle },
  xor: { name: 'xor', fn: xor },
};

export function makeData(key, n = 120, seed = 7) {
  const rng = makeRng(seed);
  const fn = (DATASETS[key] || DATASETS.moons).fn;
  return fn(n, rng);
}

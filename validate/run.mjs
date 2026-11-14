// npm run validate
//
// Checks that the piece computes what it says it computes. Exits non-zero
// if any check fails.
//
//  1. The scalar autograd in validate/value.js agrees with finite differences.
//  2. The hand-written backpropagation in src/net.js (the one the piece runs)
//     gives the same ∇C as that autograd, for every one of the 371 numbers.
//  3. It also agrees with finite differences of C itself.
//  4. The readouts under the network: ∂C/∂w assembled as the essay's chain
//     (the average over dots of a · σ′(z) · ∂C₀/∂a) equals ∇C for every
//     weight and bias, one dot's share equals the finite difference of that
//     dot's own cost C₀, and the neuron readouts (z, a) match the forward pass.
//  5. Gradient descent with the piece's settings learns every preset.
//  6. Retraining on drawn dots: add a stroke of new dots, carry on from the
//     trained weights, and it learns them too.
//  7. The map is honest at both ends: the height it shows at the start and
//     at the end of the route is the network's real cost there, and nearly
//     all of the route lies in the drawn plane.
//  8. The opening sheet shipped in src/opening.json is what the current code
//     produces.

import { readFileSync } from 'node:fs';
import { Value } from './value.js';
import { initParams, cost, costAndGrad, accuracy, rng, probe, chainTerm, chainGrad, paramIndex, forward } from '../src/net.js';
import { makeNetwork, train, ETAS, STEPS } from '../src/run.js';
import { PRESETS, pack } from '../src/data.js';
import { planeOfRun, heights } from '../src/survey.js';

let failures = 0;
function check(ok, msg) {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failures++;
}
const fmt = (x) => x.toExponential(2);

const net = makeNetwork();
const P = net.nParams;
const spiral = pack(PRESETS.spiral());

// θ after `steps` steps of the piece's own training from `start`.
function trained(data, start, steps = STEPS, eta = ETAS.medium) {
  return train({ net, data, eta, start, steps });
}

// 1. the reference autograd against finite differences
console.log('\n1. scalar autograd against finite differences');
console.log('   f(a,b,c) = tanh(ab + c²)·σ(a − b) + eᶜ/(b + 2) − log(a² + 1)');
{
  const f = (a, b, c) =>
    a.mul(b).add(c.pow(2)).tanh().mul(a.sub(b).sigmoid())
      .add(c.exp().div(b.add(2)))
      .sub(a.pow(2).add(1).log());
  const x = [0.7, -1.3, 0.4];
  const vs = x.map((v) => new Value(v));
  f(...vs).backward();
  const h = 1e-6;
  ['a', 'b', 'c'].forEach((name, i) => {
    const up = x.map((v, j) => new Value(v + (i === j ? h : 0)));
    const dn = x.map((v, j) => new Value(v - (i === j ? h : 0)));
    const fd = (f(...up).data - f(...dn).data) / (2 * h);
    const err = Math.abs(vs[i].grad - fd);
    check(err < 1e-7, `∂f/∂${name}: autograd ${vs[i].grad.toFixed(8)}, finite difference ${fd.toFixed(8)}, |Δ| ${fmt(err)}`);
  });
}

// The same cost as src/net.js, built out of Values.
function valueCost(th, data) {
  const params = Array.from(th, (v) => new Value(v));
  let total = new Value(0);
  for (let i = 0; i < data.n; i++) {
    let a = [new Value(data.xs[i]), new Value(data.ys[i])];
    for (const L of net.layers) {
      const next = [];
      for (let j = 0; j < L.nout; j++) {
        let z = params[L.b + j];
        for (let k = 0; k < L.nin; k++) z = z.add(params[L.w + j * L.nin + k].mul(a[k]));
        next.push(z.sigmoid());
      }
      a = next;
    }
    a.forEach((aj, j) => {
      total = total.add(aj.sub(j === data.labels[i] ? 1 : 0).pow(2));
    });
  }
  const C = total.mul(1 / data.n);
  C.backward();
  return { C: C.data, grad: params.map((p) => p.grad) };
}

// Part-way through training, where every term of the gradient is live.
const mid = trained(spiral, initParams(net, 4), 200);

// 2. the piece's backprop against the scalar autograd
console.log(`\n2. the piece's backpropagation against the scalar autograd (${P} weights and biases)`);
for (const [label, th] of [['random start (seed 4)', initParams(net, 4)], ['after 200 steps', mid]]) {
  const g = new Float64Array(P);
  const C = costAndGrad(net, th, spiral, g);
  const ref = valueCost(th, spiral);
  let worst = 0;
  let norm = 0;
  for (let p = 0; p < P; p++) {
    worst = Math.max(worst, Math.abs(g[p] - ref.grad[p]));
    norm = Math.max(norm, Math.abs(ref.grad[p]));
  }
  check(Math.abs(C - ref.C) < 1e-12, `${label}: C = ${C.toFixed(10)} both ways`);
  check(worst < 1e-12 * Math.max(1, norm), `${label}: largest disagreement in ∇C ${fmt(worst)} (largest component ${norm.toFixed(4)})`);
}

// 3. against finite differences of C
console.log('\n3. the piece\'s ∇C against finite differences of C (40 components chosen at random)');
{
  const th = initParams(net, 9);
  const g = new Float64Array(P);
  costAndGrad(net, th, spiral, g);
  const r = rng(2024);
  const h = 1e-6;
  let worst = 0;
  for (let i = 0; i < 40; i++) {
    const p = Math.floor(r() * P);
    const keep = th[p];
    th[p] = keep + h;
    const up = cost(net, th, spiral);
    th[p] = keep - h;
    const dn = cost(net, th, spiral);
    th[p] = keep;
    worst = Math.max(worst, Math.abs((up - dn) / (2 * h) - g[p]));
  }
  check(worst < 1e-8, `largest disagreement ${fmt(worst)}`);
}

// 4. the readouts
console.log('\n4. the equations under the network');
{
  const th = mid;
  const g = new Float64Array(P);
  costAndGrad(net, th, spiral, g);
  // Every weight and bias, assembled as the chain.
  let worst = 0;
  let count = 0;
  for (let l = 1; l <= net.layers.length; l++) {
    const L = net.layers[l - 1];
    for (let j = 0; j < L.nout; j++) {
      for (let k = -1; k < L.nin; k++) {
        worst = Math.max(worst, Math.abs(chainGrad(net, th, spiral, l, j, k) - g[paramIndex(net, l, j, k)]));
        count++;
      }
    }
  }
  check(count === P && worst < 1e-15, `∂C/∂w as (1/n) Σ a·σ′(z)·∂C₀/∂a matches ∇C on all ${count} weights and biases, worst ${fmt(worst)}`);

  // One dot's share, against the finite difference of that dot's C₀.
  const one = (i) => ({ n: 1, xs: [spiral.xs[i]], ys: [spiral.ys[i]], labels: [spiral.labels[i]] });
  const r = rng(77);
  const h = 1e-6;
  let worstDot = 0;
  for (let t = 0; t < 40; t++) {
    const i = Math.floor(r() * spiral.n);
    const l = 1 + Math.floor(r() * 3);
    const L = net.layers[l - 1];
    const j = Math.floor(r() * L.nout);
    const k = Math.floor(r() * (L.nin + 1)) - 1;
    const p = probe(net, th, spiral.xs[i], spiral.ys[i], spiral.labels[i]);
    const shown = chainTerm(p, l, j, k).value;
    const idx = paramIndex(net, l, j, k);
    const keep = th[idx];
    th[idx] = keep + h;
    const up = cost(net, th, one(i));
    th[idx] = keep - h;
    const dn = cost(net, th, one(i));
    th[idx] = keep;
    worstDot = Math.max(worstDot, Math.abs((up - dn) / (2 * h) - shown));
  }
  check(worstDot < 1e-8, `∂C₀/∂w for one dot, a·σ′(z)·∂C₀/∂a, against finite differences of C₀ (40 random dots and weights), worst ${fmt(worstDot)}`);

  // The forward readouts and C₀ itself.
  let worstFwd = 0;
  let worstC0 = 0;
  for (let i = 0; i < spiral.n; i += 7) {
    const p = probe(net, th, spiral.xs[i], spiral.ys[i], spiral.labels[i]);
    const out = forward(net, th, spiral.xs[i], spiral.ys[i]);
    for (let j = 0; j < 3; j++) worstFwd = Math.max(worstFwd, Math.abs(out[j] - p.a[3][j]));
    worstC0 = Math.max(worstC0, Math.abs(p.C0 - cost(net, th, one(i))));
  }
  check(worstFwd === 0 && worstC0 < 1e-15, `a(O) and C₀ in the readouts match the forward pass and the cost exactly (worst ${fmt(worstC0)})`);
}

// 5. learning, every preset
console.log(`\n5. gradient descent learns every preset (η = ${ETAS.medium}, ${STEPS} steps, five random starts each)`);
const outcomes = {};
for (const name of Object.keys(PRESETS)) {
  const data = pack(PRESETS[name]());
  let worstEnd = 0;
  let worstAcc = 1;
  for (let seed = 1; seed <= 5; seed++) {
    const start = initParams(net, seed);
    const c0 = cost(net, start, data);
    const end = trained(data, start);
    worstEnd = Math.max(worstEnd, cost(net, end, data) / c0);
    worstAcc = Math.min(worstAcc, accuracy(net, end, data));
  }
  outcomes[name] = { worstEnd, worstAcc };
  check(worstEnd < 0.1 && worstAcc >= 0.95, `${name} (${data.n} dots): C falls to ${(worstEnd * 100).toFixed(1)}% of its start or less, at least ${(worstAcc * 100).toFixed(1)}% of dots right`);
}

// 6. drawn dots
console.log('\n6. retraining on drawn dots (spiral, trained from seed 1, plus a stroke)');
{
  const before = PRESETS.spiral();
  const theta = trained(pack(before), initParams(net, 1));
  // Strokes laid down the way the page lays them: one dot every 0.055.
  const line = (count, f) => Array.from({ length: count }, (_, i) => f(i * 0.055));
  const strokes = {
    'green, across the upper left': line(21, (t) => [-0.95 + t, 0.8, 1]),
    'ochre, down the right edge': line(15, (t) => [0.9, -0.5 + t, 0]),
    'green, along the very top': line(21, (t) => [-0.95 + t, 0.95 - 0.1 * (-0.95 + t), 1]),
  };
  const wrongOf = (th, pts) =>
    pts.filter(([x, y, c]) => {
      const a = forward(net, th, x, y);
      return a.indexOf(Math.max(...a)) !== c;
    }).length;
  const results = Object.entries(strokes).map(([name, drawn]) => {
    const data = pack([...before, ...drawn]);
    const warm = trained(data, theta);
    const fresh = trained(data, initParams(net, 5));
    return { name, drawn, data, warm, fresh, wrongBefore: wrongOf(theta, drawn) };
  });
  for (const { name, drawn, data, warm, wrongBefore } of results.slice(0, 2)) {
    const cBefore = cost(net, theta, data);
    const cAfter = cost(net, warm, data);
    const wrongAfter = wrongOf(warm, drawn);
    check(wrongBefore > drawn.length / 2 && cAfter < cBefore / 3 && wrongAfter <= 2,
      `${name}: carrying on, C ${cBefore.toFixed(3)} → ${cAfter.toFixed(3)}, drawn dots wrong ${wrongBefore} → ${wrongAfter} of ${drawn.length}`);
  }
  const top = results[2];
  const cWarm = cost(net, top.warm, top.data);
  const wrongWarm = wrongOf(top.warm, top.drawn);
  const wrongFresh = wrongOf(top.fresh, top.drawn);
  check(wrongFresh <= 1, `${top.name}: from fresh random weights, ${wrongFresh} of ${top.drawn.length} drawn dots wrong after 1500 steps`);
  console.log(`  note ${top.name}: carrying on instead gets stuck, C ends at ${cWarm.toFixed(3)} with ${wrongWarm} of ${top.drawn.length} drawn dots still wrong (saturated neurons; see README)`);
}

// 7. the map
console.log('\n7. the map is the real cost where the route starts and ends');
const opening = JSON.parse(readFileSync(new URL('../src/opening.json', import.meta.url)));
{
  const data = pack(PRESETS[opening.preset]());
  const start = initParams(net, opening.seed);
  const all = new Float32Array((STEPS + 1) * P);
  const end = train({ net, data, eta: ETAS[opening.eta], start, onChunk: (f, th) => all.set(th, f * P) });
  const { dirs, path, inPlane } = planeOfRun({ net, start, all, end, steps: STEPS });
  // Map heights are stored as float32, hence tolerances near 1e-6 here.
  const at = (a, b) => heights(net, data, end, dirs, { a0: a, a1: a + 1, b0: b - 1, b1: b }, 2, 0, 1)[0];
  const c0 = cost(net, start, data);
  const m0 = at(path[0], path[1]);
  const cEnd = cost(net, end, data);
  const mEnd = at(0, 0);
  check(Math.abs(m0 - c0) < 1e-5, `start: map ${m0.toFixed(6)}, network ${c0.toFixed(6)}`);
  check(Math.abs(mEnd - cEnd) < 1e-6, `end: map ${mEnd.toFixed(6)}, network ${cEnd.toFixed(6)}`);
  check(inPlane > 0.95, `${(inPlane * 100).toFixed(1)}% of the route's squared length lies in the drawn plane`);

  // 8. the shipped opening sheet
  console.log('\n8. src/opening.json matches the current code');
  let worst = 0;
  for (let p = 0; p < P; p++) worst = Math.max(worst, Math.abs(end[p] - opening.end[p]));
  check(worst < 1e-5, `final weights agree to ${fmt(worst)}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);

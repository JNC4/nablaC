// npm run validate
//
// Checks that the piece computes what it says it computes. Exits non-zero
// if any check fails.
//
//  1. The scalar autograd in validate/value.js agrees with finite differences.
//  2. The hand-written backpropagation in src/net.js (the one the piece runs)
//     gives the same ∇C as that autograd, for every one of the 371 numbers.
//  3. It also agrees with finite differences of C itself.
//  4. Gradient descent with the piece's settings actually learns the task.
//  5. The map is honest at both ends: the height it shows at the start and
//     at the end of the route is the network's real cost there, and nearly
//     all of the route lies in the drawn plane.
//  6. The opening sheet shipped in src/opening.json is what the current code
//     produces.

import { readFileSync } from 'node:fs';
import { Value } from './value.js';
import { makeNet, initParams, cost, costAndGrad, accuracy, rng } from '../src/net.js';
import { setup, train, ETAS, STEPS } from '../src/run.js';
import { planeOfRun, heights } from '../src/survey.js';

let failures = 0;
function check(ok, msg) {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`);
  if (!ok) failures++;
}
const fmt = (x) => x.toExponential(2);

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
function valueCost(net, th, data) {
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

// 2 and 3. the piece's backprop
const { net, data } = setup();
const P = net.nParams;
console.log(`\n2. the piece's backpropagation against the scalar autograd (${P} weights and biases)`);
{
  // Part-way through training, where every term of the gradient is live.
  const mid = new Float64Array(P);
  train({ net, data, eta: ETAS.medium, seed: 4, steps: 200, chunk: 201, onChunk: (f, th) => mid.set(th.subarray(200 * P)) });
  for (const [label, th] of [['random start (seed 4)', initParams(net, 4)], ['after 200 steps', mid]]) {
    const g = new Float64Array(P);
    const C = costAndGrad(net, th, data, g);
    const ref = valueCost(net, th, data);
    let worst = 0;
    let norm = 0;
    for (let p = 0; p < P; p++) {
      worst = Math.max(worst, Math.abs(g[p] - ref.grad[p]));
      norm = Math.max(norm, Math.abs(ref.grad[p]));
    }
    check(Math.abs(C - ref.C) < 1e-12, `${label}: C = ${C.toFixed(10)} both ways`);
    check(worst < 1e-12 * Math.max(1, norm), `${label}: largest disagreement in ∇C ${fmt(worst)} (largest component ${norm.toFixed(4)})`);
  }
}

console.log('\n3. the piece\'s ∇C against finite differences of C (40 components chosen at random)');
{
  const th = initParams(net, 9);
  const g = new Float64Array(P);
  costAndGrad(net, th, data, g);
  const r = rng(2024);
  const h = 1e-6;
  let worst = 0;
  for (let i = 0; i < 40; i++) {
    const p = Math.floor(r() * P);
    const keep = th[p];
    th[p] = keep + h;
    const up = cost(net, th, data);
    th[p] = keep - h;
    const dn = cost(net, th, data);
    th[p] = keep;
    worst = Math.max(worst, Math.abs((up - dn) / (2 * h) - g[p]));
  }
  check(worst < 1e-8, `largest disagreement ${fmt(worst)}`);
}

// 4. learning
console.log(`\n4. gradient descent learns the task (η = ${ETAS.medium}, ${STEPS} steps, ten random starts)`);
{
  let worstEnd = 0;
  let worstAcc = 1;
  for (let seed = 1; seed <= 10; seed++) {
    let c0 = 0;
    let c1 = 0;
    const end = train({ net, data, eta: ETAS.medium, seed, onChunk: (f, th, cs) => { if (f === 0) c0 = cs[0]; c1 = cs[cs.length - 1]; } });
    worstEnd = Math.max(worstEnd, c1 / c0);
    worstAcc = Math.min(worstAcc, accuracy(net, end, data));
  }
  check(worstEnd < 0.05, `C falls to under 5% of where it started on every start (worst ${(worstEnd * 100).toFixed(1)}%)`);
  check(worstAcc >= 0.98, `every start ends with at least 98% of dots right (worst ${(worstAcc * 100).toFixed(1)}%)`);
}

// 5. the map
console.log('\n5. the map is the real cost where the route starts and ends');
const opening = JSON.parse(readFileSync(new URL('../src/opening.json', import.meta.url)));
{
  const seed = opening.seed;
  const all = new Float32Array((STEPS + 1) * P);
  const end = train({ net, data, eta: ETAS[opening.eta], seed, onChunk: (f, th) => all.set(th, f * P) });
  const { dirs, path, inPlane } = planeOfRun({ net, seed, all, end, steps: STEPS });
  // Map heights are stored as float32, hence tolerances near 1e-6 here.
  const at = (a, b) => heights(net, data, end, dirs, { a0: a, a1: a + 1, b0: b - 1, b1: b }, 2, 0, 1)[0];
  const c0 = cost(net, initParams(net, seed), data);
  const m0 = at(path[0], path[1]);
  const cEnd = cost(net, end, data);
  const mEnd = at(0, 0);
  check(Math.abs(m0 - c0) < 1e-5, `start: map ${m0.toFixed(6)}, network ${c0.toFixed(6)}`);
  check(Math.abs(mEnd - cEnd) < 1e-6, `end: map ${mEnd.toFixed(6)}, network ${cEnd.toFixed(6)}`);
  check(inPlane > 0.95, `${(inPlane * 100).toFixed(1)}% of the route's squared length lies in the drawn plane`);

  // 6. the shipped opening sheet
  console.log('\n6. src/opening.json matches the current code');
  let worst = 0;
  for (let p = 0; p < P; p++) worst = Math.max(worst, Math.abs(end[p] - opening.end[p]));
  check(worst < 1e-5, `final weights agree to ${fmt(worst)}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);

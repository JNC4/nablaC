// Correctness harness for the autograd. Run with: npm run check
//
// Two checks:
//  1) Numerical gradient check: compare every analytic ∂y/∂x against a finite
//     difference. If our backward() is wrong, these disagree.
//  2) Learning check: a tiny MLP on two-moons should drive its loss down. If
//     the gradients don't actually point downhill, the loss won't fall.

import { Value, softmax, crossEntropy } from './autograd.js';
import { MLP } from './nn.js';
import { makeData } from './data.js';

let failures = 0;
const approx = (a, b, tol, msg) => {
  const ok = Math.abs(a - b) <= tol;
  if (!ok) {
    failures++;
    console.error(`  ✗ ${msg}: analytic ${a.toFixed(6)} vs numeric ${b.toFixed(6)} (Δ ${Math.abs(a - b).toExponential(2)})`);
  } else {
    console.log(`  ✓ ${msg}`);
  }
};

// --- 1) numerical gradient check on a gnarly scalar expression ---
// f(a,b,c) = tanh(a*b + c²) * σ(a - b) + (exp(c) / (b+2)) - log(a²+1)
function f(a, b, c) {
  const t = a.mul(b).add(c.pow(2)).tanh();
  const s = a.sub(b).sigmoid();
  const term1 = t.mul(s);
  const term2 = c.exp().div(b.add(2));
  const term3 = a.pow(2).add(1).log();
  return term1.add(term2).sub(term3);
}

console.log('gradient check: f(a,b,c) = tanh(ab+c²)·σ(a−b) + eᶜ/(b+2) − log(a²+1)');
{
  const av = 0.7, bv = -1.3, cv = 0.4;
  const a = new Value(av), b = new Value(bv), c = new Value(cv);
  const y = f(a, b, c);
  y.backward();

  const h = 1e-6;
  const numeric = (vary) => {
    const fp = f(new Value(av + (vary === 0 ? h : 0)), new Value(bv + (vary === 1 ? h : 0)), new Value(cv + (vary === 2 ? h : 0))).data;
    const fm = f(new Value(av - (vary === 0 ? h : 0)), new Value(bv - (vary === 1 ? h : 0)), new Value(cv - (vary === 2 ? h : 0))).data;
    return (fp - fm) / (2 * h);
  };
  approx(a.grad, numeric(0), 1e-4, '∂f/∂a');
  approx(b.grad, numeric(1), 1e-4, '∂f/∂b');
  approx(c.grad, numeric(2), 1e-4, '∂f/∂c');
}

// --- numerical check through softmax + cross-entropy on real MLP params ---
console.log('\ngradient check: cross-entropy(softmax(MLP(x)), label) w.r.t. a weight');
{
  const net = new MLP([2, 4, 2], { seed: 3 });
  const x = [0.5, -0.2];
  const loss = () => crossEntropy(softmax(net.forward(x)), 1);

  const L = loss();
  L.backward();
  const p = net.parameters()[5]; // some interior weight
  const analytic = p.grad;

  const h = 1e-6;
  const orig = p.data;
  p.data = orig + h; const Lp = loss().data;
  p.data = orig - h; const Lm = loss().data;
  p.data = orig;
  approx(analytic, (Lp - Lm) / (2 * h), 1e-4, '∂loss/∂w[5]');
}

// --- 2) learning check: loss must actually fall on two-moons ---
console.log('\nlearning check: MLP[2,8,8,2] on two-moons for 800 steps');
{
  const net = new MLP([2, 8, 8, 2], { seed: 5 });
  const data = makeData('moons', 200, 11);
  const lr = 0.3;

  const epochLoss = () => {
    let tot = 0;
    for (const pt of data) tot += crossEntropy(softmax(net.forward([pt.x, pt.y])), pt.label).data;
    return tot / data.length;
  };

  const before = epochLoss();
  for (let step = 0; step < 800; step++) {
    net.zeroGrad();
    let loss = new Value(0);
    for (const pt of data) loss = loss.add(crossEntropy(softmax(net.forward([pt.x, pt.y])), pt.label));
    loss = loss.div(data.length);
    loss.backward();
    for (const p of net.parameters()) p.data -= lr * p.grad;
  }
  const after = epochLoss();

  let correct = 0;
  for (const pt of data) {
    const out = net.forward([pt.x, pt.y]).map((v) => v.data);
    const pred = out[0] > out[1] ? 0 : 1;
    if (pred === pt.label) correct++;
  }
  const acc = correct / data.length;

  console.log(`  loss ${before.toFixed(4)} → ${after.toFixed(4)}, accuracy ${(acc * 100).toFixed(1)}%`);
  if (after >= before) { failures++; console.error('  ✗ loss did not decrease'); }
  else console.log('  ✓ loss decreased');
  if (acc < 0.85) { failures++; console.error(`  ✗ accuracy ${(acc * 100).toFixed(1)}% below 85%`); }
  else console.log('  ✓ accuracy ≥ 85%');
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);

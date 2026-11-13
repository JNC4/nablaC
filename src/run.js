// One training run: random weights, then `steps` steps of full-batch
// gradient descent, θ ← θ − η ∇C. Shared by the browser worker and the
// offline checks so both run exactly the same thing.

import { makeNet, initParams, costAndGrad } from './net.js';
import { makeData } from './data.js';

export const HIDDEN = 16;
export const STEPS = 1500;
export const TASK = 'arms';

// The three step sizes on offer. All three are real outcomes of the same
// network on the same data; only η changes.
export const ETAS = { small: 0.5, medium: 3, large: 25 };

export function setup(task = TASK) {
  const net = makeNet([2, HIDDEN, HIDDEN, 3]);
  const data = makeData(task);
  return { net, data };
}

// Runs the descent, calling onChunk(from, thetas, costs) every `chunk` steps
// with the parameters θ_t (as rows of a Float32Array) and C(θ_t) for those
// steps. Returns the final parameters in full precision.
export function train({ net, data, eta, seed, steps = STEPS, chunk = 100, onChunk }) {
  const P = net.nParams;
  const th = initParams(net, seed);
  const g = new Float64Array(P);
  let from = 0;
  let thetas = new Float32Array(chunk * P);
  let costs = new Float32Array(chunk);
  let k = 0;
  for (let t = 0; t <= steps; t++) {
    const c = costAndGrad(net, th, data, g);
    thetas.set(th, k * P);
    costs[k] = c;
    k++;
    if (k === chunk || t === steps) {
      onChunk?.(from, thetas.subarray(0, k * P), costs.subarray(0, k));
      from += k;
      k = 0;
      thetas = new Float32Array(chunk * P);
      costs = new Float32Array(chunk);
    }
    if (t < steps) for (let p = 0; p < P; p++) th[p] -= eta * g[p];
  }
  return th;
}

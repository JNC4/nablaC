// One training run: a starting θ, then `steps` steps of full-batch gradient
// descent, θ ← θ − η ∇C. Shared by the browser worker and the offline checks
// so both run exactly the same thing.

import { makeNet, costAndGrad } from './net.js';

export const SIZES = [2, 16, 16, 3];
export const STEPS = 1500;

// The three step sizes on offer. All three are real outcomes of the same
// network on the same data; only η changes.
export const ETAS = { small: 0.5, medium: 3, large: 25 };

export const makeNetwork = () => makeNet(SIZES);

// Runs the descent from `start` (copied, not changed), calling
// onChunk(from, thetas, costs) every `chunk` steps with θ_t as rows of a
// Float32Array and C(θ_t) for those steps. Returns the final θ in full
// precision.
export function train({ net, data, eta, start, steps = STEPS, chunk = 100, onChunk }) {
  const P = net.nParams;
  const th = Float64Array.from(start);
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

// nablaC: the live trainer.
//
// Holds a network + a dataset and advances training one gradient step at a time,
// the way the UI advances the markov walk one token at a time. Each step is the
// full ritual: forward every point, measure the cost C, call backward() to fill
// ∇C across every parameter, then take one step downhill (θ ← θ − η∇C).
//
// After a step, the live numbers the renderer wants are sitting right on the
// objects: every weight's `.data` (value) and `.grad` (its share of ∇C), every
// neuron's `.act`. The trainer just exposes them plus the loss history.

import { Value, softmax, crossEntropy } from './autograd.js';
import { MLP } from './nn.js';

export class Trainer {
  constructor({ sizes, data, lr = 0.3, seed = 1, hidden = 'tanh' } = {}) {
    this.net = new MLP(sizes, { seed, hidden });
    this.data = data;
    this.lr = lr;
    this.sizes = sizes;
    this.step = 0;
    this.loss = NaN;
    this.acc = 0;
    this.gradNorm = 0; // ‖∇C‖, the length of the gradient (its "urge to change")
    this.lossHistory = [];
    this.gradHistory = []; // ‖∇C‖ per step, kept in lockstep with lossHistory
  }

  // One full-batch gradient-descent step. Returns a trace snapshot of this step.
  trainStep() {
    const net = this.net;
    net.zeroGrad();

    let loss = new Value(0);
    for (const pt of this.data) {
      const probs = softmax(net.forward([pt.x, pt.y]));
      loss = loss.add(crossEntropy(probs, pt.label));
    }
    loss = loss.div(this.data.length);
    loss.backward();

    // ‖∇C‖ over all parameters, then the descent step.
    const params = net.parameters();
    let sq = 0;
    for (const p of params) sq += p.grad * p.grad;
    this.gradNorm = Math.sqrt(sq);
    for (const p of params) p.data -= this.lr * p.grad;

    this.loss = loss.data;
    this.step++;
    this.lossHistory.push(loss.data);
    this.gradHistory.push(this.gradNorm);
    if (this.lossHistory.length > 2000) { this.lossHistory.shift(); this.gradHistory.shift(); }
    this.acc = this.accuracy();
    return this.loss;
  }

  // Full-batch cost C computed numerically (no autograd graph), used by the
  // landscape view to probe C at hypothetical parameter settings, cheaply.
  lossNumeric() {
    let tot = 0;
    for (const pt of this.data) {
      const o = this.net.logits([pt.x, pt.y]);
      const m = Math.max(o[0], o[1]);
      const e0 = Math.exp(o[0] - m), e1 = Math.exp(o[1] - m);
      const probLabel = (pt.label === 0 ? e0 : e1) / (e0 + e1);
      tot += -Math.log(Math.max(probLabel, 1e-12));
    }
    return tot / this.data.length;
  }

  // Fraction of points classified correctly (numeric forward, no graph).
  accuracy() {
    let correct = 0;
    for (const pt of this.data) {
      const o = this.net.logits([pt.x, pt.y]);
      const pred = o[0] > o[1] ? 0 : 1;
      if (pred === pt.label) correct++;
    }
    return correct / this.data.length;
  }

  // Run a probe input through the net, recording activations for the renderer's
  // forward "lighting up" sweep, and return P(class 1).
  probe(x, y) {
    const o = this.net.logits([x, y], { record: true });
    const m = Math.max(o[0], o[1]);
    const e0 = Math.exp(o[0] - m), e1 = Math.exp(o[1] - m);
    return e1 / (e0 + e1);
  }

  // Evaluate P(class 1) over a res×res grid of the input square [-lim,lim]².
  // This is the decision boundary, flattened row-major.
  grid(res, lim = 1.4) {
    const out = new Float32Array(res * res);
    for (let j = 0; j < res; j++) {
      const y = lim - (2 * lim * j) / (res - 1);
      for (let i = 0; i < res; i++) {
        const x = -lim + (2 * lim * i) / (res - 1);
        const o = this.net.logits([x, y]);
        const m = Math.max(o[0], o[1]);
        const e0 = Math.exp(o[0] - m), e1 = Math.exp(o[1] - m);
        out[j * res + i] = e1 / (e0 + e1);
      }
    }
    return out;
  }
}

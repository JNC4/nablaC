// nablaC: the network.
//
// A multilayer perceptron assembled out of nothing but the autograd `Value`.
// A Neuron is a weighted sum of its inputs plus a bias, squashed by a
// nonlinearity: a = φ(Σ wᵢxᵢ + b). A Layer is a row of neurons; an MLP is a
// stack of layers. Stack enough of these, descend ∇C, and you get what the
// field calls "intelligence." Here it's a decision boundary, but it's the same
// machine all the way up.

import { Value } from './autograd.js';

// Deterministic RNG so a seed reproduces an initialisation (mulberry32).
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Neuron {
  // nonlin: 'tanh' | 'relu' | 'linear'
  constructor(nin, nonlin, rng) {
    // Small symmetric init scaled by fan-in (keeps early activations sane).
    const scale = 1 / Math.sqrt(nin);
    this.w = Array.from({ length: nin }, () => new Value((rng() * 2 - 1) * scale));
    this.b = new Value(0);
    this.nonlin = nonlin;
    // Filled in on the most recent forward pass, for the renderer.
    this.act = 0; // activation (post-nonlinearity)
    this.pre = 0; // pre-activation (the weighted sum + bias)
  }

  forward(x) {
    let sum = this.b;
    for (let i = 0; i < this.w.length; i++) sum = sum.add(this.w[i].mul(x[i]));
    this.pre = sum.data;
    let out;
    if (this.nonlin === 'tanh') out = sum.tanh();
    else if (this.nonlin === 'relu') out = sum.relu();
    else out = sum; // linear (output layer feeds softmax)
    this.act = out.data;
    return out;
  }

  parameters() {
    return [...this.w, this.b];
  }
}

export class Layer {
  constructor(nin, nout, nonlin, rng) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, nonlin, rng));
  }

  forward(x) {
    return this.neurons.map((n) => n.forward(x));
  }

  parameters() {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

export class MLP {
  // sizes: e.g. [2, 8, 8, 2] is 2 inputs, two hidden layers of 8, 2 logits out.
  // Hidden layers are tanh; the final layer is linear (softmax happens in loss).
  constructor(sizes, { seed = 1, hidden = 'tanh' } = {}) {
    const rng = makeRng(seed);
    this.sizes = sizes;
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const last = i === sizes.length - 2;
      this.layers.push(new Layer(sizes[i], sizes[i + 1], last ? 'linear' : hidden, rng));
    }
  }

  // x: array of numbers (or Values). Returns the output layer's array of Values.
  forward(x) {
    let h = x.map((v) => (v instanceof Value ? v : new Value(v)));
    for (const layer of this.layers) h = layer.forward(h);
    return h;
  }

  parameters() {
    return this.layers.flatMap((l) => l.parameters());
  }

  zeroGrad() {
    for (const p of this.parameters()) p.grad = 0;
  }

  // Fast, graph-free forward: plain arithmetic straight off the parameter
  // values. Used to sweep the decision-boundary grid and score accuracy, where
  // we want the numbers but not the gradient machinery. Also records each
  // neuron's activation/pre-activation so the renderer can light up a probe.
  logits(x, { record = false } = {}) {
    let h = x;
    for (let li = 0; li < this.layers.length; li++) {
      const layer = this.layers[li];
      const out = new Array(layer.neurons.length);
      for (let ni = 0; ni < layer.neurons.length; ni++) {
        const n = layer.neurons[ni];
        let s = n.b.data;
        for (let wi = 0; wi < n.w.length; wi++) s += n.w[wi].data * h[wi];
        let a;
        if (n.nonlin === 'tanh') a = Math.tanh(s);
        else if (n.nonlin === 'relu') a = s < 0 ? 0 : s;
        else a = s;
        if (record) { n.pre = s; n.act = a; }
        out[ni] = a;
      }
      h = out;
    }
    return h;
  }
}

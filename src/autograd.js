// nablaC: the autograd.
//
// Reverse-mode automatic differentiation from scratch. This is the core of the
// whole project: no library, no magic. A `Value` is a
// node in a computation graph; every operation records how to push the gradient
// back through it. Calling `.backward()` on the final loss walks the graph in
// reverse-topological order and accumulates ∂(loss)/∂(node) into every `.grad`.
//
// That number, the gradient, is what the whole UI is built around. ∇C is just
// the vector of these grads over the parameters. Learning is subtracting a
// sliver of it.
//
// Scalar-valued (à la Karpathy's micrograd) on purpose: every multiply and add
// is visible, so the machinery can be drawn rather than hidden behind a tensor.

export class Value {
  constructor(data, children = [], op = '') {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = children;
    this.op = op;
  }

  add(other) {
    other = toValue(other);
    const out = new Value(this.data + other.data, [this, other], '+');
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    other = toValue(other);
    const out = new Value(this.data * other.data, [this, other], '*');
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  pow(k) {
    // k is a plain number (constant exponent).
    const out = new Value(Math.pow(this.data, k), [this], `**${k}`);
    out._backward = () => {
      this.grad += k * Math.pow(this.data, k - 1) * out.grad;
    };
    return out;
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    return this.add(toValue(other).neg());
  }

  div(other) {
    return this.mul(toValue(other).pow(-1));
  }

  exp() {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out._backward = () => {
      this.grad += e * out.grad; // d/dx e^x = e^x
    };
    return out;
  }

  log() {
    const out = new Value(Math.log(this.data), [this], 'log');
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad; // 1 - tanh^2
    };
    return out;
  }

  relu() {
    const r = this.data < 0 ? 0 : this.data;
    const out = new Value(r, [this], 'relu');
    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  sigmoid() {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], 'σ');
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  // Reverse-topological walk: seed the output's grad with 1, then let every node
  // push its gradient to its inputs exactly once, parents before children.
  backward() {
    const topo = [];
    const visited = new Set();
    const build = (v) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._prev) build(child);
      topo.push(v);
    };
    build(this);

    for (const v of topo) v.grad = 0;
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

export const toValue = (x) => (x instanceof Value ? x : new Value(x));
export const V = (x) => new Value(x);

// Numerically stable softmax over an array of Values → array of Values.
export function softmax(logits) {
  const maxData = Math.max(...logits.map((l) => l.data));
  const shifted = logits.map((l) => l.sub(maxData).exp());
  let denom = shifted[0];
  for (let i = 1; i < shifted.length; i++) denom = denom.add(shifted[i]);
  return shifted.map((e) => e.div(denom));
}

// Cross-entropy loss for a softmax distribution `probs` against an integer label.
export function crossEntropy(probs, label) {
  return probs[label].log().neg();
}

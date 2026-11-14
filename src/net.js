// The network and its gradient, written out the way the essay derives them.
//
// A layer computes z = W a + b and a' = σ(z). The cost for one example is
// Σ_j (a_j − y_j)², averaged over the training set to give C. Backpropagation
// is the chain rule run from the output back to the input:
//
//   δ(out)  = ∂C/∂z(out)  = 2 (a − y) ⊙ σ'(z)
//   ∂C/∂W   = δ a(prev)ᵀ          ∂C/∂b = δ
//   δ(prev) = (Wᵀ δ) ⊙ σ'(z(prev))
//
// with σ'(z) = σ(z)(1 − σ(z)). Every weight and bias lives in one flat
// Float64Array θ, so ∇C is a vector the same length as θ and a step of
// gradient descent is θ ← θ − η ∇C, one subtraction per number.

export function makeNet(sizes) {
  const layers = [];
  let off = 0;
  for (let l = 0; l < sizes.length - 1; l++) {
    const nin = sizes[l];
    const nout = sizes[l + 1];
    layers.push({ nin, nout, w: off, b: off + nin * nout });
    off += nin * nout + nout;
  }
  const widest = Math.max(...sizes);
  return { sizes, layers, nParams: off, widest };
}

// mulberry32: a small seeded generator, so a seed always rebuilds the same run.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Random starting weights, uniform in ±√(6 / (fan-in + fan-out)) (Glorot),
// biases zero. This is the "complete trash" network the essay starts from.
export function initParams(net, seed) {
  const r = rng(seed);
  const th = new Float64Array(net.nParams);
  for (const L of net.layers) {
    const s = Math.sqrt(6 / (L.nin + L.nout));
    for (let i = 0; i < L.nin * L.nout; i++) th[L.w + i] = (r() * 2 - 1) * s;
  }
  return th;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Scratch buffers per network, so the hot loop allocates nothing.
function scratch(net) {
  if (!net._s) {
    net._s = {
      a: net.sizes.map((n) => new Float64Array(n)),
      d: net.sizes.map((n) => new Float64Array(n)),
    };
  }
  return net._s;
}

// Forward one input (x, y) through θ. Leaves every layer's activations in
// the scratch buffers and returns the output layer.
export function forward(net, th, x, y) {
  const { a } = scratch(net);
  a[0][0] = x;
  a[0][1] = y;
  for (let l = 0; l < net.layers.length; l++) {
    const L = net.layers[l];
    const src = a[l];
    const dst = a[l + 1];
    for (let j = 0; j < L.nout; j++) {
      let z = th[L.b + j];
      const row = L.w + j * L.nin;
      for (let k = 0; k < L.nin; k++) z += th[row + k] * src[k];
      dst[j] = sigmoid(z);
    }
  }
  return a[net.sizes.length - 1];
}

// C(θ) over a dataset {xs, ys, labels, n}: mean over examples of Σ_j (a_j − y_j)².
export function cost(net, th, data) {
  let total = 0;
  const nOut = net.sizes[net.sizes.length - 1];
  for (let i = 0; i < data.n; i++) {
    const out = forward(net, th, data.xs[i], data.ys[i]);
    const lab = data.labels[i];
    for (let j = 0; j < nOut; j++) {
      const e = out[j] - (j === lab ? 1 : 0);
      total += e * e;
    }
  }
  return data.n ? total / data.n : 0;
}

// C(θ) and ∇C(θ), the full-batch gradient written into `grad`.
export function costAndGrad(net, th, data, grad) {
  const { a, d } = scratch(net);
  const nL = net.layers.length;
  const nOut = net.sizes[nL];
  grad.fill(0);
  let total = 0;
  for (let i = 0; i < data.n; i++) {
    forward(net, th, data.xs[i], data.ys[i]);
    const lab = data.labels[i];
    // Output layer: δ = 2 (a − y) σ'(z), and σ'(z) = a (1 − a).
    const out = a[nL];
    const dOut = d[nL];
    for (let j = 0; j < nOut; j++) {
      const e = out[j] - (j === lab ? 1 : 0);
      total += e * e;
      dOut[j] = 2 * e * out[j] * (1 - out[j]);
    }
    // Walk back one layer at a time.
    for (let l = nL - 1; l >= 0; l--) {
      const L = net.layers[l];
      const dl = d[l + 1];
      const prev = a[l];
      for (let j = 0; j < L.nout; j++) {
        const dj = dl[j];
        grad[L.b + j] += dj;
        const row = L.w + j * L.nin;
        for (let k = 0; k < L.nin; k++) grad[row + k] += dj * prev[k];
      }
      if (l > 0) {
        const dp = d[l];
        for (let k = 0; k < L.nin; k++) {
          let s = 0;
          for (let j = 0; j < L.nout; j++) s += th[L.w + j * L.nin + k] * dl[j];
          dp[k] = s * prev[k] * (1 - prev[k]);
        }
      }
    }
  }
  const inv = data.n ? 1 / data.n : 0;
  for (let p = 0; p < grad.length; p++) grad[p] *= inv;
  return total * inv;
}

// One input followed all the way through and, if it has a label, all the way
// back: every z and a, and for a labelled dot the derivative of its own cost
// C₀ = Σ_j (a_j − y_j)² with respect to every activation, ∂C₀/∂a. Layer 0 is
// the input. Used for the drawing of the network and for its equations.
export function probe(net, th, x, y, label = -1) {
  const nL = net.layers.length;
  const a = [Float64Array.of(x, y)];
  const z = [null];
  for (let l = 0; l < nL; l++) {
    const L = net.layers[l];
    const zl = new Float64Array(L.nout);
    const al = new Float64Array(L.nout);
    for (let j = 0; j < L.nout; j++) {
      let s = th[L.b + j];
      for (let k = 0; k < L.nin; k++) s += th[L.w + j * L.nin + k] * a[l][k];
      zl[j] = s;
      al[j] = sigmoid(s);
    }
    z.push(zl);
    a.push(al);
  }
  const out = { a, z, label, C0: null, dCda: null };
  if (label < 0) return out;
  // ∂C₀/∂a(O) = 2 (a − y), then back one layer at a time:
  // ∂C₀/∂a_k(l−1) = Σ_j w_jk(l) σ′(z_j(l)) ∂C₀/∂a_j(l).
  const dCda = a.map((al) => new Float64Array(al.length));
  let C0 = 0;
  for (let j = 0; j < a[nL].length; j++) {
    const e = a[nL][j] - (j === label ? 1 : 0);
    C0 += e * e;
    dCda[nL][j] = 2 * e;
  }
  for (let l = nL; l >= 1; l--) {
    const L = net.layers[l - 1];
    for (let k = 0; k < L.nin; k++) {
      let s = 0;
      for (let j = 0; j < L.nout; j++) {
        const sp = a[l][j] * (1 - a[l][j]);
        s += th[L.w + j * L.nin + k] * sp * dCda[l][j];
      }
      dCda[l - 1][k] = s;
    }
  }
  out.C0 = C0;
  out.dCda = dCda;
  return out;
}

// One dot's share of ∂C/∂w for the weight from neuron k of layer l−1 to
// neuron j of layer l (k = −1 means the bias b_j), as the essay's chain:
//   ∂z_j/∂w · ∂a_j/∂z_j · ∂C₀/∂a_j = a_k(l−1) · σ′(z_j(l)) · ∂C₀/∂a_j(l).
export function chainTerm(p, l, j, k) {
  const sp = p.a[l][j] * (1 - p.a[l][j]);
  const dz = k < 0 ? 1 : p.a[l - 1][k];
  return { dz, sp, dCda: p.dCda[l][j], value: dz * sp * p.dCda[l][j] };
}

// ∂C/∂w as the average of that chain over every training dot. A separate
// route to the number costAndGrad finds; validate/run.mjs checks they agree.
export function chainGrad(net, th, data, l, j, k) {
  if (!data.n) return 0;
  let s = 0;
  for (let i = 0; i < data.n; i++) s += chainTerm(probe(net, th, data.xs[i], data.ys[i], data.labels[i]), l, j, k).value;
  return s / data.n;
}

// Where weight (l, j, k) lives in θ; k = −1 is the bias.
export function paramIndex(net, l, j, k) {
  const L = net.layers[l - 1];
  return k < 0 ? L.b + j : L.w + j * L.nin + k;
}

// Fraction of examples whose brightest output neuron is the right one.
export function accuracy(net, th, data) {
  if (!data.n) return 1;
  let ok = 0;
  for (let i = 0; i < data.n; i++) {
    const out = forward(net, th, data.xs[i], data.ys[i]);
    let best = 0;
    for (let j = 1; j < out.length; j++) if (out[j] > out[best]) best = j;
    if (best === data.labels[i]) ok++;
  }
  return ok / data.n;
}

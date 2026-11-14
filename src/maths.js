// The equations under the network, with the live numbers in them, in the
// essay's notation: a for activations, w and b for weights and biases, z for
// the weighted sum, C₀ for one dot's cost, C for the average over every dot,
// ∇C for the gradient. Layers are (0) for the input, (1) and (2) for the two
// hidden layers and (O) for the output, as in the essay.

import { v, num, factor, frac, sum, d, sigma, sigmaPrime, row, op } from './tex.js';
import { chainTerm, chainGrad, paramIndex } from './net.js';

const lname = (l, nL) => (l === nL ? 'O' : String(l));
const GROUP = ['g0', 'g1', 'g2'];

// `ctx`: { net, th, data, p (a probe), eta, C, target }
export function renderMaths(ctx) {
  const { target } = ctx;
  if (!target) return overview(ctx);
  if (target.type === 'node') return target.l === 0 ? inputNode(ctx) : neuron(ctx);
  return weight(ctx);
}

function overview({ net, p, C, eta, data, touch }) {
  const nL = net.layers.length;
  const out = p.a[nL];
  const whole = `${v('a', '', 'O')} ${op('=')} ${sigma(`${v('W', '', 'O')} ${sigma(`${v('W', '', '2')} ${sigma(`${v('W', '', '1')}${v('a', '', '0')} ${op('+')} ${v('b', '', '1')}`)} ${op('+')} ${v('b', '', '2')}`)} ${op('+')} ${v('b', '', 'O')}`)}`;
  const io =
    `${v('a', '', '0')} ${op('=')} (${num(p.a[0][0])}, ${num(p.a[0][1])}) ${op('→')} ` +
    `${v('a', '', 'O')} ${op('=')} (${[...out].map((x, j) => `<span class="${GROUP[j]}">${num(x)}</span>`).join(', ')})`;
  const rows = [row(whole, 'eq--lead'), row(io)];
  if (p.label >= 0) {
    const terms = [...out].map((x, j) => `(${num(x)} ${op('−')} ${j === p.label ? 1 : 0})<sup>2</sup>`).join(` ${op('+')} `);
    rows.push(row(`${v('C', '0')} ${op('=')} ${terms} ${op('=')} ${num(p.C0, 3)}`));
  }
  if (!data.n) return rows.join('');
  rows.push(row(`${v('C')} ${op('=')} ${frac('1', v('n'))}${sum('')}${v('C', '0')} ${op('=')} ${num(C, 3)}<span class="aside">over all ${data.n} dots</span>`));
  rows.push(row(`${v('w')} ${op('←')} ${v('w')} ${op('−')} ${v('η')} ${d('C', v('w'))}<span class="aside">for every <i>w</i> and <i>b</i> at once, <i>η</i> = ${eta}</span>`, 'eq--tall'));
  rows.push(row(`<span class="aside aside--solo">${touch ? 'Tap' : 'Point at'} a neuron or a line for its own equation.</span>`, 'eq--hint'));
  return rows.join('');
}

function inputNode({ target, p }) {
  const name = target.j === 0 ? 'x' : 'y';
  return [
    row(`${v('a', String(target.j), '0')} ${op('=')} ${v(name)} ${op('=')} ${num(p.a[0][target.j])}`, 'eq--lead'),
    row(`<span class="aside aside--solo">The ${name} coordinate of the point being fed in.</span>`),
  ].join('');
}

function neuron({ net, th, data, p, eta, target }) {
  const nL = net.layers.length;
  const { l, j } = target;
  const L = net.layers[l - 1];
  const ln = lname(l, nL);
  const lp = lname(l - 1, nL);
  const z = p.z[l][j];
  const a = p.a[l][j];
  const rows = [];
  rows.push(row(`${v('a', j, ln)} ${op('=')} ${sigma(v('z', j, ln))} ${op('=')} ${sigma(num(z))} ${op('=')} ${num(a)}`, 'eq--lead'));

  // z as a sum, with its biggest terms written out.
  const terms = [];
  for (let k = 0; k < L.nin; k++) terms.push({ k, w: th[L.w + j * L.nin + k], a: p.a[l - 1][k] });
  const shown = L.nin <= 3 ? terms : [...terms].sort((s, t) => Math.abs(t.w * t.a) - Math.abs(s.w * s.a)).slice(0, 2);
  const written = shown.map((t) => `${factor(t.w)}${factor(t.a)}`).join(` ${op('+')} `);
  const rest = L.nin > shown.length ? ` ${op('+')} ⋯` : '';
  rows.push(row(`${v('z', j, ln)} ${op('=')} ${sum('k')}${v('w', `${j},k`, ln)}${v('a', 'k', lp)} ${op('+')} ${v('b', j, ln)}`));
  rows.push(row(`${op('=')} ${written}${rest} ${op('+')} ${factor(th[L.b + j])} ${op('=')} ${num(z)}`, 'eq--cont'));

  if (p.label >= 0) {
    if (l === nL) {
      const y = j === p.label ? 1 : 0;
      rows.push(row(`${d(v('C', '0'), v('a', j, ln))} ${op('=')} 2(${v('a', j, ln)} ${op('−')} ${v('y', j)}) ${op('=')} 2(${num(a)} ${op('−')} ${y}) ${op('=')} ${num(p.dCda[l][j], 3)}`));
    } else {
      const ln1 = lname(l + 1, nL);
      rows.push(row(`${d(v('C', '0'), v('a', j, ln))} ${op('=')} ${sum('i')}${v('w', `i,${j}`, ln1)}${sigmaPrime(v('z', 'i', ln1))}${d(v('C', '0'), v('a', 'i', ln1))} ${op('=')} ${num(p.dCda[l][j], 3)}`));
    }
  }
  const g = chainGrad(net, th, data, l, j, -1);
  const b = th[paramIndex(net, l, j, -1)];
  rows.push(row(`${d('C', v('b', j, ln))} ${op('=')} ${frac('1', v('n'))}${sum('')}${sigmaPrime(v('z', j, ln))}${d(v('C', '0'), v('a', j, ln))} ${op('=')} ${num(g, 4)}`));
  rows.push(row(`${v('b', j, ln)} ${op('←')} ${num(b)} ${op('−')} ${eta} ${op('×')} ${factor(g, 4)} ${op('=')} ${num(b - eta * g, 3)}`));
  return rows.join('');
}

function weight({ net, th, data, p, eta, target }) {
  const nL = net.layers.length;
  const { l, j, k } = target;
  const ln = lname(l, nL);
  const lp = lname(l - 1, nL);
  const w = th[paramIndex(net, l, j, k)];
  const wv = v('w', `${j},${k}`, ln);
  const g = chainGrad(net, th, data, l, j, k);
  const rows = [];
  rows.push(row(`${wv} ${op('=')} ${num(w)}<span class="aside">from ${v('a', k, lp)} to ${v('a', j, ln)}</span>`, 'eq--lead'));
  rows.push(row(`${d('C', wv)} ${op('=')} ${frac('1', v('n'))}${sum('')}${d(v('z', j, ln), wv)}${d(v('a', j, ln), v('z', j, ln))}${d(v('C', '0'), v('a', j, ln))}`, 'eq--tall'));
  rows.push(row(`${op('=')} ${frac('1', v('n'))}${sum('')}${v('a', k, lp)}${sigmaPrime(v('z', j, ln))}${d(v('C', '0'), v('a', j, ln))} ${op('=')} ${num(g, 4)}`, 'eq--cont eq--tall'));
  if (p.label >= 0) {
    const t = chainTerm(p, l, j, k);
    rows.push(row(`<span class="aside aside--solo">For the dot under the crosshair alone:</span>`, 'eq--note'));
    rows.push(row(`${d(v('C', '0'), wv)} ${op('=')} ${num(t.dz)} ${op('×')} ${num(t.sp)} ${op('×')} ${factor(t.dCda, 3)} ${op('=')} ${num(t.value, 4)}`));
  }
  rows.push(row(`${wv} ${op('←')} ${num(w)} ${op('−')} ${eta} ${op('×')} ${factor(g, 4)} ${op('=')} ${num(w - eta * g, 3)}`));
  return rows.join('');
}

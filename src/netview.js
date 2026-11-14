// The network itself, drawn as a textbook figure: inputs on the left, two
// hidden layers of sixteen, three outputs on the right. Every weight is a
// line, ink for positive and blue for negative, heavier the bigger it is.
// Neurons fill with ink as they activate for the point being fed in, which
// arrives as a wave from left to right. While it trains, ∇C runs back from
// the outputs as a red pulse, each line glowing in proportion to how hard
// the gradient is pulling on that weight.

import { GROUPS } from './territory.js';

const INK = [42, 36, 30];
const POS = [52, 42, 34];
const NEG = [56, 98, 150];
const RED = [200, 48, 42];
const PAPER = [246, 241, 231];
const rgba = (c, a) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
const smooth = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

export function makeNetView(canvas, net) {
  const ctx = canvas.getContext('2d');
  const sizes = net.sizes;
  const nL = sizes.length - 1;
  let W = 0;
  let H = 0;
  let dpr = 1;
  let pos = [];
  let r = 6;

  function resize(cssW, cssH, ratio) {
    dpr = ratio;
    W = Math.round(cssW * ratio);
    H = Math.round(cssH * ratio);
    canvas.width = W;
    canvas.height = H;
    const left = 44 * dpr;
    const right = 48 * dpr;
    const top = 10 * dpr;
    const bottom = 34 * dpr;
    const span = H - top - bottom;
    const xs = [0, 0.36, 0.7, 1].map((f) => left + f * (W - left - right));
    const widest = Math.max(...sizes);
    const pitch = span / widest;
    r = Math.max(3.2 * dpr, Math.min(8.5 * dpr, pitch * 0.34));
    pos = sizes.map((n, l) => {
      // Small layers spread out a little, but stay near the middle.
      const gap = n === widest ? pitch : Math.min(span / n, pitch * 3.2);
      const y0 = top + span / 2 - (gap * (n - 1)) / 2;
      return Array.from({ length: n }, (_, j) => [xs[l], y0 + gap * j]);
    });
  }

  // The edge from neuron k of layer l−1 to neuron j of layer l.
  const ends = (l, j, k) => [pos[l - 1][k], pos[l][j]];

  // Lines are gathered into a few dozen batches of nearly equal colour and
  // width, so a frame is a few dozen strokes rather than three hundred.
  const batches = new Map();
  function line(colour, alpha, width, [[x0, y0], [x1, y1]]) {
    const a = Math.round(Math.min(1, alpha) * 24) / 24;
    if (a <= 0) return;
    const wq = Math.round(width * 4) / 4;
    const key = `${colour.join(',')},${a}|${wq}`;
    let path = batches.get(key);
    if (!path) batches.set(key, (path = new Path2D()));
    path.moveTo(x0, y0);
    path.lineTo(x1, y1);
  }
  function flush() {
    for (const [key, path] of batches) {
      const [c, wq] = key.split('|');
      ctx.strokeStyle = `rgba(${c})`;
      ctx.lineWidth = +wq;
      ctx.stroke(path);
    }
    batches.clear();
  }

  // `s`: { th, p (probe), from (the probe before it moved, or null),
  //        probeAge (ms since it moved), grad, gscale,
  //        pulse (0..1 front position, or null), hover }
  function draw(s) {
    if (!W) return;
    ctx.clearRect(0, 0, W, H);
    const { th, p, from, grad, gscale, pulse, hover } = s;
    const reveal = (l) => (p ? smooth((s.probeAge - 110 * l) / 220) : 0);
    // Each layer changes over from the old point's values to the new one's
    // as the signal reaches it.
    const act = (l, j) => {
      const rv = reveal(l);
      const old = from ? from.a[l][j] : 0;
      return old + (p.a[l][j] - old) * rv;
    };

    // Weights.
    ctx.lineCap = 'round';
    for (let l = 1; l <= nL; l++) {
      const L = net.layers[l - 1];
      let wmax = 1e-9;
      for (let i = 0; i < L.nin * L.nout; i++) wmax = Math.max(wmax, Math.abs(th[L.w + i]));
      // How much each line is carrying for the current point, |w a|.
      let cmax = 1e-9;
      const prev = p ? Array.from({ length: L.nin }, (_, k) => act(l - 1, k)) : null;
      if (p) for (let j = 0; j < L.nout; j++) for (let k = 0; k < L.nin; k++) cmax = Math.max(cmax, Math.abs(th[L.w + j * L.nin + k] * prev[k]));
      // The signal is on these lines while the layer before has it and the
      // layer after does not yet.
      const travel = p ? Math.max(0, reveal(l - 1) - reveal(l)) : 0;
      for (let j = 0; j < L.nout; j++) {
        for (let k = 0; k < L.nin; k++) {
          const w = th[L.w + j * L.nin + k];
          const m = Math.abs(w) / wmax;
          let alpha = 0.1 + 0.5 * m;
          let carry = 0;
          if (p) {
            carry = Math.abs(w * prev[k]) / cmax;
            alpha *= 0.18 + 0.82 * carry;
            alpha = Math.min(1, alpha + travel * carry * 0.8);
          }
          line(w >= 0 ? POS : NEG, alpha, (0.3 + 1.9 * m + travel * carry * 1.6) * dpr, ends(l, j, k));
        }
      }
    }
    flush();

    // ∇C, running backwards.
    if (pulse !== null && grad && gscale > 0) {
      const xin = pos[0][0][0];
      const xout = pos[nL][0][0];
      const front = xout + (xin - xout) * pulse;
      const width = 0.2 * (xout - xin);
      for (let l = 1; l <= nL; l++) {
        const L = net.layers[l - 1];
        const xm = (pos[l - 1][0][0] + pos[l][0][0]) / 2;
        const glow = Math.exp(-(((xm - front) / width) ** 2));
        if (glow < 0.02) continue;
        for (let j = 0; j < L.nout; j++) {
          for (let k = 0; k < L.nin; k++) {
            const g = Math.abs(grad[L.w + j * L.nin + k]);
            const strength = Math.min(1, g / gscale) ** 0.4;
            const a = glow * strength;
            if (a < 0.03) continue;
            line(RED, 0.75 * a, (0.6 + 2.4 * strength) * dpr, ends(l, j, k));
          }
        }
      }
      flush();
    }

    // The line being pointed at.
    if (hover?.type === 'edge') {
      const [[x0, y0], [x1, y1]] = ends(hover.l, hover.j, hover.k);
      ctx.strokeStyle = rgba(PAPER, 0.9);
      ctx.lineWidth = 5 * dpr;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.strokeStyle = rgba(RED, 1);
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
    }

    // Neurons.
    for (let l = 0; l <= nL; l++) {
      pos[l].forEach(([x, y], j) => {
        let fill = PAPER;
        if (p && l > 0) {
          const a = act(l, j);
          const base = l === nL ? hex(GROUPS[j].deep) : INK;
          const k = l === nL ? a : a ** 0.85 * 0.92;
          fill = PAPER.map((c, q) => c + (base[q] - c) * k);
        }
        ctx.beginPath();
        ctx.arc(x, y, l === 0 || l === nL ? r * 1.25 : r, 0, Math.PI * 2);
        ctx.fillStyle = rgba(fill, 1);
        ctx.fill();
        const hot = hover?.type === 'node' && hover.l === l && hover.j === j;
        ctx.lineWidth = (hot ? 2.2 : 1) * dpr;
        ctx.strokeStyle = hot ? rgba(RED, 1) : rgba(INK, 0.9);
        ctx.stroke();
      });
    }

    // Labels: the input values on the left, the three answers on the right,
    // the layer names underneath.
    const fs = Math.round(11.5 * dpr);
    ctx.textBaseline = 'middle';
    ctx.fillStyle = rgba(INK, 0.95);
    ctx.font = `italic ${fs}px "Text", Georgia, serif`;
    ctx.textAlign = 'right';
    ['x', 'y'].forEach((name, j) => {
      const [x, y] = pos[0][j];
      ctx.fillText(name, x - r * 1.25 - 6 * dpr, y - fs * 0.6);
      if (p) {
        ctx.font = `${fs}px "Text", Georgia, serif`;
        ctx.fillText(fmt(p.a[0][j]), x - r * 1.25 - 6 * dpr, y + fs * 0.65);
        ctx.font = `italic ${fs}px "Text", Georgia, serif`;
      }
    });
    ctx.textAlign = 'left';
    ctx.font = `${fs}px "Text", Georgia, serif`;
    pos[nL].forEach(([x, y], j) => {
      if (!p) return;
      ctx.fillStyle = GROUPS[j].deep;
      ctx.fillText(fmt(act(nL, j)), x + r * 1.25 + 6 * dpr, y);
    });
    ctx.fillStyle = rgba(INK, 0.75);
    ['0', '1', '2', 'O'].forEach((name, l) => {
      supLabel(ctx, 'a', `(${name})`, pos[l][0][0], H - 12 * dpr, fs, dpr);
    });
  }

  function hit(cx, cy) {
    const x = cx * dpr;
    const y = cy * dpr;
    for (let l = 0; l <= nL; l++) {
      for (let j = 0; j < pos[l].length; j++) {
        const [px, py] = pos[l][j];
        if (Math.hypot(px - x, py - y) <= r * 1.3 + 3 * dpr) return { type: 'node', l, j };
      }
    }
    let best = null;
    let bestD = 4.5 * dpr;
    for (let l = 1; l <= nL; l++) {
      const x0 = pos[l - 1][0][0];
      const x1 = pos[l][0][0];
      if (x < x0 || x > x1) continue;
      const L = net.layers[l - 1];
      for (let j = 0; j < L.nout; j++) {
        for (let k = 0; k < L.nin; k++) {
          const dd = segDist(x, y, pos[l - 1][k], pos[l][j]);
          if (dd < bestD) {
            bestD = dd;
            best = { type: 'edge', l, j, k };
          }
        }
      }
    }
    return best;
  }

  return { resize, draw, hit };
}

function hex(s) {
  return [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
}

function fmt(x) {
  return (x < 0 ? '−' : '') + Math.abs(x).toFixed(2);
}

function supLabel(ctx, base, sup, x, y, fs, dpr) {
  ctx.font = `italic ${fs}px "Text", Georgia, serif`;
  const bw = ctx.measureText(base).width;
  ctx.font = `${Math.round(fs * 0.72)}px "Text", Georgia, serif`;
  const sw = ctx.measureText(sup).width;
  const x0 = x - (bw + sw + dpr) / 2;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.font = `italic ${fs}px "Text", Georgia, serif`;
  ctx.fillText(base, x0, y);
  ctx.font = `${Math.round(fs * 0.72)}px "Text", Georgia, serif`;
  ctx.fillText(sup, x0 + bw + dpr, y - fs * 0.42);
}

function segDist(x, y, [x0, y0], [x1, y1]) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const t = Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(x - x0 - t * dx, y - y0 - t * dy);
}

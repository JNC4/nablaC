// nablaC: the NETWORK view.
//
// The net drawn as what it is: columns of neurons, every pair joined by a
// weight. A weight's *value* is its resting look: warmth/coolness for sign,
// thickness for magnitude. A weight's *gradient* (its entry in ∇C) is the one
// thing that glows: a cyan wavefront floods backward through the graph on every
// heartbeat, brightest where the cost most wants that weight to move.
//
// forward sweep (→): neurons light up as signal flows in.
// backward sweep (←): ∇C floods out. this is the whole show.

const NS = 'http://www.w3.org/2000/svg';
const POS = '#d4a05f'; // excitatory weight (w > 0), warm
const NEG = '#6f93c4'; // inhibitory weight (w < 0), cool
const ACT = '#f1ead6'; // activation glow, bone-white
const GRAD = '#46d6ff'; // ∇C, the one electric accent

const el = (tag, attrs = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
};
const bump = (d, w) => Math.exp(-((d / w) ** 2)); // gaussian wavefront kernel

export function createNetworkView(svg) {
  const gEdge = el('g', {});
  const gGrad = el('g', {});
  const gNode = el('g', {});
  const gLabel = el('g', {});
  svg.append(gEdge, gGrad, gNode, gLabel);

  let net = null;
  let cols = []; // [{x, nodes:[{y, neuron, layerIdx, idx}]}]
  let edges = []; // {line, glow, from, to, srcCol, dstCol, w:weightValue}
  let nodes = []; // {circle, halo, neuron, col}
  let W = 1000, H = 600;

  function size() {
    const r = svg.getBoundingClientRect();
    W = Math.max(320, r.width);
    H = Math.max(240, r.height);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  }

  function build(network) {
    net = network;
    [gEdge, gGrad, gNode, gLabel].forEach((g) => (g.textContent = ''));
    cols = []; edges = []; nodes = [];
    size();

    const sizes = net.sizes;
    const nCols = sizes.length;
    const padX = 86, padY = 48;
    const colGap = (W - padX * 2) / (nCols - 1);

    // node positions, column by column
    for (let c = 0; c < nCols; c++) {
      const count = sizes[c];
      const x = padX + c * colGap;
      const usable = H - padY * 2;
      const gap = count > 1 ? usable / (count - 1) : 0;
      const y0 = count > 1 ? padY : H / 2;
      const colNodes = [];
      for (let k = 0; k < count; k++) {
        const y = count > 1 ? y0 + k * gap : y0;
        colNodes.push({ x, y, col: c, idx: k });
      }
      cols.push({ x, nodes: colNodes });
    }

    // edges: layer L's neuron weights connect column L (inputs) → column L+1
    net.layers.forEach((layer, li) => {
      layer.neurons.forEach((neuron, ni) => {
        const dst = cols[li + 1].nodes[ni];
        neuron.w.forEach((wv, wi) => {
          const src = cols[li].nodes[wi];
          const line = el('line', { x1: src.x, y1: src.y, x2: dst.x, y2: dst.y, class: 'edge' });
          const glow = el('line', { x1: src.x, y1: src.y, x2: dst.x, y2: dst.y, class: 'edge-grad' });
          gEdge.append(line);
          gGrad.append(glow);
          edges.push({ line, glow, w: wv, srcCol: li, dstCol: li + 1, neuron });
        });
      });
    });

    // nodes (drawn over edges) + halos for activation glow
    cols.forEach((col, c) => {
      col.nodes.forEach((nd) => {
        const neuron = c === 0 ? null : net.layers[c - 1].neurons[nd.idx];
        const halo = el('circle', { cx: nd.x, cy: nd.y, r: 18, class: 'halo' });
        const r = c === 0 || c === cols.length - 1 ? 9 : 7;
        const circle = el('circle', { cx: nd.x, cy: nd.y, r, class: 'node' });
        gNode.append(halo, circle);
        nodes.push({ circle, halo, neuron, col: c, x: nd.x, y: nd.y, idx: nd.idx });
      });
    });

    // column captions
    const caps = ['input', ...sizes.slice(1, -1).map(() => 'hidden'), 'output'];
    cols.forEach((col, c) => {
      const t = el('text', { x: col.x, y: H - 16, class: 'col-cap' });
      t.textContent = caps[c];
      gLabel.append(t);
    });
    // input / output token labels
    const inLabels = ['x', 'y'];
    cols[0].nodes.forEach((nd, k) => {
      const t = el('text', { x: nd.x - 18, y: nd.y + 4, class: 'io-lab' });
      t.textContent = inLabels[k] ?? '';
      gLabel.append(t);
    });
    const outLabels = ['A', 'B'];
    cols[cols.length - 1].nodes.forEach((nd, k) => {
      const t = el('text', { x: nd.x + 18, y: nd.y + 4, class: 'io-lab out' });
      t.textContent = outLabels[k] ?? '';
      gLabel.append(t);
    });

    render(0);
  }

  // phase ∈ [0,2): [0,1) forward wavefront, [1,2) backward (∇C) wavefront.
  function render(phase) {
    if (!net) return;
    const nCols = cols.length;
    const fwd = phase < 1;
    const front = fwd ? phase * (nCols - 1) : (2 - phase) * (nCols - 1);

    // edges: resting look from weight value; cyan overlay from |grad| on backward
    for (const e of edges) {
      const w = e.w.data;
      const aw = Math.abs(w);
      const width = 0.35 + Math.min(aw, 3) * 1.15;
      e.line.setAttribute('stroke', w >= 0 ? POS : NEG);
      e.line.setAttribute('stroke-width', width.toFixed(2));
      e.line.setAttribute('stroke-opacity', (0.12 + Math.min(aw, 2) * 0.32).toFixed(3));

      // gradient overlay: brightness follows the backward wavefront at dstCol
      const g = Math.abs(e.w.grad);
      let glow = 0;
      if (!fwd) glow = bump(front - e.dstCol, 0.6) * Math.min(1, g * 14);
      e.glow.setAttribute('stroke', GRAD);
      e.glow.setAttribute('stroke-width', (0.6 + Math.min(g * 30, 6)).toFixed(2));
      e.glow.setAttribute('stroke-opacity', (glow * 0.9).toFixed(3));
    }

    // nodes: warm activation glow on the forward wavefront
    for (const n of nodes) {
      const a = n.neuron ? n.neuron.act : 0;
      const mag = Math.min(1, Math.abs(a));
      let lit = fwd ? bump(front - n.col, 0.55) * mag : 0.18 * mag;
      n.halo.setAttribute('r', (10 + lit * 22).toFixed(1));
      n.halo.setAttribute('fill', ACT);
      n.halo.setAttribute('fill-opacity', (lit * 0.5).toFixed(3));
      // resting node fill nudges warm/cool with activation sign
      const f = n.col === 0 ? 0.5 : 0.5 + 0.5 * Math.tanh(a);
      n.circle.setAttribute('fill-opacity', (0.25 + f * 0.6).toFixed(3));
    }
  }

  return { build, render, resize: size, get cols() { return cols; } };
}

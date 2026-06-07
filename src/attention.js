// nablaC: the ATTENTION view, the top of the climb.
//
// The MLP is the whole machine in miniature. Stack it and add one more idea (let
// every token decide which other tokens to look at) and you get the
// transformer, the architecture under modern language models. This view is a
// forward pass of one self-attention block: each token is projected to a
// query, key and value; attention(i,j) = softmax(Qᵢ·Kⱼ / √d). It isn't trained,
// the projections are fixed; it's here to *show the operation*, the routing of
// information from token to token, the same softmax the MLP used, now deciding
// who attends to whom.

const ACCENT = '#46d6ff';
const TOKENS = ['the', 'blind', 'god', 'learns', 'to', 'attend']; // a nod to YALDABAOTH
const D_MODEL = 12;
const HEADS = 3;
const D_K = 6;

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hash = (s) => { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };

function vec(n, r) { return Array.from({ length: n }, () => r() * 2 - 1); }
function mat(rows, cols, r) { return Array.from({ length: rows }, () => vec(cols, r)); }
function matvec(M, v) { return M.map((row) => row.reduce((s, x, i) => s + x * v[i], 0)); }
function dot(a, b) { return a.reduce((s, x, i) => s + x * b[i], 0); }
function softmax(xs) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const z = e.reduce((s, x) => s + x, 0);
  return e.map((x) => x / z);
}

// fixed embeddings + per-head projections → attention matrices
function computeHeads() {
  const E = TOKENS.map((t) => vec(D_MODEL, rng(hash(t))));
  const heads = [];
  for (let h = 0; h < HEADS; h++) {
    const rh = rng(1000 + h * 7);
    const Wq = mat(D_K, D_MODEL, rh), Wk = mat(D_K, D_MODEL, rh), Wv = mat(D_K, D_MODEL, rh);
    const Q = E.map((e) => matvec(Wq, e));
    const K = E.map((e) => matvec(Wk, e));
    const V = E.map((e) => matvec(Wv, e));
    const attn = Q.map((qi) => softmax(K.map((kj) => dot(qi, kj) / Math.sqrt(D_K))));
    heads.push({ attn, V });
  }
  return heads;
}

export function createAttentionView(root) {
  let heads = null;
  let head = 0;
  let query = 0;

  function build() {
    if (!heads) heads = computeHeads();
    render();
  }

  function cell(p) {
    const a = 0.06 + 0.9 * p;
    return `rgba(70,214,255,${a.toFixed(3)})`;
  }

  function render() {
    const attn = heads[head].attn;
    const n = TOKENS.length;

    let html = `
      <div class="att-head">
        <h3>self-attention · <span class="ax">softmax(Q·Kᵀ / √d)·V</span></h3>
        <p>each token (a query) decides how much to read from every token (a key). the same softmax the network used to choose a class now routes information between words. fixed projections, a forward pass, not trained.</p>
        <div class="head-tabs">${heads.map((_, i) => `<button class="head-tab${i === head ? ' on' : ''}" data-h="${i}">head ${i + 1}</button>`).join('')}</div>
      </div>
      <div class="att-grid">
        <table class="attn-matrix"><thead><tr><th class="corner">Q＼K</th>${TOKENS.map((t) => `<th>${t}</th>`).join('')}</tr></thead><tbody>`;
    for (let i = 0; i < n; i++) {
      html += `<tr class="qrow${i === query ? ' sel' : ''}" data-q="${i}"><th>${TOKENS[i]}</th>`;
      for (let j = 0; j < n; j++) {
        const p = attn[i][j];
        html += `<td style="background:${cell(p)}" title="${TOKENS[i]} → ${TOKENS[j]} = ${p.toFixed(2)}">${p >= 0.18 ? p.toFixed(2).replace(/^0/, '') : ''}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;

    // routing diagram for the selected query token
    html += `<div class="routing"><div class="routing-cap">routing · <b>${TOKENS[query]}</b> attends to:</div>${routingSvg(attn[query])}</div>`;
    html += `</div>`;
    root.innerHTML = html;

    root.querySelectorAll('.head-tab').forEach((b) => b.addEventListener('click', () => { head = +b.dataset.h; render(); }));
    root.querySelectorAll('.qrow').forEach((tr) => tr.addEventListener('click', () => { query = +tr.dataset.q; render(); }));
  }

  function routingSvg(weights) {
    const n = TOKENS.length;
    const W = 560, rowY = 34, keyY = 150, pad = 40;
    const xAt = (i) => pad + (i / (n - 1)) * (W - pad * 2);
    let s = `<svg viewBox="0 0 ${W} 190" class="route-svg">`;
    weights.forEach((w, j) => {
      s += `<line x1="${xAt(query)}" y1="${rowY}" x2="${xAt(j)}" y2="${keyY}" stroke="${ACCENT}" stroke-width="${(0.5 + w * 7).toFixed(2)}" stroke-opacity="${(0.12 + w * 0.8).toFixed(2)}"/>`;
    });
    TOKENS.forEach((t, i) => {
      s += `<circle cx="${xAt(i)}" cy="${rowY}" r="${i === query ? 6 : 3}" fill="${i === query ? ACCENT : '#444b57'}"/>`;
      s += `<text x="${xAt(i)}" y="${rowY - 12}" class="rt ${i === query ? 'on' : ''}">${t}</text>`;
      s += `<circle cx="${xAt(i)}" cy="${keyY}" r="${3 + weights[i] * 8}" fill="${ACCENT}" fill-opacity="${(0.2 + weights[i] * 0.8).toFixed(2)}"/>`;
      s += `<text x="${xAt(i)}" y="${keyY + 22}" class="rt">${t}</text>`;
    });
    s += `</svg>`;
    return s;
  }

  return { build };
}

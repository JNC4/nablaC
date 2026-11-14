import './style.css';
import { makeNetwork, ETAS, STEPS } from './run.js';
import { initParams, costAndGrad, probe as follow, rng } from './net.js';
import { PRESETS, pack, MAX_POINTS } from './data.js';
import { makeTerritory } from './territory.js';
import { makeNetView } from './netview.js';
import { renderMaths } from './maths.js';
import { makeProfile } from './profile.js';
import { paintRelief } from './relief.js';
import { drawTerrain } from './terrain.js';
import opening from './opening.json';

const FONT = '"Text", Georgia, serif';
const RATE = STEPS / 7; // steps shown per second
const HOLD = 6500; // ms to rest at the bottom before setting out again
const TOUR = 2600; // ms the network looks at each dot when left alone
const PULSE = 1500; // ms for ∇C to run back through the network once
const SURVEYORS = 3;
const reduce = matchMedia('(prefers-reduced-motion: reduce)');

const net = makeNetwork();
const P = net.nParams;

const $ = (s) => document.querySelector(s);
const terrainCv = $('#terrain');
const tctx = terrainCv.getContext('2d');
const beliefCv = $('#belief');
const netCv = $('#net');
const territory = makeTerritory(beliefCv);
const netview = makeNetView(netCv, net);
const profile = makeProfile($('#profile'));
const readout = $('#readout');
const tally = $('#wrong');
const mathsEl = $('#maths');

// The training set. The reader can paint into it; every edit makes a new pack.
let preset = opening.preset;
let points = PRESETS[preset]();
let data = pack(points);
let pristine = true; // still exactly the opening preset

let trainer = null;
let surveyors = [];
let runId = 0;
const runs = new Map(); // every run the workers are busy with, by id
let run = null; // the one on screen
let eta = 'medium';
let seed = opening.seed;
let dpr = 1;
let S = 0; // terrain canvas size in device pixels
let dirty = true;

// What is being fed through the network: a dot, or any point of the plane.
const tourRng = rng(7);
const probeAt = { x: 0, y: 0, dot: -1, source: 'tour', since: -1e9 };
let p = null; // the probe's forward (and backward) pass at the step shown
let lastP = null; // the last one computed
let pFrom = null; // what the network showed before the point moved
let netHover = null;
let netPin = null;
let tool = null; // null to look, 0..2 to paint a colour, 'erase'
let stroke = null;
let lastMaths = '';
let mathsAt = 0;
const grad = new Float64Array(P);

// Section: runs.

function workers() {
  trainer?.terminate();
  surveyors.forEach((w) => w.terminate());
  runs.clear();
  trainer = new Worker(new URL('./trainer.worker.js', import.meta.url), { type: 'module' });
  trainer.onmessage = onTrainer;
  surveyors = Array.from({ length: SURVEYORS }, () => {
    const w = new Worker(new URL('./survey.worker.js', import.meta.url), { type: 'module' });
    w.onmessage = onSurvey;
    return w;
  });
}

function blankRun(start) {
  return {
    id: ++runId,
    eta: ETAS[eta],
    data,
    start,
    thetas: new Float32Array((STEPS + 1) * P),
    costs: new Float32Array(STEPS + 1),
    known: 0,
    t0: 0,
    plane: null,
    grid: null,
    relief: null,
    reliefKey: '',
    reliefAt: 0,
    finishedAt: 0,
    shown: -1,
    drawnAt: 0,
    gscale: 0,
    next: null,
  };
}

function makeRun(start, isOpening = false) {
  const r = blankRun(start);
  // The opening sheet was surveyed at build time by the same code; use it
  // until the live run confirms it (see onTrainer).
  if (isOpening) {
    r.plane = { path: Float32Array.from(opening.path), win: opening.win, provisional: true };
    r.grid = { n: opening.n, h: Float32Array.from(opening.h), pending: 0 };
  }
  runs.set(r.id, r);
  trainer.postMessage({ type: 'run', id: r.id, eta: r.eta, steps: STEPS, start, data: r.data });
  return r;
}

// The network as it stands on screen, to carry on from after an edit.
function currentTheta() {
  if (!run || !run.known) return initParams(net, seed);
  const s = Math.max(0, run.shown);
  return Float64Array.from(run.thetas.subarray(s * P, (s + 1) * P));
}

// kind: 'fresh' (new random weights from `seed`) or 'warm' (carry on).
function restart(kind) {
  const start = kind === 'warm' ? currentTheta() : initParams(net, seed);
  workers();
  if (!data.n) {
    // Nothing to learn from: hold the network still where it is.
    const r = blankRun(start);
    r.thetas.set(Float32Array.from(start), 0);
    r.known = 1;
    r.empty = true;
    show(r);
    return;
  }
  const isOpening = pristine && kind === 'fresh' && seed === opening.seed && eta === opening.eta && STEPS === opening.steps;
  show(makeRun(start, isOpening));
}

function show(r) {
  run = r;
  run.t0 = 0;
  run.shown = -1;
  if (run.relief) run.reliefAt = -1e9;
  paint(run);
  dirty = true;
}

function onTrainer({ data: msg }) {
  const r = runs.get(msg.id);
  if (!r) return;
  if (msg.type === 'steps') {
    r.thetas.set(msg.thetas, msg.from * P);
    r.costs.set(msg.costs, msg.from);
    r.known = msg.from + msg.costs.length;
  } else if (msg.type === 'plane') {
    if (r.plane?.provisional && sameWeights(msg.end, opening.end)) {
      r.plane.provisional = false;
      return;
    }
    r.plane = msg;
    r.relief = null;
    r.reliefKey = '';
    survey(r, 28);
  }
}

function sameWeights(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-4) return false;
  return true;
}

// Measure the ground: a coarse pass so there is a map quickly, then a fine one.
function survey(r, n) {
  const { end, dirs, win } = r.plane;
  r.grid = { n, h: new Float32Array(n * n), pending: SURVEYORS };
  const band = Math.ceil(n / SURVEYORS);
  surveyors.forEach((w, i) => {
    const r0 = i * band;
    const r1 = Math.min(n, r0 + band);
    w.postMessage({ id: r.id, data: r.data, end, dirs, win, n, r0, r1 });
  });
}

function onSurvey({ data: msg }) {
  const r = runs.get(msg.id);
  if (!r || !r.grid || msg.n !== r.grid.n) return;
  r.grid.h.set(msg.h, msg.r0 * msg.n);
  if (--r.grid.pending > 0) return;
  if (r === run) paint(r);
  if (r.grid.n < 72) survey(r, 72);
}

// The whole route in device pixels, every tenth step and the last.
function routePx(r) {
  const { path, win } = r.plane;
  const out = [];
  const push = (t) => out.push(((path[t * 2] - win.a0) / (win.a1 - win.a0)) * S, ((win.b1 - path[t * 2 + 1]) / (win.b1 - win.b0)) * S);
  for (let t = 0; t < STEPS; t += 10) push(t);
  push(STEPS);
  return out;
}

function paint(r) {
  if (!r?.grid || r.grid.pending > 0 || !r.plane || !S) return;
  const key = `${r.grid.n}:${S}:${document.fonts.status}`;
  if (r.reliefKey === key) return;
  r.reliefKey = key;
  const first = !r.relief;
  r.relief = paintRelief({ h: r.grid.h, n: r.grid.n, S, dpr, font: FONT, route: routePx(r) });
  if (first) r.reliefAt = r.known > 1 && r.shown > 0 ? performance.now() : -1e9;
  dirty = true;
}

// Section: the probe.

function setProbe(x, y, dot, source, now, wave) {
  const changed = dot !== probeAt.dot || source !== probeAt.source || dot < 0;
  probeAt.x = x;
  probeAt.y = y;
  probeAt.dot = dot;
  probeAt.source = source;
  if (wave && changed) {
    probeAt.since = now;
    pFrom = lastP;
  }
  else if (!wave) probeAt.since = now - 1e4;
  p = null;
}

// Left alone, the network looks at one dot after another, mostly at the
// ones it still gets wrong.
function nextTourDot(now) {
  if (!data.n) {
    setProbe(0, 0, -1, 'tour', now, true);
    return;
  }
  const wrong = [];
  for (let i = 0; i < data.n; i++) if (territory.isWrong(i)) wrong.push(i);
  const pool = wrong.length && tourRng() < 0.75 ? wrong : null;
  const i = pool ? pool[Math.floor(tourRng() * pool.length)] : Math.floor(tourRng() * data.n);
  setProbe(data.xs[i], data.ys[i], i, 'tour', now, true);
}

function nearestDot(x, y, within) {
  let best = -1;
  let bestD = within;
  for (let i = 0; i < data.n; i++) {
    const dd = Math.hypot(data.xs[i] - x, data.ys[i] - y);
    if (dd < bestD) {
      bestD = dd;
      best = i;
    }
  }
  return best;
}

// Section: layout.

function layout() {
  dpr = Math.min(2.5, window.devicePixelRatio || 1);
  S = Math.round($('#terrain-box').clientWidth * dpr);
  terrainCv.width = terrainCv.height = S;
  territory.resize($('#belief-box').clientWidth, dpr);
  const nb = $('#net-box');
  netview.resize(nb.clientWidth, nb.clientHeight, dpr);
  const pb = $('#profile-box');
  profile.resize(pb.clientWidth, pb.clientHeight, dpr);
  if (run) {
    paint(run);
    run.shown = -1;
  }
  dirty = true;
}

// Section: the frame.

function frame(now) {
  requestAnimationFrame(frame);
  if (!run || !run.known) return;
  const still = reduce.matches;
  if (!run.t0) run.t0 = now;
  const last = run.known - 1;
  const s = still ? last : Math.max(0, Math.min(last, Math.floor(((now - run.t0) / 1000) * RATE)));
  const th = run.thetas.subarray(s * P, (s + 1) * P);

  // Several steps pass per frame; redrawing the network's opinion about 30
  // times a second is plenty.
  if (s !== run.shown && (now - run.drawnAt > 30 || s === STEPS || run.shown < 0 || run.empty)) {
    if (run.shown < 0 && !run.empty) {
      costAndGrad(net, run.thetas.subarray(0, P), run.data, grad);
      let gmax = 0;
      for (const L of net.layers) for (let i = 0; i < L.nin * L.nout; i++) gmax = Math.max(gmax, Math.abs(grad[L.w + i]));
      run.gscale = gmax * 0.6;
    }
    run.shown = s;
    run.drawnAt = now;
    dataChanged(th);
    if (!run.empty) {
      profile.draw(run.costs, s, STEPS, FONT);
      readout.textContent = `C = ${run.costs[s].toFixed(3)} after ${s} steps`;
    } else {
      profile.draw(run.costs, -1, STEPS, FONT);
      readout.textContent = '';
    }
    dirty = true;
  }

  // The tour moves on.
  if (probeAt.source === 'tour' && (now - probeAt.since > TOUR || probeAt.dot >= data.n) && !(still && probeAt.dot >= 0)) nextTourDot(now);
  if (probeAt.source === 'pin' && now - probeAt.since > 15000) nextTourDot(now);
  if (!p) {
    const label = probeAt.dot >= 0 ? data.labels[probeAt.dot] : -1;
    p = follow(net, th, probeAt.x, probeAt.y, label);
    lastP = p;
    territory.compose(data, [probeAt.x, probeAt.y], brushNow);
    mathsAt = 0;
  }

  const training = !still && !run.empty && s < STEPS && run.known > 1;
  const pulse = training ? ((now - run.t0) % PULSE) / PULSE : null;
  netview.draw({
    th,
    p,
    from: pFrom,
    probeAge: still ? 1e9 : now - probeAt.since,
    grad,
    gscale: run.gscale,
    pulse,
    hover: netHover || netPin,
  });

  if (now - mathsAt > 120) {
    mathsAt = now;
    const html = renderMaths({ net, th, data, p, eta: run.eta, C: run.costs[s], target: netHover || netPin });
    if (html !== lastMaths) {
      mathsEl.innerHTML = html;
      lastMaths = html;
    }
  }

  const fading = run.relief && now - run.reliefAt < 650;
  if (dirty || fading) {
    drawTerrain(tctx, S, dpr, run.empty ? null : run, s, now, still, run.empty ? 'Draw some dots to learn from' : 'Measuring the ground…');
    dirty = false;
  }

  if (run.empty) return;
  if (s === STEPS && !run.finishedAt) run.finishedAt = now;
  if (!run.finishedAt || still) return;
  // While it rests at the bottom, the next start is trained and surveyed in
  // the background, so the new sheet is ready the moment it is shown.
  if (!run.next && run.grid && run.grid.pending === 0 && run.grid.n >= 72) {
    seed++;
    run.next = makeRun(initParams(net, seed));
  }
  const next = run.next;
  if (next && now - run.finishedAt > HOLD && next.grid?.n >= 72 && next.grid.pending === 0) {
    runs.delete(run.id);
    show(next);
  }
}

// θ or the dots changed: redraw the opinion, the gradient and the probe.
function dataChanged(th) {
  const wrong = territory.render(net, th, data);
  if (!run.empty) costAndGrad(net, th, data, grad);
  else grad.fill(0);
  p = null;
  tally.textContent = !data.n
    ? 'No dots. Pick a colour and draw some.'
    : wrong === 0
      ? `All ${data.n} dots right.`
      : `${wrong} of ${data.n} dots wrong.`;
}

// Section: drawing dots.

const ERASE_RADIUS = 0.09;
const SPACING = 0.055;
let brushNow = null;
let retrainTimer = 0;

function edited() {
  data = pack(points);
  pristine = false;
  preset = null;
  for (const b of document.querySelectorAll('[data-preset]')) b.setAttribute('aria-pressed', 'false');
  if (run) dataChanged(run.thetas.subarray(Math.max(0, run.shown) * P, (Math.max(0, run.shown) + 1) * P));
  p = null;
}

function paintAt(x, y) {
  if (tool === 'erase') {
    const before = points.length;
    points = points.filter(([px, py]) => Math.hypot(px - x, py - y) > ERASE_RADIUS);
    if (points.length !== before) edited();
    return;
  }
  if (points.length >= MAX_POINTS) return;
  if (stroke.last && Math.hypot(stroke.last[0] - x, stroke.last[1] - y) < SPACING) return;
  const jitter = () => (Math.random() - 0.5) * 0.03;
  points.push([x + jitter(), y + jitter(), tool]);
  stroke.last = [x, y];
  edited();
}

function planeAt(e) {
  const rect = beliefCv.getBoundingClientRect();
  return territory.toPlane(e.clientX - rect.left, e.clientY - rect.top);
}

beliefCv.addEventListener('pointerdown', (e) => {
  const [x, y] = planeAt(e);
  const now = performance.now();
  if (tool === null) {
    const dot = nearestDot(x, y, 0.07);
    if (dot >= 0) setProbe(data.xs[dot], data.ys[dot], dot, 'pin', now, true);
    else setProbe(x, y, -1, 'pin', now, true);
    return;
  }
  e.preventDefault();
  try {
    beliefCv.setPointerCapture(e.pointerId);
  } catch {
    // Synthetic or already-ended pointers cannot be captured; drawing still works.
  }
  stroke = { last: null };
  clearTimeout(retrainTimer);
  paintAt(x, y);
});

beliefCv.addEventListener('pointermove', (e) => {
  const [x, y] = planeAt(e);
  const now = performance.now();
  if (stroke) {
    // Fill in between samples so a quick stroke still leaves a line of dots.
    const from = stroke.last;
    if (from && tool !== 'erase') {
      const steps = Math.floor(Math.hypot(x - from[0], y - from[1]) / SPACING);
      for (let i = 1; i <= steps; i++) paintAt(from[0] + ((x - from[0]) * i) / steps, from[1] + ((y - from[1]) * i) / steps);
    } else {
      paintAt(x, y);
    }
  }
  brushNow = tool === 'erase' ? { x, y, radius: ERASE_RADIUS } : null;
  if (e.pointerType === 'mouse' && !stroke && probeAt.source !== 'pin') {
    const dot = nearestDot(x, y, 0.05);
    if (dot >= 0) setProbe(data.xs[dot], data.ys[dot], dot, 'pointer', now, true);
    else setProbe(x, y, -1, 'pointer', now, false);
  }
  p = null;
});

function endStroke() {
  if (!stroke) return;
  stroke = null;
  clearTimeout(retrainTimer);
  retrainTimer = setTimeout(() => restart('warm'), 250);
}
beliefCv.addEventListener('pointerup', endStroke);
beliefCv.addEventListener('pointercancel', endStroke);
beliefCv.addEventListener('pointerleave', () => {
  brushNow = null;
  if (probeAt.source === 'pointer') nextTourDot(performance.now());
  p = null;
});

function setTool(next) {
  tool = tool === next ? null : next;
  for (const b of document.querySelectorAll('[data-tool]')) {
    const v = b.dataset.tool === 'erase' ? 'erase' : Number(b.dataset.tool);
    b.setAttribute('aria-pressed', String(v === tool));
  }
  beliefCv.classList.toggle('is-drawing', tool !== null);
  brushNow = null;
  p = null;
}
for (const b of document.querySelectorAll('[data-tool]')) {
  b.addEventListener('click', () => setTool(b.dataset.tool === 'erase' ? 'erase' : Number(b.dataset.tool)));
}
for (const b of document.querySelectorAll('[data-preset]')) {
  b.addEventListener('click', () => {
    preset = b.dataset.preset;
    points = PRESETS[preset]();
    data = pack(points);
    pristine = false;
    for (const o of document.querySelectorAll('[data-preset]')) o.setAttribute('aria-pressed', String(o === b));
    seed++;
    restart('fresh');
    nextTourDot(performance.now());
  });
}
$('#clear').addEventListener('click', () => {
  points = [];
  edited();
  restart('warm');
  nextTourDot(performance.now());
});

// Section: pointing at the network.

netCv.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse') return;
  const rect = netCv.getBoundingClientRect();
  netHover = netview.hit(e.clientX - rect.left, e.clientY - rect.top);
  mathsAt = 0;
});
netCv.addEventListener('pointerleave', () => {
  netHover = null;
  mathsAt = 0;
});
netCv.addEventListener('pointerdown', (e) => {
  const rect = netCv.getBoundingClientRect();
  const t = netview.hit(e.clientX - rect.left, e.clientY - rect.top);
  netPin = t && netPin && JSON.stringify(t) === JSON.stringify(netPin) ? null : t;
  mathsAt = 0;
});

// Section: controls.

for (const b of document.querySelectorAll('[data-eta]')) {
  b.addEventListener('click', () => {
    eta = b.dataset.eta;
    for (const o of document.querySelectorAll('[data-eta]')) o.setAttribute('aria-pressed', String(o === b));
    restart('fresh');
  });
}
$('#again').addEventListener('click', () => {
  seed++;
  restart('fresh');
});

let resizeTimer = 0;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(layout, 80);
}).observe($('#sheet'));
document.fonts.ready.then(layout);

layout();
restart('fresh');
requestAnimationFrame(frame);

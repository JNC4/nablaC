import './style.css';
import { setup, ETAS, STEPS, TASK } from './run.js';
import { makeTerritory } from './territory.js';
import { makeProfile } from './profile.js';
import { paintRelief } from './relief.js';
import opening from './opening.json';

const FONT = '"Text", Georgia, serif';
const RATE = STEPS / 7; // steps shown per second
const HOLD = 6500; // ms to rest at the bottom before setting out again
const SURVEYORS = 3;
const reduce = matchMedia('(prefers-reduced-motion: reduce)');

const { net, data } = setup(TASK);
const P = net.nParams;

const $ = (s) => document.querySelector(s);
const terrainCv = $('#terrain');
const tctx = terrainCv.getContext('2d');
const territory = makeTerritory($('#belief'));
const profile = makeProfile($('#profile'));
const readout = $('#readout');
const tally = $('#wrong');

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

function makeRun(runSeed) {
  const r = {
    id: ++runId,
    seed: runSeed,
    eta: ETAS[eta],
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
    next: null,
  };
  // The opening sheet was surveyed at build time by the same code; use it
  // until the live run confirms it (see onTrainer).
  if (runSeed === opening.seed && eta === opening.eta && STEPS === opening.steps) {
    r.plane = { path: Float32Array.from(opening.path), win: opening.win, provisional: true };
    r.grid = { n: opening.n, h: Float32Array.from(opening.h), pending: 0 };
  }
  runs.set(r.id, r);
  trainer.postMessage({ type: 'run', id: r.id, task: TASK, eta: r.eta, seed: runSeed, steps: STEPS });
  return r;
}

// A fresh start chosen by the reader: drop everything in flight.
function restart() {
  workers();
  show(makeRun(seed));
}

function show(r) {
  run = r;
  run.t0 = run.known ? performance.now() : 0;
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
    if (r === run && !r.t0) r.t0 = performance.now();
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
    w.postMessage({ id: r.id, task: TASK, end, dirs, win, n, r0, r1 });
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

function routeEnd(r) {
  const { win } = r.plane;
  return [((0 - win.a0) / (win.a1 - win.a0)) * S, ((win.b1 - 0) / (win.b1 - win.b0)) * S];
}

function paint(r) {
  if (!r?.grid || r.grid.pending > 0 || !r.plane || !S) return;
  const key = `${r.grid.n}:${S}:${document.fonts.status}`;
  if (r.reliefKey === key) return;
  r.reliefKey = key;
  const first = !r.relief;
  r.relief = paintRelief({ h: r.grid.h, n: r.grid.n, S, dpr, font: FONT, avoid: routeEnd(r) });
  if (first) r.reliefAt = r.known ? performance.now() : -1e9;
  dirty = true;
}

function layout() {
  dpr = Math.min(2.5, window.devicePixelRatio || 1);
  S = Math.round($('#terrain-box').clientWidth * dpr);
  terrainCv.width = terrainCv.height = S;
  territory.resize($('#belief-box').clientWidth, dpr);
  const pb = $('#profile-box');
  profile.resize(pb.clientWidth, pb.clientHeight, dpr);
  if (run) {
    paint(run);
    run.shown = -1;
  }
  dirty = true;
}

function drawTerrain(s, now) {
  tctx.fillStyle = '#efe9da';
  tctx.fillRect(0, 0, S, S);
  if (!run.relief) {
    tctx.fillStyle = '#8a7f72';
    tctx.font = `italic ${Math.round(14 * dpr)}px ${FONT}`;
    tctx.textAlign = 'center';
    tctx.textBaseline = 'middle';
    tctx.fillText('Measuring the ground…', S / 2, S / 2);
    return;
  }
  const fade = reduce.matches ? 1 : Math.min(1, (now - run.reliefAt) / 600);
  tctx.globalAlpha = fade;
  tctx.drawImage(run.relief.canvas, 0, 0);
  tctx.globalAlpha = 1;

  const { path, win } = run.plane;
  const X = (a) => ((a - win.a0) / (win.a1 - win.a0)) * S;
  const Y = (b) => ((win.b1 - b) / (win.b1 - win.b0)) * S;

  // The way it has come: a red track with a pale casing, as on a walking map.
  tctx.lineJoin = 'round';
  tctx.lineCap = 'round';
  tctx.beginPath();
  tctx.moveTo(X(path[0]), Y(path[1]));
  const every = Math.max(1, Math.ceil(s / 300)); // the route is smooth; 300 points draw it
  for (let t = every; t < s; t += every) tctx.lineTo(X(path[t * 2]), Y(path[t * 2 + 1]));
  tctx.lineTo(X(path[s * 2]), Y(path[s * 2 + 1]));
  tctx.strokeStyle = 'rgba(251, 247, 238, 0.8)';
  tctx.lineWidth = 4.6 * dpr;
  tctx.stroke();
  tctx.strokeStyle = '#c8302a';
  tctx.lineWidth = 2.2 * dpr;
  tctx.setLineDash([7 * dpr, 4 * dpr]);
  tctx.stroke();
  tctx.setLineDash([]);

  // Where it set out from.
  const sx = X(path[0]);
  const sy = Y(path[1]);
  tctx.beginPath();
  tctx.arc(sx, sy, 4.2 * dpr, 0, Math.PI * 2);
  tctx.fillStyle = '#fbf7ee';
  tctx.fill();
  tctx.lineWidth = 1.5 * dpr;
  tctx.strokeStyle = '#2b2620';
  tctx.stroke();
  tctx.font = `italic ${Math.round(12.5 * dpr)}px ${FONT}`;
  tctx.fillStyle = '#2b2620';
  const right = sx < S * 0.72;
  tctx.textAlign = right ? 'left' : 'right';
  tctx.textBaseline = 'middle';
  tctx.fillText('start', sx + (right ? 9 : -9) * dpr, sy);

  // Where it is now.
  tctx.beginPath();
  tctx.arc(X(path[s * 2]), Y(path[s * 2 + 1]), 6 * dpr, 0, Math.PI * 2);
  tctx.fillStyle = '#c8302a';
  tctx.fill();
  tctx.lineWidth = 1.8 * dpr;
  tctx.strokeStyle = '#fbf7ee';
  tctx.stroke();
}

window.__perf = [];
function frame(now) {
  const __t = performance.now();
  try { frame0(now); } finally { window.__perf.push(performance.now() - __t); }
}
function frame0(now) {
  requestAnimationFrame(frame);
  if (!run || !run.known) return;
  if (!run.t0) run.t0 = now;
  const last = run.known - 1;
  const s = reduce.matches ? last : Math.min(last, Math.floor(((now - run.t0) / 1000) * RATE));
  // Several steps pass per frame; redrawing the network about 30 times a
  // second is plenty and halves the work.
  if (s !== run.shown && (now - run.drawnAt > 30 || s === STEPS || run.shown < 0)) {
    run.shown = s;
    run.drawnAt = now;
    const wrong = territory.draw(net, run.thetas.subarray(s * P, (s + 1) * P), data);
    profile.draw(run.costs, s, STEPS, FONT);
    readout.textContent = `C = ${run.costs[s].toFixed(3)} after ${s} steps`;
    tally.textContent = wrong === 0 ? `All ${data.n} dots right.` : `${wrong} of ${data.n} dots wrong.`;
    dirty = true;
  }
  const fading = run.relief && now - run.reliefAt < 650;
  if (dirty || fading) {
    drawTerrain(s, now);
    dirty = false;
  }
  if (s === STEPS && !run.finishedAt) run.finishedAt = now;
  if (!run.finishedAt || reduce.matches) return;
  // While it rests at the bottom, the next start is trained and surveyed in
  // the background, so the new sheet is ready the moment it is shown.
  if (!run.next && run.grid && run.grid.pending === 0 && run.grid.n >= 72) run.next = makeRun(++seed);
  const next = run.next;
  if (next && now - run.finishedAt > HOLD && next.grid?.n >= 72 && next.grid.pending === 0) {
    runs.delete(run.id);
    show(next);
  }
}

for (const b of document.querySelectorAll('[data-eta]')) {
  b.addEventListener('click', () => {
    eta = b.dataset.eta;
    for (const o of document.querySelectorAll('[data-eta]')) o.setAttribute('aria-pressed', String(o === b));
    restart();
  });
}
$('#again').addEventListener('click', () => {
  seed++;
  restart();
});

let resizeTimer = 0;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(layout, 80);
}).observe($('#sheet'));
document.fonts.ready.then(layout);

layout();
restart();
requestAnimationFrame(frame);

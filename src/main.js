import './styles.css';
import { DATASETS, makeData } from './data.js';
import { Trainer } from './trainer.js';
import { createNetworkView } from './network-view.js';
import { createBoundaryView, createLossView } from './panels.js';
import { createLandscapeView } from './landscape.js';
import { createAttentionView } from './attention.js';

const $ = (id) => document.getElementById(id);

const ARCHS = {
  '[2·6·2]': [2, 6, 2],
  '[2·8·8·2]': [2, 8, 8, 2],
  '[2·12·12·2]': [2, 12, 12, 2],
  '[2·16·16·16·2]': [2, 16, 16, 16, 2],
};

const netView = createNetworkView($('net'));
const boundary = createBoundaryView($('boundary'));
const lossView = createLossView($('loss'));
const landscape = createLandscapeView($('landscape'), $('landscape-eq'));
const attention = createAttentionView($('attention'));

let trainer = null;
let running = false;
let activeView = 'network';
let phase = 0;
let lastBeat = 0;
let probeIdx = 0;
const GRID_RES = 56;

// ---- options off the controls ----
function opts() {
  return {
    dataset: $('dataset').value,
    sizes: ARCHS[$('arch').value],
    activation: $('activation').value,
    lr: +$('lr').value,
    speed: +$('speed').value,
    seed: Math.abs(parseInt($('seed').value, 10) || 1),
  };
}

function syncReadouts() {
  const o = opts();
  $('lrOut').value = o.lr.toFixed(2);
  $('speedOut').value = o.speed === 0 ? 'crawl' : o.speed < 70 ? 'slow' : o.speed < 150 ? 'fast' : 'blur';
  $('meta').textContent = `${$('arch').value} / η ${o.lr.toFixed(2)}`;
}

function beatMs() {
  return Math.max(40, 440 - opts().speed * 2);
}
function stepsPerBeat() {
  const s = opts().speed;
  return s > 160 ? 5 : s > 110 ? 2 : 1;
}

// ---- build / reset ----
function reset() {
  const o = opts();
  const data = makeData(o.dataset, 160, o.seed);
  trainer = new Trainer({ sizes: o.sizes, data, lr: o.lr, seed: o.seed, hidden: o.activation });
  probeIdx = 0;
  netView.build(trainer.net);
  boundary.setData(data);
  trainer.probe(0, 0);
  drawBoundary();
  drawLoss();
  updateReadouts();
  phase = 0;
  setStatus('ready · press TRAIN');
  equation();
}

function drawBoundary() {
  if (!trainer) return;
  boundary.draw(trainer.grid(GRID_RES), GRID_RES);
}
function drawLoss() {
  if (!trainer) return;
  lossView.draw(trainer.lossHistory, trainer.gradHistory);
}

function updateReadouts() {
  if (!trainer) return;
  $('roStep').textContent = trainer.step;
  $('roLoss').textContent = isNaN(trainer.loss) ? '—' : trainer.loss.toFixed(4);
  $('roAcc').textContent = trainer.step ? (trainer.acc * 100).toFixed(0) + '%' : '—';
  $('roGrad').textContent = trainer.step ? trainer.gradNorm.toFixed(3) : '—';
}

function equation() {
  if (!trainer || !trainer.step) { $('equation').innerHTML = '<em>θ ← θ − η ∇C</em>'; return; }
  $('equation').innerHTML =
    `C = <b>${trainer.loss.toFixed(4)}</b> &nbsp;·&nbsp; ` +
    `<span class="ax">‖∇C‖</span> = <b>${trainer.gradNorm.toFixed(3)}</b> &nbsp;·&nbsp; ` +
    `<em>θ ← θ − η∇C</em> &nbsp;·&nbsp; step ${trainer.step} &nbsp;·&nbsp; acc ${(trainer.acc * 100).toFixed(0)}%`;
}

function setStatus(s) { $('status').textContent = s; }

// ---- the heartbeat: one (or a few) gradient steps + the data the views need ----
function beat() {
  for (let i = 0; i < stepsPerBeat(); i++) trainer.trainStep();

  // probe a rotating data point so the forward sweep shows live activations
  const pt = trainer.data[probeIdx % trainer.data.length];
  probeIdx++;
  trainer.probe(pt.x, pt.y);

  drawBoundary();
  drawLoss();
  updateReadouts();
  equation();
  if (trainer.acc >= 0.999) setStatus(`converged · ${trainer.step} steps · 100%`);
  else setStatus(`descending · C ${trainer.loss.toFixed(3)}`);
}

// ---- animation loop ----
function frame(t) {
  if (running && trainer) {
    const bm = beatMs();
    if (t - lastBeat >= bm) { beat(); lastBeat = t; }
    phase = (((t - lastBeat) / bm) * 2) % 2;
  }
  if (activeView === 'network') netView.render(running ? phase : 0.04);
  else if (activeView === 'landscape') landscape.frame(running);
  requestAnimationFrame(frame);
}

// ---- views ----
function showView(v) {
  activeView = v;
  document.querySelectorAll('.view').forEach((s) => s.classList.toggle('active', s.id === `view-${v}`));
  document.querySelectorAll('#viewtoggle button').forEach((b) => b.classList.toggle('active', b.dataset.view === v));
  if (v === 'network') { netView.resize(); netView.build(trainer.net); drawBoundary(); drawLoss(); }
  if (v === 'landscape') landscape.build(trainer);
  if (v === 'attention') attention.build();
}

function toggleRun() {
  running = !running;
  $('run').textContent = running ? 'PAUSE' : 'TRAIN';
  $('run').classList.toggle('running', running);
  if (running) { lastBeat = performance.now(); setStatus('descending…'); }
  else setStatus(`paused · step ${trainer.step}`);
}

// ---- init ----
function fillSelect(sel, entries) {
  for (const [val, label] of entries) {
    const o = document.createElement('option');
    o.value = val; o.textContent = label; sel.append(o);
  }
}

function init() {
  fillSelect($('dataset'), Object.entries(DATASETS).map(([k, v]) => [k, v.name]));
  fillSelect($('arch'), Object.keys(ARCHS).map((k) => [k, k]));
  $('arch').value = '[2·8·8·2]';
  $('dataset').value = 'moons';

  syncReadouts();
  reset();

  ['lr', 'speed'].forEach((id) => $(id).addEventListener('input', () => { syncReadouts(); if (trainer) trainer.lr = opts().lr; }));
  ['dataset', 'arch', 'activation', 'seed'].forEach((id) =>
    $(id).addEventListener('change', () => { syncReadouts(); running = false; $('run').textContent = 'TRAIN'; $('run').classList.remove('running'); reset(); })
  );
  $('reseed').addEventListener('click', () => { $('seed').value = String(1 + Math.floor(Math.random() * 9998)); reset(); });
  $('run').addEventListener('click', toggleRun);
  $('stepBtn').addEventListener('click', () => { if (trainer) { beat(); netView.render(1.999); } });
  $('reset').addEventListener('click', () => { running = false; $('run').textContent = 'TRAIN'; $('run').classList.remove('running'); reset(); });
  document.querySelectorAll('#viewtoggle button').forEach((b) =>
    b.addEventListener('click', () => showView(b.dataset.view))
  );

  let rz;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => {
      if (activeView === 'network') { netView.build(trainer.net); drawBoundary(); drawLoss(); }
      if (activeView === 'landscape') landscape.build(trainer);
    }, 120);
  });

  requestAnimationFrame(frame);

  const qs = new URLSearchParams(location.search);
  // ?auto → animate training; ?auto=N → run N steps instantly then show result.
  const auto = qs.get('auto');
  if (auto !== null) {
    const n = parseInt(auto, 10);
    if (n > 0) {
      for (let i = 0; i < n; i++) trainer.trainStep();
      trainer.probe(0.4, 0.2);
      drawBoundary(); drawLoss(); updateReadouts(); equation();
      netView.render(0.04);
      setStatus(`trained ${n} steps · acc ${(trainer.acc * 100).toFixed(0)}%`);
    } else {
      toggleRun();
    }
  }

  // ?view=landscape|attention deep-links straight into a view.
  const view = qs.get('view');
  if (view && ['network', 'landscape', 'attention'].includes(view)) showView(view);
}

init();

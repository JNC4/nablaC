// Surveys the opening sheet ahead of time so the page can show its map on
// the first frame. Same code as the browser: train, find the plane, measure
// the ground. Runs before every build. The page still trains the opening
// run live, checks that it lands on these weights, and surveys again itself
// if it does not.

import { writeFileSync } from 'node:fs';
import { makeNetwork, train, ETAS, STEPS } from '../src/run.js';
import { initParams } from '../src/net.js';
import { PRESETS, pack } from '../src/data.js';
import { planeOfRun, heights } from '../src/survey.js';

export const OPENING = { preset: 'spiral', seed: 1, eta: 'medium' };
const N = 72;

const t0 = performance.now();
const net = makeNetwork();
const data = pack(PRESETS[OPENING.preset]());
const P = net.nParams;
const start = initParams(net, OPENING.seed);
const all = new Float32Array((STEPS + 1) * P);
const costs = new Float32Array(STEPS + 1);
const end = train({
  net, data, eta: ETAS[OPENING.eta], start, steps: STEPS,
  onChunk(from, thetas, cs) {
    all.set(thetas, from * P);
    costs.set(cs, from);
  },
});
const { dirs, path, inPlane, win } = planeOfRun({ net, start, all, end, steps: STEPS });
const h = heights(net, data, end, dirs, win, N);

const round = (arr, dp) => Array.from(arr, (v) => +v.toFixed(dp));
const out = {
  ...OPENING,
  steps: STEPS,
  end: round(end, 6),
  win,
  inPlane: +inPlane.toFixed(4),
  path: round(path, 4),
  n: N,
  h: round(h, 4),
};
writeFileSync(new URL('../src/opening.json', import.meta.url), JSON.stringify(out));
console.log(
  `opening sheet: ${OPENING.preset}, seed ${OPENING.seed}, η = ${ETAS[OPENING.eta]}, C ${costs[0].toFixed(3)} → ${costs[STEPS].toFixed(4)},` +
    ` ${(inPlane * 100).toFixed(1)}% of the route in the plane, ${((performance.now() - t0) / 1000).toFixed(1)} s`,
);

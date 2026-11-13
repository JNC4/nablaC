// The route profile, as printed under a walking route in a guidebook:
// height (the cost C) against distance along the route (the step number).

import { BANDS, WATER, WATER_TINT } from './relief.js';

export function makeProfile(canvas) {
  const ctx = canvas.getContext('2d');
  let W = 0;
  let H = 0;
  let dpr = 1;

  function resize(cssW, cssH, ratio) {
    dpr = ratio;
    W = Math.round(cssW * ratio);
    H = Math.round(cssH * ratio);
    canvas.width = W;
    canvas.height = H;
  }

  function draw(costs, upto, steps, font) {
    if (!W) return;
    ctx.clearRect(0, 0, W, H);
    const padL = 40 * dpr;
    const padR = 14 * dpr;
    const padT = 8 * dpr;
    const padB = 18 * dpr;
    const top = 1.05;
    const x = (t) => padL + (t / steps) * (W - padL - padR);
    const y = (c) => padT + (1 - Math.min(c, top) / top) * (H - padT - padB);

    const fs = Math.round(10.5 * dpr);
    ctx.font = `${fs}px ${font}`;
    ctx.fillStyle = '#6d6358';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.strokeStyle = 'rgba(80, 64, 48, 0.18)';
    ctx.lineWidth = 1;
    for (const c of [0, 0.5, 1]) {
      ctx.beginPath();
      ctx.moveTo(padL, Math.round(y(c)) + 0.5);
      ctx.lineTo(W - padR, Math.round(y(c)) + 0.5);
      ctx.stroke();
      ctx.fillText(c === 0 ? '0' : c === 1 ? 'C 1.0' : c.toFixed(1), padL - 6 * dpr, y(c));
    }
    ctx.textBaseline = 'alphabetic';
    for (const t of [0, 500, 1000, 1500]) {
      if (t > steps) continue;
      const last = t === steps;
      ctx.textAlign = last ? 'right' : 'center';
      ctx.fillText(last ? `${t} steps` : String(t), last ? W - padR + 2 * dpr : x(t), H - 3 * dpr);
    }

    if (upto < 0) return;
    // The ground under the route, in the same height tints as the map.
    ctx.save();
    // One point per CSS pixel of width, keeping the highest value in each, so
    // a brief jump uphill is never smoothed away.
    const per = Math.max(1, Math.floor(steps / (W / dpr)));
    const pts = [];
    for (let t0 = 0; t0 <= upto; t0 += per) {
      let hiT = t0;
      for (let t = t0 + 1; t < Math.min(upto + 1, t0 + per); t++) if (costs[t] > costs[hiT]) hiT = t;
      pts.push(x(hiT), y(costs[hiT]));
    }
    pts.push(x(upto), y(costs[upto]));
    const line = () => {
      ctx.moveTo(pts[0], pts[1]);
      for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
    };
    ctx.beginPath();
    line();
    ctx.lineTo(x(upto), y(0));
    ctx.lineTo(x(0), y(0));
    ctx.closePath();
    ctx.clip();
    ctx.fillStyle = WATER_TINT;
    ctx.fillRect(0, y(WATER), W, y(0) - y(WATER) + 1);
    for (let k = 0; k < BANDS.length; k++) {
      const c0 = BANDS[k][0];
      const c1 = k + 1 < BANDS.length ? BANDS[k + 1][0] : top;
      if (c0 >= top) break;
      ctx.fillStyle = BANDS[k][1];
      ctx.fillRect(0, y(Math.min(c1, top)), W, y(c0) - y(Math.min(c1, top)) + 1);
    }
    ctx.restore();
    ctx.beginPath();
    line();
    ctx.strokeStyle = '#7d4a20';
    ctx.lineWidth = 1.4 * dpr;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x(upto), y(costs[upto]), 3.6 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = '#c8302a';
    ctx.fill();
    ctx.lineWidth = 1.2 * dpr;
    ctx.strokeStyle = '#fbf7ee';
    ctx.stroke();
  }

  return { resize, draw };
}

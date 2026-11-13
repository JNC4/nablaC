// The cost, drawn as a layer-coloured contour sheet in the manner of
// Bartholomew's half-inch maps: a fixed tint for each band of height (green
// lowland, buff and brown hills, plum summits), brown contours with the
// heavier ones numbered, a soft Imhof-style relief shading lit from the upper
// left, water where C is nearly zero, and a spot height on the summit.
//
// The bands are fixed values of C, not fractions of whatever this sheet
// happens to span, so the same colour means the same error on every sheet.

import { contour } from './contour.js';

export const WATER = 0.05;

const hex = (s) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));

// Band edges (values of C) and the tint from that edge up to the next.
export const BANDS = [
  [WATER, '#a4c690'],
  [0.1, '#b6cf95'],
  [0.2, '#c8d8a0'],
  [0.3, '#d9dfaa'],
  [0.4, '#e7e2b0'],
  [0.5, '#eedaa3'],
  [0.6, '#ecce95'],
  [0.7, '#e6c087'],
  [0.8, '#dfb17b'],
  [0.9, '#d6a271'],
  [1.0, '#cc9469'],
  [1.25, '#bf8766'],
  [1.5, '#b07c67'],
  [1.75, '#a27369'],
  [2.0, '#966d6e'],
  [2.5, '#8a6872'],
];
export const WATER_TINT = '#c2dbe6';
const INDEX = new Set([0.5, 1, 1.5, 2, 2.5]);
const TINTS = BANDS.map(([, c]) => hex(c));
const SHADOW = hex('#5a5068');
const LIGHT = hex('#fffbea');
const WATER_FILL = hex(WATER_TINT);
const WATER_DEEP = hex('#a3c6d9');

const INK = {
  contour: 'rgba(128, 80, 40, 0.5)',
  index: 'rgba(112, 64, 28, 0.88)',
  label: '#6e3f1a',
  shore: '#4f86a8',
  water: '#2f6a90',
  spot: '#2b2620',
};

export function bandOf(e) {
  let k = 0;
  while (k + 1 < BANDS.length && e >= BANDS[k + 1][0]) k++;
  return k;
}

// Catmull-Rom resampling of an n×n grid onto an m×m grid (corner aligned),
// done separably: every row first, then every column of the result.
export function resample(src, n, m) {
  const s = (n - 1) / (m - 1);
  const idx = new Int32Array(m);
  const frac = new Float32Array(m);
  for (let j = 0; j < m; j++) {
    const x = Math.min(n - 1.000001, j * s);
    idx[j] = Math.floor(x);
    frac[j] = x - idx[j];
  }
  const cl = (k) => (k < 0 ? 0 : k > n - 1 ? n - 1 : k);
  const cr = (p0, p1, p2, p3, t) =>
    p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
  const mid = new Float32Array(n * m); // n rows, m columns
  for (let r = 0; r < n; r++) {
    const o = r * n;
    for (let j = 0; j < m; j++) {
      const c = idx[j];
      mid[r * m + j] = cr(src[o + cl(c - 1)], src[o + c], src[o + cl(c + 1)], src[o + cl(c + 2)], frac[j]);
    }
  }
  const out = new Float32Array(m * m);
  for (let i = 0; i < m; i++) {
    const r = idx[i];
    const t = frac[i];
    const r0 = cl(r - 1) * m;
    const r1 = r * m;
    const r2 = cl(r + 1) * m;
    const r3 = cl(r + 2) * m;
    for (let j = 0; j < m; j++) out[i * m + j] = cr(mid[r0 + j], mid[r1 + j], mid[r2 + j], mid[r3 + j], t);
  }
  return out;
}

function extent(h) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of h) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi];
}

function trace(ctx, polys, toPx) {
  for (const { pts, closed } of polys) {
    ctx.moveTo(toPx(pts[0]), toPx(pts[1]));
    for (let i = 2; i < pts.length; i += 2) ctx.lineTo(toPx(pts[i]), toPx(pts[i + 1]));
    if (closed) ctx.closePath();
  }
}

// Paints the sheet into a canvas of S×S device pixels. `route` is the walk
// in device pixels, flat [x0, y0, x1, y1, ...]; labels keep clear of it, and
// well clear of where it ends.
export function paintRelief({ h, n, S, dpr, font, route }) {
  const endX = route[route.length - 2];
  const endY = route[route.length - 1];
  const clear = (x, y, dRoute, dEnd) => {
    if (Math.hypot(endX - x, endY - y) < dEnd) return false;
    for (let i = 0; i < route.length; i += 2) if (Math.hypot(route[i] - x, route[i + 1] - y) < dRoute) return false;
    return true;
  };
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const ctx = cv.getContext('2d');
  const [lo, hi] = extent(h);

  // Raster: resample, tint by band, shade.
  const R = Math.min(S, 720);
  const E = resample(h, n, R);
  const img = ctx.createImageData(R, R);
  const px = img.data;
  const alt = (40 * Math.PI) / 180;
  const lx = -Math.cos(alt) * Math.SQRT1_2;
  const ly = -Math.cos(alt) * Math.SQRT1_2;
  const lz = Math.sin(alt);
  const Z = 0.22 * R; // pixels of relief per unit of C
  let seed = 12345;
  const grain = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff - 0.5) * 6;
  };
  for (let i = 0; i < R; i++) {
    const up = Math.max(0, i - 1) * R;
    const dn = Math.min(R - 1, i + 1) * R;
    for (let j = 0; j < R; j++) {
      const k = i * R + j;
      const e = E[k];
      let r;
      let g;
      let b;
      if (e < WATER) {
        const depth = Math.min(1, (WATER - e) / WATER) * 0.7;
        r = WATER_FILL[0] + (WATER_DEEP[0] - WATER_FILL[0]) * depth;
        g = WATER_FILL[1] + (WATER_DEEP[1] - WATER_FILL[1]) * depth;
        b = WATER_FILL[2] + (WATER_DEEP[2] - WATER_FILL[2]) * depth;
      } else {
        const c = TINTS[bandOf(e)];
        r = c[0];
        g = c[1];
        b = c[2];
        const dx = ((E[i * R + Math.min(R - 1, j + 1)] - E[i * R + Math.max(0, j - 1)]) / 2) * Z;
        const dy = ((E[dn + j] - E[up + j]) / 2) * Z;
        const shade = (-dx * lx - dy * ly + lz) / Math.sqrt(dx * dx + dy * dy + 1);
        const d = shade - lz;
        if (d < 0) {
          const a = Math.min(0.34, -d * 0.7);
          r += (SHADOW[0] - r) * a;
          g += (SHADOW[1] - g) * a;
          b += (SHADOW[2] - b) * a;
        } else {
          const a = Math.min(0.55, d * 1.1);
          r += (LIGHT[0] - r) * a;
          g += (LIGHT[1] - g) * a;
          b += (LIGHT[2] - b) * a;
        }
      }
      const n0 = grain();
      px[k * 4] = r + n0;
      px[k * 4 + 1] = g + n0;
      px[k * 4 + 2] = b + n0;
      px[k * 4 + 3] = 255;
    }
  }
  if (R === S) ctx.putImageData(img, 0, 0);
  else {
    const tmp = document.createElement('canvas');
    tmp.width = tmp.height = R;
    tmp.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmp, 0, 0, S, S);
  }

  // Contours from a finer resampling of the grid, so the lines are smooth.
  const G = Math.min(260, n * 4);
  const F = resample(h, n, G);
  const toPx = (g) => (g / (G - 1)) * S;
  const lines = BANDS.slice(1)
    .map(([v]) => v)
    .filter((v) => v > lo && v < hi)
    .map((v) => ({ v, index: INDEX.has(v), polys: contour(F, G, G, v) }));

  // Number the heavy contours, at the middle of each long enough line.
  const fs = Math.round(10.5 * dpr);
  ctx.font = `italic ${fs}px ${font}`;
  const labels = [];
  for (const { v, index, polys } of lines) {
    if (!index) continue;
    const text = v.toFixed(1);
    const tw = ctx.measureText(text).width;
    for (const poly of polys) {
      const pts = poly.pts;
      if (polyLength(pts) * (S / (G - 1)) < tw * 5) continue;
      for (const frac of [0.5, 0.35, 0.65, 0.2, 0.8]) {
        const at = pointAt(pts, frac);
        const x = toPx(at.x);
        const y = toPx(at.y);
        if (x < tw || x > S - tw || y < fs * 2 || y > S - fs * 2) continue;
        if (!clear(x, y, tw * 0.9, 40 * dpr)) continue;
        if (labels.some((l) => Math.hypot(l.x - x, l.y - y) < tw * 4)) continue;
        let ang = at.angle;
        if (ang > Math.PI / 2) ang -= Math.PI;
        if (ang < -Math.PI / 2) ang += Math.PI;
        labels.push({ x, y, ang, text, w: tw + fs * 0.8, h: fs * 1.3 });
        break;
      }
    }
  }

  // Lines, broken where the numbers sit.
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, S, S);
  for (const l of labels) {
    const c = Math.cos(l.ang);
    const s = Math.sin(l.ang);
    const hw = l.w / 2;
    const hh = l.h / 2;
    const corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([u, w]) => [l.x + u * c - w * s, l.y + u * s + w * c]);
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 3; i >= 0; i--) ctx.lineTo(corners[i][0], corners[i][1]);
    ctx.closePath();
  }
  ctx.clip('evenodd');
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const { index, polys } of lines) {
    ctx.strokeStyle = index ? INK.index : INK.contour;
    ctx.lineWidth = (index ? 1.2 : 0.65) * dpr;
    ctx.beginPath();
    trace(ctx, polys, toPx);
    ctx.stroke();
  }
  ctx.restore();

  ctx.fillStyle = INK.label;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const l of labels) {
    ctx.save();
    ctx.translate(l.x, l.y);
    ctx.rotate(l.ang);
    ctx.fillText(l.text, 0, 0);
    ctx.restore();
  }

  // Shoreline, and the water's name near the middle of its largest stretch,
  // nudged off the route if they would collide.
  if (lo < WATER) {
    ctx.strokeStyle = INK.shore;
    ctx.lineWidth = 1.1 * dpr;
    ctx.beginPath();
    trace(ctx, contour(F, G, G, WATER), toPx);
    ctx.stroke();

    let sx = 0;
    let sy = 0;
    let count = 0;
    for (let i = 0; i < G * G; i++) {
      if (F[i] < WATER) {
        sx += i % G;
        sy += Math.floor(i / G);
        count++;
      }
    }
    if (count / (G * G) > 0.012) {
      const text = `C below ${WATER}`;
      ctx.font = `italic ${fs}px ${font}`;
      const tw = ctx.measureText(text).width;
      const cx = toPx(sx / count);
      const cy = toPx(sy / count);
      const fit = (x, y) => [
        Math.min(S - tw / 2 - 8 * dpr, Math.max(tw / 2 + 8 * dpr, x)),
        Math.min(S - 12 * dpr, Math.max(12 * dpr, y)),
      ];
      let spot = fit(cx, cy);
      // Offsets: a fraction of the label's width across, CSS pixels down.
      for (const [fx, dy] of [[0, 0], [0, 24], [0, -24], [0, 44], [0, -44], [-0.6, 24], [0.6, 24], [-0.6, -24], [0.6, -24]]) {
        const p = fit(cx + fx * tw, cy + dy * dpr);
        if (F[Math.round((p[1] / S) * (G - 1)) * G + Math.round((p[0] / S) * (G - 1))] >= WATER) continue;
        if (clear(p[0], p[1], tw * 0.65, 34 * dpr)) {
          spot = p;
          break;
        }
      }
      ctx.fillStyle = INK.water;
      ctx.fillText(text, spot[0], spot[1]);
    }
  }

  // Spot height on the highest ground.
  let top = 0;
  for (let i = 1; i < h.length; i++) if (h[i] > h[top]) top = i;
  const inset = 22 * dpr;
  const tx = Math.min(S - inset * 1.5, Math.max(inset, ((top % n) / (n - 1)) * S));
  const ty = Math.min(S - inset, Math.max(inset, (Math.floor(top / n) / (n - 1)) * S));
  ctx.fillStyle = INK.spot;
  ctx.beginPath();
  ctx.arc(tx, ty, 2 * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = `${fs}px ${font}`;
  ctx.textAlign = tx > S / 2 ? 'right' : 'left';
  ctx.fillText(h[top].toFixed(2), tx + (tx > S / 2 ? -5 : 5) * dpr, ty - 8 * dpr);

  return { canvas: cv, lo, hi };
}

function polyLength(p) {
  let s = 0;
  for (let i = 2; i < p.length; i += 2) s += Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]);
  return s;
}

function pointAt(p, frac) {
  const target = polyLength(p) * frac;
  let s = 0;
  for (let i = 2; i < p.length; i += 2) {
    const seg = Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]);
    if (s + seg >= target) {
      // Tangent from a short run either side, for a steadier angle.
      const a = Math.max(0, i - 6);
      const b = Math.min(p.length - 2, i + 4);
      return { x: p[i], y: p[i + 1], angle: Math.atan2(p[b + 1] - p[a + 1], p[b] - p[a]) };
    }
    s += seg;
  }
  return { x: p[0], y: p[1], angle: 0 };
}

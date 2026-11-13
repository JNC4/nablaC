// Marching squares: the lines along which a gridded field equals `level`,
// joined into polylines. Points are in grid units (x = column, y = row).

export function contour(v, w, h, level) {
  const nH = h * (w - 1);
  const E = nH + (h - 1) * w;
  const link = new Int32Array(E * 2).fill(-1);
  const px = new Float32Array(E);
  const py = new Float32Array(E);
  const seen = new Uint8Array(E);
  const used = [];

  const hEdge = (r, c) => r * (w - 1) + c; // (r,c)-(r,c+1)
  const vEdge = (r, c) => nH + r * w + c; // (r,c)-(r+1,c)

  const place = (e, x0, y0, x1, y1, a, b) => {
    if (seen[e]) return;
    seen[e] = 1;
    used.push(e);
    const t = (level - a) / (b - a);
    px[e] = x0 + (x1 - x0) * t;
    py[e] = y0 + (y1 - y0) * t;
  };
  const join = (e1, e2) => {
    link[e1 * 2 + (link[e1 * 2] < 0 ? 0 : 1)] = e2;
    link[e2 * 2 + (link[e2 * 2] < 0 ? 0 : 1)] = e1;
  };

  for (let r = 0; r < h - 1; r++) {
    for (let c = 0; c < w - 1; c++) {
      const tl = v[r * w + c];
      const tr = v[r * w + c + 1];
      const br = v[(r + 1) * w + c + 1];
      const bl = v[(r + 1) * w + c];
      const idx = (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0);
      if (idx === 0 || idx === 15) continue;
      const T = hEdge(r, c);
      const B = hEdge(r + 1, c);
      const L = vEdge(r, c);
      const R = vEdge(r, c + 1);
      const on = { T: false, B: false, L: false, R: false };
      const seg = (a, b) => {
        on[a] = on[b] = true;
        pairs.push(a, b);
      };
      const pairs = [];
      switch (idx) {
        case 1: case 14: seg('L', 'B'); break;
        case 2: case 13: seg('B', 'R'); break;
        case 3: case 12: seg('L', 'R'); break;
        case 4: case 11: seg('T', 'R'); break;
        case 6: case 9: seg('T', 'B'); break;
        case 7: case 8: seg('L', 'T'); break;
        case 5: case 10: {
          const mid = (tl + tr + br + bl) / 4 > level;
          // Corners tl and br above (10) or tr and bl above (5).
          if ((idx === 10) === mid) { seg('L', 'B'); seg('T', 'R'); }
          else { seg('L', 'T'); seg('B', 'R'); }
          break;
        }
      }
      if (on.T) place(T, c, r, c + 1, r, tl, tr);
      if (on.B) place(B, c, r + 1, c + 1, r + 1, bl, br);
      if (on.L) place(L, c, r, c, r + 1, tl, bl);
      if (on.R) place(R, c + 1, r, c + 1, r + 1, tr, br);
      const id = { T, B, L, R };
      for (let i = 0; i < pairs.length; i += 2) join(id[pairs[i]], id[pairs[i + 1]]);
    }
  }

  // Walk the links into polylines, starting from open ends first.
  const done = new Uint8Array(E);
  const lines = [];
  const walk = (start) => {
    const pts = [];
    let prev = -1;
    let cur = start;
    while (cur >= 0 && !done[cur]) {
      done[cur] = 1;
      pts.push(px[cur], py[cur]);
      const a = link[cur * 2];
      const b = link[cur * 2 + 1];
      const next = a !== prev ? a : b;
      prev = cur;
      cur = next;
    }
    const closed = cur >= 0 && cur === start;
    if (pts.length >= 4) lines.push({ pts, closed });
  };
  for (const e of used) if (!done[e] && link[e * 2 + 1] < 0) walk(e);
  for (const e of used) if (!done[e]) walk(e);
  return lines;
}

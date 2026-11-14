// The cost map on screen: the surveyed sheet, then the route walked so far
// on top of it, the start, and the red dot that is the network now.

const FONT = '"Text", Georgia, serif';

export function drawTerrain(ctx, S, dpr, run, s, now, still, message) {
  ctx.fillStyle = '#efe9da';
  ctx.fillRect(0, 0, S, S);
  if (!run?.relief) {
    ctx.fillStyle = '#8a7f72';
    ctx.font = `italic ${Math.round(14 * dpr)}px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, S / 2, S / 2);
    return;
  }
  const fade = still ? 1 : Math.min(1, (now - run.reliefAt) / 600);
  ctx.globalAlpha = fade;
  ctx.drawImage(run.relief.canvas, 0, 0);
  ctx.globalAlpha = 1;

  const { path, win } = run.plane;
  const X = (a) => ((a - win.a0) / (win.a1 - win.a0)) * S;
  const Y = (b) => ((win.b1 - b) / (win.b1 - win.b0)) * S;

  // The way it has come: a red track with a pale casing, as on a walking map.
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(X(path[0]), Y(path[1]));
  const every = Math.max(1, Math.ceil(s / 300)); // the route is smooth; 300 points draw it
  for (let t = every; t < s; t += every) ctx.lineTo(X(path[t * 2]), Y(path[t * 2 + 1]));
  ctx.lineTo(X(path[s * 2]), Y(path[s * 2 + 1]));
  ctx.strokeStyle = 'rgba(251, 247, 238, 0.8)';
  ctx.lineWidth = 4.6 * dpr;
  ctx.stroke();
  ctx.strokeStyle = '#c8302a';
  ctx.lineWidth = 2.2 * dpr;
  ctx.setLineDash([7 * dpr, 4 * dpr]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Where it set out from.
  const sx = X(path[0]);
  const sy = Y(path[1]);
  ctx.beginPath();
  ctx.arc(sx, sy, 4.2 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = '#fbf7ee';
  ctx.fill();
  ctx.lineWidth = 1.5 * dpr;
  ctx.strokeStyle = '#2b2620';
  ctx.stroke();
  ctx.font = `italic ${Math.round(12.5 * dpr)}px ${FONT}`;
  ctx.fillStyle = '#2b2620';
  const right = sx < S * 0.72;
  ctx.textAlign = right ? 'left' : 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('start', sx + (right ? 9 : -9) * dpr, sy);

  // Where it is now.
  ctx.beginPath();
  ctx.arc(X(path[s * 2]), Y(path[s * 2 + 1]), 6 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = '#c8302a';
  ctx.fill();
  ctx.lineWidth = 1.8 * dpr;
  ctx.strokeStyle = '#fbf7ee';
  ctx.stroke();
}

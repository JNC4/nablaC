// Trains one run off the main thread, streaming every step back as it goes,
// then works out the plane the map will be drawn in.

import { setup, train } from './run.js';
import { planeOfRun } from './survey.js';

self.onmessage = ({ data: msg }) => {
  if (msg.type !== 'run') return;
  const { id, eta, seed, steps } = msg;
  const { net, data } = setup(msg.task);
  const P = net.nParams;
  const all = new Float32Array((steps + 1) * P);

  const end = train({
    net, data, eta, seed, steps,
    onChunk(from, thetas, costs) {
      all.set(thetas, from * P);
      // Copies go to the main thread; `all` stays here for the survey.
      self.postMessage({ type: 'steps', id, from, thetas: thetas.slice(), costs: costs.slice() });
    },
  });

  const { dirs, path, inPlane, win } = planeOfRun({ net, seed, all, end, steps });
  self.postMessage({ type: 'plane', id, end, dirs, path, inPlane, win });
};

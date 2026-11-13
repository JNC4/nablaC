// Measures the cost over a band of rows of the map grid.

import { setup } from './run.js';
import { heights } from './survey.js';

self.onmessage = ({ data: msg }) => {
  const { id, task, end, dirs, win, n, r0, r1 } = msg;
  const { net, data } = setup(task);
  const h = heights(net, data, end, dirs, win, n, r0, r1);
  self.postMessage({ id, n, r0, r1, h }, [h.buffer]);
};

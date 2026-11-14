// Measures the cost over a band of rows of the map grid.

import { makeNetwork } from './run.js';
import { heights } from './survey.js';

const net = makeNetwork();

self.onmessage = ({ data: msg }) => {
  const { id, data, end, dirs, win, n, r0, r1 } = msg;
  const h = heights(net, data, end, dirs, win, n, r0, r1);
  self.postMessage({ id, n, r0, r1, h }, [h.buffer]);
};

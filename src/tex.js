// A very small typesetter for the equations: italic variables with stacked
// superscript and subscript, built-up fractions, Σ with its index below, and
// numbers with a true minus sign. Output is HTML, styled in style.css.

export function num(v, dp = 2) {
  if (!Number.isFinite(v)) return '?';
  const a = Math.abs(v);
  let s;
  if (a !== 0 && a < 0.5 * 10 ** -dp) {
    const e = Math.floor(Math.log10(a));
    s = `${(a / 10 ** e).toFixed(1)}×10<sup>−${-e}</sup>`;
  } else {
    s = a.toFixed(dp);
  }
  const neg = v < 0 && s !== (0).toFixed(dp);
  return `<span class="n">${neg ? '−' : ''}${s}</span>`;
}

// A number that can stand next to a multiplication: negatives get brackets.
export const factor = (v, dp = 2) => (v < 0 ? `(${num(v, dp)})` : num(v, dp));

// A variable: v('a', 'j', '2') → a with (2) above and j below.
export function v(base, sub = '', layer = '') {
  const sup = layer === '' ? '' : `(${layer})`;
  if (!sub && !sup) return `<i class="v">${base}</i>`;
  return `<i class="v">${base}</i><span class="ss"><span class="ss__sup">${sup}</span><span class="ss__sub">${sub}</span></span>`;
}

export const frac = (n, d) => `<span class="frac"><span class="frac__n">${n}</span><span class="frac__d">${d}</span></span>`;

export const sum = (index) => `<span class="sum"><span class="sum__s">Σ</span><span class="sum__i">${index}</span></span>`;

export const d = (top, bottom) => frac(`∂${top}`, `∂${bottom}`);

export const sigma = (arg) => `<i class="v">σ</i>(${arg})`;
export const sigmaPrime = (arg) => `<i class="v">σ</i>′(${arg})`;

export const row = (html, cls = '') => `<div class="eq${cls ? ` ${cls}` : ''}">${html}</div>`;

export const op = (s) => `<span class="op">${s}</span>`;

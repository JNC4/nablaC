# ∇C

**In ten seconds:** on the left, a small neural network's guesses about some coloured dots go from muddle to a clean fit; on the right, a red dot walks down a contour map of the network's error until it reaches the lake at the bottom. Learning is that walk downhill.

The companion piece to the essay *Constructing ∇C*, which builds neural-network learning from high-school maths. Same notation: C is the cost, ∇C its gradient, and every step of learning is θ ← θ − η∇C.

```bash
npm install
npm run dev        # http://localhost:5102
npm run build      # regenerates the opening sheet, then builds to dist/
npm run validate   # the checks below; exits non-zero on failure
```

## What you are looking at

- **What it thinks** (left). The plane the dots live in, coloured by the group the network would answer at each spot, paler where it is unsure, a thin line where its answer changes. A ring marks every dot it currently gets wrong, and the count sits under the map.
- **How wrong it is** (right). A contour map of the cost. Every point on it is one complete setting of the network's 371 weights and biases; the height there is the error C that setting makes on the 150 dots. The red dot is the network on the left, and the dashed track is the way it has come. Water is where C is below 0.05.
- **The route profile** under the map: C against step number, coloured with the same height tints.

It runs by itself: a random start, 1500 steps played over seven seconds, a rest at the bottom, then a new random start (trained and surveyed in the background while you look at the last one). Two controls:

- **Step size** (η): small (0.5) never gets off the plateau in 1500 steps and never reaches the water; medium (3) gets there; large (25) jumps straight uphill onto a flat plateau where every output neuron is saturated, chatters there, and then usually crashes down to a lower floor than medium reaches. On two of the first twelve random starts it never comes back down. All three are real outcomes; nothing is staged.
- **Start again**: new random weights.

## Art direction

A sheet from a mid-century layer-coloured map: Bartholomew's half-inch maps of Britain for the hypsometric tints (green lowland through buff and brown to plum summits, one flat tint per band of height), Eduard Imhof's Swiss relief shading for the soft modelling lit from the upper left, brown contours with every fifth one heavier and numbered along the line, a blue shoreline, a spot height on the summit, and the route drawn the way Swiss walking maps draw paths: red dashes with a pale casing. The network's beliefs on the left use the vocabulary of a geological map instead: flat colour washes for each unit and thin black lines at the boundaries. Cream paper with a faint grain, Source Serif 4 for all text, set in italic for captions on the map as a cartographer would.

Why it fits: the essay already reaches for this picture. It describes gradient descent as rolling a ball down hills and says that in more dimensions "the peaks and valleys look more like a topographic map". A contour map is the one diagram of a landscape that almost everyone can read without being taught, so the claim "learning is walking downhill on your error" can be shown rather than stated. Layer tints in particular do work here: green means low error and brown means high on every sheet, because the bands are fixed values of C, not fractions of whatever a sheet happens to span.

## What the maths actually does

**Network.** 2 inputs → 16 → 16 → 3 outputs, sigmoid everywhere, as in the essay. 2·16 + 16 + 16·16 + 16 + 16·3 + 3 = 371 parameters, held in one flat vector θ. Starting weights are Glorot-uniform, biases zero, from a seeded mulberry32 generator.

```
z(l) = W(l) a(l−1) + b(l)        a(l) = σ(z(l))        σ(z) = 1 / (1 + e^−z)
```

**Cost.** The essay's squared error against a one-hot target, averaged over the 150 training dots:

```
C(θ) = (1/N) Σ_n Σ_j ( a_j(out) − y_j )²
```

**Gradient.** Backpropagation, written out by hand in `src/net.js` exactly as the essay derives it with the chain rule:

```
δ(out)  = 2 (a(out) − y) ⊙ σ'(z(out))          σ'(z) = a (1 − a)
∂C/∂W(l) = δ(l) a(l−1)ᵀ        ∂C/∂b(l) = δ(l)
δ(l−1)  = (W(l)ᵀ δ(l)) ⊙ σ'(z(l−1))
```

summed over the dots and divided by N. Full batch, no momentum, no tricks: θ ← θ − η ∇C, 1500 times. A second, unrelated gradient engine (a scalar reverse-mode autograd, carried over from the first version of this piece) lives in `validate/value.js` purely to check this one.

**Data.** Three spiral arms, 50 dots each, a third of a turn apart, with Gaussian noise (σ = 0.05). Seeded, so identical on every machine.

**The map.** C lives in 371 dimensions, so the map is a flat slice, chosen after the walk:

```
d₁ = (θ₀ − θ_end) / ‖θ₀ − θ_end‖
d₂ = top principal direction of the steps θ_t − θ_end once their d₁ component is removed
height(α, β) = C(θ_end + α d₁ + β d₂)
```

Using the start-to-end line as the first axis means both ends of the route sit exactly on the surveyed ground; the second axis is the direction in which the route strays furthest from that line. (Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018, use the top two principal directions; this is a variant that pins the start.) The principal direction is found by power iteration on the Gram matrix of every third step. The route drawn is every step projected onto the plane. The window is square with equal scale on both axes, so a contour crossing at right angles really is the steepest way down within the slice. Heights are measured on a 72 × 72 grid (each cell is a full forward pass over all 150 dots), split across three Web Workers with a quick 28 × 28 pass first, then Catmull-Rom resampled for the raster and the contours. Contours are marching squares, joined into polylines.

**Timing.** Training runs in a worker and streams every step back; the page plays them at about 214 steps a second. The map cannot exist until the walk is over (the plane depends on the whole route), so the opening sheet (seed 1, η = 3) is surveyed by `scripts/opening.mjs` at build time, using the same modules. The page still trains that run live, compares its final weights against the shipped ones (tolerance 1e-4), and surveys again itself if they differ.

## The numbers

There is no published number to reproduce here; the old piece did not have one either. What `npm run validate` checks instead is that the piece computes what it claims. Output from the current code:

```
1. scalar autograd against finite differences
   f(a,b,c) = tanh(ab + c²)·σ(a − b) + eᶜ/(b + 2) − log(a² + 1)
  ok   ∂f/∂a: autograd -1.68939623, finite difference -1.68939623, |Δ| 9.16e-12
  ok   ∂f/∂b: autograd -2.61002391, finite difference -2.61002391, |Δ| 2.47e-10
  ok   ∂f/∂c: autograd 2.55155497, finite difference 2.55155497, |Δ| 2.06e-11

2. the piece's backpropagation against the scalar autograd (371 weights and biases)
  ok   random start (seed 4): C = 0.8825267234 both ways
  ok   random start (seed 4): largest disagreement in ∇C 2.50e-16 (largest component 0.1484)
  ok   after 200 steps: C = 0.3788677277 both ways
  ok   after 200 steps: largest disagreement in ∇C 4.08e-17 (largest component 0.0017)

3. the piece's ∇C against finite differences of C (40 components chosen at random)
  ok   largest disagreement 7.10e-10

4. gradient descent learns the task (η = 3, 1500 steps, ten random starts)
  ok   C falls to under 5% of where it started on every start (worst 3.0%)
  ok   every start ends with at least 98% of dots right (worst 99.3%)

5. the map is the real cost where the route starts and ends
  ok   start: map 0.801982, network 0.801982
  ok   end: map 0.015312, network 0.015312
  ok   98.9% of the route's squared length lies in the drawn plane

6. src/opening.json matches the current code
  ok   final weights agree to 4.98e-7

all checks passed
```

Outcomes over the first twelve random starts (`η`, final C after 1500 steps):

| step size | η | final C | dots right | notes |
|---|---|---|---|---|
| small | 0.5 | 0.36 to 0.38 | 63% to 69% | still on the plateau, never reaches the water |
| medium | 3 | 0.014 to 0.021 | 99.3% to 100% | small uphill bumps up to +0.05 along the way |
| large | 25 | 0.0003 to 0.0005, or stuck at 1.000 | 100%, or 33% | 2 of 12 starts saturate every output and never recover |

## What is approximate, and what was cut

- **The map is a slice, not the landscape.** It is the true cost on a plane through 371-dimensional space. The route is projected onto that plane; for the opening run 98.9% of its squared length lies in the plane, but in between its ends the dot can sit slightly above or below the drawn ground, and it can appear to cross a contour sideways. The profile under the map is the true C at every step, not a reading off the map.
- **"Steepest downhill" is in all 371 dimensions.** −∇C is the steepest direction in the full space; its shadow on the slice need not be perpendicular to the drawn contours.
- **The map is drawn after the fact.** The network only ever knew the slope where it stood. The page trains the whole run (well under a second on a laptop, about two seconds in headless Chromium) and plays it back at a watchable pace; the walk is computed, not animated toward a known answer, but you are watching a replay a few hundred milliseconds behind the computation.
- **Map resolution.** 72 × 72 measured heights, interpolated. Thin features between grid points (the cliff the large step size walks along) are smoothed. The spot height is the highest measured grid value, not a true maximum.
- **The water threshold** (C < 0.05) is a choice, labelled on the map.
- **Medium usually ends with one dot wrong.** The noise puts one dot inside another arm; the network fits around it on some starts and not others. It is shown as it is.
- **Cut from the old piece:** the network diagram, the attention view, the dataset/shape/activation form and the TRAIN button. The task is fixed (three spirals) and the network is fixed (two hidden layers of 16) so that the only thing that changes is the thing the piece is about. Cross-entropy and softmax were replaced by the essay's squared error and sigmoid outputs.
- **Verified only in headless Chromium**, with and without SwiftShader, at 1440 × 900 and 390 × 844. Not yet looked at in Firefox or Safari, or on a real phone. Under SwiftShader the canvases are slow to composite and screenshot timing is unreliable; with `--disable-gpu` the per-frame work measured about 15 ms median on a loaded machine.

## Dependencies

None at runtime. Vite is a dev dependency. Fonts are self-hosted and subset: Source Serif 4 (roman and italic, weights 400 to 600, about 160 KB), plus the single ∇ glyph from Libertinus Serif because Source Serif has none; both OFL, licences in `src/fonts/`. Built output is about 270 KB including fonts.

| file | lines | |
|---|---|---|
| `src/net.js` | 153 | network, cost, hand-written backprop |
| `src/run.js` | 48 | one training run, shared by browser and checks |
| `src/data.js` | 46 | the three arms |
| `src/survey.js` | 177 | the plane and the height grid |
| `src/contour.js` | 93 | marching squares |
| `src/relief.js` | 347 | the layer-coloured sheet |
| `src/territory.js` | 123 | the beliefs map |
| `src/profile.js` | 102 | the route profile |
| `src/main.js` | 295 | workers, playback, drawing the route |
| `src/*.worker.js` | 36 | training and surveying off the main thread |
| `scripts/opening.mjs` | 44 | surveys the opening sheet at build time |
| `validate/` | 284 | scalar autograd and the checks |

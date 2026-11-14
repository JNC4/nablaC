# ∇C

**In ten seconds:** on the left, a small neural network's guesses about some coloured dots go from muddle to a clean fit; in the middle, the network itself, its neurons lighting up for one dot at a time and a red pulse of ∇C running back through its weights; on the right, a red dot walks down a contour map of the network's error until it reaches the lake at the bottom. Learning is that walk downhill.

The companion piece to the essay *Constructing ∇C*, which builds neural-network learning from high-school maths. Same notation: C is the cost, C₀ the cost of one example, ∇C the gradient, layers written a⁽ˡ⁾ with (O) for the output, and every step of learning is w ← w − η ∂C/∂w.

```bash
npm install
npm run dev        # http://localhost:5102
npm run build      # regenerates the opening sheet, then builds to dist/
npm run validate   # the checks below; exits non-zero on failure
```

## What you are looking at

Three figures side by side on a wide screen (map, network, map), stacked on a phone.

- **What it thinks** (left). The plane the dots live in, coloured by the group the network would answer at each spot, paler where it is unsure, a thin line where its answer changes. A ring marks every dot it currently gets wrong, and the count sits under the map. A red crosshair marks the point being fed through the network.
- **The network** (middle). 2 inputs, two hidden layers of 16, 3 outputs, every one of the 336 weights drawn as a line: ink for positive, blue for negative, heavier the bigger it is. Neurons fill with ink as they activate for the crosshaired point; the three outputs fill with their colour. When the point moves, the change travels through the drawing left to right, layer by layer, and the lines carrying the most signal (largest |w·a|) darken. While the network trains, a red pulse runs right to left every 1.5 s: each line glows in proportion to how hard ∇C is pulling on that weight at the current step, so the pulses fade as the gradient shrinks.
- **The equations** (under the network), typeset, with the live numbers in them. Left alone they show the whole network as one function, the crosshaired dot's way in and way out, its own cost C₀, the average cost C and the update rule. Point at (or tap) a neuron and they become that neuron's a = σ(z), its weighted sum z with the biggest terms written out, ∂C₀/∂a for the crosshaired dot, ∂C/∂b, and the bias it will have after this step. Point at a line and they become that weight's ∂C/∂w as the chain rule ∂z/∂w · ∂a/∂z · ∂C₀/∂a averaged over every dot, the same chain for the crosshaired dot alone with its three factors, and the weight's next value.
- **How wrong it is** (right). A contour map of the cost. Every point on it is one complete setting of the network's 371 weights and biases; the height there is the error C that setting makes on the dots. The red dot is the network, and the dashed track is the way it has come. Water is where C is below 0.05. Under it, the route profile: C against step number.

It runs by itself: a random start, 1500 steps played over seven seconds, the crosshair touring the dots (mostly the ones still wrong) every 2.6 s, a rest at the bottom, then a new random start that was trained and surveyed in the background while you looked at the last one.

What you can do:

- **Draw.** Pick a colour and click or drag on the left map (or touch, on a phone) to lay down dots, about one every 0.055 units along the stroke; "rub out" erases under a dashed circle; "clear" removes everything. A quarter of a second after the stroke ends the network carries on training from where it stands, on the new dots, and the cost map is surveyed again. Presets: spiral (the opening), rings, islands. Up to 360 dots.
- **Look.** Hover the left map to feed any point through the network (it snaps to a dot when you are near one, so the dot's own cost appears); click to pin it. Hover or tap the network for a neuron's or a weight's equations.
- **Step size** (η): small (0.5) never gets off the plateau in 1500 steps; medium (3) reaches the water; large (25) jumps straight uphill onto a flat plateau where the outputs are saturated, chatters there, and usually crashes down to a lower floor than medium reaches. On two of the first twelve random starts it never comes back down.
- **Start again**: new random weights.

## Art direction

A sheet from a mid-century printed atlas or textbook. The cost map borrows Bartholomew's layer-coloured half-inch maps for its tints (green lowland through buff and brown to plum summits, one flat tint per band of height), Eduard Imhof's Swiss relief shading for the soft modelling lit from the upper left, brown contours numbered along the line, a blue shoreline, a spot height on the summit, and the route drawn the way Swiss walking maps draw paths: red dashes with a pale casing. The beliefs map uses a geological map's vocabulary: flat washes and thin black boundaries. The network is drawn the way an old textbook figure would draw it, in two inks on the same paper, with no glow and no gradients of its own: circles filled like halftone by their activation, lines weighted by size, labelled a⁽⁰⁾ a⁽¹⁾ a⁽²⁾ a⁽ᴼ⁾ underneath. The equations are set in Source Serif 4 with real built-up fractions, stacked indices and a Σ with its index below, so that they read like the essay's own equations rather than a readout. Red is used for one idea throughout: the step. It is the route on the map, the dot that walks it, the gradient pulse in the network and the thing you are pointing at.

Why it fits: the essay describes gradient descent as rolling a ball down hills and says that in more dimensions "the peaks and valleys look more like a topographic map", and the essay itself is a textbook. A contour map is the one diagram of a landscape that almost everyone can read without being taught, and a textbook figure is the register the network and its maths came from.

## What the maths actually does

**Network.** 2 inputs → 16 → 16 → 3 outputs, sigmoid everywhere, as in the essay. 2·16 + 16 + 16·16 + 16 + 16·3 + 3 = 371 parameters, held in one flat vector θ. Starting weights are Glorot-uniform, biases zero, from a seeded mulberry32 generator.

```
z(l) = W(l) a(l−1) + b(l)        a(l) = σ(z(l))        σ(z) = 1 / (1 + e^−z)
```

**Cost.** The essay's squared error against a one-hot target y, per dot and averaged:

```
C₀ = Σ_j ( a_j(O) − y_j )²          C = (1/n) Σ_dots C₀
```

**Gradient.** Backpropagation, written out by hand in `src/net.js` (`costAndGrad`) as the essay derives it with the chain rule:

```
δ(O)    = 2 (a(O) − y) ⊙ σ'(z(O))          σ'(z) = a (1 − a)
∂C/∂W(l) = δ(l) a(l−1)ᵀ        ∂C/∂b(l) = δ(l)
δ(l−1)  = (W(l)ᵀ δ(l)) ⊙ σ'(z(l−1))
```

summed over the dots and divided by n. Full batch, no momentum: θ ← θ − η ∇C, 1500 times per run.

**The readouts** use a second, deliberately separate route to the same numbers, written the way the essay writes it for a single weight. `probe()` runs one point forward, keeping every z and a, and for a labelled dot backward, keeping ∂C₀/∂a for every neuron (∂C₀/∂a_j(O) = 2(a_j − y_j), then ∂C₀/∂a_k(l−1) = Σ_j w_jk σ′(z_j) ∂C₀/∂a_j). `chainGrad()` then forms ∂C/∂w_jk(l) as the average over dots of a_k(l−1) · σ′(z_j(l)) · ∂C₀/∂a_j(l). The validation checks this equals `costAndGrad` on every weight and bias, and that one dot's factor equals a finite difference of that dot's C₀. A third engine, a scalar reverse-mode autograd carried over from the first version of this piece, lives in `validate/value.js` purely to check `costAndGrad`.

**Drawing.** Every edit rebuilds the training set. When a stroke ends, a new run starts from the weights currently on screen (not from scratch), so you watch the network adjust rather than relearn. A new preset, a step-size change, Start again and the automatic loop all start from fresh random weights.

**Data.** Spiral: three arms of 50 dots, a third of a turn apart, Gaussian noise σ = 0.05. Rings: a disc and two annuli (36, 52 and 62 dots). Islands: five Gaussian clusters for three colours, two colours split across opposite corners. All seeded.

**The map.** C lives in 371 dimensions, so the map is a flat slice, chosen after the walk:

```
d₁ = (θ₀ − θ_end) / ‖θ₀ − θ_end‖
d₂ = top principal direction of the steps θ_t − θ_end once their d₁ component is removed
height(α, β) = C(θ_end + α d₁ + β d₂)
```

Using the start-to-end line as the first axis means both ends of the route sit exactly on the surveyed ground. (Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018, use the top two principal directions; this variant pins the start.) Power iteration on the Gram matrix of every third step; the route is every step projected onto the plane; the window is square with equal scale on both axes. Heights are measured on a 72 × 72 grid (each cell a full forward pass over every dot), split across three Web Workers after a quick 28 × 28 pass, then Catmull-Rom resampled. Contours are marching squares joined into polylines.

**Timing.** Training runs in a worker and streams every step back; the page plays them at about 214 steps a second, starting as soon as the first steps arrive, and the map fades in when its survey is done. The opening sheet (spiral, seed 1, η = 3) is surveyed by `scripts/opening.mjs` at build time with the same modules; the page still trains that run live, compares its final weights against the shipped ones (tolerance 1e-4), and surveys again itself if they differ.

## The numbers

There is no published number to reproduce here; the old piece did not have one either. `npm run validate` checks instead that the piece computes what it shows. Output from the current code:

```
1. scalar autograd against finite differences
   f(a,b,c) = tanh(ab + c²)·σ(a − b) + eᶜ/(b + 2) − log(a² + 1)
  ok   ∂f/∂a: autograd -1.68939623, finite difference -1.68939623, |Δ| 9.16e-12
  ok   ∂f/∂b: autograd -2.61002391, finite difference -2.61002391, |Δ| 2.47e-10
  ok   ∂f/∂c: autograd 2.55155497, finite difference 2.55155497, |Δ| 2.06e-11

2. the piece's backpropagation against the scalar autograd (371 weights and biases)
  ok   random start (seed 4): C = 0.8825267234 both ways
  ok   random start (seed 4): largest disagreement in ∇C 2.50e-16 (largest component 0.1484)
  ok   after 200 steps: C = 0.3788677278 both ways
  ok   after 200 steps: largest disagreement in ∇C 6.32e-17 (largest component 0.0017)

3. the piece's ∇C against finite differences of C (40 components chosen at random)
  ok   largest disagreement 7.10e-10

4. the equations under the network
  ok   ∂C/∂w as (1/n) Σ a·σ′(z)·∂C₀/∂a matches ∇C on all 371 weights and biases, worst 1.81e-17
  ok   ∂C₀/∂w for one dot, a·σ′(z)·∂C₀/∂a, against finite differences of C₀ (40 random dots and weights), worst 2.13e-10
  ok   a(O) and C₀ in the readouts match the forward pass and the cost exactly (worst 0.00e+0)

5. gradient descent learns every preset (η = 3, 1500 steps, five random starts each)
  ok   spiral (150 dots): C falls to 2.4% of its start or less, at least 99.3% of dots right
  ok   rings (150 dots): C falls to 2.7% of its start or less, at least 100.0% of dots right
  ok   islands (150 dots): C falls to 0.5% of its start or less, at least 100.0% of dots right

6. retraining on drawn dots (spiral, trained from seed 1, plus a stroke)
  ok   green, across the upper left: carrying on, C 0.239 → 0.051, drawn dots wrong 21 → 2 of 21
  ok   ochre, down the right edge: carrying on, C 0.189 → 0.031, drawn dots wrong 15 → 1 of 15
  ok   green, along the very top: from fresh random weights, 1 of 21 drawn dots wrong after 1500 steps
  note green, along the very top: carrying on instead gets stuck, C ends at 0.150 with 21 of 21 drawn dots still wrong (saturated neurons; see README)

7. the map is the real cost where the route starts and ends
  ok   start: map 0.801982, network 0.801982
  ok   end: map 0.015312, network 0.015312
  ok   98.9% of the route's squared length lies in the drawn plane

8. src/opening.json matches the current code
  ok   final weights agree to 4.98e-7

all checks passed
```

Outcomes over the first twelve random starts on the spiral (final C after 1500 steps):

| step size | η | final C | dots right | notes |
|---|---|---|---|---|
| small | 0.5 | 0.36 to 0.38 | 63% to 69% | still on the plateau, never reaches the water |
| medium | 3 | 0.014 to 0.021 | 99.3% to 100% | small uphill bumps up to +0.05 along the way |
| large | 25 | 0.0003 to 0.0005, or stuck at 1.000 | 100%, or 33% | 2 of 12 starts saturate every output and never recover |

## What is approximate, and what was cut

- **Carrying on after a stroke can get stuck.** Retraining starts from the weights on screen so the change is visible as an adjustment. A trained sigmoid network has saturated neurons, and some strokes land where those neurons barely respond: check 6 above draws one such stroke, and 1500 steps from the trained weights leave all 21 dots wrong while fresh random weights learn them. The piece shows this as it happens. The next automatic restart, or Start again, begins from fresh weights and usually fixes it.
- **The map is a slice, not the landscape.** The true cost on a plane through 371 dimensions. For the opening run 98.9% of the route's squared length lies in the plane, but between its ends the dot can sit slightly above or below the drawn ground and appear to cross a contour sideways. The profile under the map is the true C at every step.
- **"Steepest downhill" is in all 371 dimensions.** −∇C is steepest in the full space; its shadow on the slice need not cross the drawn contours at right angles.
- **The map is drawn after the fact,** because the plane depends on the whole route. The network only ever knew the slope where it stood. Each run is computed in a worker (about two seconds for 150 dots in headless Chromium on a busy machine, longer in proportion to the number of dots) and played back at a watchable pace a little behind the computation. After a stroke the map reads "Measuring the ground" for a second or two while it is surveyed.
- **The gradient pulse** is the true |∂C/∂w| at the step on screen, but the brightness curve is chosen: each line is drawn at √(|∂C/∂w| / (0.6 × the largest |∂C/∂w| at that step)), and the whole pulse dims as (largest now / largest at the start of the run)^0.3, never below a quarter. So which lines glow, and in what order, is real; how bright the pulse is in absolute terms is a drawing choice that keeps it visible to the end. Biases have no lines, so their gradients appear only in the equations.
- **Edge hover on dense layers.** The 16 × 16 layer has 256 lines in a narrow column; the nearest line within about 4.5 pixels is the one shown. On a phone tapping a particular line there is fiddly.
- **The z equation** writes out the two largest terms of a 16-term sum and elides the rest with ⋯; the total printed is the full sum.
- **Map resolution.** 72 × 72 measured heights, interpolated. The spot height is the highest grid value, not a true maximum. The water threshold (C < 0.05) is a choice, labelled on the map.
- **Cut:** the old piece's attention view and the dataset/shape/activation form. The network shape is fixed (2-16-16-3); what you change is the data and the step size.
- **Verified only in headless Chromium** (software rendering, `--disable-gpu`), at 1440 × 900, 1024 × 768 and 390 × 844, with drawing driven by synthetic mouse and touch pointer events. Not looked at in Firefox or Safari or on a real phone, and real-finger drawing on a touch screen is untested. In that software renderer the page ran at about 40 frames a second with no long tasks after the opening; a GPU-backed browser should do better.

## Dependencies

None at runtime. Vite is a dev dependency. Fonts are self-hosted and subset: Source Serif 4 (roman and italic, weights 400 to 600, Latin plus σ, Σ, δ, η, θ, ∂, ′ and arrows, about 165 KB), plus ∇, ⋅ and ∈ from Libertinus Serif because Source Serif lacks them; both OFL, licences in `src/fonts/`. Built output is about 285 KB including fonts.

| file | lines | |
|---|---|---|
| `src/net.js` | 226 | network, cost, hand-written backprop, the probe and the chain-rule readouts |
| `src/run.js` | 43 | one training run, shared by browser and checks |
| `src/data.js` | 82 | presets and packing drawn dots |
| `src/survey.js` | 177 | the plane and the height grid |
| `src/contour.js` | 93 | marching squares |
| `src/relief.js` | 355 | the layer-coloured sheet |
| `src/terrain.js` | 68 | the route on the sheet |
| `src/territory.js` | 178 | the beliefs map, dots, crosshair |
| `src/netview.js` | 271 | the network figure |
| `src/maths.js`, `src/tex.js` | 143 | the equations and their typesetting |
| `src/profile.js` | 102 | the route profile |
| `src/main.js` | 547 | workers, playback, the tour, drawing, pointing |
| `src/*.worker.js` | 38 | training and surveying off the main thread |
| `scripts/opening.mjs` | 48 | surveys the opening sheet at build time |
| `validate/` | 394 | scalar autograd and the checks |

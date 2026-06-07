# nablaC

*∇C, the gradient of the cost, made visible.*

A neural network that trains in your browser with its machinery drawn instead of
hidden. No library, no pretrained weights. A from-scratch reverse-mode autograd
builds the network out of scalar `Value`s and you watch it learn: the forward
pass lights up the neurons, the backward pass floods **∇C** back through every
weight, and the decision boundary bends around the data while the cost curve
drops to the floor.

The name is the point. ∇C is the gradient of the cost function, one vector that
tells every parameter which way is downhill. Subtract a sliver of it enough times
and a pile of random numbers turns into something that classifies, then something
that attends, then the thing the field likes to call "intelligence." nablaC
doesn't claim any magic, because there isn't any. You compute a gradient, take a
small step downhill, and repeat until the loss stops falling.

It's a sibling to [YALDABAOTH](https://github.com/JNC4/markovsw), which drew a Markov chain's decision
the same way this draws a network's gradient: take the mechanism people treat as
a black box and make it legible.

## One accent colour

Everything that has settled on screen is dim and monochrome. The one bright thing
is the live gradient, electric cyan, used only for ∇C as it floods backward.
Weight sign reads warm (excitatory) or cool (inhibitory), magnitude is edge
thickness, and neuron activation is a warm glow on the forward sweep. Cyan only
ever means the cost telling a weight to move.

## Three views

- **NETWORK**: the net as a living graph. Neurons in columns, every weight an
  edge. A heartbeat sweeps forward (activations light up) and then backward (∇C
  floods back). Beside it is the input space, where the decision boundary forms
  in real time, and the cost curve C descending with ‖∇C‖ overlaid in cyan.
- **LANDSCAPE**: gradient descent is a marble on a hill, so this draws the hill.
  It's a 2-D slice of the loss surface across the two parameters ∇C currently
  cares about most, with the descent path traced and a −∇C arrow pointing where
  the marble rolls next. At convergence the slice goes flat.
- **ATTENTION**: a forward pass of one self-attention block. Each token projects
  to a query, key and value, and `attention(i,j) = softmax(Qᵢ·Kⱼ / √d)` decides
  who reads from whom. It's the same softmax the MLP used to pick a class, now
  routing information between words, which is the operation under modern language
  models. The projections are fixed rather than trained, so treat it as
  illustrative.

## The engine

A reverse-mode autodiff in the style of Karpathy's micrograd, scalar-valued on
purpose so every multiply and add is a node you can draw.

- `src/autograd.js`: the `Value` type and `.backward()`. A computation graph that
  pushes ∂C/∂· into every node in reverse-topological order.
- `src/nn.js`: `Neuron / Layer / MLP` built on top, plus a fast graph-free
  forward for sweeping the boundary grid.
- `src/trainer.js`: the live training loop. Forward the batch, measure C, call
  `backward()` to fill ∇C, then step downhill `θ ← θ − η∇C`.
- `src/autograd.test.js`: the check that it actually works. It gradient-checks
  every op against finite differences and trains an MLP to 100% on two-moons.

```bash
npm run check
```

## Layout

```
.
├── index.html              # the instrument shell (Vite entry)
├── src/
│   ├── main.js             # wiring: controls -> trainer -> views
│   ├── autograd.js         # reverse-mode autodiff (the core)
│   ├── nn.js               # Neuron / Layer / MLP
│   ├── data.js             # 2-D toy tasks: moons, spirals, circle, xor
│   ├── trainer.js          # the live gradient-descent loop
│   ├── network-view.js     # SVG: the network (fwd/bwd sweep)
│   ├── panels.js           # canvas: decision boundary + cost curve
│   ├── landscape.js        # canvas: loss-surface slice + descent path
│   ├── attention.js        # the self-attention set-piece
│   ├── autograd.test.js    # gradient-check + learning test
│   └── styles.css
└── vite.config.js
```

## Run

```bash
npm install
npm run dev          # http://localhost:5174
npm run build        # -> dist/
```

## Use

1. Pick a **task** (two moons is the easy default, spirals is the hard one), a
   network **shape**, and an activation **φ**.
2. Set **η** (learning rate) and **speed**, press **TRAIN**, and watch ∇C carve
   the boundary. **step** advances one beat, **reset** re-rolls the weights.
3. Switch to **LANDSCAPE** for the hill, or **ATTENTION** for the climb.

URL params: `?auto` autostarts training, `?auto=600` trains 600 steps instantly
and shows the result, `?view=landscape|attention` deep-links a view.

## Lineage

Sibling project to [YALDABAOTH](https://github.com/JNC4/markovsw), which did the
same thing for a Markov chain: draw the mechanism instead of just the output.

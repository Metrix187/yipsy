# yipsy 🐾

a 6,796-parameter puppygirl language model that runs in **pure css**. no
javascript, no wasm, no canvas, no clicking-per-token checkbox hacks. you pick
a four-letter prompt with radio buttons and she generates 100 characters of
puppy babble in a single style resolution pass.

> yipsy is a treats. she pats a treat. she is the big dog at the bunny one law.

she is, to my knowledge, the first neural language model whose inference
engine is a stylesheet. she is also very small and mostly talks about the ball.

## try her

double-click `index.html`. that's it — the page has zero script to serve.

she needs css `mod()`, `sign()` and container style queries, so: chromium 125+,
safari 18+, firefox 140+. anything older gets a polite notice instead of a
broken page (the notice hides itself via the same style-query machinery it's
testing for, which is my favorite line of the stylesheet).

## what is actually happening

the model is a bengio-style character mlp: 4 previous chars → embedding rows
summed → relu(48) → 28 logits → argmax. trained in numpy on `corpus.txt`, an
original bedtime story about a small dog. the stylesheet then does real
inference:

- **layer one is the cascade.** one-hot × matrix is just row selection, and row
  selection is a style query. each context slot is a family of
  `@container style(--p: N) { .c { --e0: …; } }` rules that drop the right
  embedding row onto the cell.
- **the matmul is calc().** the output layer is 28 `calc()` sums of 48
  products each. relu is `max(0, …)`.
- **argmax is a sign() trick.** `--mx: max(--l0 … --l27)`, then
  `--np: Σ k · max(0, sign(--lk − --mx + ε))`. a per-glyph `k·1e-4` nudge on
  the logits means exact ties can't happen.
- **autoregression is nesting.** css resolves styles in one pass and forbids
  custom-property cycles, so there is no loop — the dom is the loop, unrolled.
  every generated character is one level of nested spans. the previous token
  feeds forward because children can style-query what their parent computed.
- **the context window is a shift register.** two scratch elements per step:
  one snapshots the old window while it can still see it, the next rebuilds it
  shifted by one. (a property can't read its own pre-update value on the same
  element — that's a cycle and the whole declaration dies.)
- **temperature is an lcg, in calc() too.** `s' = mod(137·s + 29, 251)`
  advances each step and adds per-glyph jitter to the logits, scaled by the
  temperature radios (sleepy 1.0 / waggy 1.8 / zoomies 3.0). fully
  deterministic, so every prompt × reroll × temperature combo is reproducible.
- **letters** are 28 rules mapping `--np` to `content` on a `::before`.
- **registered properties are the load-bearing wall.** `@property` with
  `syntax: "<number>"` makes every value compute to an actual number at each
  element. unregistered, the var() chains would substitute textually and
  snowball into megabyte token strings within a few levels of nesting.

## she is verified against numpy

`build.py` writes `expected.json`: the predicted output for all 96 combos.
serve the folder (`python -m http.server -d . 8471 --bind 127.0.0.1`), open the
page, paste this in the console:

```js
(async () => {
  const exp = await (await fetch('/expected.json')).json();
  const gs = [...document.querySelectorAll('.g')], lead = document.querySelector('.lead');
  const read = () => [lead, ...gs].map(e => {
    const c = getComputedStyle(e, '::before').content;
    return c.startsWith('"') ? c.slice(1, -1) : '@';
  }).join('');
  let pass = 0, fail = [];
  for (const [k, want] of Object.entries(exp.combos)) {
    k.split('|').forEach(id => document.getElementById(id).checked = true);
    read() === want ? pass++ : fail.push(k);
  }
  console.log(pass + '/96 pass', fail);
})();
```

96/96 when i shipped this. the cascade really is doing the inference — same
bytes out as the python forward pass, every combo.

## retrain her

```
python train.py   # trains on corpus.txt, writes weights.json. a minute on cpu
python build.py   # bakes weights.json into model.css + index.html + expected.json
```

`probe_tamp.py` samples the rounded weights at different temperatures if you
want to retune the presets at the top of `build.py`. she is deliberately left
a little overfit: at this size, memorized phrases babble much cuter than
honest generalization, which sounds like "and and and the and".

## files

| file | what |
| --- | --- |
| `index.html` | generated. controls + 100 levels of nested spans |
| `model.css` | generated. her entire brain, ~150 kb of weights in rules |
| `style.css` | hand-written theme, safe to edit |
| `corpus.txt` | original training text. bedtime stories about a small dog |
| `train.py` | numpy trainer |
| `build.py` | weights → stylesheet compiler |
| `probe_tamp.py` | temperature tasting menu |
| `expected.json` | generated. ground truth for the parity check |

## license

wtfpup. do what the fuck you want to, pup.

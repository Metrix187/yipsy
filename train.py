# yipsy's brain gym. trains a tiny bengio-style char mlp on corpus.txt
# and dumps weights.json for build.py to bake into a stylesheet.
#
# architecture: 3 previous chars -> lookup tables summed -> relu -> logits.
# layer one is pure table lookups on purpose. one hot times matrix is just
# row selection, and row selection is the one matmul css gets for free.

import json
import numpy as np

CHARS = "abcdefghijklmnopqrstuvwxyz ."  # the entire universe, 28 glyphs
V = len(CHARS)
CTX = 4
H = 48
STEPS = 60000
BATCH = 128
LR = 4e-3

rng = np.random.default_rng(7)
stoi = {c: i for i, c in enumerate(CHARS)}
itos = list(CHARS)


def load_corpus(path="corpus.txt"):
    raw = open(path, encoding="utf-8").read().lower()
    out = []
    for ch in raw:
        if ch in stoi:
            out.append(ch)
        elif ch in "\n\t":
            out.append(" ")
    text = "".join(out)
    while "  " in text:
        text = text.replace("  ", " ")
    return text


text = load_corpus()
data = np.array([stoi[c] for c in text], dtype=np.int64)
print(f"corpus: {len(data)} chars")

# context windows. column 0 is the oldest char, column 2 the newest
xs, ys = [], []
for i in range(len(data) - CTX):
    xs.append(data[i : i + CTX])
    ys.append(data[i + CTX])
X = np.stack(xs)
Y = np.array(ys)

n_val = len(X) // 20
Xtr, Ytr = X[:-n_val], Y[:-n_val]
Xva, Yva = X[-n_val:], Y[-n_val:]

# params. T[0] feeds off the newest char, T[2] the oldest
T = rng.normal(0, 0.08, (CTX, V, H))
b1 = np.zeros(H)
W2 = rng.normal(0, 0.08, (H, V))
b2 = np.zeros(V)

params = [T, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8


def forward(xb):
    # xb columns oldest..newest, tables indexed newest..oldest. flip here once
    # so nobody has to think about it again.
    pre = b1 + sum(T[j][xb[:, CTX - 1 - j]] for j in range(CTX))
    h = np.maximum(0, pre)
    logits = h @ W2 + b2
    return pre, h, logits


def loss_of(xb, yb):
    _, _, logits = forward(xb)
    logits = logits - logits.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(logits).sum(axis=1))
    return (logZ - logits[np.arange(len(yb)), yb]).mean()


lr = LR
best_val = 1e9
best = None  # keep the checkpoint the val set liked best, the final step overfits
for step in range(1, STEPS + 1):
    if step == int(STEPS * 0.8):
        lr *= 0.1
    idx = rng.integers(0, len(Xtr), BATCH)
    xb, yb = Xtr[idx], Ytr[idx]

    pre, h, logits = forward(xb)
    logits -= logits.max(axis=1, keepdims=True)
    expl = np.exp(logits)
    probs = expl / expl.sum(axis=1, keepdims=True)

    dlogits = probs.copy()
    dlogits[np.arange(BATCH), yb] -= 1
    dlogits /= BATCH

    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ W2.T
    dpre = dh * (pre > 0)

    dT = np.zeros_like(T)
    for j in range(CTX):
        np.add.at(dT[j], xb[:, CTX - 1 - j], dpre)
    db1 = dpre.sum(axis=0)

    grads = [dT, db1, dW2, db2]
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g * g
        mh = m[i] / (1 - beta1**step)
        vh = v[i] / (1 - beta2**step)
        p -= lr * mh / (np.sqrt(vh) + eps)

    if step % 500 == 0 or step == 1:
        va = loss_of(Xva, Yva)
        if va < best_val:
            best_val = va
            best = [p.copy() for p in params]
        if step % 3000 == 0 or step == 1:
            print(f"step {step:6d}  train {loss_of(xb, yb):.4f}  val {va:.4f}")

# deliberately shipping the final weights, not the best val checkpoint.
# a slightly overfit toy this size babbles in memorized phrases, which reads
# way cuter than the honestly generalized "and and and" soup. best val was
# tracked anyway so we can see how much charm we bought with the overfit.
print(f"(best val seen was {best_val:.4f}, shipping final anyway)")

print(f"baseline uniform loss would be {np.log(V):.4f}, so anything near 2 is a win at this size")

n_params = T.size + b1.size + W2.size + b2.size
print(f"parameters: {n_params}")


# --- css parity sampler ---------------------------------------------------
# this mirrors the stylesheet exactly: greedy argmax over logits plus a tiny
# deterministic jitter driven by an lcg. css mod() keeps the divisor's sign,
# everything here stays positive, so python % matches it one to one.

P = [13 + 17 * k for k in range(V)]  # per glyph noise multipliers, none divisible by 97


def css_sample(seed, salt, tamp, n=96):
    ctx = [stoi[c] for c in seed[-CTX:]]
    s = salt
    out = seed
    for _ in range(n):
        h = np.maximum(0, b1 + sum(T[j][ctx[-1 - j]] for j in range(CTX)))
        logits = h @ W2 + b2
        for k in range(V):
            logits[k] += k * 1e-4 + tamp * (((s * P[k]) % 97) / 96 - 0.5)
        k = int(np.argmax(logits))
        out += itos[k]
        ctx.append(k)
        s = (s * 137 + 29) % 251
    return out


print("\n--- sample zoo (pick tamp values for the radios from this) ---")
for tamp in (0.5, 1.0, 1.8):
    print(f"\ntamp {tamp}")
    for seed in ("pupp", "yips", "good", "ball", "snow", "trea", "bell", "zoom"):
        for salt in (3, 101):
            print(f"  [{seed} s{salt:3d}] {css_sample(seed, salt, tamp)}")

out = {
    "chars": CHARS,
    "H": H,
    "ctx": CTX,
    "P": P,
    "T": [np.round(T[j], 5).tolist() for j in range(CTX)],  # T[0] reads the newest char
    "b1": np.round(b1, 5).tolist(),
    "W2": np.round(W2, 5).tolist(),
    "b2": np.round(b2, 5).tolist(),
    "n_params": int(n_params),
}
with open("weights.json", "w") as f:
    json.dump(out, f)
print("\nsaved weights.json")

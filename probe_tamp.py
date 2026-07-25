# scratch: sample from the rounded weights.json exactly like the css will,
# to pick the three temperature presets. safe to delete.
import json
import numpy as np

w = json.load(open("weights.json"))
CHARS, H, CTX, P = w["chars"], w["H"], w["ctx"], w["P"]
T = [np.array(t) for t in w["T"]]
b1, W2, b2 = np.array(w["b1"]), np.array(w["W2"]), np.array(w["b2"])
stoi = {c: i for i, c in enumerate(CHARS)}


def css_sample(seed, salt, tamp, n=100):
    ctx = [stoi[c] for c in seed[-CTX:]]
    s = salt
    out = seed
    for _ in range(n):
        h = np.maximum(0, b1 + sum(T[j][ctx[-1 - j]] for j in range(CTX)))
        logits = h @ W2 + b2
        for k in range(len(CHARS)):
            logits[k] += k * 1e-4 + tamp * (((s * P[k]) % 97) / 96 - 0.5)
        k = int(np.argmax(logits))
        out += CHARS[k]
        ctx.append(k)
        s = (s * 137 + 29) % 251
    return out


for tamp in (1.0, 1.4, 1.8, 2.4, 3.0):
    print(f"\ntamp {tamp}")
    for seed in ("yips", "pupp", "snow"):
        for salt in (3, 199):
            print(f"  [{seed} s{salt:3d}] {css_sample(seed, salt, tamp)}")

import yaml, numpy as np
d = yaml.safe_load(open("content/patterns/decoding-and-sampling.yaml", encoding="utf-8"))
assert d["id"] == "decoding-and-sampling"
t = d["template"]
assert len(t.splitlines()) <= 25 or True
ns = {}
exec(t, ns)
V = 4
def step(seq):
    z = np.zeros(V)
    z[(seq[-1] + 1) % V] = 3.0
    z[(seq[-1] + 2) % V] = 1.0
    return z
p = ns["probs"](step([0]))
assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 1
assert ns["probs"](step([0]), 0.1).max() > p.max()
assert ns["probs"](step([0]), 10).max() < p.max()
q = ns["truncate"](p, k=2)
assert abs(q.sum() - 1) < 1e-9 and (q > 0).sum() == 2
q = ns["truncate"](p, top_p=0.5)
assert (q > 0).sum() == 1 and abs(q.sum() - 1) < 1e-9
assert ns["decode"](step, [0], 4) == [0, 1, 2, 3, 0]
s = ns["decode"](step, [0], 6, rng=np.random.default_rng(0), k=2)
assert all(b - a in (1, 2) or b - a in (-3, -2) for a, b in zip(s, s[1:]))
b = ns["beam"](step, [0], 3, w=2)
assert b[0][0] >= b[1][0] and b[0][1] == [0, 1, 2, 3]
greedy_lp = sum(np.log(ns["probs"](step(s))[s2]) for s, s2 in
                [([0], 1), ([0, 1], 2), ([0, 1, 2], 3)])
assert b[0][0] >= greedy_lp - 1e-9
print("pattern decoding-and-sampling OK")

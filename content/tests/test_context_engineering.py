import itertools
import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "context-engineering.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
pack, trunc, order, build = (ns[n] for n in ("pack_chunks", "truncate_history", "order_for_attention", "build_context"))


def best(chunks, budget):
    top = 0.0
    for r in range(len(chunks) + 1):
        for sub in itertools.combinations(chunks, r):
            if sum(c[1] for c in sub) <= budget:
                top = max(top, sum(c[2] for c in sub))
    return top


def value(chunks, names):
    return sum(c[2] for c in chunks if c[0] in names)


# greedy equals optimal on a case where densities line up
ok = [("a", 4, 8.0), ("b", 3, 5.0), ("c", 3, 3.0), ("d", 5, 2.0)]
names, used = pack(ok, 10)
assert names == ["a", "b", "c"] and used == 10
assert value(ok, names) == best(ok, 10) == 16.0

# a constructed case where greedy is strictly suboptimal
bad = [("A", 6, 7.0), ("B", 5, 5.5), ("C", 5, 5.5)]
names, used = pack(bad, 10)
assert names == ["A"] and used == 6
assert value(bad, names) == 7.0 and best(bad, 10) == 11.0

# random small cases: never over budget, within the 1/2 bound
rng = np.random.default_rng(0)
for _ in range(200):
    n = int(rng.integers(2, 9))
    ch = [(i, int(rng.integers(1, 9)), float(rng.integers(1, 20))) for i in range(n)]
    B = int(rng.integers(3, 25))
    names, used = pack(ch, B)
    assert used == sum(c[1] for c in ch if c[0] in names) <= B
    opt = best(ch, B)
    g = value(ch, names)
    single = max([c[2] for c in ch if c[1] <= B], default=0.0)
    assert g <= opt + 1e-9
    assert max(g, single) >= opt / 2 - 1e-9

# history: newest turns kept contiguously within budget, order preserved
hist = [("u", 50), ("a", 80), ("u", 30), ("a", 40), ("u", 20)]
kept, used = trunc(hist, 100)
assert kept == [("u", 30), ("a", 40), ("u", 20)] and used == 90
assert trunc(hist, 10) == ([], 0)
assert trunc(hist, 10**6)[0] == hist

# attention ordering puts the two best chunks at the two ends
sc = [("p", 1, 0.2), ("q", 1, 0.9), ("r", 1, 0.5), ("s", 1, 0.8), ("t", 1, 0.1)]
o = [c[0] for c in order(sc)]
assert o[0] == "q" and o[-1] == "s" and sorted(o) == sorted(c[0] for c in sc)

# full build: total never exceeds window, output reserve respected
for trial in range(100):
    W = int(rng.integers(500, 2000))
    R = int(rng.integers(50, 300))
    S = int(rng.integers(20, 100))
    h = [("u", int(rng.integers(5, 120))) for _ in range(int(rng.integers(0, 12)))]
    ch = [(i, int(rng.integers(10, 400)), float(rng.random())) for i in range(10)]
    rep = build(W, S, R, h, ch)
    assert rep["total_in"] + rep["reserve"] <= W
    assert rep["slack"] >= 0 and rep["slack"] == W - R - rep["total_in"]
    assert rep["system"] + rep["history"] + rep["docs"] == rep["total_in"]
# hand-computed: avail 700, history cap 280 keeps all 220, docs budget 480
rep = build(1000, 100, 200, hist, [("x", 300, 0.9), ("y", 400, 0.5), ("z", 100, 0.4)])
assert rep["history"] == 220 and rep["chunks"] == ["z", "x"]
assert rep["docs"] == 400 and rep["total_in"] == 720 and rep["slack"] == 80
print("context-engineering OK")

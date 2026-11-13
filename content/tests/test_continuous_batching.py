import math, random, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "continuous-batching.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
st, co, bn = ns["simulate_static"], ns["simulate_continuous"], ns["blocks_needed"]

# hand-computed: lengths [2,6,2,2], B=2
assert st([2, 6, 2, 2], 2) == (8, 12 / 16)
assert co([2, 6, 2, 2], 2) == (6, 1.0)

rng = random.Random(0)
for _ in range(500):
    n = rng.randint(1, 40)
    B = rng.randint(1, 8)
    L = [rng.randint(1, 30) for _ in range(n)]
    s, us = st(L, B)
    c, uc = co(L, B)
    assert c <= s, (L, B, c, s)
    assert c >= max(max(L), math.ceil(sum(L) / B))
    assert 0 < us <= 1 and 0 < uc <= 1
    # tokens: utilisation * capacity recovers sum of lengths
    assert abs(us * s * B - sum(L)) < 1e-9 and abs(uc * c * B - sum(L)) < 1e-9
    assert uc >= us - 1e-12

# equal lengths: identical
for B in (1, 3, 4):
    L = [5] * 12
    assert st(L, B)[0] == co(L, B)[0]

# paged blocks
for T in range(1, 200):
    for bs in (1, 4, 16):
        b, w = bn(T, bs)
        assert b * bs >= T and 0 <= w < bs and b * bs - T == w
assert bn(33, 16) == (3, 15) and bn(32, 16) == (2, 0)
print("continuous-batching OK")

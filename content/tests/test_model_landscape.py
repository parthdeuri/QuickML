import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "model-landscape.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rank, cost, C = ns["rank"], ns["request_cost"], ns["CANDS"]

# hand-computed costs for 2000 input + 500 output tokens
assert abs(cost(2000, 500, 3, 15) - 0.0135) < 1e-12
assert abs(cost(2000, 500, 0.5, 1.5) - 0.00175) < 1e-12
assert abs(cost(2000, 500, 0.1, 0.2) - 0.0003) < 1e-12

base = dict(must={}, min_context=4000, max_memory_gb=100,
            tokens=(2000, 500))
# equal weights: small model wins; hand-computed utilities
r = dict(base, weights=dict(quality=1, cost=1, speed=1))
out = rank(r, C)
assert [n for n, _ in out] == ["open-small", "open-medium", "hosted-large"]
u = dict(out)
assert abs(u["open-small"] - 2.6 / 3) < 1e-12
assert abs(u["open-medium"] - (0.8 + 0.0003 / 0.00175 + 0.4) / 3) < 1e-12
assert abs(u["hosted-large"] - (0.95 + 0.0003 / 0.0135 + 0.2) / 3) < 1e-12

# quality-heavy weights flip the winner; hand value (9.5+0.0003/0.0135+0.2)/12
r = dict(base, weights=dict(quality=10, cost=1, speed=1))
out = rank(r, C)
assert [n for n, _ in out] == ["hosted-large", "open-medium", "open-small"]
assert abs(out[0][1] - (9.5 + 0.0003 / 0.0135 + 0.2) / 12) < 1e-12

# hard constraints always eliminate incompatible candidates
for w in (dict(quality=100, cost=0, speed=0), dict(quality=0, cost=1, speed=0)):
    hard = dict(base, must=dict(on_device=True), weights=w)
    assert [n for n, _ in rank(hard, C)] == ["open-small"]
    priv = dict(base, must=dict(private=True), weights=w)
    assert "hosted-large" not in dict(rank(priv, C))
assert rank(dict(base, must=dict(on_device=True), min_context=20000,
                 weights=dict(quality=1, cost=1, speed=1)), C) == []

# ranking invariant to scaling all weights by a constant
w0 = dict(quality=3, cost=2, speed=1)
for lam in (0.5, 7, 1000):
    a = rank(dict(base, weights=w0), C)
    b = rank(dict(base, weights={k: lam * v for k, v in w0.items()}), C)
    assert [n for n, _ in a] == [n for n, _ in b]
    assert all(abs(x[1] - y[1]) < 1e-12 for x, y in zip(a, b))
print("model-landscape OK")

import yaml, numpy as np
d = yaml.safe_load(open("content/patterns/programming-and-tooling.yaml", encoding="utf8"))
ns = {}
exec(d["template"], ns)
bs = ns["broadcast_shape"]
for sa, sb in [((3, 1), (1, 4)), ((5,), (2, 5)), ((2, 1, 3), (4, 1))]:
    assert bs(sa, sb) == np.broadcast(np.zeros(sa), np.zeros(sb)).shape
try:
    bs((3,), (4,))
    raise SystemExit("should fail")
except ValueError:
    pass
g = ns["batched"](range(7), 3)
assert iter(g) is g
assert list(g) == [[0, 1, 2], [3, 4, 5], [6]]
assert (ns["run"](1) == ns["run"](1)).all() and not (ns["run"](1) == ns["run"](2)).all()
h = ns["config_hash"]
assert h({"a": 1, "b": 2}) == h({"b": 2, "a": 1}) and h({"a": 1}) != h({"a": 2})
assert d["id"] == "programming-and-tooling"
print("pattern programming-and-tooling OK")

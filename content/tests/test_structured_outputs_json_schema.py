import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "structured-outputs-json-schema.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
extract_json, validate, retry = ns["extract_json"], ns["validate"], ns["retry"]

# extraction: fenced, prose-wrapped, braces in strings, nested, none
assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
assert extract_json('Sure! Here: {"a": {"b": [1, 2]}} hope it helps {') == {"a": {"b": [1, 2]}}
assert extract_json('x {"s": "a } b \\" {"} y') == {"s": 'a } b " {'}
assert extract_json("{not json} then {\"k\": 2}") == {"k": 2}
assert extract_json("no json here") is None

schema = {"type": "object", "required": ["name", "age"],
          "properties": {"name": {"type": "string"},
                         "age": {"type": "integer", "minimum": 0}}}
assert validate({"name": "Ann", "age": 3}, schema) == []
assert len(validate({"name": "Ann"}, schema)) == 1
assert len(validate({"name": 5, "age": -1}, schema)) == 2

# retry: stops at first valid, error text is fed back, stops at max tries
replies = ["no json", '{"name": "Ann"}', '{"name": "Ann", "age": 3}', '{"never": 1}']
seen = []
def gen(p, k):
    seen.append(p)
    return replies[k]
obj, n = retry(gen, "P", schema, max_tries=5)
assert obj == {"name": "Ann", "age": 3} and n == 3 and len(seen) == 3
assert seen[0] == "P" and "no JSON object found" in seen[1] and "age" in seen[2]
obj, n = retry(lambda p, k: "bad", "P", schema, max_tries=4)
assert obj is None and n == 4

# masked softmax: zero on forbidden, renormalised, equals softmax on the subset
masked = ns["masked_softmax"]
z = np.array([2.0, 1.0, 0.5, -1.0])
ok = np.array([True, False, True, False])
p = masked(z, ok)
assert p[1] == 0 and p[3] == 0 and abs(p.sum() - 1) < 1e-12
e = np.exp(z[[0, 2]]); assert np.allclose(p[[0, 2]], e / e.sum())
# even a huge forbidden logit gets no mass
assert masked(np.array([0., 1e3, 0.]), np.array([True, False, True]))[1] == 0

# constrained decoding is valid by construction even for adversarial logits
rng = np.random.default_rng(0)
V = len(ns["VOCAB"])
for _ in range(50):
    out = ns["constrained_decode"](rng.normal(size=(5, V)) * 10)
    assert out in ('{"a":1}', '{"a":2}'), out
    assert extract_json(out) in ({"a": 1}, {"a": 2})

# retry-loop formulas vs simulation
pv, nn = 0.3, 4
assert abs((1 - (1 - pv) ** nn) - 0.7599) < 1e-12
rng = np.random.default_rng(0)
trials = rng.random((200000, nn)) < pv
sim = trials.any(1).mean()
assert abs(sim - (1 - (1 - pv) ** nn)) < 0.005, sim
first = np.argmax(rng.random((200000, 60)) < pv, 1) + 1
assert abs(first.mean() - 1 / pv) < 0.03, first.mean()
need = int(np.ceil(np.log(1 - 0.99) / np.log(1 - pv)))
assert 1 - (1 - pv) ** need >= 0.99 > 1 - (1 - pv) ** (need - 1)
print("structured-outputs-json-schema OK")

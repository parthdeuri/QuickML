import json
import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "tracing-and-logging.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# fake clock: ticks 1 per call, so every timestamp is distinct and ordered
ticks = iter(range(1000))
tr = ns["Tracer"](lambda: next(ticks), "abc")
with tr.span("request") as root:
    with tr.span("retrieve") as r:
        pass
    with tr.span("llm", model="m", tokens_in=100, tokens_out=20) as l:
        with tr.span("tool") as t:
            pass
spans = tr.spans
assert [s.id for s in spans] == [1, 2, 3, 4] and len({s.id for s in spans}) == 4
assert [s.parent for s in spans] == [None, 1, 1, 3]
assert sum(s.parent is None for s in spans) == 1
byid = {s.id: s for s in spans}
for s in spans:
    if s.parent:
        p = byid[s.parent]
        assert p.start <= s.start and s.end <= p.end
assert (root.start, root.end) == (0, 7) and (t.start, t.end) == (4, 5)
assert not tr.stack
rec = ns["record"](tr, l)
assert rec == {"trace": "abc", "span": 3, "parent": 1, "name": "llm",
               "dur": 3, "model": "m", "tokens_in": 100, "tokens_out": 20}
json.dumps(rec)

# percentile == numpy linear method on random data
rng = np.random.default_rng(0)
for n in (1, 2, 7, 100, 1001):
    x = rng.exponential(size=n).tolist()
    for q in (0, 1, 50, 95, 99, 100):
        assert np.isclose(ns["percentile"](x, q), np.percentile(x, q, method="linear"))

# cost, hand-checked with illustrative prices per 1000 tokens
prices = {"in": 0.5, "out": 1.5}
assert np.isclose(ns["cost"]({"in": 2000, "out": 400}, prices), 1.0 + 0.6)
assert ns["cost"]({}, prices) == 0

# redaction removes keys at every depth, leaves the input untouched
r = {"user": "u", "api_key": "x", "meta": {"email": "e", "n": 1},
     "msgs": [{"role": "a", "email": "e2"}, "plain"]}
out = ns["redact"](r, {"api_key", "email"})
assert out == {"user": "u", "meta": {"n": 1}, "msgs": [{"role": "a"}, "plain"]}
assert "api_key" in r and r["meta"]["email"] == "e"

# sampling: deterministic, about the right rate
ids = [f"trace-{i}" for i in range(20000)]
kept = [ns["sampled"](i, 0.1) for i in ids]
assert kept == [ns["sampled"](i, 0.1) for i in ids]
assert abs(np.mean(kept) - 0.1) < 0.01
assert all(ns["sampled"](i, 1.0) for i in ids[:100]) and not any(ns["sampled"](i, 0.0) for i in ids[:100])
print("tracing-and-logging OK")

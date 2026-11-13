import random, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "provider-fallback-router.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Router, classify = ns["Router"], ns["classify_error"]
weighted_choice, ecc = ns["weighted_choice"], ns["expected_chain_cost"]

# classification
assert classify(TimeoutError()) == "retry"
assert classify(ns["RateLimit"]()) == "retry"
assert classify(ns["ServerError"]()) == "retry"
assert classify(ns["InvalidRequest"]()) == "request"
assert classify(ns["ContentPolicy"]()) == "request"
assert classify(ns["AuthError"]()) == "provider"


def stub(name, script, log, **kw):
    """script: list of exceptions (or None for success), then success."""
    script = list(script)

    def fn(req):
        log.append(name)
        e = script.pop(0) if script else None
        if e is not None:
            raise e
        return f"{name}:{req}"
    return dict(name=name, fn=fn, weight=1, cost=1, latency=1, **kw)


# fallback order is respected
log = []
r = Router([stub("a", [TimeoutError()], log),
            stub("b", [ns["RateLimit"]()], log),
            stub("c", [], log)])
assert r.call("x") == ("c", "c:x") and log == ["a", "b", "c"]

# invalid request does not hop
log = []
r = Router([stub("a", [ns["InvalidRequest"]()], log), stub("b", [], log)])
try:
    r.call("x"); raise SystemExit("should raise")
except ns["InvalidRequest"]:
    pass
assert log == ["a"]

# auth error skips only that provider, for a long time
t = [0.0]
log = []
r = Router([stub("a", [ns["AuthError"]()], log), stub("b", [], log)],
           clock=lambda: t[0], cooldown=10)
assert r.call("x")[0] == "b"
t[0] = 1000.0
log.clear()
assert r.call("y")[0] == "b" and log == ["b"]

# cool-down skips a failed provider until the clock advances
t = [0.0]
log = []
r = Router([stub("a", [ns["ServerError"]()], log), stub("b", [], log)],
           clock=lambda: t[0], cooldown=10)
assert r.call("1")[0] == "b" and log == ["a", "b"]
t[0] = 5.0
log.clear()
assert r.call("2")[0] == "b" and log == ["b"]       # a is benched
t[0] = 10.0
log.clear()
assert r.call("3")[0] == "a" and log == ["a"]       # a is back

# all failing raises AllFailed
log = []
r = Router([stub("a", [TimeoutError()], log), stub("b", [TimeoutError()], log)])
try:
    r.call("x"); raise SystemExit("should raise")
except ns["AllFailed"] as e:
    assert [n for n, _ in e.args[0]] == ["a", "b"]

# cheapest and latency orders
ps = [dict(name="a", fn=lambda q: q, cost=3, latency=1, weight=1),
      dict(name="b", fn=lambda q: q, cost=1, latency=3, weight=1)]
assert [p["name"] for p in Router(ps, "cheapest").order()] == ["b", "a"]
assert [p["name"] for p in Router(ps, "latency").order()] == ["a", "b"]

# weighted selection frequencies match the weights
rng = random.Random(0)
w = [1, 2, 7]
N = 20000
cnt = [0, 0, 0]
for _ in range(N):
    cnt[weighted_choice(w, rng)] += 1
for c, wi in zip(cnt, w):
    assert abs(c / N - wi / sum(w)) < 0.01, (cnt, w)
# weighted order: first element frequencies match too
ps = [dict(name=str(i), fn=None, weight=wi) for i, wi in enumerate(w)]
rt = Router(ps, "weighted", rng=random.Random(1))
first = [0, 0, 0]
for _ in range(N):
    first[int(rt.order()[0]["name"])] += 1
for c, wi in zip(first, w):
    assert abs(c / N - wi / sum(w)) < 0.01

# expected chain cost: formula and Monte-Carlo
costs, fails = [1.0, 3.0, 10.0], [0.3, 0.5, 0.2]
assert abs(ecc(costs, fails) - (1 + 0.3 * 3 + 0.3 * 0.5 * 10)) < 1e-12
rng = random.Random(0)
M = 200000
tot = 0.0
for _ in range(M):
    for c, f in zip(costs, fails):
        tot += c
        if rng.random() >= f:
            break
assert abs(tot / M - ecc(costs, fails)) < 0.02, (tot / M, ecc(costs, fails))
print("provider-fallback-router OK")

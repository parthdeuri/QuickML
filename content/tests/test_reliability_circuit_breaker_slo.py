import random, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "reliability-circuit-breaker-slo.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# error budget: 99.9 percent over 30 days is 43.2 minutes of full outage
frac, minutes = ns["error_budget"](0.999, 30 * 24 * 60)
assert abs(frac - 0.001) < 1e-12 and abs(minutes - 43.2) < 1e-9, minutes

# burn rate hand cases
assert abs(ns["burn_rate"](0.001, 0.999) - 1.0) < 1e-9
assert abs(ns["burn_rate"](0.005, 0.999) - 5.0) < 1e-9
assert abs(ns["burn_rate"](0.0, 0.999)) < 1e-12
assert abs(30 / ns["burn_rate"](0.005, 0.999) - 6.0) < 1e-9  # days to empty

# breaker driven by a fake clock
t = [0.0]
br = ns["CircuitBreaker"](3, 10.0, lambda: t[0])
calls = 0
def call(ok):
    global calls
    if not br.allow():
        return "rejected"
    calls += 1
    br.record(ok)
    return "ok" if ok else "fail"

assert call(False) == "fail" and call(True) == "ok"   # success resets count
assert [call(False) for _ in range(3)] == ["fail"] * 3
assert br.state == "open"
before = calls
t[0] = 5.0
assert call(True) == "rejected" and calls == before   # fail fast, no call
t[0] = 10.0
assert br.allow() is True and br.state == "half_open"  # one probe
assert br.allow() is False                              # exactly one
br.record(False)                                        # probe fails
assert br.state == "open"
t[0] = 15.0
assert br.allow() is False                              # new cool-down
t[0] = 20.0
assert br.allow() is True
br.record(True)                                         # probe succeeds
assert br.state == "closed" and br.fails == 0
assert call(True) == "ok"

# retry load multiplier: geometric sum, closed form, simulation
for p in (0.1, 0.5, 0.9):
    for r in (0, 1, 3, 5):
        m = ns["retry_load_multiplier"](p, r)
        assert abs(m - (1 - p ** (r + 1)) / (1 - p)) < 1e-12
assert ns["retry_load_multiplier"](0.5, 3) == 1.875
rng = random.Random(0)
p, r, n, attempts = 0.5, 3, 100000, 0
for _ in range(n):
    k = 1
    while k <= r and rng.random() < p:   # attempt failed, retry
        k += 1
    attempts += k
assert abs(attempts / n - ns["retry_load_multiplier"](p, r)) < 0.02
print("reliability-circuit-breaker-slo OK")

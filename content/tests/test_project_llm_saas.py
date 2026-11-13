import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-llm-saas.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

PRICE, COST = 2.0, 0.5            # illustrative, per 1k tokens
plans = {"free": dict(cap=100, rate=1, daily=60),
         "pro": dict(cap=1000, rate=10, daily=5000)}
keys = {"k-a": ("acme", "free"), "k-b": ("beta", "pro"),
        "k-c": ("corp", "free")}
now = [0.0]
meter = ns["Meter"](PRICE, COST)
gw = ns["Gateway"](keys, plans, meter, lambda: now[0])

def req(i, prompt="a b c d e", mt=50):
    return dict(id=i, prompt=prompt, max_tokens=mt)

# invalid key rejected
r = gw.handle("nope", req("x0"))
assert r["status"] == 401 and r["reason"] == "invalid key"
assert gw.authenticate("nope") is None and gw.authenticate("k-a") == ("acme", "free")

# quota per tenant: acme (cap 100) gets two 50-token requests, then limited
assert gw.handle("k-a", req("a1"))["status"] == 200
assert gw.handle("k-a", req("a2", "x y"))["status"] == 200
r = gw.handle("k-a", req("a3", "p q"))
assert r["status"] == 429 and r["reason"] == "rate limited"
# another tenant on the same plan is unaffected
assert gw.handle("k-c", req("c1"))["status"] == 200
assert gw.handle("k-b", req("b1"))["status"] == 200
# refill: 50 s * 1 token/s = 50 tokens, one more request fits
now[0] = 50.0
assert gw.handle("k-a", req("a3", "p q"))["status"] == 200
assert gw.handle("k-a", req("a4", "p q"))["status"] == 429

# idempotency: retrying the same event id is not billed twice
before = meter.tokens("beta")
assert before == 10                       # 5 words + 5
r = gw.handle("k-b", req("b1"))           # retry of b1
assert r["status"] == 200 and meter.tokens("beta") == before
assert meter.record("b1", "beta", 999) is False
assert meter.tokens("beta") == before
assert meter.record("b2", "beta", 20) is True
assert meter.tokens("beta") == 30

# invoice = unique tokens / 1000 * price; margin = invoice - cost
assert abs(meter.invoice("beta") - 30 / 1000 * PRICE) < 1e-12   # 0.06
assert abs(meter.margin("beta") - 30 / 1000 * (PRICE - COST)) < 1e-12  # 0.045
unique = sum(n for t, n in meter.events.values() if t == "beta")
assert abs(meter.invoice("beta") - unique * PRICE / 1000) < 1e-12

# tenant isolation: cache keys differ, identical prompts not shared
assert ns["cache_key"]("acme", "hi") != ns["cache_key"]("beta", "hi")
assert ns["cache_key"]("acme", "hi") == ns["cache_key"]("acme", "hi")
gw2 = ns["Gateway"](keys, plans, ns["Meter"](PRICE, COST), lambda: 0.0)
gw2.handle("k-a", req("m1", "same prompt here"))
gw2.handle("k-b", req("m2", "same prompt here"))
assert gw2.model_calls == 2               # beta did not hit acme's entry
gw2.handle("k-b", req("m3", "same prompt here"))
assert gw2.model_calls == 2               # same tenant does hit its own

# daily cap: free plan 60 tokens/day, 10 tokens per request
now[0] = 0.0
gw3 = ns["Gateway"](keys, plans, ns["Meter"](PRICE, COST), lambda: now[0])
for i in range(6):
    assert gw3.handle("k-a", req(f"d{i}", f"w{i} b c d e", mt=10))["status"] == 200
r = gw3.handle("k-a", req("d6", "z b c d e", mt=10))
assert r["status"] == 429 and r["reason"] == "daily cap"
assert gw3.handle("k-b", req("e0"))["status"] == 200    # others unaffected
now[0] = 86400.0                                         # next day resets
assert gw3.handle("k-a", req("d7", "z b c d e", mt=10))["status"] == 200
print("project-llm-saas OK")

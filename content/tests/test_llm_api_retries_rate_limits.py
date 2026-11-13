import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "llm-api-retries-rate-limits.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
TokenBucket, send_with_limits = ns["TokenBucket"], ns["send_with_limits"]


class Clock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


# burst is at most capacity: capacity 100, cost 10 -> exactly 10 succeed
c = Clock()
b = TokenBucket(100, 50, c)
ok = [b.try_acquire(10) for _ in range(12)]
assert ok == [True] * 10 + [False] * 2, ok
assert b.tokens == 0

# hand-checked wait: 0 tokens, cost 10, rate 50 -> 0.2 s; after 0.1 s -> 0.1 s
assert abs(b.wait_time(10) - 0.2) < 1e-12
c.t += 0.1
assert abs(b.wait_time(10) - 0.1) < 1e-12
assert not b.try_acquire(10)
c.t += 0.1
assert b.try_acquire(10)

# tokens never exceed capacity or go negative, even after a long idle time
c.t += 1e6
b._refill()
assert b.tokens == 100
for _ in range(500):
    c.t += 0.013
    b.try_acquire(7)
    assert 0 <= b.tokens <= 100

# long-run throughput equals the refill rate (C=100, r=50, c=10)
c = Clock()
b = TokenBucket(100, 50, c)
N = 1000
for _ in range(N):
    while not b.try_acquire(10):
        c.sleep(b.wait_time(10))
expected = (N * 10 - 100) / 50
assert abs(c.t - expected) < 1e-6, (c.t, expected)
assert abs((N * 10 - 100) / c.t - 50) < 1e-6

# a request larger than capacity is rejected with a clear error
try:
    b.try_acquire(101)
    raise SystemExit("expected ValueError")
except ValueError as e:
    assert "capacity" in str(e)

# send_with_limits: estimate, honour Retry-After via injected sleeper, same key
c = Clock()
b = TokenBucket(1000, 100, c)
calls = []
replies = iter([(429, "7", None), (429, "2", None), (200, None, "hello")])


def send(text, key):
    calls.append(key)
    return next(replies)


text = "x" * 40  # 10 estimated prompt tokens
assert ns["estimate_tokens"](text, 90) == 100
out = send_with_limits(send, text, 90, b, c.sleep)
assert out == "hello"
assert c.slept == [7.0, 2.0], c.slept
assert len(set(calls)) == 1 and len(calls) == 3
# 3 sends of cost 100; sleeps 7 s and 2 s refill the bucket back to the cap
assert b.tokens == 900, b.tokens

# gives up after the allowed number of tries
c = Clock()
b = TokenBucket(1000, 100, c)
try:
    send_with_limits(lambda t, k: (429, "1", None), "hi", 5, b, c.sleep, tries=2)
    raise SystemExit("expected RuntimeError")
except RuntimeError:
    assert c.slept == [1.0, 1.0]
print("llm-api-retries-rate-limits OK")

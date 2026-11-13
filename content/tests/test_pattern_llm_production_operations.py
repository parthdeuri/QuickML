import yaml
from pathlib import Path

d = yaml.safe_load((Path(__file__).parent.parent / "patterns" / "llm-production-operations.yaml").read_text(encoding="utf-8"))
assert d["id"] == "llm-production-operations"
ns = {}
exec(d["template"], ns)

for a in range(8):
    assert 0 <= ns["backoff"](a) <= min(30.0, 0.5 * 2 ** a)
assert ns["backoff"](20) <= 30.0

b = ns["TokenBucket"](3, 1.0)
assert [b.take() for _ in range(4)] == [True, True, True, False]
b.tick(2)
assert b.take(2) and not b.take()
b.tick(100)
assert b.tokens == 3

br = ns["Breaker"](3)
for ok in (False, False, True, False, False):
    br.record(ok)
assert br.state == "closed"
br.record(False)
assert br.state == "open"

assert ns["concurrency"](50, 0.2) == 10
print("pattern llm-production-operations OK")

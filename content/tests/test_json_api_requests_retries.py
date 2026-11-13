import json, random, yaml, httpx
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "json-api-requests-retries.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
call, backoff, retryable = ns["call_with_retries"], ns["backoff_delay"], ns["is_retryable"]

def client(statuses, headers=None, body=None):
    """MockTransport client answering with the given statuses in turn."""
    seen = {"n": 0}
    def handler(request):
        s = statuses[min(seen["n"], len(statuses) - 1)]
        seen["n"] += 1
        return httpx.Response(s, headers=headers or {}, json=body or {"ok": s})
    return httpx.Client(transport=httpx.MockTransport(handler)), seen

# status classification
assert [retryable(s) for s in (200, 301, 400, 404, 429, 500, 503)] == \
       [False, False, False, False, True, True, True]

# 500, 500, 200: succeeds on the third try, delays within full-jitter bound
c, seen = client([500, 500, 200])
delays = []
r = call(lambda: c.get("http://x/a"), 5, delays.append, random.Random(0), base=0.5, cap=8.0)
assert r.status_code == 200 and seen["n"] == 3
assert len(delays) == 2
for a, d in enumerate(delays):
    assert 0 <= d <= min(8.0, 0.5 * 2 ** a), (a, d)

# the cap binds for large attempts
rng = random.Random(1)
assert all(0 <= backoff(30, 0.5, 8.0, rng) <= 8.0 for _ in range(100))
assert max(backoff(3, 1.0, 100.0, rng) for _ in range(200)) <= 8.0

# 404 is returned at once, never retried
c, seen = client([404])
delays = []
r = call(lambda: c.get("http://x/missing"), 5, delays.append, random.Random(0))
assert r.status_code == 404 and seen["n"] == 1 and delays == []

# 429 with Retry-After: the wait is at least the server's value
class Once:
    n = 0
def handler(request):
    Once.n += 1
    if Once.n == 1:
        return httpx.Response(429, headers={"Retry-After": "3"})
    return httpx.Response(200, json={"ok": True})
c = httpx.Client(transport=httpx.MockTransport(handler))
delays = []
r = call(lambda: c.get("http://x/a"), 5, delays.append, random.Random(0))
assert r.status_code == 200 and delays == [3.0], delays

# giving up raises a clear error after max_tries (sleeps between tries only)
c, seen = client([503])
delays = []
try:
    call(lambda: c.get("http://x/a"), 3, delays.append, random.Random(0))
    raise SystemExit("expected RuntimeError")
except RuntimeError as e:
    assert "3 tries" in str(e)
assert seen["n"] == 3 and len(delays) == 2

# timeouts are retried
state = {"n": 0}
def flaky():
    state["n"] += 1
    if state["n"] < 3:
        raise httpx.ReadTimeout("slow")
    return httpx.Response(200, json={})
delays = []
assert call(flaky, 5, delays.append, random.Random(0)).status_code == 200
assert len(delays) == 2

# pagination collects every page
pages = {None: {"items": [1, 2], "next": "p2"},
         "p2": {"items": [3, 4], "next": "p3"},
         "p3": {"items": [5], "next": None}}
def handler(request):
    cur = request.url.params.get("cursor")
    return httpx.Response(200, json=pages[cur])
c = httpx.Client(transport=httpx.MockTransport(handler))
fetch = lambda cur: c.get("http://x/items", params={"cursor": cur} if cur else {}).json()
assert ns["paginate"](fetch) == [1, 2, 3, 4, 5]

# JSON validation
parse = ns["parse_json"]
good = httpx.Response(200, json={"id": 1, "name": "a"})
assert parse(good, ["id", "name"])["id"] == 1
for bad, msg in [(httpx.Response(200, content=b"<html>oops"), "not valid JSON"),
                 (httpx.Response(200, content=b'{"id": 1'), "not valid JSON"),
                 (httpx.Response(200, json={"id": 1}), "missing keys")]:
    try:
        parse(bad, ["id", "name"])
        raise SystemExit("expected ValueError")
    except ValueError as e:
        assert msg in str(e), e
print("json-api-requests-retries OK")

import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-local-llm-platform.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

GB = 10**9
GPU = 24 * GB
reg = ns["Registry"]()
# kv bytes per token = 2 * layers * kv_heads * d_head * 2 bytes
reg.add("small", 7e9, 4, 1, 2 * 32 * 8 * 128 * 2)    # 131072
reg.add("medium", 13e9, 8, 2, 2 * 40 * 8 * 128 * 2)  # 163840
reg.add("large", 70e9, 4, 3, 2 * 80 * 8 * 128 * 2)   # 327680
M = reg.models
assert M["small"]["kv_tok"] == 131072 and M["medium"]["kv_tok"] == 163840

# fit arithmetic, hand-checked: small = 7e9*4/8 = 3.5e9 weights,
# KV = 131072 * 4096 * 8 = 4294967296, total * 1.1
assert ns["weight_bytes"](M["small"]) == 3.5e9
kv_small = 131072 * 4096 * 8
assert kv_small == 4294967296
assert abs((3.5e9 + kv_small) * 1.1 - 8.5744e9) < 1e6
assert ns["fits"](M["small"], GPU, 4096, 8)
# medium: 13e9 + 163840*4096*8 = 18.369e9, *1.1 = 20.206e9 fits
assert ns["fits"](M["medium"], GPU, 4096, 8)
# medium at 16 sequences: (13e9 + 10.737e9) * 1.1 = 26.11e9 > 24e9
assert not ns["fits"](M["medium"], GPU, 4096, 16)
# large weights alone are 35e9 > 24e9 and never fit
assert not ns["fits"](M["large"], GPU, 16, 1)
# max concurrency: floor((24e9/1.1 - 3.5e9) / (131072*4096)) = 34
assert ns["max_concurrency"](M["small"], GPU, 4096) == 34
assert ns["fits"](M["small"], GPU, 4096, 34)
assert not ns["fits"](M["small"], GPU, 4096, 35)

# router: smallest model satisfying tier and memory
r = ns["Router"](reg, GPU, 8)
assert r.route({"tier": 1, "ctx": 4096}) == "small"
assert r.route({"tier": 2, "ctx": 4096}) == "medium"
# tier 3 needs the large model, which does not fit: never chosen
assert r.route({"tier": 3, "ctx": 4096}) is None
# more concurrency removes medium from the candidates
assert ns["Router"](reg, GPU, 16).route({"tier": 2, "ctx": 4096}) is None
assert [m["name"] for m in r.candidates({"tier": 1, "ctx": 4096})] == ["small", "medium"]

# health-based fallback to the next candidate
M["small"]["healthy"] = False
assert r.route({"tier": 1, "ctx": 4096}) == "medium"
M["medium"]["healthy"] = False
assert r.route({"tier": 1, "ctx": 4096}) is None
M["small"]["healthy"] = M["medium"]["healthy"] = True
assert r.route({"tier": 1, "ctx": 4096}) == "small"

# Little's law sizes the concurrency limit: 5 qps * 2 s = 10
assert ns["littles_limit"](5, 2.0) == 10
assert ns["littles_limit"](5, 2.0, 1.25) == 13
assert ns["littles_limit"](4, 2.0) == 8
# 8 qps * 2 s = 16 concurrent: medium no longer fits that limit
limit = ns["littles_limit"](8, 2.0)
assert limit == 16
assert ns["Router"](reg, GPU, limit).route({"tier": 2, "ctx": 4096}) is None

# admission control: 2 running, 2 queued, then reject with a reason
ac = ns["AdmissionControl"](max_active=2, max_queue=2)
assert [ac.submit(i) for i in range(2)] == [("run", None)] * 2
assert [ac.submit(i) for i in range(2, 4)] == [("queued", None)] * 2
assert ac.submit(4) == ("rejected", "queue full")
assert ac.release() == 2          # a slot frees; queued request starts
assert ac.active == 2 and len(ac.queue) == 1
assert ac.submit(5) == ("queued", None)   # accepted again
assert ac.submit(6) == ("rejected", "queue full")
print("project-local-llm-platform OK")

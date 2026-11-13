import math, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "ai-system-design-capacity-estimation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
size, avail = ns["size_system"], ns["replica_availability"]

# worked example, checked by hand at each step
r = size(864000, 10, 3, 500, 250, 10000, 0.5)
assert r["avg_qps"] == 100.0            # 8.64e6 requests / 86400 s
assert r["peak_qps"] == 300.0           # x3 peak factor
assert r["tok_per_s"] == 225000.0       # 300 * 750 tokens
assert r["gpus"] == 45                  # 225000 / (10000 * 0.5), exact
assert math.isclose(r["latency_s"], 5.3)  # 0.2 + 500/5000 + 250/50
assert math.isclose(r["concurrency"], 1590.0)  # Little: 300 * 5.3
assert r["seqs_per_replica"] == 36      # ceil(1590 / 45) = ceil(35.33)
weights = 8e9 * 2
kv_seq = 131072 * 750
assert r["mem_bytes"] == weights + 36 * kv_seq == 19538944000
assert r["monthly_cost"] == 45 * 2.0 * 730 == 65700.0
assert avail(0.01, 2) == 1 - 0.01 ** 2
assert math.isclose(avail(0.01, 2), 0.9999) and avail(0.5, 1) == 0.5
assert r["availability"] == avail(0.01, 45)

# ceiling behaviour: exact multiple needs no extra GPU, one more token does
assert size(864000, 10, 3, 500, 250, 10000, 0.5)["gpus"] == 45
assert size(864000, 10, 3, 500, 251, 10000, 0.5)["gpus"] == 46

# monotonicity: more users never need fewer GPUs; more utilisation fewer
prev = 0
for dau in range(100000, 5000001, 100000):
    g = size(dau, 10, 3, 500, 250, 10000, 0.5)["gpus"]
    assert g >= prev
    prev = g
prev = 10 ** 9
for u in (0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
    g = size(864000, 10, 3, 500, 250, 10000, u)["gpus"]
    assert g <= prev
    prev = g
assert size(864000, 10, 3, 500, 250, 10000, 1.0)["gpus"] == 23  # ceil(22.5)

# Little's law: doubling latency (via decode speed) doubles concurrency
a = size(864000, 10, 3, 500, 250, 10000, 0.5, queue_s=0, prefill_tps=1e12, decode_tps=250)
b = size(864000, 10, 3, 500, 250, 10000, 0.5, queue_s=0, prefill_tps=1e12, decode_tps=125)
assert math.isclose(b["concurrency"], 2 * a["concurrency"], rel_tol=1e-6)
assert math.isclose(a["concurrency"], 300 * 1.0, rel_tol=1e-6)  # 250/250 = 1 s

# more replicas never lower availability
assert avail(0.05, 3) > avail(0.05, 2) > avail(0.05, 1)
print("ai-system-design-capacity-estimation OK")

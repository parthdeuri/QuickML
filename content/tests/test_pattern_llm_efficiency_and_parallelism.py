import yaml, numpy as np
d = yaml.safe_load(open("content/patterns/llm-efficiency-and-parallelism.yaml", encoding="utf-8"))
assert d["id"] == "llm-efficiency-and-parallelism"
ns = {}
exec(d["template"], ns)
assert ns["weight_bytes"](7e9, 2) == 14e9
assert ns["weight_bytes"](7e9, 1) == ns["weight_bytes"](7e9, 2) / 2
assert ns["kv_bytes"](2, 4, 8, 10, 3) == 2 * 2 * 4 * 8 * 10 * 3 * 2
assert ns["kv_bytes"](2, 4, 8, 20, 3) == 2 * ns["kv_bytes"](2, 4, 8, 10, 3)
x = np.random.default_rng(0).normal(size=1000)
q, s = ns["quantize"](x)
err = np.abs(ns["dequantize"](q, s) - x).max()
assert q.dtype == np.int8 and err <= s / 2 + 1e-12
assert ns["attainable"](100, 2, 10) == 20 and ns["attainable"](100, 2, 80) == 100
t1 = ns["decode_tokens_per_s"](14e9, 1e12)
assert abs(t1 - 1e12 / 14e9) < 1e-9
assert ns["decode_tokens_per_s"](7e9, 1e12) > t1
assert ns["decode_tokens_per_s"](14e9, 1e12, batch=8) == 8 * t1
assert ns["shard_bytes"](80e9, 4) == 20e9
print("pattern llm-efficiency-and-parallelism OK")

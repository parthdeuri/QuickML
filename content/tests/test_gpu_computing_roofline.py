import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gpu-computing-roofline.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# hand-checked: 2*4*3*5 = 120 flops; bytes = 2*(12+15+20) = 94
assert ns["matmul_flops"](4, 3, 5) == 120
assert ns["matmul_bytes"](4, 3, 5, 2) == 94
# FLOP count equals a counted multiply-add loop on a tiny case
m, k, n = 3, 4, 5
cnt = 0
for i in range(m):
    for j in range(n):
        for p in range(k):
            cnt += 2  # one multiply + one add
assert cnt == ns["matmul_flops"](m, k, n)
# attainable and ridge
P, B = 100e12, 1e12
assert ns["ridge_point"](P, B) == 100.0
assert ns["attainable"](P, B, 10) == 10e12
assert ns["attainable"](P, B, 500) == P
# intensity grows with size and crosses the ridge
ai = lambda s: ns["arithmetic_intensity"](ns["matmul_flops"](s, s, s), ns["matmul_bytes"](s, s, s, 2))
vals = [ai(s) for s in (8, 64, 512, 4096)]
assert vals == sorted(vals) and vals[0] < 100 < vals[-1]
# square matmul intensity is 2s/(3b) exactly: 2s^3 / (3*b*s^2)
assert abs(ai(300) - 2 * 300 / (3 * 2)) < 1e-9
# elementwise add: n flops, 3n values moved -> 1/(3b)
for b in (2, 4):
    nel = 1000
    assert abs(ns["arithmetic_intensity"](nel, 3 * nel * b) - 1 / (3 * b)) < 1e-15
# decode bound: linear in batch until the compute roof
wb = 14e9
t1 = ns["decode_tokens_per_s"](wb, 1e12, 1)
assert abs(t1 - 1e12 / wb) < 1e-9
assert abs(ns["decode_tokens_per_s"](wb, 1e12, 8) - 8 * t1) < 1e-9
fl = 2 * (wb / 2)  # 2 flops per weight, 2 bytes per weight
caps = [ns["decode_tokens_per_s"](wb, 1e12, bt, peak=P, flops_per_token=fl) for bt in (1, 2, 4, 1024)]
assert abs(caps[1] - 2 * caps[0]) < 1e-9 and abs(caps[2] - 4 * caps[0]) < 1e-9
assert abs(caps[3] - P / fl) < 1e-9 and caps[3] < 1024 * t1
print("gpu-computing-roofline OK")

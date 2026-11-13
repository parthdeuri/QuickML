import yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gpu-memory-estimation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# hand-checked: 7e9 params fp16 = 14e9 B; KV = 2*32*8*128*4096*2*2 B
P = 7_000_000_000
kv = 2 * 32 * 8 * 128 * 4096 * 2 * 2
assert kv == 1_073_741_824 * 1 * 1 and ns["inference_bytes"](P, 32, 8, 128, 4096, 2) == 2 * P + kv
assert ns["inference_bytes"](P, 32, 8, 128, 4096, 2, overhead=0.25) == 1.25 * (2 * P + kv)

# weights bytes equal real parameter nbytes
torch.manual_seed(0)
m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
n = sum(p.numel() for p in m.parameters())
assert sum(p.nbytes for p in m.parameters()) == ns["training_bytes"](n, "fp32")["weights"]
assert sum(p.nbytes for p in m.half().parameters()) == ns["training_bytes"](n)["weights"]
m = m.float()

# optimiser state equals real Adam state numel * 4 bytes
opt = torch.optim.Adam(m.parameters())
m(torch.randn(5, 4)).sum().backward()
opt.step()
state = sum(s[k].numel() for s in opt.state.values() for k in ("exp_avg", "exp_avg_sq"))
assert state * 4 == ns["training_bytes"](n, "fp32")["optim"] == 8 * n
t = ns["training_bytes"](n)
assert sum(t.values()) == 16 * n

# ZeRO
G, P = 8, 8_000_000
tot = 16 * P
assert ns["zero_bytes_per_gpu"](0, G, P) == tot
assert abs(ns["zero_bytes_per_gpu"](1, G, P) - (4 * P + 12 * P / G)) < 1e-6
assert abs(ns["zero_bytes_per_gpu"](2, G, P) - (2 * P + 14 * P / G)) < 1e-6
assert abs(ns["zero_bytes_per_gpu"](3, G, P) - tot / G) < 1e-6
# activations are not sharded
assert abs(ns["zero_bytes_per_gpu"](3, G, P, 1000) - (tot / G + 1000)) < 1e-6

# max_batch: largest that fits
args = (32, 8, 128, 4096)
for budget, ov in [(80e9, 0.0), (40e9, 0.25), (24e9, 0.1)]:
    b = ns["max_batch"](budget, 7e9, *args, overhead=ov)
    assert b >= 1
    assert ns["inference_bytes"](7e9, *args, b, overhead=ov) <= budget
    assert ns["inference_bytes"](7e9, *args, b + 1, overhead=ov) > budget
assert ns["max_batch"](1e9, 7e9, *args) == 0
print("gpu-memory-estimation OK")

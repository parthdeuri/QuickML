import yaml, torch
import torch.nn as nn
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "parameter-count-flops.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)


class Block(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)


class Model(nn.Module):
    def __init__(self, L, d, d_ff, V, tied):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d, d_ff) for _ in range(L))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)  # no position table (RoPE)
        if tied:
            self.head.weight = self.emb.weight


def real_count(m):
    return sum(p.numel() for p in m.parameters())  # parameters() dedups tied


for L, d, d_ff, V in [(2, 8, 32, 50), (3, 16, 64, 100), (4, 12, 24, 37)]:
    for tied in (True, False):
        m = Model(L, d, d_ff, V, tied)
        assert ns["count_params"](L, d, d_ff, V, tied) == real_count(m), (L, d, tied)

# 12 L d^2 + V d rule is close to the exact count when d_ff = 4d
L, d, V = 4, 128, 100
exact = ns["count_params"](L, d, 4 * d, V, tied=True)
rule = ns["rule_of_thumb"](L, d, V)
assert abs(exact - rule) / exact < 0.02, (exact, rule)

N, D = 1_000_000, 10_000
assert ns["train_flops"](N, D) == 6 * N * D
assert ns["infer_flops_per_token"](N) == 2 * N
assert ns["memory_bytes"](N, 4) == 4_000_000
assert ns["memory_bytes"](N, 2) == 2_000_000
assert ns["memory_bytes"](N, 1) == 1_000_000
assert ns["memory_bytes"](N, training=True) == 16_000_000
print("parameter-count-flops OK")

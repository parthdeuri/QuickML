import math
import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "agent-evaluation-safety.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
pass_at_k, pass_pow_k = ns["pass_at_k"], ns["pass_pow_k"]
f1, tm, gate = ns["tool_call_f1"], ns["trajectory_match"], ns["permission_gate"]

# pass@k equals the combinatorial formula
for n, c, k in [(10, 3, 1), (10, 3, 5), (20, 7, 4), (5, 0, 3), (5, 5, 2), (6, 4, 3)]:
    want = 1 - math.comb(n - c, k) / math.comb(n, k)
    assert abs(pass_at_k(n, c, k) - want) < 1e-12, (n, c, k)
assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-12 and pass_at_k(5, 0, 3) == 0.0
assert pass_at_k(6, 4, 3) == 1.0  # fewer than k failures: always a hit

# seeded Monte-Carlo: draw k of n samples without replacement
rng = np.random.default_rng(0)
n, c, k = 20, 7, 4
samples = np.array([1] * c + [0] * (n - c))
hits = sum(rng.permutation(samples)[:k].any() for _ in range(40000))
assert abs(hits / 40000 - pass_at_k(n, c, k)) < 0.01, hits / 40000

# pass^k arithmetic and the contrast with pass@k
assert abs(pass_pow_k(0.9, 5) - 0.59049) < 1e-12
assert pass_pow_k(0.5, 3) == 0.125 and pass_pow_k(1.0, 10) == 1.0
assert 1 - (1 - 0.9) ** 5 > 0.999 > pass_pow_k(0.9, 5)

# tool-call F1 hand cases
ref = [("search", {"q": "a"}), ("read", {"id": 1})]
assert f1(ref, ref) == (1.0, 1.0, 1.0)
extra = ref + [("search", {"q": "b"})]
p, r, f = f1(extra, ref)  # one extra call: precision 2/3, recall 1
assert abs(p - 2 / 3) < 1e-12 and r == 1.0 and abs(f - 0.8) < 1e-12
p, r, f = f1(ref[:1], ref)  # one missing call: precision 1, recall 1/2
assert p == 1.0 and r == 0.5 and abs(f - 2 / 3) < 1e-12
wrong_arg = [("search", {"q": "z"}), ("read", {"id": 1})]
assert f1(wrong_arg, ref)[2] == 0.5            # arguments differ
assert f1(wrong_arg, ref, args=False)[2] == 1.0  # names only
assert f1([], ref) == (0.0, 0.0, 0.0)
# a duplicated call is counted as a multiset, not a set
assert f1([("read", {"id": 1})] * 2, [("read", {"id": 1})])[0] == 0.5

# trajectory modes
rf = ["search", "read", "answer"]
assert tm(rf, rf, "exact") and not tm(["search", "read", "x", "answer"], rf, "exact")
assert tm(["search", "x", "read", "answer"], rf, "in_order")
assert not tm(["read", "search", "answer"], rf, "in_order")
assert tm(["read", "search", "answer"], rf, "any_order")
assert not tm(["search", "answer"], rf, "any_order")
assert not tm(["search", "read", "read"], rf, "any_order")
assert tm(["read", "read", "search", "answer"], rf, "any_order")
try:
    tm(rf, rf, "fuzzy")
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# permission gate: fail closed
policy = {"tiers": {"read_file": "read", "write_file": "write",
                    "delete_db": "destructive", "weird": "admin"},
          "auto": {"read", "write"}}
assert gate("read_file", policy) == "allow"
assert gate("write_file", policy) == "allow"
assert gate("delete_db", policy) == "confirm"
assert gate("delete_db", policy, confirmed=False) == "confirm"
assert gate("delete_db", policy, confirmed=True) == "allow"
assert gate("launch_missiles", policy, confirmed=True) == "deny"  # unknown
assert gate("weird", policy, confirmed=True) == "deny"  # unknown tier
read_only = {"tiers": policy["tiers"], "auto": {"read"}}
assert gate("write_file", read_only) == "deny"
assert ns["within_budget"](5, 0.5, 5, 1.0) and not ns["within_budget"](6, 0.5, 5, 1.0)
assert not ns["within_budget"](1, 2.0, 5, 1.0)
print("agent-evaluation-safety OK")

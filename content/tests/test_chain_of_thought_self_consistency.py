import numpy as np, yaml
from scipy.stats import binom
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "chain-of-thought-self-consistency.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
acc, vote, ext = ns["majority_accuracy"], ns["majority_vote"], ns["extract_final_answer"]

# exact binomial tail
for p in (0.1, 0.4, 0.5, 0.6, 0.7, 0.9):
    for n in (1, 3, 5, 11, 21):
        assert abs(acc(p, n) - binom.sf(n // 2, n, p)) < 1e-12, (p, n)
assert abs(acc(0.7, 1) - 0.7) < 1e-12
assert [round(acc(0.7, n), 3) for n in (1, 5, 21)] == [0.7, 0.837, 0.974]

# monotone in n: up for p>0.5, down for p<0.5 (odd n)
ns_ = [1, 3, 5, 7, 9, 21, 51]
up = [acc(0.6, n) for n in ns_]
dn = [acc(0.4, n) for n in ns_]
assert all(a < b for a, b in zip(up, up[1:]))
assert all(a > b for a, b in zip(dn, dn[1:]))
assert abs(acc(0.5, 5) - 0.5) < 1e-12

# Monte-Carlo with fixed seed: majority of n votes, wrong answers spread
rng = np.random.default_rng(0)
p, n, T = 0.6, 11, 40000
correct = rng.random((T, n)) < p
mc = (correct.sum(1) > n // 2).mean()
assert abs(mc - acc(p, n)) < 0.01, (mc, acc(p, n))
# plurality over 10 spread wrong answers is at least the majority bound
ans = np.where(correct, 0, rng.integers(1, 11, size=(T, n)))
pl = np.mean([np.bincount(r).argmax() == 0 for r in ans])
assert pl >= acc(p, n) - 0.01

# answer parsing on messy strings
assert ext("Step 1... so the total is 12.\nAnswer: 42") == "42"
assert ext("answer:   42.") == "42"
assert ext("ANSWER: $1,200 dollars") == "1200"
assert ext("Answer: -3.5, roughly") == "-3.5"
assert ext("I think... Answer: 7\nWait, recheck.\nAnswer: 8") == "8"
assert ext("Answer: Paris.") == "paris"
assert ext("no final line here") is None

# vote: majority, None ignored, tie goes to first seen
assert vote(["4", "5", "4", None, "6"]) == "4"
assert vote([None, None]) is None
assert vote(["a", "b", "b", "a"]) == "a"
print("chain-of-thought-self-consistency OK")

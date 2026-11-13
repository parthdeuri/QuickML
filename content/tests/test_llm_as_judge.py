import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "llm-as-judge.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
swap, kappa = ns["pairwise_with_swap"], ns["cohen_kappa"]

rng = np.random.default_rng(0)
for _ in range(20):
    n, k = int(rng.integers(20, 200)), int(rng.integers(2, 5))
    a, b = rng.integers(0, k, n), rng.integers(0, k, n)
    assert np.isclose(kappa(a, b), cohen_kappa_score(a, b)), (kappa(a, b), cohen_kappa_score(a, b))
# identical labels -> 1, independent labels -> about 0
x = rng.integers(0, 3, 500)
assert kappa(x, x) == 1.0
big1, big2 = rng.integers(0, 3, 20000), rng.integers(0, 3, 20000)
assert abs(kappa(big1, big2)) < 0.02
# pure position bias always picks the first slot -> tie after swapping
assert swap(lambda x, y: "first", 3, 7) == "tie"
assert swap(lambda x, y: "second", 3, 7) == "tie"
# unbiased judge prefers the larger value -> consistent wins
better = lambda x, y: "first" if x > y else "second"
assert swap(better, 9, 2) == "A" and swap(better, 2, 9) == "B"
# noisy judge: position-biased on close calls, content-driven on clear ones
close = lambda x, y: "first" if abs(x - y) < 2 else better(x, y)
assert swap(close, 5, 6) == "tie" and swap(close, 9, 2) == "A"
print("llm-as-judge OK")

import numpy as np, yaml
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "dbscan.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)


def same_partition(a, b):
    """Equal up to renaming of clusters; noise (-1) must match exactly."""
    if not np.array_equal(a == -1, b == -1):
        return False
    pairs = {(x, y) for x, y in zip(a, b) if x != -1}
    return len({x for x, _ in pairs}) == len(pairs) == len({y for _, y in pairs})


rng = np.random.default_rng(0)
cases = [
    (make_moons(300, noise=0.07, random_state=0)[0], 0.15, 5),
    (np.r_[rng.normal(0, 0.4, (80, 2)), rng.normal(4, 0.4, (80, 2)),
           rng.uniform(-3, 7, (20, 2))], 0.5, 4),
    (rng.uniform(0, 5, (120, 2)), 0.4, 3),
]
for X, eps, mp in cases:
    mine = ns["dbscan"](X, eps, mp)
    ref = DBSCAN(eps=eps, min_samples=mp).fit(X).labels_
    assert same_partition(mine, ref), (eps, mp)
    assert (mine == -1).any() or len(set(mine)) > 1
assert len(set(ns["dbscan"](cases[0][0], 0.15, 5)) - {-1}) == 2  # two moons
print("dbscan OK")

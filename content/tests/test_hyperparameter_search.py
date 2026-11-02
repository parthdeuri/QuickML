import numpy as np, yaml
from pathlib import Path
from sklearn.model_selection import ParameterGrid, ParameterSampler

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "hyperparameter-search.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
space = {"lr": [0.001, 0.01, 0.1], "depth": [2, 4, 6, 8], "bootstrap": [True, False]}

# Grid: identical set AND order to ParameterGrid (sorted keys, last varies fastest)
mine = list(ns["grid"](space))
ref = list(ParameterGrid(space))
assert mine == ref and len(mine) == 3 * 4 * 2

# Random: every draw lies in the grid, same keys as ParameterSampler output
rs = list(ns["random_search"](space, 50, np.random.default_rng(0)))
assert len(rs) == 50 and all(c in ref for c in rs)
sk = list(ParameterSampler(space, 10, random_state=0))
assert all(set(c) == set(rs[0]) for c in sk)
# DIFFERENCE: ParameterSampler is without replacement for all-list spaces
# (distinct configs); this card's sampler draws with replacement.
assert len({tuple(sorted(c.items())) for c in sk}) == 10
assert len({tuple(sorted(c.items())) for c in rs}) < 50
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # sklearn warns and caps at grid size
    capped = list(ParameterSampler(space, 50, random_state=0))
assert len(capped) == len(ref) and all(c in ref for c in capped)

# Callable samplers (log-uniform) give a fresh value per trial; grid reuses values
cont = {"lr": lambda r: 10 ** r.uniform(-4, 0), "depth": [2, 4, 6]}
trials = list(ns["random_search"](cont, 9, np.random.default_rng(1)))
assert len({c["lr"] for c in trials}) == 9
g = list(ns["grid"]({"lr": [1e-4, 1e-2, 1], "depth": [2, 4, 6]}))
assert len({c["lr"] for c in g}) == 3 and len(g) == 9

# best() on a score that depends only on lr: random finds lr closer to the optimum
score = lambda c: -abs(np.log10(c["lr"]) + 2.4)
sg, _ = ns["best"](score, g)
sr, _ = ns["best"](score, ns["random_search"](cont, 9, np.random.default_rng(1)))
assert sr > sg, (sr, sg)
bs, bc = ns["best"](lambda c: c["depth"], ns["grid"](space))
assert bc["depth"] == 8 and bs == 8
print("hyperparameter-search OK")

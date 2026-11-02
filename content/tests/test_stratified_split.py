import numpy as np, yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "stratified-split.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
y = rng.choice(3, size=1000, p=[0.9, 0.09, 0.01])
frac = 0.2
tr, te = ns["stratified_split"](y, frac, seed=0)
# a clean partition
assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y)
# deterministic for a fixed seed
tr2, te2 = ns["stratified_split"](y, frac, seed=0)
assert np.array_equal(te, te2)
share = np.bincount(y) / len(y)
tol = 3 / len(te)  # stated tolerance: rounding is under one sample per class
for part in (tr, te):
    assert np.abs(np.bincount(y[part], minlength=3) / len(part) - share).max() <= tol
# per-class test count is within half a sample of frac * class size
counts = np.bincount(y)
assert np.abs(np.bincount(y[te], minlength=3) - frac * counts).max() <= 0.5 + 1e-9
# sklearn is close, not identical (it allocates counts to hit n_test exactly)
_, te_sk = train_test_split(np.arange(len(y)), test_size=frac, stratify=y, random_state=0)
d = np.abs(np.bincount(y[te], minlength=3) / len(te)
           - np.bincount(y[te_sk], minlength=3) / len(te_sk))
assert d.max() <= tol, d
assert abs(len(te) - len(te_sk)) <= 3
# unstratified random splits can miss the rare class; the stratified one never does
misses = 0
for s in range(200):
    _, t = train_test_split(np.arange(len(y)), test_size=frac, random_state=s)
    misses += (y[t] == 2).sum() == 0
assert (y[te] == 2).sum() == round(frac * counts[2])
print("stratified-split OK; random splits missing the rare class:", misses, "/ 200")

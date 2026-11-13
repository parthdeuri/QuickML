import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "benchmark-contamination.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ngram_set, rate, ratio, gap = (ns[k] for k in ("ngram_set", "contamination_rate", "memorisation_ratio", "accuracy_gap"))

assert ngram_set("a b c d".split(), 2) == {("a", "b"), ("b", "c"), ("c", "d")}
assert ngram_set(["a"], 3) == set()
train = ["a b c d e f g".split(), "the cat sat on the mat today".split()]
bench = ["x c d e y".split(), "p q r s".split(), "a b c d".split()]
r, flags = rate(train, bench, n=3)
assert flags == [True, False, True] and np.isclose(r, 2 / 3)
# 4-grams: only the third item shares one ("a b c d")
r4, f4 = rate(train, bench, n=4)
assert f4 == [False, False, True] and np.isclose(r4, 1 / 3)
# planted item with 13-gram inside a long doc is flagged, a clean one is not
rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(5000)]
item = list(rng.choice(vocab, 20))
doc = list(rng.choice(vocab, 300)) + item + list(rng.choice(vocab, 300))
clean = list(rng.choice(vocab, 20))
r, f = rate([doc], [item, clean])
assert f == [True, False] and r == 0.5

# memorisation stub: low loss on the exact item, higher on rewordings
loss = {"orig": 0.4, "p1": 2.0, "p2": 2.2, "p3": 1.8}
assert np.isclose(ratio(loss.get, "orig", ["p1", "p2", "p3"]), 0.4 / 2.0)
assert ratio(loss.get, "orig", ["p1", "p2", "p3"]) < 1
loss2 = {"orig": 2.0, "p1": 2.0, "p2": 2.2, "p3": 1.8}
assert np.isclose(ratio(loss2.get, "orig", ["p1", "p2", "p3"]), 1.0)

correct = [1, 1, 1, 0, 1, 0, 0, 0]
cont = [1, 1, 1, 1, 0, 0, 0, 0]
assert np.isclose(gap(correct, cont), 0.75 - 0.25)
print("benchmark-contamination OK")

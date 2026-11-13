import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "hallucination-detection.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
mc, ent, ov = ns["majority_confidence"], ns["answer_entropy"], ns["overlap_score"]

a = ["Paris", "Paris", "Lyon", "Paris"]
assert mc(a) == 0.75
assert np.isclose(ent(a), -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
assert np.isclose(ent(a), 0.5623351446)

assert ent(["x"] * 5) == 0.0 and mc(["x"] * 5) == 1.0
for k in (2, 3, 7):
    diff = [str(i) for i in range(k)]
    assert np.isclose(ent(diff), np.log(k))
    assert np.isclose(mc(diff), 1 / k)

rng = np.random.default_rng(0)
for _ in range(50):
    ans = list(rng.integers(0, 4, size=rng.integers(1, 12)))
    assert 0.0 <= mc(ans) <= 1.0
    assert 0.0 <= ent(ans) <= np.log(len(ans)) + 1e-12

claim = "the capital is paris"
samples = ["Paris is the capital", "the capital is Lyon"]
assert np.isclose(ov(claim, samples), (4 / 4 + 3 / 4) / 2)
assert ov(claim, [claim] * 3) == 1.0
assert ov(claim, ["no shared words here"]) == 0.0
print("hallucination-detection OK")

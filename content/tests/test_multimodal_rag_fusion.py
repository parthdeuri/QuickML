import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "multimodal-rag-fusion.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# toy 2-D CLIP-like space: image j sits at angle 90*j degrees (varied norms),
# its caption sits near the same direction
ang = np.deg2rad([0, 90, 180, 270])
images = np.stack([np.cos(ang), np.sin(ang)], 1) * np.array([[1], [3], [0.5], [2]])
off = np.deg2rad([8, -6, 5, -9])
texts = np.stack([np.cos(ang + off), np.sin(ang + off)], 1)
for j in range(4):
    top, sc = ns["cross_modal_retrieve"](texts[j], images, 2)
    assert top[0] == j, (j, top)
    assert sc[0] >= sc[1] and top.shape == (2,)
    assert np.all(sc <= 1 + 1e-12)

# min-max lies in [0, 1] and hits both ends; constant list gives zeros
rng = np.random.default_rng(0)
s = rng.normal(size=20) * 50 + 7
m = ns["minmax"](s)
assert m.min() == 0 and m.max() == 1 and np.all((m >= 0) & (m <= 1))
assert np.array_equal(ns["minmax"](np.ones(4)), np.zeros(4))
z = ns["zscore"](s)
assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12

# weight (1,0) reproduces list 1's ranking, (0,1) list 2's
A, B = rng.normal(size=10) * 100, rng.normal(size=10)
for norm in ("minmax", "zscore", None):
    f10 = ns["late_fusion"]([A, B], (1, 0), norm)
    f01 = ns["late_fusion"]([A, B], (0, 1), norm)
    assert np.array_equal(np.argsort(-f10), np.argsort(-A))
    assert np.array_equal(np.argsort(-f01), np.argsort(-B))

# rrf hand-computed: a=1/61+1/63, b=1/62+1/61, c=1/63+1/62
out = ns["rrf"]([["a", "b", "c"], ["b", "c", "a"]])
exp = {"a": 1/61 + 1/63, "b": 1/62 + 1/61, "c": 1/63 + 1/62}
assert [d for d, _ in out] == ["b", "a", "c"]
assert all(abs(dict(out)[d] - exp[d]) < 1e-15 for d in exp)
assert abs(dict(ns["rrf"]([["x"]], k=10))["x"] - 1/11) < 1e-15

# scale dominance: item 2 is second in both lists, item 1 wins list A
A = np.array([100.0, 300.0, 200.0])       # large scale
B = np.array([0.9, 0.0, 0.5])             # unit scale, prefers item 0
raw = ns["late_fusion"]([A, B], (1, 1), None)
nrm = ns["late_fusion"]([A, B], (1, 1), "minmax")
assert raw.argmax() == A.argmax() == 1          # A decides everything
assert np.array_equal(np.argsort(-raw), np.argsort(-A))
assert nrm.argmax() == 2                        # both lists count
print("multimodal-rag-fusion OK")

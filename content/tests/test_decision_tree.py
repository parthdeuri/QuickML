import numpy as np, yaml
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "decision-tree.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = ((X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] + 0.3 * rng.normal(size=200)) > 0.3).astype(int)
Xt = rng.normal(size=(100, 4))
for depth in (1, 2, 3, 4):
    tree = ns["grow"](X, y, depth)
    ref = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    for A in (X, Xt):
        mine = np.array([ns["predict_one"](tree, x) for x in A])
        assert (mine == ref.predict(A)).all(), depth
print("decision-tree OK")

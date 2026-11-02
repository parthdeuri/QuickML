import numpy as np, yaml
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "random-forest.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
X, y = make_classification(n_samples=300, n_features=8, n_informative=4,
                           flip_y=0.05, random_state=0)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.4, random_state=0)
trees = ns["fit"](Xa, ya, B=25, depth=6)
mine = np.array([ns["predict_one"](trees, x) for x in Xb])
ref = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0).fit(Xa, ya)
acc, racc = (mine == yb).mean(), ref.score(Xb, yb)
single = np.array([ns["predict_one"](trees[:1], x) for x in Xb])
print(acc, racc, (single == yb).mean())
assert abs(acc - racc) <= 0.06
assert acc >= (single == yb).mean()
assert acc > 0.8
print("random-forest OK")

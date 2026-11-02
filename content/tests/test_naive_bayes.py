import numpy as np, yaml
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "naive-bayes.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                           n_classes=3, random_state=0)
model = ns["fit"](X, y)
ref = GaussianNB().fit(X, y)
assert np.allclose(ns["predict_proba"](model, X), ref.predict_proba(X), atol=1e-8)
assert (model[0][ns["log_joint"](model, X).argmax(1)] == ref.predict(X)).all()
print("naive-bayes OK")

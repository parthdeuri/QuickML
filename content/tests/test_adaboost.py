import warnings
import numpy as np, yaml
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore")
card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "adaboost.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

X, y01 = make_classification(300, n_features=4, n_informative=3, n_redundant=0,
                             flip_y=0.1, class_sep=1.0, random_state=0)
y = 2 * y01 - 1
T = 15
model = ns["adaboost"](X, y, T)
pred = ns["predict"](model, X)
assert (pred != 0).all()


class Stump(BaseEstimator, ClassifierMixin):
    """The card's exact weighted-error stump, wrapped for sklearn."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.p_ = ns["stump"](X, y, sample_weight / sample_weight.sum())[1:]
        return self

    def predict(self, X):
        j, t, s = self.p_
        return np.where(X[:, j] <= t, s, -s)


# Boosting loop vs sklearn SAMME with the same weak learner. sklearn's vote
# weight is log((1-e)/e) for K=2, i.e. twice the card's 1/2 log odds; the
# normalised sample weights and the sign of the vote are identical.
ref = AdaBoostClassifier(estimator=Stump(), n_estimators=T, algorithm="SAMME",
                         random_state=0).fit(X, y)
assert len(ref.estimators_) == T
assert np.allclose(2 * np.array([m[0] for m in model]), ref.estimator_weights_[:T])
assert np.array_equal(pred, ref.predict(X))
for (a, j, t, s), est in zip(model, ref.estimators_):
    assert tuple(est.p_) == (j, t, s)

# Difference: sklearn's DecisionTreeClassifier stump minimises gini, not the
# weighted error, so only loose agreement (>0.7 of predictions) is required.
ref_tree = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                              n_estimators=T, algorithm="SAMME",
                              random_state=0).fit(X, y)
agree = (ref_tree.predict(X) == pred).mean()
assert agree > 0.7, agree
acc, acc_tree = (pred == y).mean(), ref_tree.score(X, y)
assert abs(acc - acc_tree) < 0.1, (acc, acc_tree)
# Boosting beats a single stump
single = ns["predict"](ns["adaboost"](X, y, 1), X)
assert acc > (single == y).mean()
print("adaboost OK; agreement with gini-tree stumps:", round(agree, 3))

import dataclasses, numpy as np, yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler as SkScaler

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "python-oop-protocols-dataclasses.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Estimator, TrainConfig = ns["Estimator"], ns["TrainConfig"]
StandardScaler, MeanRegressor, Pipeline = ns["StandardScaler"], ns["MeanRegressor"], ns["Pipeline"]

# runtime_checkable Protocol is structural
assert isinstance(MeanRegressor(), Estimator)
assert not isinstance(StandardScaler(), Estimator)  # no predict
class Duck:  # unrelated class with the right methods
    def fit(self, X, y=None): return self
    def predict(self, X): return X
assert isinstance(Duck(), Estimator)

# frozen dataclass and validation
cfg = TrainConfig(lr=0.1)
try:
    cfg.lr = 0.5
    raise AssertionError("should be frozen")
except dataclasses.FrozenInstanceError:
    pass
assert TrainConfig().tags == [] and TrainConfig().tags is not TrainConfig().tags
for bad in ({"lr": 0}, {"lr": -1.0}, {"epochs": 0}):
    try:
        TrainConfig(**bad)
        raise AssertionError(bad)
    except ValueError:
        pass
cfg.tags.append("x")  # frozen blocks rebinding, not in-place mutation
assert cfg.tags == ["x"]

# scaler equals sklearn, including a constant column
rng = np.random.default_rng(0)
X = rng.normal(size=(30, 4)) * [1, 5, 0.1, 3] + [0, 10, -2, 1]
X[:, 3] = 4.0
sk = SkScaler().fit(X)
mine = StandardScaler().fit(X)
assert np.allclose(mine.transform(X), sk.transform(X))
assert np.allclose(mine.mean_, sk.mean_) and np.allclose(mine.scale_, sk.scale_)
Xt = rng.normal(size=(5, 4))
assert np.allclose(mine.transform(Xt), sk.transform(Xt))

# pipeline equals manual composition
y = rng.normal(size=30)
pipe = Pipeline([StandardScaler()], MeanRegressor()).fit(X, y)
sc = StandardScaler().fit(X)
reg = MeanRegressor().fit(sc.transform(X), y)
assert np.allclose(pipe.predict(Xt), reg.predict(sc.transform(Xt)))
assert np.allclose(pipe.predict(Xt), y.mean())
assert pipe.predict(Xt).shape == (5,)
print("python-oop-protocols-dataclasses OK")

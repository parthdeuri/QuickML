import numpy as np, pandas as pd, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "pandas-groupby-merge-window.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
keys = rng.choice(list("abcde"), size=60).tolist()
vals = rng.normal(size=60).tolist()
df = pd.DataFrame({"k": keys, "v": vals})
for name, fn in [("mean", np.mean), ("sum", np.sum), ("max", np.max), ("count", len)]:
    got = ns["groupby_agg"](keys, vals, fn)
    want = df.groupby("k")["v"].agg(name)
    assert list(got) == list(want.index)
    assert np.allclose(list(got.values()), want.values), name

# left join with duplicate right keys and missing keys
left = [{"id": int(i), "x": float(i) * 1.5} for i in rng.integers(0, 6, size=12)]
right = [{"id": 0, "y": 10.0}, {"id": 1, "y": 11.0}, {"id": 1, "y": 12.0},
         {"id": 2, "y": 13.0}, {"id": 4, "y": 14.0}, {"id": 4, "y": 15.0}, {"id": 4, "y": 16.0}]
got = ns["left_join"](left, right, "id")
want = pd.DataFrame(left).merge(pd.DataFrame(right), on="id", how="left")
assert len(got) == len(want) > len(left)
gdf = pd.DataFrame(got)
assert list(gdf.columns) == list(want.columns)
assert np.allclose(gdf["y"].fillna(-1), want["y"].fillna(-1))
assert (gdf["id"].values == want["id"].values).all()
# row count formula: sum over left rows of max(matches, 1)
cnt = pd.Series([r["id"] for r in right]).value_counts()
assert len(got) == sum(max(cnt.get(l["id"], 0), 1) for l in left)

# rolling mean and lag
x = rng.normal(size=25)
s = pd.Series(x)
for w in (1, 3, 7):
    r = ns["rolling_mean"](x, w)
    assert np.isnan(r[:w - 1]).all() and not np.isnan(r[w - 1:]).any()
    assert np.allclose(r, s.rolling(w).mean().values, equal_nan=True)
for k in (1, 2, 5, 30):
    assert np.allclose(ns["lag"](x, k), s.shift(k).values, equal_nan=True)

# look-ahead leakage: a lagged feature at t ignores x_t, a rolling one does not
x2 = x.copy(); x2[10] += 100
assert ns["lag"](x, 1)[10] == ns["lag"](x2, 1)[10]
assert ns["rolling_mean"](x, 3)[10] != ns["rolling_mean"](x2, 3)[10]
assert ns["rolling_mean"](ns["lag"](x, 1), 3)[10] == ns["rolling_mean"](ns["lag"](x2, 1), 3)[10]
print("pandas-groupby-merge-window OK")

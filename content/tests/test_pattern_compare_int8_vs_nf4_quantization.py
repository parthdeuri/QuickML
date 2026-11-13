import numpy as np
import yaml
d = yaml.safe_load(open("content/patterns/compare-int8-vs-nf4-quantization.yaml", encoding="utf-8"))
ns = {}
exec(d["template"], ns)
w = np.random.default_rng(0).standard_normal(64 * 256)
mse = lambda lv: float(np.mean((ns["roundtrip"](w, lv) - w) ** 2))
e_u, e_n, e_8 = mse(ns["uniform4"]), mse(ns["nf4"]), mse(ns["int8"])
assert len(ns["nf4"]) == 16 and 0.0 in ns["nf4"]
assert abs(ns["nf4"].max() - 1) < 1e-9 and abs(ns["nf4"].min() + 1) < 1e-9
assert e_n < e_u, (e_n, e_u)
assert e_8 < e_n
p = ns["prune"](w, 0.5)
assert abs(np.mean(p == 0) - 0.5) < 0.01
assert np.all(p[p != 0] == w[p != 0])
print("pattern compare-int8-vs-nf4-quantization OK")

import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "batch-normalization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
d = 5
gamma, beta = rng.normal(size=d), rng.normal(size=d)
bn = torch.nn.BatchNorm1d(d).double()
bn.weight.data = torch.tensor(gamma)
bn.bias.data = torch.tensor(beta)
run = {"mean": np.zeros(d), "var": np.ones(d)}
bn.train()
for _ in range(3):  # several batches so running stats accumulate
    x = rng.normal(size=(16, d)) * 2 + 1
    out = ns["bn_train"](x, gamma, beta, run)
    assert np.allclose(out, bn(torch.tensor(x)).detach().numpy())
assert np.allclose(run["mean"], bn.running_mean.numpy())
assert np.allclose(run["var"], bn.running_var.numpy())
bn.eval()
x = rng.normal(size=(6, d))
assert np.allclose(ns["bn_eval"](x, gamma, beta, run), bn(torch.tensor(x)).detach().numpy())
print("batch-normalization OK")

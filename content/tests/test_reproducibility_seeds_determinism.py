import math, random, yaml
import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "reproducibility-seeds-determinism.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)


# module level so DataLoader workers can import them (spawn on Windows)
def seed_worker(worker_id):
    ns["seed_worker"](worker_id)


class Noisy(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        # uses both NumPy and Python randomness inside the worker
        return float(i) + np.random.rand() + random.random()


def batches(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(Noisy(), batch_size=4, shuffle=True, num_workers=2,
                    worker_init_fn=seed_worker, generator=g)
    return [b.tolist() for b in dl]


def main():
    train, config_hash, fingerprint = ns["train"], ns["config_hash"], ns["fingerprint"]

    # same seed -> identical loss history; different seed -> different
    a, b, c = train(0), train(0), train(1)
    assert a == b and len(a) == 20
    assert a != c
    assert a[-1] < a[0]  # the tiny run actually trains

    # config hash: stable under key order, sensitive to every value
    h = config_hash({"lr": 0.1, "bs": 32, "opt": {"name": "sgd", "m": 0.9}})
    assert h == config_hash({"opt": {"m": 0.9, "name": "sgd"}, "bs": 32, "lr": 0.1})
    assert len(h) == 64
    for cfg in ({"lr": 0.2, "bs": 32, "opt": {"name": "sgd", "m": 0.9}},
                {"lr": 0.1, "bs": 33, "opt": {"name": "sgd", "m": 0.9}},
                {"lr": 0.1, "bs": 32, "opt": {"name": "sgd", "m": 0.99}},
                {"lr": 0.1, "bs": 32}):
        assert config_hash(cfg) != h

    # array fingerprint: tied to values, dtype and shape
    x = np.arange(12, dtype=np.float64)
    fx = fingerprint(x)
    assert fx == fingerprint(x.copy())
    y = x.copy(); y[3] += 1e-12
    assert fingerprint(y) != fx
    assert fingerprint(x.astype(np.float32)) != fx
    assert fingerprint(x.reshape(3, 4)) != fx

    # floating-point addition is not associative: order changes the sum
    vals = [1.0, 1e16, -1e16]
    def fold(xs):  # plain left fold (Python 3.12+ sum() compensates)
        t = 0.0
        for v in xs:
            t += v
        return t
    assert fold(vals) == 0.0 and fold(reversed(vals)) == 1.0
    assert fold(vals) != fold(reversed(vals))
    # on ordinary data the orders agree within the error bound
    r = np.random.default_rng(0).normal(size=10000)
    s1 = np.float64(0.0)
    for v in r:
        s1 += v
    s2 = np.float64(0.0)
    for v in r[::-1]:
        s2 += v
    exact = math.fsum(r)
    bound = (len(r) - 1) * 2.0 ** -53 * np.abs(r).sum()
    assert abs(s1 - exact) <= bound and abs(s2 - exact) <= bound
    assert np.isclose(s1, s2, rtol=0, atol=1e-10)

    # DataLoader: same generator seed -> identical batches (workers too)
    assert batches(7) == batches(7)
    assert batches(7) != batches(8)

    print("reproducibility-seeds-determinism OK")


if __name__ == "__main__":
    main()

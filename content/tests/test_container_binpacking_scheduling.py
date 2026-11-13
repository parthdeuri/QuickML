import random, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "container-binpacking-scheduling.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ff, ffd, bf = ns["first_fit"], ns["first_fit_decreasing"], ns["best_fit"]
lb, opt, place = ns["lower_bound"], ns["optimal_bins"], ns["place_pods"]

# hand case: sizes 4,8,1,4,2,1 in capacity 10 -> sum 20, optimum 2
items = [4, 8, 1, 4, 2, 1]
assert lb(items, 10) == 2 and opt(items, 10) == 2
assert ff(items, 10) == [10, 10] and len(ffd(items, 10)) == 2

rng = random.Random(0)
worst = 0.0
for _ in range(400):
    n = rng.randint(1, 8)
    cap = rng.choice([10, 20, 100])
    items = [rng.randint(1, cap) for _ in range(n)]
    o = opt(items, cap)
    results = {"ff": ff(items, cap), "ffd": ffd(items, cap), "bf": bf(items, cap)}
    for name, bins in results.items():
        assert all(0 < b <= cap for b in bins), (name, items, bins)
        assert sum(bins) == sum(items)
        assert lb(items, cap) <= len(bins), (name, items)
        assert o <= len(bins)
    assert lb(items, cap) <= o
    assert len(results["ffd"]) <= 11 / 9 * o + 6 / 9 + 1e-9, (items, cap)
    worst = max(worst, len(results["ffd"]) / o)

# multi-resource placement: (cpu, memory GiB), each dimension checked
nodes = [(4, 8), (4, 4)]
pods = [(2, 8), (2, 6), (2, 4), (2, 2)]
where, free = place(pods, nodes)
# pod 2 fits CPU on node 0 but memory is exhausted there
assert where == [0, None, 1, None], where
assert all(x >= 0 for f in free for x in f)
# a pod that fits CPU but not memory is not placed
w, _ = place([(1, 100)], [(8, 16)])
assert w == [None]
# and one that fits memory but not CPU
w, _ = place([(16, 1)], [(8, 16)])
assert w == [None]
# capacity never exceeded across random multi-resource instances
for _ in range(200):
    nodes = [(rng.randint(4, 16), rng.randint(8, 64)) for _ in range(4)]
    pods = [(rng.randint(1, 8), rng.randint(1, 32)) for _ in range(12)]
    where, free = place(pods, nodes)
    for i, nd in enumerate(nodes):
        used = [sum(p[d] for p, w in zip(pods, where) if w == i) for d in (0, 1)]
        assert all(u <= c for u, c in zip(used, nd))
        assert list(nd) == [u + f for u, f in zip(used, free[i])]

# indivisible GPUs strand capacity: 6 GPUs free in total, no pod of 4 fits
nodes = [(8,), (8,)]
where, free = place([(5,), (5,), (4,)], nodes)
assert where == [0, 1, None] and sum(f[0] for f in free) == 6
assert free[0][0] < 4 and free[1][0] < 4   # stranded 3 + 3
print("container-binpacking-scheduling OK")

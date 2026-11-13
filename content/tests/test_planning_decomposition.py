import itertools, random, yaml
from graphlib import TopologicalSorter
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "planning-decomposition.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
topo_order, levels, critical_path, replan = (ns[n] for n in ("topo_order", "levels", "critical_path", "replan"))


def forward(order, deps):
    pos = {t: i for i, t in enumerate(order)}
    return sorted(order) == sorted(deps) and all(pos[d] < pos[t] for t, ds in deps.items() for d in ds)


def rand_dag(rng, n):
    return {i: [j for j in range(i) if rng.random() < 0.4] for i in range(n)}


def all_paths(deps, dur):
    kids = {t: [u for u, ds in deps.items() if t in ds] for t in deps}
    best = (0, [])

    def walk(t, path, tot):
        nonlocal best
        path, tot = path + [t], tot + dur[t]
        if tot > best[0]:
            best = (tot, path)
        for u in kids[t]:
            walk(u, path, tot)
    for t in deps:
        walk(t, [], 0)
    return best


def descendants(deps, x):
    out, stack = set(), [x]
    while stack:
        y = stack.pop()
        for t, ds in deps.items():
            if y in ds and t not in out:
                out.add(t)
                stack.append(t)
    return out


# fixed example
deps = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"], "e": []}
assert forward(topo_order(deps), deps)
assert levels(deps) == [["a", "e"], ["b", "c"], ["d"]]
dur = {"a": 2, "b": 5, "c": 1, "d": 3, "e": 4}
assert critical_path(dur, deps) == (10, ["a", "b", "d"])
assert replan(deps, "b") == {"a": [], "c": ["a"], "e": []}

rng = random.Random(0)
for _ in range(60):
    n = rng.randint(1, 8)
    deps = rand_dag(rng, n)
    perm = list(deps)
    rng.shuffle(perm)
    deps = {t: deps[t] for t in perm}  # dict order must not matter
    order = topo_order(deps)
    assert forward(order, deps)
    ref = list(TopologicalSorter(deps).static_order())
    assert forward(ref, deps) and sorted(order) == sorted(ref)
    lv = levels(deps)
    level_of = {t: i for i, layer in enumerate(lv) for t in layer}
    assert sorted(level_of) == sorted(deps)
    for t, ds in deps.items():
        assert level_of[t] == 1 + max((level_of[d] for d in ds), default=-1)
    dur = {t: rng.randint(1, 9) for t in deps}
    length, path = critical_path(dur, deps)
    bf_len, _ = all_paths(deps, dur)
    assert length == bf_len, (length, bf_len)
    assert sum(dur[t] for t in path) == length
    for a_, b_ in zip(path, path[1:]):
        assert a_ in deps[b_]
    for x in deps:
        new = replan(deps, x)
        assert set(deps) - set(new) == {x} | descendants(deps, x)
        assert all(d in new for ds in new.values() for d in ds)

# cycles raise
for bad in ({"a": ["b"], "b": ["a"]}, {"a": ["a"]}, {"a": [], "b": ["c"], "c": ["d"], "d": ["b"]}):
    try:
        topo_order(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle not detected")
print("planning-decomposition OK")

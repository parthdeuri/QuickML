import itertools, sys, functools, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "python-generators-functional.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
batched, compose, memoize = ns["batched"], ns["compose"], ns["memoize"]

# laziness: items are consumed only on demand
pulled = []
def source():
    for i in range(100):
        pulled.append(i)
        yield i
g = batched(source(), 4)
assert pulled == []
assert next(g) == [0, 1, 2, 3] and pulled == [0, 1, 2, 3]
assert next(g) == [4, 5, 6, 7] and len(pulled) == 8

# batched matches the reference (itertools.batched where available)
def ref_batched(xs, n):
    xs = list(xs)
    return [tuple(xs[i:i + n]) for i in range(0, len(xs), n)]
for n_items in (0, 1, 7, 8, 9):
    for n in (1, 3, 8):
        got = [tuple(b) for b in batched(range(n_items), n)]
        assert got == ref_batched(range(n_items), n)
        if hasattr(itertools, "batched"):
            assert got == list(itertools.batched(range(n_items), n))
assert [len(b) for b in batched(range(10), 4)] == [4, 4, 2]

# compose applies left to right
f, g_, h = (lambda x: x + 1), (lambda x: x * 2), (lambda x: x - 3)
assert compose(f, g_, h)(5) == h(g_(f(5))) == 9
assert compose(h, g_, f)(5) == f(g_(h(5))) == 5
assert compose()(7) == 7

# memoize call counts match functools.lru_cache
calls = {"m": 0, "l": 0}
@memoize
def sq(x):
    calls["m"] += 1
    return x * x
@functools.lru_cache(maxsize=None)
def sq2(x):
    calls["l"] += 1
    return x * x
args = [1, 2, 1, 3, 2, 1, 4]
assert [sq(a) for a in args] == [sq2(a) for a in args]
assert calls["m"] == calls["l"] == len(set(args)) == 4
assert sq2.cache_info().hits == len(args) - 4

# one-shot exhaustion
gen = (i for i in range(5))
assert sum(gen) == 10 and sum(gen) == 0 and list(gen) == []

# memory: generator is tiny compared with the list
N = 100_000
assert sys.getsizeof(i for i in range(N)) * 100 < sys.getsizeof(list(range(N)))

# lazy pipeline: read, clean, batch
lines = ["  Hello ", "", "WORLD", "  ", "Foo", "bar ", "BAZ"]
out = list(ns["pipeline"](lines, 2))
assert out == [["hello", "world"], ["foo", "bar"], ["baz"]]
assert list(ns["flatten"](ns["pipeline"](lines, 2))) == ["hello", "world", "foo", "bar", "baz"]
seen = []
def tracked():
    for l in lines:
        seen.append(l)
        yield l
p = ns["pipeline"](tracked(), 2)
next(p)
assert seen == lines[:3]  # only what the first batch needed
print("python-generators-functional OK")

import numpy as np, yaml
from math import comb
from scipy.stats import binom
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "agentic-workflow-patterns.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
chain, route, vote = ns["chain"], ns["route"], ns["parallel_vote"]
orch, evalopt = ns["orchestrator_workers"], ns["evaluator_optimizer"]

# chain: gate stops early and returns the reason; later steps never run
calls = []
def mk(f):
    def s(x):
        calls.append(x)
        return f(x)
    return s
steps = [mk(lambda x: x + 1), mk(lambda x: x * 10), mk(lambda x: x - 3)]
assert chain(steps, lambda i, x: (True, ""), 1) == (17, "done")
calls.clear()
out, why = chain(steps, lambda i, x: (x < 15, "too big"), 1)
assert (out, why) == (20, "stop at step 1: too big") and len(calls) == 2

# routing: right handler, unknown class falls back to default
h = {"math": lambda x: "M" + x, "code": lambda x: "C" + x}
clf = lambda x: x.split(":")[0]
assert route(clf, h, lambda x: "D" + x, "math:1") == "Mmath:1"
assert route(clf, h, lambda x: "D" + x, "code:2") == "Ccode:2"
assert route(clf, h, lambda x: "D" + x, "poem:3") == "Dpoem:3"

# voting: majority, deterministic tie-break (smallest value)
W = lambda a: (lambda x: a)
assert vote([W("a"), W("b"), W("a")], 0) == "a"
assert vote([W(2), W(1), W(2), W(1)], 0) == 1
assert vote([W(3)], 0) == 3

# orchestrator-workers: split, each part once, synthesise in order
seen = []
def worker(p):
    seen.append(p)
    return p.upper()
res = orch(lambda x: x.split(), worker, lambda r: "+".join(r), "a b c")
assert res == "A+B+C" and seen == ["a", "b", "c"]

# evaluator-optimizer: stops on accept, and at max_rounds
gen = lambda fb: 0 if fb is None else fb + 1
assert evalopt(gen, lambda d: (d >= 2, d), 10) == (2, 3)
assert evalopt(gen, lambda d: (False, d), 4) == (3, 4)
assert evalopt(gen, lambda d: (True, d), 4) == (0, 1)

# maths: majority accuracy is the binomial tail
rng = np.random.default_rng(0)
for n in (1, 3, 5, 9):
    for p in (0.3, 0.6, 0.8):
        tail = sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range((n + 1) // 2, n + 1))
        assert np.isclose(tail, binom.sf(n // 2, n, p))
        sim = (rng.random((100000, n)) < p).sum(1) > n / 2
        assert abs(sim.mean() - tail) < 0.01, (n, p)
# chain success s^k
s, k = 0.95, 10
sim = (rng.random((100000, k)) < s).all(1).mean()
assert abs(sim - s**k) < 0.01 and round(s**k, 2) == 0.60
print("agentic-workflow-patterns OK")

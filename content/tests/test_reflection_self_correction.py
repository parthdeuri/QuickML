import random, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "reflection-self-correction.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
refine_loop, expected_success = ns["refine_loop"], ns["expected_success"]


def scripted(answers, good):
    """generate -> answers[0]; each revise -> next scripted answer."""
    calls = {"gen": 0, "crit": 0, "rev": 0, "ver": 0}
    it = iter(answers[1:])

    def generate():
        calls["gen"] += 1
        return answers[0]

    def critique(a):
        calls["crit"] += 1
        return "fb:" + str(a)

    def revise(a, fb):
        calls["rev"] += 1
        return next(it)

    def verify(a):
        calls["ver"] += 1
        return a in good
    return (generate, critique, revise, verify), calls


# first answer verifies: no critique or revise at all
fns, c = scripted(["ok"], {"ok"})
assert refine_loop(*fns, 5) == ("ok", 0, "verified")
assert c["rev"] == 0 and c["crit"] == 0 and c["gen"] == 1

# fixed on round 2; revise is never called after the pass
fns, c = scripted(["a", "b", "ok", "zzz", "zzz"], {"ok"})
assert refine_loop(*fns, 5) == ("ok", 2, "verified")
assert c["rev"] == 2

# max_iters
fns, c = scripted(["a", "b", "c", "d", "e"], set())
assert refine_loop(*fns, 3) == ("d", 3, "max_iters") and c["rev"] == 3
fns, c = scripted(["a"], set())
assert refine_loop(*fns, 0) == ("a", 0, "max_iters") and c["rev"] == 0

# no improvement: same answer twice
fns, c = scripted(["a", "b", "b", "ok"], {"ok"})
assert refine_loop(*fns, 9) == ("b", 2, "no_improvement") and c["rev"] == 2

# expected_success vs seeded Monte Carlo
def simulate(p, q, n, trials, rng):
    wins = 0
    for _ in range(trials):
        ok = rng.random() < p
        for _ in range(n):
            if not ok and rng.random() < q:
                ok = True
        wins += ok
    return wins / trials

rng = random.Random(0)
for p, q, n in [(0.3, 0.4, 3), (0.5, 0.2, 5), (0.1, 0.5, 2)]:
    mc = simulate(p, q, n, 40000, rng)
    assert abs(mc - expected_success(p, q, n)) < 0.01, (p, q, n, mc)
assert expected_success(0.37, 0.0, 9) == 0.37
assert expected_success(0.37, 0.8, 0) == 0.37
assert abs(expected_success(0.3, 0.4, 2) - (1 - 0.7 * 0.36)) < 1e-12
vals = [expected_success(0.2, 0.3, n) for n in range(15)]
assert all(b >= a for a, b in zip(vals, vals[1:])) and vals[-1] < 1
print("reflection-self-correction OK")

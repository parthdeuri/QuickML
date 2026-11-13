import numpy as np, yaml
from pathlib import Path
from scipy.stats import binomtest

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "prompt-versioning.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ph, Registry = ns["prompt_hash"], ns["Registry"]

# hash: stable, ignores trailing whitespace only, changes with any edit
t = "You are a helpful bot.\nAnswer briefly: {q}"
assert ph(t) == ph(t) and len(ph(t)) == 12
assert ph(t) == ph("  " + t + "\n\n") and ph(t) == ph(t.replace("\n", " \r\n"))
for i in range(len(t)):
    assert ph(t[:i] + "#" + t[i + 1:]) != ph(t)
assert ph(t + " ") == ph(t) and ph(t + "!") != ph(t)

# registry: exact text, immutability, semantic latest (1.10 > 1.9)
reg = Registry()
texts = {"1.0.0": "a {x}", "1.9.0": "b {x}", "1.10.0": "c  {x}\n second"}
for v, s in texts.items():
    assert reg.register("qa", v, s, note="n") == ph(s)
for v, s in texts.items():
    assert reg.get("qa", v) == s
assert reg.latest("qa") == ("1.10.0", texts["1.10.0"])
reg.register("other", "9.0.0", "z")
assert reg.latest("qa")[0] == "1.10.0"
try:
    reg.register("qa", "1.0.0", "changed")
    raise SystemExit("overwrite allowed")
except ValueError:
    pass
assert reg.get("qa", "1.0.0") == "a {x}"

# gate
gate = ns["regression_gate"]
same = [True, False] * 20
ok, p, b, c = gate(same, same)
assert ok and p == 1.0 and (b, c) == (0, 0)
old = [True] * 40 + [False] * 10
new = [False] * 14 + [True] * 26 + [False] * 10   # loses 14 examples, wins none
ok, p, b, c = gate(old, new)
assert not ok and (b, c) == (0, 14)
assert p == binomtest(14, 14, 0.5, alternative="greater").pvalue
# mixed case compared against scipy directly
rng = np.random.default_rng(1)
o = rng.random(200) < 0.7
n = np.where(rng.random(200) < 0.2, ~o, o)
ok, p, b, c = gate(o.tolist(), n.tolist())
bb, cc = int((n & ~o).sum()), int((o & ~n).sum())
assert (b, c) == (bb, cc)
assert np.isclose(p, binomtest(cc, bb + cc, 0.5, alternative="greater").pvalue)
assert ok == (p >= 0.05)
# a better prompt passes
assert gate(new, old)[0]

# canary
route = ns["route_canary"]
users = [f"user{i}" for i in range(20000)]
r1 = [route(u, 0.1) for u in users]
assert r1 == [route(u, 0.1) for u in users]
assert abs(np.mean([x == "new" for x in r1]) - 0.1) < 0.01
assert all(route(u, 0.0) == "old" and route(u, 1.0) == "new" for u in users[:200])
# monotone: raising the fraction only adds users to the canary
r2 = [route(u, 0.3) for u in users]
assert all(b2 == "new" for a, b2 in zip(r1, r2) if a == "new")
print("prompt-versioning OK")

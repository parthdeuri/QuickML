import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "output-validation-guardrails.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# hand cases per checker
assert ns["length"](5)("12345")[0] and not ns["length"](5)("123456")[0]
b = ns["banned"]([r"password", r"rm\s+-rf"])
assert b("Hello there")[0] and not b("my PASSWORD is")[0] and not b("run rm  -rf /")[0]
assert ns["no_pii"]("call me maybe")[0]
assert not ns["no_pii"]("mail a.b@x.com")[0] and not ns["no_pii"]("555-123-4567")[0]
assert ns["valid_json"]('{"a": 1}')[0] and not ns["valid_json"]("{a:")[0]
g = ns["grounded"](["The cache stores keys and values for tokens"], 0.8)
assert g("cache stores keys and values")[0]
assert not g("quantum banana rockets")[0]
assert not g("")[0]   # no content words: scored 0, so rejected

# pipeline reports every failing reason, not just the first
guards = [ns["length"](10), ns["banned"]([r"secret"]), ns["no_pii"]]
ok, why = ns["run_guards"]("secret mail a@b.co", guards)
assert not ok and len(why) == 3
ok, why = ns["run_guards"]("fine", guards)
assert ok and why == []

# fail-closed blocks on a checker exception, fail-open passes
def boom(o):
    raise RuntimeError("down")
assert ns["run_guards"]("fine", [boom], "closed")[0] is False
assert ns["run_guards"]("fine", [boom], "open") == (True, [])
ok, why = ns["run_guards"]("fine", [boom, ns["length"](2)], "open")
assert not ok and why == ["over 2 chars"]

# retry with feedback, then fallback
seen = []
def gen(feedback):
    seen.append(list(feedback))
    return "ok" if feedback else "x" * 50
assert ns["guarded"](gen, [ns["length"](10)], "closed", 2, "FB") == "ok"
assert seen == [[], ["over 10 chars"]]
assert ns["guarded"](lambda f: "x" * 50, [ns["length"](10)], "closed", 2, "FB") == "FB"

# series rates: closed form and seeded Monte-Carlo (independent checks)
fpr, miss = [0.02, 0.05, 0.10], [0.30, 0.50, 0.20]
F, M = ns["series_rates"](fpr, miss)
assert np.isclose(F, 1 - 0.98 * 0.95 * 0.90) and np.isclose(M, 0.03)
rng = np.random.default_rng(0)
N = 400000
rej = np.zeros(N, bool)
for f in fpr:
    rej |= rng.random(N) < f
slip = np.ones(N, bool)
for m in miss:
    slip &= rng.random(N) < m
assert abs(rej.mean() - F) < 4 * np.sqrt(F * (1 - F) / N)
assert abs(slip.mean() - M) < 4 * np.sqrt(M * (1 - M) / N)
print("output-validation-guardrails OK")

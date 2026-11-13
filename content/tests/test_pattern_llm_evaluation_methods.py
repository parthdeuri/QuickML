import yaml, math
d = yaml.safe_load(open("content/patterns/llm-evaluation-methods.yaml", encoding="utf-8"))
assert d["id"] == "llm-evaluation-methods"
ns = {}
exec(d["template"], ns)
assert abs(ns["perplexity"]([math.log(0.25)] * 5) - 4.0) < 1e-9
assert ns["perplexity"]([-0.1, -0.2]) < ns["perplexity"]([-2.0, -3.0])
assert ns["ngram_precision"]("a b c d".split(), "a b c d".split(), 2) == 1.0
assert abs(ns["ngram_precision"]("a a a".split(), "a b".split(), 1) - 1 / 3) < 1e-12
assert abs(ns["sign_test_p"](9, 1) - 22 / 1024) < 1e-12
assert ns["sign_test_p"](5, 5) == 1.0
assert ns["sign_test_p"](20, 0) < 1e-5
a, b = ns["elo_update"](1000, 1000, 1.0)
assert (a, b) == (1016.0, 984.0) and abs(a + b - 2000) < 1e-9
a2, _ = ns["elo_update"](1400, 1000, 1.0)
assert 0 < a2 - 1400 < 16
print("pattern llm-evaluation-methods OK")

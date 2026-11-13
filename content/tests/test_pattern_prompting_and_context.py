import yaml
d = yaml.safe_load(open("content/patterns/prompting-and-context.yaml", encoding="utf-8"))
assert d["id"] == "prompting-and-context"
ns = {}
exec(d["template"], ns)
assert ns["render"]("Q: {q}\nA:", q="2+2") == "Q: 2+2\nA:"
try:
    ns["render"]("{a} {b}", a=1)
    raise SystemExit("missing field should fail")
except KeyError:
    pass
chunks = [("a", 50, 5.0), ("b", 30, 6.0), ("c", 40, 1.0), ("d", 20, 4.0)]
kept, used = ns["pack"](chunks, 100)
assert used <= 100 and kept == ["b", "d", "a"] and used == 100
assert ns["pack"](chunks, 0) == ([], 0)
assert ns["parse"]('{"label": "x", "n": 1}', ["label"]) == {"label": "x", "n": 1}
assert ns["parse"]('{"n": 1}', ["label"]) is None
assert ns["parse"]("not json", ["label"]) is None
assert ns["parse"]("[1]", ["label"]) is None
assert ns["paired"]([1, 1, 0, 1], [1, 0, 0, 0]) == (2, 0)
assert ns["paired"]([1, 0], [0, 1]) == (1, 1)
print("pattern prompting-and-context OK")

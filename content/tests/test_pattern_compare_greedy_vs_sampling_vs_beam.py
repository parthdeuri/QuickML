import yaml
d = yaml.safe_load(open("content/patterns/compare-greedy-vs-sampling-vs-beam.yaml", encoding="utf-8"))
ns = {}
exec(d["template"], ns)
gs, gp = ns["greedy"]()
bs, bp = ns["beam"](2)
assert gs == "Ax" and abs(gp - 0.2) < 1e-9
assert bs == "Bx" and abs(bp - 0.36) < 1e-9 and bp > gp
assert ns["beam"](1)[0] == gs
assert len({ns["sample"](1.0, 1.0, s) for s in range(200)}) > 3
assert {ns["sample"](0.01, 1.0, s) for s in range(50)} == {gs}
assert {ns["sample"](1.0, 0.4, s)[0] for s in range(50)} == {"A"}
print("pattern compare-greedy-vs-sampling-vs-beam OK")

import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "state-graph-workflows.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Graph, END = ns["Graph"], ns["END"]

# linear graph runs nodes in order; list reducer appends, default overwrites
g = Graph(reducers={"log": lambda a, b: a + b})
g.add_node("a", lambda s: {"log": ["a"], "x": 1})
g.add_node("b", lambda s: {"log": ["b"], "x": s["x"] + 10})
g.add_node("c", lambda s: {"log": ["c"], "x": s["x"] * 2})
g.set_entry("a")
g.add_edge("a", "b")
g.add_edge("b", "c")
final, pending = g.run({"log": [], "x": 0})
assert pending is None
assert final["log"] == ["a", "b", "c"] and final["x"] == 22
init = {"log": [], "x": 0}
g.run(init)
assert init == {"log": [], "x": 0}  # input state is not mutated

# conditional loop exits when the condition is met
def build(limit):
    h = Graph(reducers={"log": lambda a, b: a + b, "n": lambda a, b: a + b})
    h.add_node("work", lambda s: {"n": 1, "log": [f"w{s['n']}"]})
    h.add_node("done", lambda s: {"log": ["done"]})
    h.set_entry("work")
    h.add_conditional_edges("work", lambda s: "done" if s["n"] >= limit else "work")
    return h

h = build(3)
final, _ = h.run({"n": 0, "log": []})
assert final["n"] == 3 and final["log"] == ["w0", "w1", "w2", "done"]

# resuming from checkpoint k equals the uninterrupted run
cps = []
full, _ = h.run({"n": 0, "log": []}, checkpoints=cps)
assert len(cps) == 4 and cps[-1][0] == END and cps[-1][1] == full
for node, snap in cps:
    resumed, p = h.run(snap, start=node)
    assert p is None and resumed == full, (node, resumed)

# a cycle with no exit raises a clear step-limit error
loop = Graph()
loop.add_node("p", lambda s: {})
loop.add_node("q", lambda s: {})
loop.set_entry("p")
loop.add_edge("p", "q")
loop.add_edge("q", "p")
try:
    loop.run({}, max_steps=7)
    raise AssertionError("expected step limit")
except RuntimeError as e:
    assert "step limit 7" in str(e), e

# interrupt before a node stops, and continues with edited state
g2 = Graph(reducers={"log": lambda a, b: a + b})
g2.add_node("draft", lambda s: {"log": ["draft"], "text": "v1"})
g2.add_node("send", lambda s: {"log": ["send:" + s["text"]]})
g2.set_entry("draft")
g2.add_edge("draft", "send")
st, pend = g2.run({"log": []}, interrupt_before=("send",))
assert pend == "send" and st["log"] == ["draft"] and st["text"] == "v1"
st["text"] = "v1-edited"
end, pend = g2.run(st, start="send")
assert pend is None and end["log"] == ["draft", "send:v1-edited"]
# interrupting before the entry node returns the state untouched
st0, pend0 = g2.run({"log": []}, interrupt_before=("draft",))
assert pend0 == "draft" and st0 == {"log": []}
print("state-graph-workflows OK")

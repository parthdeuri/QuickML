import yaml
from pathlib import Path

d = yaml.safe_load((Path(__file__).parent.parent / "patterns" / "agent-building-blocks.yaml").read_text(encoding="utf-8"))
assert d["id"] == "agent-building-blocks"
ns = {}
exec(d["template"], ns)

assert ns["call_tool"]("add", (2, 3)) == 5
assert ns["call_tool"]("nope", ()).startswith("error: unknown")
assert ns["call_tool"]("neg", (1, 2)).startswith("error: bad")

m = ns["script"]([("call", "add", (2, 3)), ("call", "neg", (5,)), ("final", "done -5")])
ans, log = ns["run"](m)
assert ans == "done -5" and log == [("add", 5), ("neg", -5)]

loop = lambda log: ("call", "add", (1, 1))
ans, log = ns["run"](loop, max_steps=4)
assert ans is None and len(log) == 4

s, path = "plan", ["plan"]
while s != "done":
    s = ns["step"](s)
    path.append(s)
assert path == ["plan", "act", "check", "done"]
print("pattern agent-building-blocks OK")

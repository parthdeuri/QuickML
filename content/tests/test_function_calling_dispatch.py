import json, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "function-calling-dispatch.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ToolRegistry, run_tool_loop = ns["ToolRegistry"], ns["run_tool_loop"]

ran = []
def get_weather(city, unit="c"):
    ran.append(("weather", city, unit))
    return {"c": 21, "f": 70}[unit]
def add(a, b):
    ran.append(("add", a, b))
    return a + b
def crash():
    raise RuntimeError("down")

reg = ToolRegistry()
reg.register("get_weather", {"type": "object", "properties": {
    "city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
    "required": ["city"]}, get_weather, "Weather lookup")
reg.register("add", {"type": "object", "properties": {
    "a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"]}, add)
reg.register("crash", {"type": "object"}, crash)

def call(i, name, args):
    return {"id": i, "name": name,
            "arguments": args if isinstance(args, str) else json.dumps(args)}

def body(m):
    return json.loads(m["content"])

# declarations expose name, description and schema
decl = {d["name"]: d for d in reg.declarations()}
assert decl["get_weather"]["description"] == "Weather lookup"
assert decl["add"]["parameters"]["required"] == ["a", "b"]

# valid call executes
m = reg.dispatch(call("c1", "add", {"a": 2, "b": 3}))
assert m == {"role": "tool", "tool_call_id": "c1", "name": "add", "ok": True, "content": '{"result": 5}'}
# invalid arguments: wrong type, missing required, non-object -> structured error, function NOT run
ran.clear()
for args in ({"a": "2", "b": 3}, {"a": 1}, {"a": 1, "b": 2.5}, [1, 2], "null"):
    m = reg.dispatch(call("c2", "add", args if args != "null" else "null"))
    assert m["ok"] is False and body(m)["error"] == "invalid_arguments", (args, m)
    assert body(m)["details"]
assert ran == []
# unknown tool lists the real names
m = reg.dispatch(call("c3", "nope", {}))
assert body(m) == {"error": "unknown_tool", "available": ["add", "crash", "get_weather"]}
assert m["tool_call_id"] == "c3"
# invalid JSON
for raw in ("{a: 1", "", "{'a': 1}"):
    m = reg.dispatch(call("c4", "add", raw))
    assert body(m)["error"] == "invalid_json" and m["ok"] is False
assert ran == []
# the function raising is captured, never propagated
m = reg.dispatch(call("c5", "crash", {}))
assert body(m)["error"] == "tool_failed" and "down" in body(m)["detail"]
# malformed call dict does not raise either
assert reg.dispatch({})["ok"] is False

# scripted stub model: call, fed-back result, then a plain answer
def scripted(turns):
    it = iter(turns)
    seen = []
    def model(messages, tools):
        seen.append([m["role"] for m in messages])
        return next(it)
    model.seen = seen
    return model

ran.clear()
model = scripted([
    {"content": None, "tool_calls": [call("a", "get_weather", {"city": "Pune"})]},
    {"content": "It is 21 in Pune.", "tool_calls": []}])
msgs = [{"role": "user", "content": "weather in Pune?"}]
out = run_tool_loop(model, msgs, reg, max_steps=4)
assert out == {"stop": "answer", "steps": 2, "answer": "It is 21 in Pune."}
assert ran == [("weather", "Pune", "c")]
assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
assert model.seen[1] == ["user", "assistant", "tool"]   # result was fed back
assert body(msgs[2]) == {"result": 21}

# invalid call -> error fed back, model corrects, function run once
ran.clear()
model = scripted([
    {"content": None, "tool_calls": [call("a", "add", {"a": "1", "b": 2})]},
    {"content": None, "tool_calls": [call("b", "add", {"a": 1, "b": 2})]},
    {"content": "3", "tool_calls": []}])
msgs = [{"role": "user", "content": "1+2"}]
out = run_tool_loop(model, msgs, reg)
assert out["stop"] == "answer" and out["steps"] == 3 and ran == [("add", 1, 2)]
assert body(msgs[2])["error"] == "invalid_arguments" and body(msgs[4]) == {"result": 3}

# parallel calls: results appended in call order with matching ids
ran.clear()
model = scripted([
    {"content": None, "tool_calls": [call("p1", "add", {"a": 1, "b": 1}),
                                      call("p2", "nope", {}),
                                      call("p3", "add", {"a": 5, "b": 5})]},
    {"content": "done", "tool_calls": []}])
msgs = [{"role": "user", "content": "go"}]
run_tool_loop(model, msgs, reg)
tools = [m for m in msgs if m["role"] == "tool"]
assert [m["tool_call_id"] for m in tools] == ["p1", "p2", "p3"]
assert [m["ok"] for m in tools] == [True, False, True]
assert ran == [("add", 1, 1), ("add", 5, 5)]

# step limit: clear stop reason, exactly max_steps model calls
n = []
def endless(messages, tools):
    n.append(1)
    return {"content": None, "tool_calls": [call("x%d" % len(n), "add", {"a": 1, "b": 1})]}
msgs = [{"role": "user", "content": "loop"}]
out = run_tool_loop(endless, msgs, reg, max_steps=3)
assert out == {"stop": "max_steps", "steps": 3, "answer": None} and len(n) == 3
assert len([m for m in msgs if m["role"] == "tool"]) == 3
# immediate answer: one step
out = run_tool_loop(scripted([{"content": "hi"}]), [{"role": "user", "content": "x"}], reg)
assert out == {"stop": "answer", "steps": 1, "answer": "hi"}
print("function-calling-dispatch OK")

import json, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "model-context-protocol-jsonrpc.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Server, Client, RpcError = ns["Server"], ns["Client"], ns["RpcError"]

calls = []
def add(a, b):
    calls.append((a, b))
    return a + b
def boom(x):
    raise ValueError("bad " + x)

schema = {"type": "object",
          "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
          "required": ["a", "b"], "additionalProperties": False}
srv = Server()
srv.add_tool("add", "Add two integers", schema, add)
srv.add_tool("boom", "Always fails",
             {"type": "object", "properties": {"x": {"type": "string"}},
              "required": ["x"]}, boom)
srv.add_tool("broken", "Bad schema", {"type": "nonsense"}, lambda: 1)
cl = Client(srv)

def send(msg):
    out = srv.handle(msg if isinstance(msg, str) else json.dumps(msg))
    return None if out is None else json.loads(out)

# list round trip: name, description, inputSchema
r = cl.request("tools/list")
t = {x["name"]: x for x in r["tools"]}
assert set(t) == {"add", "boom", "broken"}
assert t["add"]["description"] == "Add two integers" and t["add"]["inputSchema"] == schema
# call round trip
r = cl.request("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
assert r == {"content": [{"type": "text", "text": "5"}], "isError": False}
assert calls == [(2, 3)]
# a failing tool is a result with isError, not a protocol error
r = cl.request("tools/call", {"name": "boom", "arguments": {"x": "y"}})
assert r["isError"] is True and "bad y" in r["content"][0]["text"]
# unknown method -> -32601
resp = send({"jsonrpc": "2.0", "id": 7, "method": "nope"})
assert resp["error"]["code"] == -32601 and resp["id"] == 7 and "result" not in resp
# bad arguments -> -32602 and the tool is NOT run
n = len(calls)
for bad in ({"a": "2", "b": 3}, {"a": 1}, {"a": 1, "b": 2, "c": 3}):
    resp = send({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                 "params": {"name": "add", "arguments": bad}})
    assert resp["error"]["code"] == -32602 and resp["id"] == 8, resp
assert len(calls) == n
resp = send({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "ghost"}})
assert resp["error"]["code"] == -32602
# malformed JSON -> -32700 with null id
resp = send("{not json")
assert resp["error"]["code"] == -32700 and resp["id"] is None
# invalid request -> -32600
assert send({"id": 1, "method": "tools/list"})["error"]["code"] == -32600
assert send([1, 2])["error"]["code"] == -32600 and send([1, 2])["id"] is None
assert send({"jsonrpc": "2.0", "id": 4, "method": 5})["id"] == 4
# internal error -> -32603 (schema itself is invalid)
resp = send({"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "broken", "arguments": {}}})
assert resp["error"]["code"] == -32603
# notifications: nothing comes back, even for unknown methods
assert srv.handle(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})) is None
assert srv.handle(json.dumps({"jsonrpc": "2.0", "method": "nope", "params": {}})) is None
cl.notify("notifications/initialized")
# the id is echoed exactly, string or integer
for i in (1, 0, 12345, "abc", "7"):
    resp = send({"jsonrpc": "2.0", "id": i, "method": "tools/list"})
    assert resp["id"] == i and type(resp["id"]) is type(i)
    assert resp["jsonrpc"] == "2.0" and ("result" in resp) != ("error" in resp)
assert cl.request("tools/list", id_="req-1")["tools"]
# client raises on an error response
try:
    cl.request("nope")
    raise SystemExit("expected RpcError")
except RpcError as e:
    assert e.code == -32601
print("model-context-protocol-jsonrpc OK")

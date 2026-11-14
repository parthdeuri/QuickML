"""Card diagrams: agents, tool calling, MCP, reliability, caching, streaming, routing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_lib import G, render  # noqa: E402


def agent_loop_react(g):
    cx = 58
    g.n("th", "Thought", cx, 78, 40)
    g.n("ac", "Action: tool[input]", cx, 62, 40, kind="param")
    g.n("ob", "Observation", cx, 46, 40, kind="io")
    g.n("gd", "steps < max?\nno repeated action?", cx, 28, 46, 12, kind="loss")
    g.n("stop", "stop: budget or loop", cx, 10, 40, kind="loss")
    g.n("fin", "Final\nAnswer", 15, 78, 22, 11, kind="hi")
    g.a(("th", "b"), ("ac", "t"))
    g.a(("ac", "b"), ("ob", "t"))
    g.a(("ob", "b"), ("gd", "t"))
    g.a(("gd", "b"), ("stop", "t"), "no", loff=(3, 0))
    g.p([("gd", "r"), (92, 28), (92, 78), ("th", "r")], dashed=True, kind="cool")
    g.text(85, 53, "yes:\nnext\nstep", "muted", fs=7)
    g.a(("th", "l"), ("fin", "r"), "done", loff=(0, 3))
    g.text(cx, 90, "one action at a time, read the result", "muted", fs=7.5)


def function_calling_dispatch(g):
    cx = 34
    g.n("call", "call: name + JSON args", cx, 92, 44, kind="io")
    g.n("s1", "1  look up name", cx, 78, 40)
    g.n("s2", "2  parse JSON", cx, 64, 40)
    g.n("s3", "3  check schema", cx, 50, 40)
    g.n("s4", "4  run function", cx, 36, 40, kind="param")
    g.n("res", "tool message (call id)", cx, 18, 46, kind="hi")
    g.n("err", "error as\ntool msg", 84, 64, 24, 40, kind="loss")
    g.a(("call", "b"), ("s1", "t"))
    for a, b in [("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "res")]:
        g.a((a, "b"), (b, "t"))
    for s in ("s1", "s2", "s3"):
        g.a((s, "r"), ("err", "l") if False else (72, g.nodes[s][1]), dashed=True, kind="loss")
    g.p([("err", "b"), (84, 18), ("res", "r")], dashed=True, kind="loss")
    g.text(cx, 5, "dispatch never raises: errors go back to the model", "muted", fs=7)


def state_graph_workflows(g):
    cx = 42
    g.n("st", "START", cx, 84, 18, 8, kind="io")
    g.n("plan", "plan", cx, 70, 30)
    g.n("act", "act", cx, 54, 30, kind="param")
    g.n("chk", "route: done?", cx, 38, 34, 10, kind="hi")
    g.n("end", "END", cx, 20, 18, 8, kind="io")
    g.a(("st", "b"), ("plan", "t"))
    g.a(("plan", "b"), ("act", "t"))
    g.a(("act", "b"), ("chk", "t"))
    g.a(("chk", "b"), ("end", "t"), "yes", loff=(4, 0))
    g.p([("chk", "r"), (80, 38), (80, 70), ("plan", "r")], dashed=True, kind="cool")
    g.text(91, 54, "not\ndone", "muted", fs=7)
    for nid in ("plan", "act", "chk"):
        y = g.nodes[nid][1]
        g.n("sn" + nid, "snapshot", 10, y, 16, 7, kind="cool", fs=7)
        g.a((nid, "l"), ("sn" + nid, "r"), dashed=True, kind="cool")
    g.text(50, 8, "state: msgs (append) · step (add)", "muted", fs=7.5)


def agentic_workflow_patterns(g):
    ys = [90, 70, 50, 30, 10]
    titles = ["1 chain\nwith gate", "2 routing", "3 parallel\nvote", "4 orchestrator\nworkers",
              "5 evaluator\noptimizer"]
    for y, t in zip(ys, titles):
        g.text(3, y, t, "muted", fs=7, ha="left")
    for y in (80, 60, 40, 20):
        g.ax.plot([3, 97], [y, y], color=g.t["muted"], lw=0.5, alpha=0.4, zorder=1)
    y = ys[0]
    g.n("c1", "LLM 1", 31, y, 14, 8, fs=7)
    g.n("cg", "gate", 52, y, 12, 8, kind="hi", fs=7)
    g.n("c2", "LLM 2", 73, y, 14, 8, kind="param", fs=7)
    g.n("co", "out", 91, y, 10, 8, kind="io", fs=7)
    g.a(("c1", "r"), ("cg", "l"))
    g.a(("cg", "r"), ("c2", "l"))
    g.a(("c2", "r"), ("co", "l"))
    y = ys[1]
    g.n("ri", "in", 30, y, 8, 8, kind="io", fs=7)
    g.n("rr", "route", 50, y, 14, 8, kind="hi", fs=7)
    g.a(("ri", "r"), ("rr", "l"))
    for k, (nm, dy) in enumerate([("math", 7), ("code", 0), ("chat", -7)]):
        g.n(f"rb{k}", nm, 82, y + dy, 14, 5.5, kind="cool", fs=6.5)
        g.a(("rr", "r"), (f"rb{k}", "l"))
    y = ys[2]
    g.n("pi", "in", 30, y, 8, 8, kind="io", fs=7)
    g.n("pv", "vote", 88, y, 14, 8, kind="hi", fs=7)
    for k, dy in enumerate([7, 0, -7]):
        g.n(f"pl{k}", "LLM", 59, y + dy, 14, 5.5, kind="param", fs=6.5)
        g.a(("pi", "r"), (f"pl{k}", "l"))
        g.a((f"pl{k}", "r"), ("pv", "l"))
    y = ys[3]
    g.n("oo", "orchestrator", 40, y, 26, 8, kind="hi", fs=7)
    g.n("oy", "out", 92, y, 10, 8, kind="io", fs=7)
    for k, dy in enumerate([7, 0, -7]):
        g.n(f"ow{k}", "worker", 68, y + dy, 14, 5.5, kind="param", fs=6.5)
        g.a(("oo", "r"), (f"ow{k}", "l"))
        g.a((f"ow{k}", "r"), ("oy", "l"))
    y = ys[4]
    g.n("eg", "generate", 38, y, 18, 8, kind="param", fs=7)
    g.n("ee", "evaluate", 64, y, 18, 8, kind="hi", fs=7)
    g.n("eo", "out", 90, y, 10, 8, kind="io", fs=7)
    g.a(("eg", "r"), ("ee", "l"))
    g.a(("ee", "r"), ("eo", "l"), "ok", loff=(0, 3))
    g.p([("ee", "t"), (64, y + 7.5), (38, y + 7.5), ("eg", "t")], dashed=True, kind="loss")


def multi_agent_orchestration(g):
    g.n("bb", "shared blackboard", 50, 43, 40, 12, kind="hi")
    g.n("pl", "1 planner", 50, 76, 24, 9, kind="param")
    g.n("co", "2 coder", 86, 43, 24, 9, kind="param")
    g.n("cr", "3 critic", 50, 10, 24, 9, kind="param")
    g.n("wr", "4 writer", 14, 43, 24, 9, kind="param")
    g.p([("pl", "r"), (86, 76), ("co", "t")], dashed=True, kind="loss")
    g.p([("co", "b"), (86, 10), ("cr", "r")], dashed=True, kind="loss")
    g.p([("cr", "l"), (14, 10), ("wr", "b")], dashed=True, kind="loss")
    g.p([("wr", "t"), (14, 76), ("pl", "l")], dashed=True, kind="loss")
    g.p([("pl", "b"), ("bb", "t")], arrow=False)
    g.p([("cr", "t"), ("bb", "b")], arrow=False)
    g.p([("co", "l"), ("bb", "r")], arrow=False)
    g.p([("wr", "r"), ("bb", "l")], arrow=False)
    g.text(50, 58, "all read and write", "muted", fs=7)
    g.text(50, 2.5, "dashed: turn order 1 to 4, then repeat", "muted", fs=7)


def model_context_protocol_jsonrpc(g):
    g.n("cl", "client", 18, 94, 28, 8, kind="io")
    g.n("sv", "server", 82, 94, 28, 8, kind="param")
    g.p([("cl", "b"), (18, 8)], arrow=False, dashed=True)
    g.p([("sv", "b"), (82, 8)], arrow=False, dashed=True)
    msgs = [(80, "r", "#1  tools/list", "line"), (68, "l", "#1  result: tools", "cool"),
            (54, "r", "#2  tools/call", "line"), (42, "l", "#2  result: content", "cool"),
            (28, "r", "#3  tools/call", "line"), (16, "l", "#3  error -32602", "loss")]
    for y, d, lab, k in msgs:
        if d == "r":
            g.a((19, y), (81, y), kind=k)
        else:
            g.a((81, y), (19, y), kind=k)
        g.text(50, y + 3.2, lab, "text" if k != "loss" else "muted", fs=7.5)
    g.text(50, 4, "the reply echoes the request id", "muted", fs=7.5)


def reliability_circuit_breaker_slo(g):
    g.n("cl", "CLOSED\ncalls pass", 22, 66, 30, 12, kind="io")
    g.n("op", "OPEN\nfail fast", 78, 66, 30, 12, kind="loss")
    g.n("ho", "HALF-OPEN\none probe", 50, 22, 36, 12, kind="hi")
    g.a(("cl", "r"), ("op", "l"))
    g.text(50, 70, "failures ≥ N", "muted", fs=7.5)
    g.a((85, 60), (67, 22), kind="line")
    g.text(88, 40, "after\ncool-down", "muted", fs=7)
    g.a((55, 28), (70, 60), kind="loss")
    g.text(50, 46, "probe\nfails", "muted", fs=7)
    g.a((33, 22), (22, 60))
    g.text(12, 40, "probe\nok", "muted", fs=7)
    g.text(50, 6, "open: no calls reach the failing service", "muted", fs=7.5)


def semantic_cache(g):
    cx = 38
    g.n("q", "query", cx, 94, 24, 8, kind="io")
    g.n("em", "embed (unit vector)", cx, 82, 40, 8)
    g.n("nn", "nearest cached\nentry (max dot)", cx, 68, 44, 11, kind="param")
    g.n("th", "score ≥ τ ?", cx, 52, 32, 10, kind="hi")
    g.a(("q", "b"), ("em", "t"))
    g.a(("em", "b"), ("nn", "t"))
    g.a(("nn", "b"), ("th", "t"))
    g.text(82, 68, "same tenant,\nnot expired", "muted", fs=7)
    g.n("hit", "return cached\nanswer", 16, 34, 26, 11, kind="io")
    g.p([("th", "l"), (16, 52), ("hit", "t")])
    g.text(19, 55, "hit", "muted", fs=7.5)
    g.n("llm", "call LLM", 76, 36, 30, 9, kind="op")
    g.n("st", "store in cache", 76, 18, 30, 9, kind="param")
    g.p([("th", "r"), (76, 52), ("llm", "t")])
    g.text(65, 55, "miss", "muted", fs=7.5)
    g.a(("llm", "b"), ("st", "t"))
    g.text(50, 4, "τ trades hit rate for wrong answers", "muted", fs=7.5)


def streaming_sse_parsing(g):
    for i, cx in enumerate([14, 40, 66]):
        g.n(f"ch{i}", f"chunk {i + 1}", cx, 84, 22, 8, kind="io", fs=8)
    g.n("bf", "UTF-8 decode\n+ line buffer", 40, 68, 40, 11, kind="param")
    g.n("cl", "collect data: lines", 40, 52, 40, 9)
    g.n("ev", "blank line =\nevent ends", 40, 38, 36, 11, kind="hi")
    g.n("js", "parse JSON delta", 40, 24, 36, 9)
    g.n("tx", "append to text", 40, 10, 36, 9, kind="io")
    for i in range(3):
        g.a((f"ch{i}", "b"), ("bf", "t"))
    for a, b in [("bf", "cl"), ("cl", "ev"), ("ev", "js"), ("js", "tx")]:
        g.a((a, "b"), (b, "t"))
    g.text(82, 68, "partial line\nstays buffered", "muted", fs=7)
    g.text(82, 52, ": comment\nlines ignored", "muted", fs=7)
    g.text(82, 10, "stop at\n[DONE]", "muted", fs=7)


def provider_fallback_router(g):
    g.n("rq", "request", 50, 94, 22, 7, kind="io")
    g.n("pv", "provider i (in order)", 50, 82, 44, 8, kind="param")
    g.n("cf", "error class?", 50, 66, 36, 11, kind="hi")
    g.a(("rq", "b"), ("pv", "t"))
    g.a(("pv", "b"), ("cf", "t"), "error", loff=(7, 0))
    g.n("o1", "timeout/429\nnext provider", 20, 44, 28, 12, kind="cool")
    g.n("o2", "auth error\nbench + next", 52, 44, 28, 12, kind="loss")
    g.n("o3", "bad request\nstop, raise", 84, 44, 28, 12, kind="loss")
    g.a((38, 61), ("o1", "t"))
    g.a(("cf", "b"), ("o2", "t"))
    g.a((62, 61), ("o3", "t"))
    g.n("nx", "next healthy provider\n(skip cooled-down)", 36, 22, 46, 12)
    g.n("rs", "raise to\ncaller", 84, 22, 28, 12, kind="loss")
    g.a((20, 38), (26, 27.5))
    g.a((52, 38), (46, 27.5))
    g.a(("o3", "b"), ("rs", "t"))
    g.p([("nx", "l"), (3, 22), (3, 82), ("pv", "l")], dashed=True, kind="cool")
    g.text(50, 6, "chain empty: one combined error", "muted", fs=7.5)


DIAGRAMS = {
    "agent-loop-react": (94, agent_loop_react),
    "function-calling-dispatch": (100, function_calling_dispatch),
    "state-graph-workflows": (92, state_graph_workflows),
    "agentic-workflow-patterns": (100, agentic_workflow_patterns),
    "multi-agent-orchestration": (86, multi_agent_orchestration),
    "model-context-protocol-jsonrpc": (100, model_context_protocol_jsonrpc),
    "reliability-circuit-breaker-slo": (80, reliability_circuit_breaker_slo),
    "semantic-cache": (100, semantic_cache),
    "streaming-sse-parsing": (92, streaming_sse_parsing),
    "provider-fallback-router": (100, provider_fallback_router),
}

if __name__ == "__main__":
    render(DIAGRAMS, sys.argv[1:] or None)

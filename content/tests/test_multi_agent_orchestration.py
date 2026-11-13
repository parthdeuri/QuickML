import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "multi-agent-orchestration.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rr, sel = ns["round_robin"], ns["selector"]
run, cost = ns["run_conversation"], ns["token_cost"]

# every agent sees exactly the entries written before its turn
seen = []
def mk(i):
    def agent(board):
        seen.append((i, len(board)))
        return f"a{i}:{len(board)}"
    return agent
agents = [mk(i) for i in range(3)]
board, why = run(agents, rr(3), 7, lambda b: False)
assert why == "max_turns" and len(board) == 7
assert [w for w, _ in board] == [0, 1, 2, 0, 1, 2, 0]  # round-robin order
assert seen == [(i % 3, i) for i in range(7)]
assert [m for _, m in board] == [f"a{i % 3}:{i}" for i in range(7)]

# selector picks the intended speaker from the board
def choose(b):
    if not b:
        return 0
    return 2 if b[-1][0] == 0 else 0 if b[-1][0] == 1 else 1
board, why = run(agents, sel(choose), 5, lambda b: False)
assert [w for w, _ in board] == [0, 2, 1, 0, 2]

# stop_fn ends early with a reason; max_turns respected
board, why = run(agents, rr(3), 10, lambda b: len(b) == 4)
assert why == "stop_fn" and len(board) == 4
board, why = run(agents, rr(3), 0, lambda b: True)
assert board == [] and why == "max_turns"
done = [lambda b: "DONE" if len(b) >= 2 else "work"] * 2
board, why = run(done, rr(2), 9, lambda b: b[-1][1] == "DONE")
assert why == "stop_fn" and len(board) == 3

# agents get a copy: mutating it does not corrupt the board
def evil(b):
    b.clear()
    return "x"
board, _ = run([evil, evil], rr(2), 3, lambda b: False)
assert len(board) == 3

# token cost equals the independent loop over turns
for n, r, c0, tau in [(1, 1, 100, 50), (3, 4, 200, 30), (4, 10, 0, 25)]:
    ctx, read, written = c0, 0, 0
    for _ in range(n * r):
        read += ctx           # context read before speaking
        ctx += tau            # message appended
        written += tau
    assert cost(n, r, c0, tau) == (read, written), (n, r)
assert cost(3, 4, 200, 30) == (12 * 200 + 30 * 66, 360)
print("multi-agent-orchestration OK")

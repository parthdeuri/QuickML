import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "pipeline-parallel-schedule.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

for p, m in [(1, 3), (2, 4), (4, 8), (4, 2), (3, 1), (5, 12)]:
    grid = ns["gpipe_forward_schedule"](p, m)
    steps = m + p - 1
    assert len(grid) == p and all(len(r) == steps for r in grid)
    # each microbatch exactly once per stage
    for s in range(p):
        assert sorted(j for j in grid[s] if j is not None) == list(range(m))
    # dependency: j at stage s only right after stage s-1
    for s in range(1, p):
        for j in range(m):
            assert grid[s].index(j) == grid[s - 1].index(j) + 1
    idle = sum(c is None for r in grid for c in r)
    assert idle == p * (p - 1)
    assert abs(idle / (p * steps) - (p - 1) / (m + p - 1)) < 1e-12
    assert abs(ns["bubble_fraction"](p, m) - idle / (p * steps)) < 1e-12

# hand check: p=4, m=4 -> 3/7
assert abs(ns["bubble_fraction"](4, 4) - 3 / 7) < 1e-12
assert ns["gpipe_forward_schedule"](2, 3) == [[0, 1, 2, None], [None, 0, 1, 2]]

# 1F1B per stage
for p, m in [(4, 8), (4, 2), (3, 3), (2, 6)]:
    for s in range(p):
        order = ns["one_f_one_b"](s, p, m)
        assert len(order) == 2 * m
        assert [j for k, j in order if k == "F"] == list(range(m))
        assert [j for k, j in order if k == "B"] == list(range(m))
        inflight, peak, done_f = 0, 0, set()
        for k, j in order:
            if k == "F":
                inflight += 1
                done_f.add(j)
            else:
                assert j in done_f  # backward after its own forward
                inflight -= 1
            peak = max(peak, inflight)
            assert inflight >= 0
        assert peak <= p - s, (p, m, s, peak)
        if m >= p:
            assert peak == p - s
assert ns["one_f_one_b"](3, 4, 3) == [("F", 0), ("B", 0), ("F", 1), ("B", 1), ("F", 2), ("B", 2)]
assert [k for k, _ in ns["one_f_one_b"](0, 3, 4)] == list("FFFBFBBB")
print("pipeline-parallel-schedule OK")

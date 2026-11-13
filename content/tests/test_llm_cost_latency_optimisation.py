import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "llm-cost-latency-optimisation.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
cost, lat = ns["request_cost"], ns["request_latency"]
pc, ce, ci = ns["prefix_cache_cost"], ns["cascade_expected_cost"], ns["cascade_is_cheaper"]

# hand-checked arithmetic (illustrative numbers)
assert cost(1000, 200, 2e-6, 8e-6) == 1000 * 2e-6 + 200 * 8e-6
assert abs(cost(1000, 200, 2e-6, 8e-6) - 0.0036) < 1e-12
assert abs(lat(0.5, 0.02, 100) - 2.5) < 1e-12
# latency additivity and output dominance
assert abs(lat(0.5, 0.02, 150) - (lat(0.5, 0.02, 100) + 50 * 0.02)) < 1e-12
assert lat(0.5, 0.02, 1000) > 10 * lat(0.5, 0.02, 0)
# streaming: total unchanged, perceived wait is the time to first token
ttft, itl, n = 0.5, 0.02, 100
stream_times = [ttft + (i + 1) * itl for i in range(n)]
assert abs(stream_times[-1] - lat(ttft, itl, n)) < 1e-12
assert stream_times[0] < lat(ttft, itl, n)

# prefix cache: hit rate 0 gives the plain cost; hit rate 1 bills d
plain = cost(800 + 200, 100, 2e-6, 8e-6)
assert abs(pc(800, 200, 100, 2e-6, 8e-6, 0.1, 0.0) - plain) < 1e-12
full = 800 * 2e-6 * 0.1 + 200 * 2e-6 + 100 * 8e-6
assert abs(pc(800, 200, 100, 2e-6, 8e-6, 0.1, 1.0) - full) < 1e-12
half = pc(800, 200, 100, 2e-6, 8e-6, 0.1, 0.5)
assert abs(half - (plain - 0.5 * 0.9 * 800 * 2e-6)) < 1e-12
assert full < half < plain

# cascade: expected cost and the cheaper-than-big-model criterion
assert abs(ce(1.0, 10.0, 0.25) - 3.5) < 1e-12
for c1 in np.linspace(0.1, 5, 12):
    for e in np.linspace(0, 1, 11):
        direct = ce(c1, 10.0, e) < 10.0
        assert ci(c1, 10.0, e) == direct, (c1, e)

# seeded Monte-Carlo simulation matches the expected cost
rng = np.random.default_rng(0)
c1, c2, e = 1.0, 10.0, 0.25
escalate = rng.random(200_000) < e
sim = (c1 + escalate * c2).mean()
assert abs(sim - ce(c1, c2, e)) < 0.03, sim
print("llm-cost-latency-optimisation OK")

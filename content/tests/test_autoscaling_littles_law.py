import heapq, math, random, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "autoscaling-littles-law.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
erlang_c, mean_wait, need = ns["erlang_c"], ns["mean_wait_mmc"], ns["servers_needed"]

# Little's law arithmetic: 50 QPS x 2 s latency = 100 concurrent
assert ns["littles_L"](50, 2.0) == 100.0

def direct_erlang_c(c, a):
    num = a ** c / math.factorial(c) * c / (c - a)
    den = sum(a ** k / math.factorial(k) for k in range(c)) + num
    return num / den

for c in (1, 2, 5, 10, 30):
    for rho in (0.1, 0.5, 0.9, 0.99):
        a = rho * c
        assert abs(erlang_c(c, a) - direct_erlang_c(c, a)) < 1e-9
assert abs(erlang_c(1, 0.7) - 0.7) < 1e-12        # M/M/1: P(wait) = rho
assert abs(erlang_c(2, 1.0) - 1 / 3) < 1e-12      # hand case

# seeded event-driven FCFS M/M/c simulation
def simulate(lam, mu, c, n, seed):
    rng = random.Random(seed)
    free = [0.0] * c
    heapq.heapify(free)
    t, waited, wait_sum, soj_sum, last = 0.0, 0, 0.0, 0.0, 0.0
    for _ in range(n):
        t += rng.expovariate(lam)
        start = max(t, heapq.heappop(free))
        done = start + rng.expovariate(mu)
        heapq.heappush(free, done)
        waited += start > t
        wait_sum += start - t
        soj_sum += done - t
        last = max(last, done)
    return waited / n, wait_sum / n, soj_sum / n, soj_sum / last, n / t

lam, mu, c = 8.0, 1.0, 10
pw, wq, W, L, lam_hat = simulate(lam, mu, c, 200000, 0)
assert abs(pw - erlang_c(c, lam / mu)) < 0.02, pw
assert abs(wq - mean_wait(lam, mu, c)) < 0.01, wq
# Little's law in the simulation: avg number in system ~ lam * avg time
assert abs(L - lam_hat * W) / L < 0.02, (L, lam_hat * W)
assert abs(L - lam * (mean_wait(lam, mu, c) + 1 / mu)) / L < 0.03

# servers_needed is minimal
for lam, mu, tgt in ((8.0, 1.0, 0.1), (90.0, 2.0, 0.01), (500.0, 5.0, 0.002)):
    c = need(lam, mu, tgt)
    assert mean_wait(lam, mu, c) <= tgt
    assert c - 1 <= lam / mu or mean_wait(lam, mu, c - 1) > tgt

# unstable systems raise a clear error
for bad in (lambda: mean_wait(10.0, 1.0, 10), lambda: mean_wait(20.0, 1.0, 10),
            lambda: erlang_c(3, 3.5)):
    try:
        bad()
        raise SystemExit("expected ValueError")
    except ValueError as e:
        assert "unstable" in str(e)

# more servers, less waiting
ws = [mean_wait(8.0, 1.0, c) for c in range(9, 20)]
assert all(x > y for x, y in zip(ws, ws[1:])), ws
print("autoscaling-littles-law OK")

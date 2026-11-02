import numpy as np, yaml
from pathlib import Path

root = Path(__file__).parent.parent / "problems"
card = yaml.safe_load((root / "value-iteration.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
qns = {}
exec(yaml.safe_load((root / "q-learning.yaml").read_text(encoding="utf-8"))["code"], qns)

# same 4x4 gridworld as test_q_learning.py
N = 4
S, A = N * N, 4
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
goal, pit = 3 * N + 3, 1 * N + 1
nxt = np.zeros((S, A), dtype=int)
rew = np.full((S, A), -1.0)
done = np.zeros(S, dtype=bool)
done[[goal, pit]] = True
for s in range(S):
    r, c = divmod(s, N)
    for a, (dr, dc) in enumerate(moves):
        r2, c2 = min(max(r + dr, 0), N - 1), min(max(c + dc, 0), N - 1)
        nxt[s, a] = r2 * N + c2
        if nxt[s, a] == pit:
            rew[s, a] = -10.0
        if nxt[s, a] == goal:
            rew[s, a] = 0.0
gamma = 0.9
V, Q = ns["value_iteration"](nxt, rew, done, gamma=gamma)
live = ~done

# brute-force Bellman optimality check, state by state
for s in range(S):
    if done[s]:
        assert V[s] == 0
        continue
    best = max(rew[s, a] + gamma * V[nxt[s, a]] for a in range(A))
    assert abs(V[s] - best) < 1e-8
    for a in range(A):
        assert abs(Q[s, a] - (rew[s, a] + gamma * V[nxt[s, a]])) < 1e-12

# greedy policy evaluated exactly (linear solve) reproduces V*
pol = Q.argmax(1)
P = np.zeros((S, S))
r_pi = np.zeros(S)
for s in range(S):
    if not done[s]:
        P[s, nxt[s, pol[s]]] = 1
        r_pi[s] = rew[s, pol[s]]
V_pi = np.linalg.solve(np.eye(S) - gamma * P, r_pi)
assert np.allclose(V_pi, V, atol=1e-8)

# contraction: sup-norm error shrinks by at most gamma per sweep
Vk = np.zeros(S)
prev = np.abs(Vk - V).max()
for _ in range(30):
    Vk = np.where(done, 0.0, (rew + gamma * Vk[nxt]).max(1))
    err = np.abs(Vk - V).max()
    assert err <= gamma * prev + 1e-12
    prev = err

# agreement with the Q-learning card on the same gridworld
Ql = qns["q_learning"](nxt, rew, done, gamma=gamma)
assert np.abs(Ql[live] - Q[live]).max() < 1e-2, np.abs(Ql[live] - Q[live]).max()
assert np.allclose(Q[live][np.arange(live.sum()), Ql[live].argmax(1)], V[live], atol=1e-6)
print("value-iteration OK")

import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "q-learning.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# 4x4 deterministic gridworld; actions: up, down, left, right.
# Bumping a wall stays in place. Step reward -1, goal (3,3) is terminal;
# state (1,1) is a pit with reward -10 on entry (also terminal).
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
Qs = np.zeros((S, A))             # value iteration reference
for _ in range(1000):
    V = np.where(done, 0.0, Qs.max(1))
    new = rew + gamma * V[nxt]
    new[done] = 0.0
    if np.abs(new - Qs).max() < 1e-12:
        break
    Qs = new

Q = ns["q_learning"](nxt, rew, done, gamma=gamma)
live = ~done
assert np.abs(Q[live] - Qs[live]).max() < 1e-2, np.abs(Q[live] - Qs[live]).max()
assert np.allclose(Q[done], 0)
# greedy policy achieves optimal value in every live state
assert np.allclose(Qs[live][np.arange(live.sum()), Q[live].argmax(1)], Qs[live].max(1), atol=1e-6)
print("q-learning OK")

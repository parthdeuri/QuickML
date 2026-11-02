import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gae-advantage.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
gae = ns["gae"]

rng = np.random.default_rng(0)
T = 30
rewards = rng.normal(size=T)
values = rng.normal(size=T)
last_value = rng.normal()
dones = np.zeros(T)
dones[[9, 19]] = 1            # two episode ends inside the rollout (third episode is cut off)

def td_errors(r, v, lv, d, gamma):
    out = []
    for t in range(len(r)):
        nxt = lv if t == len(r) - 1 else v[t + 1]
        out.append(r[t] + (0.0 if d[t] else gamma * nxt) - v[t])
    return out

def ref_recursion(r, v, lv, d, gamma, lam):
    """independent backward loop with explicit branches"""
    delta = td_errors(r, v, lv, d, gamma)
    adv = [0.0] * len(r)
    for t in range(len(r) - 1, -1, -1):
        if t == len(r) - 1 or d[t]:
            adv[t] = delta[t]
        else:
            adv[t] = delta[t] + gamma * lam * adv[t + 1]
    return np.array(adv)

def ref_sum(r, v, lv, d, gamma, lam):
    """definition: A_t = sum_l (gamma*lam)^l delta_{t+l}, truncated at episode end"""
    delta = td_errors(r, v, lv, d, gamma)
    adv = []
    for t in range(len(r)):
        total, l = 0.0, 0
        while t + l < len(r):
            total += (gamma * lam) ** l * delta[t + l]
            if d[t + l]:
                break
            l += 1
        adv.append(total)
    return np.array(adv)

for gamma, lam in ((0.99, 0.95), (0.9, 0.5), (1.0, 1.0), (0.95, 0.0), (0.8, 0.99)):
    adv, targets = gae(rewards, values, last_value, dones, gamma, lam)
    assert np.allclose(adv, ref_recursion(rewards, values, last_value, dones, gamma, lam))
    assert np.allclose(adv, ref_sum(rewards, values, last_value, dones, gamma, lam))
    assert np.allclose(targets, adv + values)

# lambda = 0: advantage is the TD error
adv, _ = gae(rewards, values, last_value, dones, 0.97, 0.0)
assert np.allclose(adv, td_errors(rewards, values, last_value, dones, 0.97))

# lambda = 1: Monte-Carlo advantage = discounted return to episode end (+ bootstrap if cut off) - V
gamma = 0.97
adv, _ = gae(rewards, values, last_value, dones, gamma, 1.0)
mc = []
for t in range(T):
    G, k = 0.0, 0
    while True:
        G += gamma ** k * rewards[t + k]
        if dones[t + k]:
            break
        if t + k == T - 1:                    # truncated rollout: bootstrap from last_value
            G += gamma ** (k + 1) * last_value
            break
        k += 1
    mc.append(G - values[t])
assert np.allclose(adv, mc)

# one terminal step: advantage = r - V exactly, no bootstrap
a1, _ = gae(np.array([2.0]), np.array([0.5]), 99.0, np.array([1.0]), 0.9, 0.9)
assert np.allclose(a1, [1.5])
print("gae-advantage OK")

import itertools
import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "ctc-greedy-decoding.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
collapse, greedy, loss = ns["ctc_collapse"], ns["greedy_decode"], ns["ctc_forward_loss"]

# collapse rule: letters a=1 b=2, blank 0
a, b = 1, 2
assert collapse([a, a, 0, b, 0, b, b]) == [a, b, b]   # 'aa-b-bb' -> 'abb'
assert collapse([a, a, a]) == [a]                      # repeats merge
assert collapse([a, 0, a]) == [a, a]                   # blank keeps both
assert collapse([0, 0, 0]) == []
assert collapse([b, 0, 0, a, a, 0]) == [b, a]

def norm_logp(rng, T, V):
    z = rng.normal(size=(T, V))
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)

# brute force over every length-T path that collapses to the target
rng = np.random.default_rng(0)
for T, V, target in [(4, 3, [1, 2]), (5, 3, [1, 1]), (5, 3, [2, 1, 2]), (3, 2, [1])]:
    lp = norm_logp(rng, T, V)
    total = 0.0
    for path in itertools.product(range(V), repeat=T):
        if collapse(path) == target:
            total += np.exp(sum(lp[t, p] for t, p in enumerate(path)))
    ours = loss(lp, target)
    assert abs(ours + np.log(total)) < 1e-10, (ours, -np.log(total))
    ref = F.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([target]),
                     torch.tensor([T]), torch.tensor([len(target)]),
                     blank=0, reduction="sum").item()
    assert abs(ours - ref) < 1e-9, (ours, ref)
    assert ours >= 0

# larger case against torch
lp = norm_logp(rng, 30, 6)
tg = [3, 3, 1, 5, 2]
ref = F.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([tg]), torch.tensor([30]), torch.tensor([5]), blank=0, reduction="sum").item()
assert abs(loss(lp, tg) - ref) < 1e-9

# greedy decode on peaky probabilities: 'a a - b - b b'
path = [a, a, 0, b, 0, b, b]
P = np.full((len(path), 3), 1e-6)
P[np.arange(len(path)), path] = 1.0
P /= P.sum(1, keepdims=True)
assert greedy(np.log(P)) == [a, b, b]

# loss is 0 only if one alignment has probability 1
P1 = np.zeros((2, 3)) + 1e-300
P1[0, a], P1[1, 0] = 1.0, 1.0
assert abs(loss(np.log(P1), [a])) < 1e-12
assert loss(norm_logp(rng, 2, 3), [a]) > 0
assert loss(norm_logp(rng, 6, 3), [a, b]) > 0
print("ctc-greedy-decoding OK")

import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "transfer-learning-fine-tuning.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)
x = rng.normal(size=(7, 5))
y = rng.integers(0, 3, size=7)
W1 = rng.normal(size=(5, 6))
W2 = rng.normal(size=(6, 3))
lr = 0.1

for frozen in [(), ("W1",), ("W2",)]:
    t = {"W1": torch.tensor(W1.copy()), "W2": torch.tensor(W2.copy())}
    for k in t:
        t[k].requires_grad_(k not in frozen)
    loss = F.cross_entropy(torch.relu(torch.tensor(x) @ t["W1"]) @ t["W2"], torch.tensor(y))
    loss.backward()
    with torch.no_grad():
        for k in t:
            if t[k].grad is not None:
                t[k] -= lr * t[k].grad
    new, grads = ns["finetune_step"]({"W1": W1, "W2": W2}, x, y, lr, frozen=frozen)
    for k in ("W1", "W2"):
        if k in frozen:
            assert k not in grads and t[k].grad is None
            assert np.array_equal(new[k], {"W1": W1, "W2": W2}[k])  # unchanged
        else:
            assert np.allclose(grads[k], t[k].grad.numpy()), (frozen, k)
        assert np.allclose(new[k], t[k].detach().numpy()), (frozen, k)
print("transfer-learning-fine-tuning OK")

import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "clip-contrastive-loss.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
N, D = 8, 16
img = rng.normal(size=(N, D))
txt = img + 0.8 * rng.normal(size=(N, D))      # correlated pairs


def ref_loss(img, txt, tau):
    i = F.normalize(torch.tensor(img), dim=-1)
    t = F.normalize(torch.tensor(txt), dim=-1)
    logits = i @ t.T / tau
    labels = torch.arange(len(img))
    li, lt = F.cross_entropy(logits, labels), F.cross_entropy(logits.T, labels)
    return (li + lt).item() / 2, li.item(), lt.item()


for tau in (0.07, 0.5, 1.0):
    out = ns["clip_loss"](img, txt, tau)
    ref = ref_loss(img, txt, tau)[0]
    assert np.isclose(out, ref), (tau, out, ref)

# matched pairs score lower loss than shuffled pairs
assert ns["clip_loss"](img, txt) < ns["clip_loss"](img, txt[rng.permutation(N)])
# invariant to embedding scale because of normalisation
assert np.isclose(ns["clip_loss"](3 * img, 0.2 * txt), ns["clip_loss"](img, txt))
print("clip-contrastive-loss OK")

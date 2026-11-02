import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bert-masked-lm.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
mask_tokens, mlm_loss = ns["mask_tokens"], ns["mlm_loss"]

# --- masking statistics (seeded; tolerances ~4-5 standard errors)
V, MASK = 1000, 0
rng = np.random.default_rng(0)
ids = rng.integers(1, V, size=(400, 500))           # 200k ordinary tokens (never the mask id)
new, labels = mask_tokens(ids, V, MASK, np.random.default_rng(1))
sel = labels != -100
n, ns_ = ids.size, sel.sum()
assert abs(sel.mean() - 0.15) < 0.004               # binomial se ~ 0.0008
assert np.array_equal(labels[sel], ids[sel])
assert np.array_equal(new[~sel], ids[~sel])         # unselected tokens untouched
is_mask = new[sel] == MASK
unchanged = new[sel] == ids[sel]
# random replacements can land on MASK (1/V) or on the original (1/V): negligible, covered by tolerance
assert abs(is_mask.mean() - 0.8) < 0.01, is_mask.mean()                 # se ~ 0.0023
assert abs(unchanged.mean() - 0.1) < 0.01, unchanged.mean()
random_rep = ~is_mask & ~unchanged
assert abs(random_rep.mean() - 0.1) < 0.01, random_rep.mean()
assert abs(is_mask.mean() + unchanged.mean() + random_rep.mean() - 1) < 1e-12

# --- loss on masked positions only vs F.cross_entropy(ignore_index=-100)
B, T, Vs = 6, 16, 50
r = np.random.default_rng(2)
ids_s = r.integers(1, Vs, size=(B, T))
_, lab = mask_tokens(ids_s, Vs, MASK, np.random.default_rng(3), p=0.3)
assert (lab != -100).sum() > 0
logits = r.normal(size=(B, T, Vs)) * 3
mine = mlm_loss(logits, lab)
ref = F.cross_entropy(torch.tensor(logits).view(-1, Vs), torch.tensor(lab).view(-1),
                      ignore_index=-100)
assert abs(mine - ref.item()) < 1e-12, (mine, ref.item())
# logits at unselected positions do not matter
l2 = logits.copy()
l2[lab == -100] = r.normal(size=(int((lab == -100).sum()), Vs)) * 100
assert abs(mlm_loss(l2, lab) - mine) < 1e-12
print("bert-masked-lm OK")

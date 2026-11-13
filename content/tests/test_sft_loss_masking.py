import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "sft-loss-masking.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# build_labels: prompt and padding are -100, response kept, padded to length
ids, lab = ns["build_labels"]([5, 6, 7], [8, 9], 7)
assert ids == [5, 6, 7, 8, 9, 0, 0]
assert lab == [-100, -100, -100, 8, 9, -100, -100]

rng = np.random.default_rng(0)
V, T = 11, 9
plens, rlens = [2, 4, 3], [5, 1, 3]
rows = [ns["build_labels"](list(rng.integers(1, V, p)), list(rng.integers(1, V, r)), T)
        for p, r in zip(plens, rlens)]
labels = np.array([r[1] for r in rows])
logits = rng.normal(size=(3, T, V))

# token mode equals torch cross_entropy with ignore_index=-100 (shifted)
lt, tt = torch.tensor(logits), torch.tensor(labels)
ref = F.cross_entropy(lt[:, :-1].reshape(-1, V), tt[:, 1:].reshape(-1), ignore_index=-100)
tok = ns["sft_loss"](logits, labels, "token")
assert np.isclose(tok, ref.item(), atol=1e-12), (tok, ref.item())

# sequence mode equals the mean of per-sequence torch losses
per = [F.cross_entropy(lt[b, :-1], tt[b, 1:], ignore_index=-100).item() for b in range(3)]
seq = ns["sft_loss"](logits, labels, "sequence")
assert np.isclose(seq, np.mean(per), atol=1e-12)

# unequal response lengths: the two reductions differ
assert abs(tok - seq) > 1e-3
# equal response lengths: they agree
rows2 = [ns["build_labels"](list(rng.integers(1, V, p)), list(rng.integers(1, V, 4)), T)
         for p in (1, 3, 2)]
lab2 = np.array([r[1] for r in rows2])
lg2 = rng.normal(size=(3, T, V))
assert np.isclose(ns["sft_loss"](lg2, lab2, "token"), ns["sft_loss"](lg2, lab2, "sequence"))

# masked positions get no gradient influence: changing prompt logits keeps loss
lg3 = logits.copy()
lg3[0, 0] += 100.0  # position 0 of row 0 predicts a prompt token (label -100)
assert np.isclose(ns["sft_loss"](lg3, labels, "token"), tok)

# packed block mask: causal inside blocks, nothing across blocks
lengths = [3, 2, 4]
M = ns["packed_block_mask"](lengths)
n = sum(lengths)
assert M.shape == (n, n)
starts = np.cumsum([0] + lengths)
for b, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
    blk = M[s:e, s:e]
    assert (blk == np.tril(np.ones((e - s, e - s), bool))).all()
    assert not M[s:e, :s].any() and not M[s:e, e:].any()
assert M.diagonal().all()
print("sft-loss-masking OK")

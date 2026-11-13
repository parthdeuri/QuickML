import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "vlm-projector-image-tokens.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
H, W, C, P = 12, 16, 3, 4
img = rng.normal(size=(H, W, C))
patches = ns["patchify"](img, P)
N = (H // P) * (W // P)
assert patches.shape == (N, P * P * C) == (12, 48)

# patchify equals F.unfold up to layout: unfold orders each patch (C, P, P),
# ours orders it (P, P, C); both scan patches row by row
u = F.unfold(torch.tensor(img).permute(2, 0, 1)[None], P, stride=P)[0].T.numpy()
mine = patches.reshape(N, P, P, C).transpose(0, 3, 1, 2).reshape(N, -1)
assert np.allclose(u, mine)

# projector equals an nn.Sequential with the same weights
dv, dh, dl = 48, 20, 10
W1, b1 = rng.normal(size=(dv, dh)), rng.normal(size=dh)
W2, b2 = rng.normal(size=(dh, dl)), rng.normal(size=dl)
feats = rng.normal(size=(N, dv))
net = torch.nn.Sequential(torch.nn.Linear(dv, dh), torch.nn.GELU(), torch.nn.Linear(dh, dl)).double()
with torch.no_grad():
    net[0].weight.copy_(torch.tensor(W1.T)); net[0].bias.copy_(torch.tensor(b1))
    net[2].weight.copy_(torch.tensor(W2.T)); net[2].bias.copy_(torch.tensor(b2))
z = ns["project"](feats, W1, b1, W2, b2)
assert z.shape == (N, dl)
assert np.allclose(z, net(torch.tensor(feats)).detach().numpy(), atol=1e-10)

# pooling: N / k^2 tokens, each the mean of its k x k group
gh, gw, k = 6, 8, 2
grid = rng.normal(size=(gh, gw, dl))
pooled = ns["pool_tokens"](grid, k)
assert pooled.shape == (gh * gw // k**2, dl) == (12, 10)
for a in range(gh // k):
    for b in range(gw // k):
        grp = grid[a*k:(a+1)*k, b*k:(b+1)*k].reshape(-1, dl)
        assert np.allclose(pooled[a * (gw // k) + b], grp.mean(0))

# build_inputs: image tokens first, then text, shapes add up
text = rng.normal(size=(5, dl))
x = ns["build_inputs"](pooled, text)
assert x.shape == (12 + 5, dl)
assert np.array_equal(x[:12], pooled) and np.array_equal(x[12:], text)

# token_count on a grid of sizes
assert ns["token_count"](336, 336, 14) == 576
assert ns["token_count"](336, 336, 14, 2) == 144
for Hh, Ww, Pp, kk in [(224, 224, 16, 1), (224, 224, 16, 2), (448, 224, 14, 4), (96, 64, 8, 2)]:
    n = (Hh // Pp) * (Ww // Pp)
    assert ns["token_count"](Hh, Ww, Pp, kk) == n // kk**2
assert ns["token_count"](224, 224, 16) == 196
print("vlm-projector-image-tokens OK")

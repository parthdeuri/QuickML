import numpy as np, yaml, torch
import torch.nn.functional as F
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "vision-transformer-patch-embed.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
C, H, W_, P, D = 3, 8, 12, 4, 10
img = rng.normal(size=(C, H, W_))
N = (H // P) * (W_ // P)
Wm = rng.normal(size=(C * P * P, D))
b = rng.normal(size=D)
cls = rng.normal(size=D)
pos = rng.normal(size=(N + 1, D))

# 1) patchify vs F.unfold (channel-major flatten, row-major patch grid)
patches = ns["patchify"](img, P)
unf = F.unfold(torch.tensor(img)[None], kernel_size=P, stride=P)[0].T.numpy()
assert patches.shape == (N, C * P * P) and np.allclose(patches, unf)

# 2) whole embedding vs Conv2d(stride = kernel = P) with copied weights
conv = torch.nn.Conv2d(C, D, P, stride=P).double()
with torch.no_grad():
    conv.weight.copy_(torch.tensor(Wm.T.reshape(D, C, P, P)))
    conv.bias.copy_(torch.tensor(b))
tok = conv(torch.tensor(img)[None]).flatten(2).transpose(1, 2)[0]
ref = torch.cat([torch.tensor(cls)[None], tok]) + torch.tensor(pos)
out = ns["patch_embed"](img, P, Wm, b, cls, pos)
assert out.shape == (N + 1, D)
assert np.allclose(out, ref.detach().numpy())
assert np.allclose(out[0], cls + pos[0])
print("vision-transformer-patch-embed OK")
